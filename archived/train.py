import torch
import random
import numpy as np

from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.slim import slim
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer. modules import blocks

def get_data(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    ny = sys.ny
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainY = train_sim['Y'][:length].reshape(nbatch, nsteps, ny)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    train_data = DictDataset({'Y': trainY, 'Y0': trainY[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devY = dev_sim['Y'][:length].reshape(nbatch, nsteps, ny)
    devY = torch.tensor(devY, dtype=torch.float32)
    dev_data = DictDataset({'Y': devY, 'Y0': devY[:, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testY = test_sim['Y'][:length].reshape(1, nsim, ny)
    testY = torch.tensor(testY, dtype=torch.float32)
    test_data = {'Y': testY, 'Y0': testY[:, 0:1, :], 'name': 'test'}

    return train_loader, dev_loader, test_data

def get_model(ny, nx_koopman, n_hidden, n_layers):
    # instantiate encoder neural net
    encode = blocks.MLP(ny, nx_koopman, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ELU,
                    hsizes=n_layers*[n_hidden])
    # initial condition encoder: output of which to be used to initialize the Koopman operator rollout
    encode_Y0 = Node(encode, ['Y0'], ['x'], name='encoder_Y0')
    # observed trajectory encoder: output of which to be used in latent trajectory prediction loss
    encode_Y = Node(encode, ['Y'], ['x_traj'], name='encoder_Y')

    # instantiate decoder neural net
    decode = blocks.MLP(nx_koopman, ny, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ELU,
                    hsizes=n_layers*[n_hidden])
    # reconstruction decoder: output of which to be used in encoder-decoder reconstruction loss
    decode_y0 = Node(decode, ['x'], ['yhat0'], name='decoder_y0')
    # predicted trajectory decoder: output of which to be used in output trajectory prediction loss
    decode_y = Node(decode, ['x'], ['yhat'], name='decoder_y')

    return encode, encode_Y0, encode_Y, decode, decode_y0, decode_y

def get_koopman_operator(stable):
    # instantiate Koopman operator matrix
    if stable:
        # SVD factorized Koopman operator with bounded eigenvalues: sigma_min <= \lambda_i <= sigma_max
        K = slim.linear.SVDLinear(nx_koopman, nx_koopman,
                            sigma_min=0.01, sigma_max=1.0, bias=False)
        # SVD penalty variable
        K_reg_error = variable(K.reg_error())
        # SVD penalty loss term
        K_reg_loss = 1.*(K_reg_error == 0.0)
        K_reg_loss.name = 'SVD_loss'

        return K, K_reg_loss
    
    else:
        # linear Koopman operator without guaranteed stability
        K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
    
        return K, None

# Seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

nsim = 2000   # number of simulation steps in the dataset
nsteps = 20   # number of prediction horizon steps in the loss function
bs = 100      # minibatching batch size

# Define system ground truth system
system = psl.systems['VanDerPol']
modelSystem = system()
ts = modelSystem.ts
nx = modelSystem.nx
ny = modelSystem.ny

train_loader, dev_loader, test_data = get_data(modelSystem, nsim, nsteps, ts, bs)

# model parameters
nx_koopman = 50
n_hidden = 60
n_layers = 2

encode, encode_Y0, encode_Y, decode, decode_y0, decode_y = get_model(ny, nx_koopman, n_hidden, n_layers)

# if True then provably stable Koopman operator
stable = True     

K, K_reg_loss = get_koopman_operator(stable)

# symbolic Koopman model
Koopman = Node(K, ['x'], ['x'], name='K')

# latent Koopmann rollout over nsteps long prediction horizon
dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

# put all nodes of the Koopman model together in a list of nodes
nodes = [encode_Y0, decode_y0, encode_Y, dynamics_model, decode_y]

# %% Constraints + losses:
Y = variable("Y")                      # observed outputs from the dataset
Y0 = variable('Y0')                    # observed initial conditions from the dataset
yhat = variable('yhat')                # predicted output by the encoder-decoder Koopman operator
yhat0 = variable('yhat0')              # reconstructed initial conditions by the encoder-decoder
x_traj = variable('x_traj')            # encoded trajectory in the latent space: \phi_{\theta_1}(y_{k+1}) 
x = variable('x')                      # Koopman latent space trajectory: K^k\phi_{\theta_1}(y_1)

# output trajectory prediction loss
y_loss = 10.*(yhat[:, 1:-1, :] == Y[:, 1:, :])^2
y_loss.name = "y_loss"

# one-step  output prediction loss
onestep_loss = 1.*(yhat[:, 1, :] == Y[:, 1, :])^2
onestep_loss.name = "onestep_loss"

# latent trajectory prediction loss
x_loss = 1.*(x[:, 1:-1, :] == x_traj[:, 1:, :])^2
x_loss.name = "x_loss"

# encoder-decoder reconstruction loss
reconstruct_loss = 1.*(Y0 == yhat0)^2
reconstruct_loss.name = "reconstruct_loss"

# aggregate list of objective terms and constraints
objectives = [y_loss, x_loss, onestep_loss, reconstruct_loss]
if stable:
    objectives.append(K_reg_loss)

# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints=[])

# construct constrained optimization problem
problem = Problem(nodes, loss)

optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout=['dev_loss', 'train_loss'])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=50,
    warmup=100,
    epochs=1000,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
)

# %% train
best_model = trainer.train()
problem.load_state_dict(best_model)

# update the rollout length based on the test data
problem.nodes[3].nsteps = test_data['Y'].shape[1]

# Test set results
test_outputs = problem.step(test_data)

pred_traj = test_outputs['yhat'][:, 1:-1, :].detach().numpy().reshape(-1, nx).T
true_traj = test_data['Y'][:, 1:, :].detach().numpy().reshape(-1, nx).T

# plot trajectories
figsize = 25
fig, ax = plt.subplots(nx, figsize=(figsize, figsize))
labels = [f'$y_{k}$' for k in range(len(true_traj))]
for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
    if nx > 1:
        axe = ax[row]
    else:
        axe = ax
    axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
    axe.plot(t1, 'c', linewidth=4.0, label='True')
    axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
    axe.tick_params(labelbottom=False, labelsize=figsize)
axe.tick_params(labelbottom=True, labelsize=figsize)
axe.legend(fontsize=figsize)
axe.set_xlabel('$time$', fontsize=figsize)
plt.tight_layout()
plt.show()