# This is in a rough draft stage
import matplotlib.pyplot as plt
import uuid
import sys
import numpy as np
from pathlib import Path
import automata as atmt
import sys
import json
import random
import optuna
import itertools
sys.path.append("..")
import autokoopman.core.trajectory as atraj
import pandas as pd
import autokoopman.estimator.deepkoopman as dk

'''
Class 1: Cellular automata which rapidly converge to a uniform state. Examples are rules 0, 32, 160 and 232.
Class 2: Cellular automata which rapidly converge to a repetitive or stable state. Examples are rules 4, 108, 218 and 250.
Class 3: Cellular automata which appear to remain in a random state. Examples are rules 22, 30, 126, 150, 182.
Class 4: Cellular automata which form areas of repetitive or stable states, but also form structures that interact
with each other in complicated ways. An example is rule 110. Rule 110 has been shown to be capable of universal computation.[4]
'''

def to_autokoopman_traj_data(state_size, traj_len, trajs):
  state_part_name = ["x"+str(i+1) for i in range(state_size)]
  cols = ["time"] + state_part_name + ["id"]
  data = []
  time_arr = np.array([j for j in range(traj_len)])

  for t in range(1, trajs.shape[0]+1):
    curr_traj = [trajs[t-1, :, k] for k in range(trajs.shape[2])]
    col_vals = [time_arr]+ curr_traj +[np.ones(shape=time_arr.shape)*t]
    row = np.array(col_vals).T
    data.append(row)

  data = np.concatenate(data)

  my_df = pd.DataFrame(
      columns=cols,
      data=np.array(data)
  )

  autokoopman_data = atraj.UniformTimeTrajectoriesData.from_pandas(my_df)
  print("Shape:", data.shape)
  return autokoopman_data


def generate_data(state_size, traj_len, rule):

    # shuffle with a fixed seed
    start_state_permutations = list(itertools.product([0, 1], repeat=state_size))
    random.Random(0).shuffle(start_state_permutations) # this is fine because our state space is usually small enough
    start_state_permutations = np.array(start_state_permutations)
    print(start_state_permutations.shape)
    end_train = int(start_state_permutations.shape[0]*0.65) 
    end_val = end_train + int(start_state_permutations.shape[0]*0.20)
    
    train_perms = start_state_permutations[:end_train]
    val_perms = start_state_permutations[end_train: end_val]
    test_perms = start_state_permutations[end_val:]
    
    print("unique train: ", len(np.unique(train_perms)))
    print("unique val: ", len(np.unique(val_perms)))
    print("unique test: ", len(np.unique(test_perms)))
    
    
    train_trajs = np.array(atmt.gather_rule_trajs_from_starts(starts=train_perms, traj_len=traj_len, rule=rule))
    val_trajs = np.array(atmt.gather_rule_trajs_from_starts(starts=val_perms, traj_len=traj_len, rule=rule))
    test_trajs = np.array(atmt.gather_rule_trajs_from_starts(starts=test_perms, traj_len=traj_len, rule=rule))

    print(train_trajs.shape, val_trajs.shape, test_trajs.shape)
    train_data = to_autokoopman_traj_data(state_size, traj_len, train_trajs)
    val_data = to_autokoopman_traj_data(state_size, traj_len, val_trajs)
    test_data = to_autokoopman_traj_data(state_size, traj_len, test_trajs)
    return train_data, val_data, test_data


def loss_val_report(koop: dk.DeepKoopman, exp_id, results_dir):
    plt.figure(figsize=(10, 8))
    plt.plot(koop.loss_hist['validation_lin_loss'], label="linearity loss")
    plt.plot(koop.loss_hist['validation_pred_loss'], label="prediction loss")
    plt.plot(koop.loss_hist['validation_total_loss'], label="total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Deep Learning Loss")
    plt.legend()
    plt.grid()
    plt.savefig("{}/loss_report_id_{}.pdf".format(results_dir, exp_id))
    

def record_experiment_details(exp_id, exp_params: dict, results_dir):
    with open('{}/experiment_params_id_{}.json'.format(results_dir, exp_id), 'w') as fp:
        json.dump(exp_params, fp)
    

def train(train_data, val_data, state_size, traj_len, rule, results_dir):
    def objective(trial):        
        exp_id = uuid.uuid4()
        
        hidden_enc_dim = trial.suggest_categorical("hidden_enc_dim", [64, 128, 256, 512])
        hidden_dim_options = [256, 512, 1024, 2048]
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_options)
        lr = trial.suggest_float("lr", 1e-4, 1e-2)
        num_hidden_layers = trial.suggest_categorical("num_hidden_layers", [1,2])
        print("hidden_dim: ", hidden_dim)
        print("lr: ", lr)
        print("hidden_enc_dim: ", hidden_enc_dim)
        print("num_hidden_layers: ", num_hidden_layers)
        
        exp_params = {"state_size": state_size,
                  "traj_len": traj_len,
                  "rule":rule,
                  "max_iter": 6000,
                  "hidden_dim": hidden_dim,
                  "hidden_enc_dim": hidden_enc_dim,
                  "lr": lr,
                  "num_hidden_layers": num_hidden_layers,
                  "metric_loss_weight": 1e-2,
                  "pred_loss_weight": 1e-2,
                  }

        koop = dk.DeepKoopman(
        state_dim=len(train_data.state_names),
        input_dim=0,
        hidden_dim=exp_params['hidden_dim'],
        max_iter=exp_params['max_iter'],
        lr=exp_params['lr'],
        hidden_enc_dim=exp_params['hidden_enc_dim'],
        num_hidden_layers=exp_params['num_hidden_layers'],
        metric_loss_weight=exp_params['metric_loss_weight'],
        pred_loss_weight=exp_params['pred_loss_weight'],
        validation_data=val_data
        )
        record_experiment_details(exp_id=exp_id, exp_params=exp_params, results_dir=results_dir)
        koop.fit(train_data) 
        print("Start val: ", koop.loss_hist['validation_total_loss'][0])
        print("End val: ", koop.loss_hist['validation_total_loss'][-1])
        val_loss = koop.loss_hist['validation_total_loss'][-1]
        loss_val_report(koop, exp_id, results_dir)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)
    
    return study.best_params, study.best_value
    
    # koop = dk.DeepKoopman(
    #     state_dim=len(train_data.state_names),
    #     input_dim=0,
    #     hidden_dim=hidden_dim,
    #     max_iter=max_iter,
    #     lr=lr,
    #     hidden_enc_dim=hidden_enc_dim,
    #     num_hidden_layers=num_hidden_layers,
    #     metric_loss_weight=metric_loss_weight,
    #     pred_loss_weight=pred_loss_weight,
    #     validation_data=val_data
    # )
    # koop.fit(train_data)
    # return koop



def run_experiment(exp_name, rule):
    # exp_params = {"state_size": 6,
    #               "traj_len": 50,
    #               "rule":22,
    #               "max_iter": 6000,
    #               "hidden_dim": 256,
    #               "hidden_enc_dim": 128,
    #               "lr": 1e-3,
    #               "num_hidden_layers": 2,
    #               "metric_loss_weight": 1e-2,
    #               "pred_loss_weight": 1e-2,
    #               }
    
    # class 1: 0
    # class 2: 108
    # class 3: 22
    # class 4: 110
    
    results_dir = "./experiment_outputs/{}".format(exp_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    data_params = {"state_size": 6,
                   "traj_len": 50,
                   "rule": rule}
    
    
    # record_experiment_details(exp_id=exp_id, exp_params=exp_params)
    # TODO: need to save test data or separate data generation from actual training.
    # for now I don't care too much to do ad-hoc experiments mvp
    train_data, val_data, _ = generate_data(state_size=data_params["state_size"], traj_len=data_params["traj_len"],
                                            rule=data_params["rule"])
    
    
    best_params, best_val_loss = train(results_dir=results_dir, train_data=train_data, val_data=val_data,
                 state_size=data_params["state_size"], traj_len=data_params["traj_len"],
                 rule = data_params["rule"])
    
    print("Best params: ", best_params)
    print("Best val loss: ", best_val_loss)
    with open('{}/optuna_result_state_{}_trajlen_{}_rule_{}.json'.format(results_dir, str(data_params['state_size']),
                                                                      str(data_params['traj_len']),
                                                                      str(data_params['rule'])), 'w') as f:
        json.dump({"best_params":best_params,
                   "best_val": best_val_loss}, f) 
    
    
def main():
    exp_name = sys.argv[1]
    rule = int(sys.argv[2])
    run_experiment(exp_name, rule)

if __name__ in "__main__":
    main()