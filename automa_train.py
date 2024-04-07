
# This is in a rough draft stage
import matplotlib.pyplot as plt
import uuid
import numpy as np
import automata as atmt
import sys
import json
sys.path.append("..")
import autokoopman.core.trajectory as atraj
import pandas as pd
import autokoopman.estimator.deepkoopman as dk



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
    train_trajs = np.array(atmt.gather_rule_trajectories(state_size=state_size,
                                        traj_len=traj_len,
                                        rule=rule,
                                        seed_list=[i for i in range(1000)]))

    val_trajs = np.array(atmt.gather_rule_trajectories(state_size=state_size,
                                        traj_len=traj_len,
                                        rule=rule,
                                        seed_list=[i for i in range(1000, 1350)]))

    test_trajs = np.array(atmt.gather_rule_trajectories(state_size=state_size,
                                        traj_len=traj_len,
                                        rule=rule,
                                        seed_list=[i for i in range(1350, 1500)]))

    print(train_trajs.shape, val_trajs.shape, test_trajs.shape)
    train_data = to_autokoopman_traj_data(state_size, traj_len, train_trajs)
    val_data = to_autokoopman_traj_data(state_size, traj_len, val_trajs)
    test_data = to_autokoopman_traj_data(state_size, traj_len, test_trajs)
    return train_data, val_data, test_data


def train(train_data, val_data, hidden_dim, hidden_enc_dim, max_iter, lr, num_hidden_layers=2,
          metric_loss_weight = 1e-2, pred_loss_weight=1e-2):
    
    koop = dk.DeepKoopman(
        state_dim=len(train_data.state_names),
        input_dim=0,
        hidden_dim=hidden_dim,
        max_iter=max_iter,
        lr=lr,
        hidden_enc_dim=hidden_enc_dim,
        num_hidden_layers=num_hidden_layers,
        metric_loss_weight=metric_loss_weight,
        pred_loss_weight=pred_loss_weight,
        validation_data=val_data
    )
    koop.fit(train_data)
    return koop

def loss_val_report(koop: dk.DeepKoopman, exp_id):
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
    plt.savefig("./experiment_outputs/loss_report_id_{}.pdf".format(exp_id))
    

def record_experiment_details(exp_id, exp_params: dict):
    with open('./experiment_outputs/experiment_params_id_{}.json'.format(exp_id), 'w') as fp:
        json.dump(exp_params, fp)
    


def run_experiment():
    exp_params = {"state_size": 6,
                  "traj_len": 50,
                  "rule":110,
                  "max_iter": 6000,
                  "hidden_dim": 256,
                  "hidden_enc_dim": 128,
                  "lr": 1e-3,
                  "num_hidden_layers": 2,
                  "metric_loss_weight": 1e-2,
                  "pred_loss_weight": 1e-2,
                  }
    
    exp_id = uuid.uuid4()
    record_experiment_details(exp_id=exp_id, exp_params=exp_params)
    # TODO: need to save test data or separate data generation from actual training.
    # for now I don't care too much to do ad-hoc experiments mvp
    train_data, val_data, _ = generate_data(state_size=exp_params["state_size"], traj_len=exp_params["traj_len"],
                                            rule=exp_params["rule"])
    
    koop = train(train_data, val_data, hidden_dim=exp_params["hidden_dim"],
          hidden_enc_dim=exp_params["hidden_enc_dim"], max_iter=exp_params["max_iter"],
          lr=exp_params["lr"], num_hidden_layers=exp_params["num_hidden_layers"],
          metric_loss_weight=exp_params["metric_loss_weight"],
          pred_loss_weight=exp_params["pred_loss_weight"])
    
    loss_val_report(koop=koop, exp_id=exp_id)
    
def main():
    run_experiment()

main()
    

    
