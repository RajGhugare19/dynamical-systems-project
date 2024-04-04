from automata import gather_rule_trajectories, plot_traj
import numpy as np

def main(params):
    file_name_train = "automata-state_size-{}-traj_len-{}-rule-{}-train1.csv".format(str(params["state_size"]),
                                                                              str(params["traj_len"]),
                                                                              str(params["rule"]))
    file_name_val = "automata-state_size-{}-traj_len-{}-rule-{}-val.csv".format(str(params["state_size"]),
                                                                              str(params["traj_len"]),
                                                                              str(params["rule"]))
    file_name_test = "automata-state_size-{}-traj_len-{}-rule-{}-test.csv".format(str(params["state_size"]),
                                                                              str(params["traj_len"]),
                                                                              str(params["rule"]))
    trajs = gather_rule_trajectories(state_size=params["state_size"],
                                     traj_len=params["traj_len"],
                                     rule=params["rule"],
                                     seed_list=params["seed_list"])
    trajs = np.asarray(trajs)
    trajs = trajs.reshape(-1, trajs.shape[-1])
    print(trajs.shape)
    num_samples = params["traj_len"]*len(params["seed_list"])
    
    train_end = int(num_samples*0.7)
    np.savetxt(file_name_train, trajs[:train_end], delimiter=",", fmt='%1.0f')
    print("train shape: ", trajs[:train_end].shape)
    
    val_end = train_end + int(num_samples*0.25)
    np.savetxt(file_name_val, trajs[train_end: val_end], delimiter=",", fmt='%1.0f')
    print("val shape: ", trajs[train_end: val_end].shape)

    
    np.savetxt(file_name_test, trajs[val_end:], delimiter=",", fmt='%1.0f')
    print("test shape: ", trajs[val_end:].shape)
    # plot_traj(traj=trajs[0:params["traj_len"]])    
    
if __name__ in "__main__":
    # TODO: this should not be a dict. This should be just command line params from the script.
    params = {"state_size": 10,
              "traj_len": 50, 
              "rule": 1,
              "seed_list": [i for i in range(5000)]}
    main(params)

