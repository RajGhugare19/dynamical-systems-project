import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
import pykoopman as pk
import pandas as pd
from pathlib import Path
import optuna
from tqdm import tqdm
import random
from copy import deepcopy
import itertools
import cellpylib as cpl
import numpy.random as rnd
from sklearn.metrics import mean_squared_error
np.random.seed(42)  # for reproducibility
import warnings
warnings.filterwarnings('ignore')
from automata import gather_rule_trajs_from_starts
import pickle as pkl

def num_unique_local_operations(trajs):
    neighbourhoods = list(itertools.product([0.0, 1.0], repeat=3)) # 3 bcs we consider neighbourhoods of size 3
    local_to_count = {s:0 for s in neighbourhoods}
    
    for tr in trajs:
        for tr_state in tr:
            str_state = "".join(str(int(n)) for n in tr_state)
            for key in local_to_count:
                substr = "".join(list(str(int(j)) for j in key))
                c = str_state.count(substr)
                local_to_count[key] += c
                
    return local_to_count

def get_arrays_bigger_than(all_arrs, size_filter):
    final_arrays = []
    for a in all_arrs:
        if(len(a) > size_filter):
            final_arrays.append(a)
    return final_arrays
        

def slice_array_at_indices(arr, indices, filter_trajs_len=1):
    final_list = []
    indices = sorted(list(set(indices)))
    
    prev_end = -1
    i = -1
    for i in indices:
        if(prev_end+1 < i): 
            final_list.append(arr[prev_end+1: i])
        prev_end = i
    
    if(len(arr[i+1:]) != 0):
        final_list.append(arr[i+1:]) # add rest of it
    
    # remove lists of len 1 or less
    final_list = get_arrays_bigger_than(final_list, 1)
    
    return final_list
            

def generate_data(state_size, traj_len, rule, compo_test_type, results_dir):

    # shuffle with a fixed seed
    start_state_permutations = list(itertools.product([0.0, 1.0], repeat=state_size))
    random.Random(0).shuffle(start_state_permutations) # this is fine because our state space is usually small enough
    end_train = int(len(start_state_permutations)*0.65) 
    end_val = end_train + int(len(start_state_permutations)*0.20)
    
    train_perms = start_state_permutations[:end_train]
    val_perms = start_state_permutations[end_train: end_val]
    test_perms = start_state_permutations[end_val:]
    
    print("Num train trajs: ", len(train_perms))
    print("Num val trajs: ", len(val_perms))
    print("Num test trajs: ", len(test_perms))
    
    
    # print("train starts: ", train_perms)
    # print("val starts: ", val_perms)
    # print("test starts: ", test_perms)
    
    train_trajs = gather_rule_trajs_from_starts(starts=train_perms, traj_len=traj_len, rule=rule)
    # val_states = [np.array(tup) for tup in val_perms]
    # test_states = [np.array(tup) for tup in test_perms]
    val_trajs = gather_rule_trajs_from_starts(starts=val_perms, traj_len=2, rule=rule)
    test_trajs = gather_rule_trajs_from_starts(starts=test_perms, traj_len=2, rule=rule)
    
    local_to_count = num_unique_local_operations(train_trajs).values()
    print(">>>>>>> Local to count", local_to_count)

    # check for local operation coverage
    if(0 in local_to_count):
        raise NotImplementedError("We did not implement a way to deal with a training set that does not cover all local evolution operations. This is the case here.\
                                  Pick another automata rule to generate a different dataset hopefully without this problem. That should be the case for all rules defined in cellpylib.")
    else:
        print("Check for local evolution operation coverage succeeded for rule {}!".format(str(rule)))
    
    # TODO slice train trajectories so that start states in test and val do not occur in train.
    # pykoopman can deal with ariable length trajectories, so this is fine
    
    print("Running compositionality-aware dataset pre-processing...")
    val_test_checking_set = set(val_perms).union(set(test_perms))
    
    
    if(compo_test_type == "weak"):
        final_train_trajs = []
        
        # go through every train traj
        for tr_idx in tqdm(range(len(train_trajs))):
            tr = train_trajs[tr_idx]
            traj_slice_indices = []
            # in traj, go through every state
            for s_idx in range(len(tr)):  
                # check if state it occurs in test or val start states
                if(tuple(tr[s_idx]) in val_test_checking_set):
                    # if yes, store index to later slice train traj into two sections: before state and after state.
                    traj_slice_indices.append(s_idx)
            
                    
            final_train_trajs.extend(slice_array_at_indices(tr, traj_slice_indices))
            
        
        with open("{}/filtered_train_trajs.pkl".format(results_dir), 'wb') as outfile:
            pkl.dump(final_train_trajs, outfile)
        
        with open("{}/val_states.pkl".format(results_dir), 'wb') as outfile:
            pkl.dump(val_trajs, outfile)
        
        with open("{}/test_states.pkl".format(results_dir), 'wb') as outfile:
            pkl.dump(test_trajs, outfile)
            
            
            # this is more for sanity checking
            # with open('train_trajs.csv', 'w') as outfile:
            #     for slice_2d in train_trajs:
            #         np.savetxt(outfile, np.asarray(slice_2d), fmt='%d')
            #         np.savetxt(outfile, np.asarray(["-------"]), fmt='%s')

            # with open('final_train_trajs.csv', 'w') as outfile:
            #     for slice_2d in final_train_trajs:
            #         np.savetxt(outfile, np.asarray(slice_2d), fmt='%d')
            #         np.savetxt(outfile, np.asarray(["-------"]), fmt='%s')
            
            # with open('val_start_states.csv', 'w') as outfile:
            #     np.savetxt(outfile, np.asarray(val_perms), fmt='%d')
                    
            # with open('test_start_states.csv', 'w') as outfile:
            #     np.savetxt(outfile, np.asarray(test_perms), fmt='%d')
            
        # Add each section as a separate trajectory in final train list
    elif(compo_test_type == "strong"):
        pass
    
    else:
        raise NotImplementedError("You did not choose a compo test type that actually exists")
        
    
    # TODO make trajectories noisy (Maybe)

    return final_train_trajs, val_trajs, test_trajs


def one_step_mse(dlk_regressor, val_trajs):
    mse_list = []
    for i in range(len(val_trajs)):
        true_val = val_trajs[i]
        pred_val = dlk_regressor.simulate(true_val[0][np.newaxis, :], n_steps = 1)
        mse = mean_squared_error(pred_val[1], true_val[1])
        mse_list.append(mse)
    
    avg_mse = sum(mse_list)/float(len(mse_list))
    return avg_mse, mse_list


'''
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "elu": nn.ELU(),
            "mish": nn.Mish(),
            "linear": nn.Identity(),
'''

def train(train_data, val_data, look_forward, intrinsic_cords, hidden_dim, num_layers, batch_size=64, max_epochs=25, patience=5):
    state_size = train_data[0].shape[1]
    dlk_regressor = pk.regression.NNDMD(dt=1.0, look_forward=look_forward,
                                        config_encoder=dict(input_size=state_size,
                                                            hidden_sizes=[hidden_dim] * num_layers,
                                                            output_size=intrinsic_cords,
                                                            activations="relu"),
                                        config_decoder=dict(input_size=intrinsic_cords, hidden_sizes=[hidden_dim] * num_layers,
                                                            output_size=state_size, activations="linear"),
                                        batch_size=batch_size, lbfgs=False,
                                        normalize=True, normalize_mode='equal',
                                        normalize_std_factor=1.0,
                                        trainer_kwargs=dict(max_epochs=1)) # keep max_epochs here as 1. we want to validate after each epoch as done below
    
    inc = 0
    avg_mse_val = np.inf
    best_val = np.inf
    
    for epoch in range(max_epochs):
        
        print(">>>>> On Epoch num : ", epoch)
        prev_mse_val = avg_mse_val
        dlk_regressor.fit(train_data)
        avg_mse_train, _ = one_step_mse(dlk_regressor, train_data)
        avg_mse_val, _ = one_step_mse(dlk_regressor, val_data)
        print("Train avg 1 step MSE: ", avg_mse_train)
        print("Val avg 1 step MSE: ", avg_mse_val)
        
        if(best_val > avg_mse_val):
            best_val = avg_mse_val
            best_regressor = deepcopy(dlk_regressor)
            
            
        if(avg_mse_val > prev_mse_val):
            inc += 1
            print(">>>>> Early stopping increment up to: ", inc)
            
        else:
            inc = 0
        
        print("Best val avg 1 step MSE: ", best_val)
        if(inc > patience):
            print("Early stopping at epoch : ", epoch)
            break

            
    return best_regressor, best_val

def save_model(trained_koop_obj, model_save_dir):
    with open("{}/saved_best_model.pkl".format(model_save_dir), 'wb') as f:
        pkl.dump(trained_koop_obj, f)

def run_experiment(params, exp_name):   
    results_dir = "./experiment_outputs/{}".format(exp_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True) 
    
    train_data, val_data, test_data = generate_data(state_size=params['state_size'],
                                                    traj_len=params['traj_len'], rule=params['rule'],
                                                    compo_test_type=params['compo_test_type'],
                                                    results_dir=results_dir)
        
    model, avg_mse = train(train_data, val_data, params['look_forward'], params['intrinsic_cords'], 
          params['hidden_dim'], params['num_layers'])
    
    save_model(model, results_dir)
    
    test_avg_mse, test_mses = one_step_mse(model, test_data)
    print("Test avg mse: ", test_avg_mse)
    print("Test MSEs: ", test_mses)
    
    np.save("{}/test_mse.npy".format(results_dir), np.array(test_mses))
    plt.figure(figsize=(5, 10))
    plt.boxplot(test_mses)
    plt.ylim(0.0, 0.6)
    plt.xticks([1], ["Rule {}".format(str(params['rule']))])
    plt.savefig("{}/test_mse_plot.pdf".format(results_dir))
    
    

if __name__ in "__main__":
    exp_name = sys.argv[1]
    rule = int(sys.argv[2])
    
    params = {'state_size': 12,
            'traj_len': 50,
            'rule': rule,
            'look_forward': 1,
            'intrinsic_cords': 512,
            'hidden_dim': 128,
            'num_layers': 2,
            'compo_test_type': "weak"
            }
    
    run_experiment(params, exp_name)
    
    
    

    

