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
import warnings
warnings.filterwarnings('ignore')
from automata import gather_rule_trajs_from_starts
import pickle as pkl
import json

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
            
def generate_data_for_strong_comp_raj(state_size, rule, results_dir, seed):
    # shuffle with a fixed seed
    start_state_permutations = list(itertools.product([0.0, 1.0], repeat=state_size))
    
    random.Random(seed).shuffle(start_state_permutations)

    train_start_state_permutations_1_3 = list(itertools.product([0.0, 1.0], repeat=3))[:4]
    val_start_state_permutations_1_3 = list(itertools.product([0.0, 1.0], repeat=3))[4:6]
    test_start_state_permutations_1_3 = list(itertools.product([0.0, 1.0], repeat=3))[6:]

    train_perms = []
    val_perms = []
    test_perms = []

    for sss in start_state_permutations:
        if sss[:3] in train_start_state_permutations_1_3:
            train_perms.append(sss)
        elif sss[:3] in val_start_state_permutations_1_3:
            val_perms.append(sss)
        else:
            assert sss[:3] in test_start_state_permutations_1_3
            test_perms.append(sss)
     
    print("Num train trajs: ", len(train_perms))
    print("Num val trajs: ", len(val_perms))
    print("Num test trajs: ", len(test_perms))

    train_trajs = gather_rule_trajs_from_starts(starts=train_perms, traj_len=2, rule=rule)
    val_trajs = gather_rule_trajs_from_starts(starts=val_perms, traj_len=2, rule=rule)
    test_trajs = gather_rule_trajs_from_starts(starts=test_perms, traj_len=2, rule=rule)
    
    local_to_count = num_unique_local_operations(train_trajs).values()
    
    # check for local operation coverage
    if(0 in local_to_count):
        raise NotImplementedError("We did not implement a way to deal with a training set that does not cover all local evolution operations. This is the case here.\
                                  Pick another automata rule to generate a different dataset hopefully without this problem. That should be the case for all rules defined in cellpylib.")
    else:
        print("Check for local evolution operation coverage succeeded for rule {}!".format(str(rule)))

    return train_trajs, val_trajs, test_trajs

def generate_data(state_size, traj_len, rule, compo_test_type, results_dir, seed):

    # shuffle with a fixed seed
    start_state_permutations = list(itertools.product([0.0, 1.0], repeat=state_size))
    random.Random(seed).shuffle(start_state_permutations) # this is fine because our state space is usually small enough
    end_train = int(len(start_state_permutations)*0.65) 
    end_val = end_train + int(len(start_state_permutations)*0.20)
    
    train_perms = start_state_permutations[:end_train]
    val_perms = start_state_permutations[end_train: end_val]
    test_perms = start_state_permutations[end_val:]
    
    print("Num train trajs: ", len(train_perms))
    print("Num val trajs: ", len(val_perms))
    print("Num test trajs: ", len(test_perms))
    
    train_trajs = gather_rule_trajs_from_starts(starts=train_perms, traj_len=2, rule=rule)
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
    
    return train_trajs, val_trajs, test_trajs


def one_step_mse(dlk_regressor, val_trajs):
    val_trajs = np.stack(val_trajs)
    pred_val = dlk_regressor.predict(val_trajs[:,0], n = 1)

    mse = np.mean((pred_val - val_trajs[:,1])**2, axis=1)

    return np.mean(mse), mse.tolist()


def level_1_strong_gen_mse(dlk_regressor, val_trajs):
    val_trajs = np.stack(val_trajs)

    pred_val = dlk_regressor.predict(val_trajs[:,0], n = 1)

    mse = (pred_val - val_trajs[:,1])**2

    in_comb_mse_mask = np.ones_like(mse, dtype=np.int32)
    in_comb_mse_mask[:,1] = 0
    in_comb_mse_mask[:,0] = 0
    in_comb_mse_mask[:,2] = 0
    in_comb_mse_mask[:,3] = 0
    in_comb_mse_mask[:,-1] = 0

    out_comb_mse_mask = np.ones_like(mse,  dtype=np.int32)
    out_comb_mse_mask[:,1] = 0
    out_comb_mse_mask = 1 - in_comb_mse_mask

    in_comb_mse = np.sum(in_comb_mse_mask * mse) / np.sum(in_comb_mse_mask)
    out_comb_mse = np.sum(out_comb_mse_mask * mse) / np.sum(out_comb_mse_mask)

    return in_comb_mse, out_comb_mse

'''
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "elu": nn.ELU(),
            "mish": nn.Mish(),
            "linear": nn.Identity(),
'''

def train(train_data, val_data, look_forward, intrinsic_cords, hidden_dim, num_layers, batch_size=64, max_epochs=25, patience=5, compo_test_type="weak"):
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

        if compo_test_type == 'strong':
            in_comb_mse, out_comb_mse = level_1_strong_gen_mse(dlk_regressor, val_data)
            print("Val avg 1 in combination MSE: ", in_comb_mse)
            print("Val avg 1 out of combination MSE: ", out_comb_mse)
            
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

def run_experiment(params, exp_name, seed):   
    results_dir = "./experiment_outputs/{}".format(exp_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True) 
    
    if(params['compo_test_type'] == "weak"):
        train_data, val_data, test_data = generate_data(state_size=params['state_size'],
                                                        traj_len=params['traj_len'], rule=params['rule'],
                                                        compo_test_type=params['compo_test_type'],
                                                        results_dir=results_dir, seed=seed)
    elif (params['compo_test_type'] == "strong"):
        train_data, val_data, test_data = generate_data_for_strong_comp_raj(state_size=params['state_size'],
                                                        rule=params['rule'],
                                                        results_dir=results_dir,  seed=seed)
    else:
        raise NotImplementedError("You did not choose a compo test type that actually exists")
    
    model, avg_mse = train(train_data, val_data, params['look_forward'], params['intrinsic_cords'], 
          params['hidden_dim'], params['num_layers'], compo_test_type=params['compo_test_type'])
    
    save_model(model, results_dir)
    
    test_avg_mse, test_mses = one_step_mse(model, test_data)
    print("Test avg mse: ", test_avg_mse)

    if params['compo_test_type'] == 'strong':
        in_comb_test_mse, out_comb_test_mse = level_1_strong_gen_mse(model, test_data)
        print("Test avg 1 in combination MSE: ", in_comb_test_mse)
        print("Test avg 1 out of combination MSE: ", out_comb_test_mse)
        
    stats_dict = dict()
    stats_dict['test_avg_mse'] = test_avg_mse
    stats_dict['in_comb_test_mse'] = in_comb_test_mse
    stats_dict['out_comb_test_mse'] = out_comb_test_mse

    out_file = open("{}/{}-test_mse.json".format(results_dir, seed), "w") 
  
    json.dump(stats_dict, out_file)

    # np.save("{}/test_mse.npy".format(results_dir), np.array(test_mses))
    # plt.figure(figsize=(5, 10))
    # plt.boxplot(test_mses)
    # plt.ylim(0.0, 0.6)
    # plt.xticks([1], ["Rule {}".format(str(params['rule']))])
    # plt.savefig("{}/test_mse_plot.pdf".format(results_dir))
    

if __name__ in "__main__":
    exp_name = sys.argv[1]
    rule = int(sys.argv[2])
    seed = int(sys.argv[3])

    np.random.seed(seed) 

    
    params = {'state_size': 12,
            'traj_len': 50,
            'rule': rule,
            'look_forward': 1,
            'intrinsic_cords': 512,
            'hidden_dim': 128,
            'num_layers': 2,
            'compo_test_type': "strong"
            }
    
    run_experiment(params, exp_name, seed)

# changes to undo
# 1) change max_epochs back to 1