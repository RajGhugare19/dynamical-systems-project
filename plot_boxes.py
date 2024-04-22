import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def read_all_seed_results(results_dir, seed_range=30):
    
    mses = []
    in_comb_mses = []
    out_comb_mses = []
    
    for seed in range(1, seed_range+1):
        curr_file_name = "{}/{}-test_mse.json".format(results_dir, seed)
        with open(curr_file_name, "r") as file:
            data = json.load(file)
            mses.append(data["test_avg_mse"])
            if(not np.isinf(data['in_comb_test_mse'])):
                in_comb_mses.append(data["in_comb_test_mse"])
            if(not np.isinf(data['out_comb_test_mse'])):
                out_comb_mses.append(data["out_comb_test_mse"])
    return mses, in_comb_mses, out_comb_mses


def plot_boxplot(rule_to_result_dir, figure_out_dir):
    
        
    all_mses = []
    all_in_comb_mses = []
    all_out_comb_mses = []
    rules = list(rule_to_result_dir.keys())
    
    for rule in rule_to_result_dir:
        
        mses, in_comb_mses, out_comb_mses = read_all_seed_results(rule_to_result_dir[rule])
        all_mses.append(mses)
        all_in_comb_mses.append(in_comb_mses)
        all_out_comb_mses.append(out_comb_mses)
        
        # mse_type_to_rule_result_map["Average Test MSE"].update({rule: mses})
        # mse_type_to_rule_result_map["In-Combination MSE"].update({rule: in_comb_mses})
        # mse_type_to_rule_result_map["Out-Combination MSE"].update({rule: out_comb_mses})
        
    plt.boxplot(all_mses)
    plt.xticks([i for i in range(1, len(all_mses)+1)], rules)
    plt.ylabel("Average Test MSE")
    plt.savefig("{}/{}".format(figure_out_dir, "Average_Test_MSE.pdf"))
    plt.clf()
    
    plt.boxplot(all_in_comb_mses)
    plt.xticks([i for i in range(1, len(all_in_comb_mses)+1)], rules)
    plt.ylabel("In-Combination MSE")
    plt.savefig("{}/{}".format(figure_out_dir, "In-Combination_MSE.pdf"))
    plt.clf()
    
    plt.boxplot(all_out_comb_mses)
    plt.xticks([i for i in range(1, len(all_out_comb_mses)+1)], rules)
    plt.ylabel("Out of Combination MSE")
    plt.savefig("{}/{}".format(figure_out_dir, "Out-of-Combination_MSE.pdf"))
    
if __name__ in "__main__":

    # test_type = "strong"
    # figure_dir = "./figures/{}".format(test_type)
    # Path(figure_dir).mkdir(parents=True, exist_ok=True) 

    # rule_to_result_dir = {"Rule 1": "experiment_outputs/exp_rule_1_final",
    #                       "Rule 2": "experiment_outputs/exp_rule_2_final",
    #                       "Rule 110": "experiment_outputs/exp_rule_110_final",
    #                       "Rule 126": "experiment_outputs/exp_rule_126_final"}
    # plot_boxplot(rule_to_result_dir, figure_dir)
    
    
    
    test_type = "weak"
    figure_dir = "./figures/{}".format(test_type)
    Path(figure_dir).mkdir(parents=True, exist_ok=True) 
    
    rule_to_result_dir = {"Rule 1": "experiment_outputs/exp_rule_1_final_weak",
                          "Rule 2": "experiment_outputs/exp_rule_2_final_weak",
                          "Rule 110": "experiment_outputs/exp_rule_110_final_weak",
                          "Rule 126": "experiment_outputs/exp_rule_126_final_weak"}
    plot_boxplot(rule_to_result_dir, figure_dir)
    
    print("Done plotting! Check figures dir!")