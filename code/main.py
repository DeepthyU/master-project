"""
This file is used to run the experiments on the email dataset.
"""
import pickle
import pandas as pd
import util
import seed_selection
import graph_generator as gg
import main_utils as mu
import networkx as nx
import argparse

imp_prob_dict = {"email": 0.3, "facebook_combined": 0.2, "twitter_combined": 0.25,
                 "socfb-Caltech36": 0.1, "CA-AstroPh": 0.25}

# main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser("main")
    parser.add_argument(dest='dataset', type=str, default="email")
    parser.add_argument(dest='k_n', type=int, default=10)
    parser.add_argument(dest='k_p', type=int, default=25)
    parser.add_argument(dest='neg_policy', type=str, default="random")
    parser.add_argument(dest='weight', type=str, default="value")
    args = parser.parse_args()
    # python main.py email 10 25 random value
    dataset = args.dataset
    k_n = args.k_n
    k_p = args.k_p
    neg_policy = args.neg_policy
    weight = args.weight
    imp_prob = imp_prob_dict[dataset]
    print(
        f"dataset: {dataset}, k_n: {k_n}, k_p: {k_p}, imp_prob: {imp_prob}, neg_policy: {neg_policy}, weight: {weight}")
    network, neg_seeds, network_new_labels, neg_seeds_new_labels = util.init_settings(dataset, weight, imp_prob,
                                                                                      neg_policy, k_n)

    folder = f"../outputs/{dataset}_{weight}_weight_{neg_policy}_neg_policy_{imp_prob}"
    methods = mu.get_methods(dataset, neg_policy, weight)
    remaining_methods = []

    print(dataset, " dataset loaded the graph successfully! There are ",
          network.number_of_nodes(), " nodes and ", network.number_of_edges(), " edges")
    print("Negative seed set is", neg_seeds)
    if True:
        for method in remaining_methods:
            print("Running method:", method)
            pos_seeds = seed_selection.select_seeds(network, method, k_p, neg_seeds,
                                                            imp_prob)
            pos_seed_file = f"{folder}/pos_seeds_{method}.txt"
            with open(pos_seed_file, 'w') as f:
                for seed in pos_seeds:
                    f.write("%s\n" % str(seed))
    if True:
        pp_dict_dict, welfare_analysis_df = mu.run_simulations(folder, neg_policy, network_new_labels, neg_seeds_new_labels, methods, remaining_methods,
                           df_columns_added=False)
    else:
        with open(f"{folder}/pp_dict_dict1.pkl", 'rb') as f:
            pp_dict_dict = pickle.load(f)
        welfare_analysis_df = pd.read_csv(f"{folder}/welfare_analysis_df1.csv", converters={'avg_sp_at_distance': eval,
                                                                                             'avg_prox_at_distance': eval,
                                                                                             'avg_vul_at_distance': eval,
                                                                                             'std_dev_sp_at_distance': eval,
                                                                                            'improved_sp_at_distance': eval,
                                                                                            'gini_coeff_list': eval
                                                                                            })
    welfare_output_folder = f"{folder}/figs/"
    gg.generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df)
