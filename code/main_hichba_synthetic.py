"""
This file contains the code for running experiments on HICH-BA graphs
"""
import networkx as nx
import util
import argparse
import seed_selection
import pickle
import pandas as pd
import graph_generator as gg


# main method
if __name__ == '__main__':
    config_list = [(200000,0.9,0.9,0.9),
                   (40000,0.9,0.9,0.9),
                   (100000,0.9,0.9,0.1),
                   (100000,0.9,0.2,0.9),
                   (100000,0.2,0.9,0.9),
                   (100000, 0.9, 0.9, 0.9)
                ]
    n = 10000
    k_n = 10
    k_p = 25
    for m,h,p_T,p_PA in config_list:
        dataset = f"HICHBA_{n}_{m}_{h}_{p_PA}_{p_T}_el"
        dataset_name = f"HICHBA_{n}_{m}_{h}_{p_PA}_{p_T}"
        # load networkx network from file
        network = nx.read_edgelist(f"../data/{dataset}.txt", nodetype=int, create_using=nx.DiGraph())
        print(dataset, " dataset loaded the graph successfully! There are ",
          network.number_of_nodes(), " nodes and ", network.number_of_edges(), " edges")

        parser = argparse.ArgumentParser("hichba")
        # python main_hichba_synthetic.py 0.3 random value
        parser.add_argument(dest='inf_prob', type=float, default=0.3)
        parser.add_argument(dest='neg_policy', type=str, default="random")
        parser.add_argument(dest='weight', type=str, default="value")

        args = parser.parse_args()
        print("HICHBA inf_prob: ", args.inf_prob)
        inf_prob = args.inf_prob
        print("HICHBA neg_policy: ", args.neg_policy)
        neg_policy = args.neg_policy
        neg_seeds = util.get_neg_seeds(network, neg_policy, k_n)
        print("HICHBA weight: ", args.weight)
        weight = args.weight
        if weight == "value":
            network = util.set_weight_value(network, inf_prob, "positive")
            network = util.set_weight_value(network, inf_prob, "negative")
        elif weight == "degree":
            network = util.set_weight_degree(network, "positive")
            network = util.set_weight_degree(network, "negative")
        network_new_labels = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
        neg_seeds_new_labels = util.get_new_labels(network_new_labels, neg_seeds)
        #out-degree of each neg_seeds:


        print(
            f"dataset: {dataset}, k_n: {k_n}, k_p: {k_p}, inf_prob: {inf_prob}, neg_policy: {neg_policy}, weight: {weight}")

        # proximity = util.compute_proximity(network_new_labels,neg_seeds_new_labels)
        # proximity_count = dict(Counter(proximity.values()))
        # diameter = nx.diameter(network_new_labels)
        # print(f"diameter = {diameter}")
        # avg_shortest_path_length = nx.average_shortest_path_length(network_new_labels)
        # print(f"avg shortest path length = {avg_shortest_path_length}")
        # r = nx.degree_assortativity_coefficient(network_new_labels)
        # print(f"Degree assortavity:{r}")
        # print("clustering coefficient: " + str(nx.average_clustering(network, network.nodes())))
        # degree_counter = Counter([network_new_labels.out_degree(node) for node in network_new_labels.nodes()])
        # print(f"degree_counter_sorted: {sorted(degree_counter.items(), key=lambda x: x[0])}")

        folder = f"../outputs/hichba/{dataset}_{weight}_weight_{neg_policy}_neg_policy_{inf_prob}"

        print("Negative seed set is", neg_seeds)
        methods = ["degree", "pagerank", "cmia-o","biog","rbf","fwrrs","fair-cmia-o","myopic_maximin","protect","naive_protect"]
        remaining_methods = []
        if True:
            for method in remaining_methods:
                pos_seeds = seed_selection.select_seeds(network, method, k_p, neg_seeds,
                                                            inf_prob,dataset_name)
                if method == "fair-cmia-o-sn":
                    method = "fair-cmia-o"
                pos_seed_file = f"{folder}/pos_seeds_{method}.txt"
                with open(pos_seed_file, 'w') as f:
                    for seed in pos_seeds:
                        f.write("%s\n" % str(seed))
        if False:
            pp_dict_dict, welfare_analysis_df = mu.run_simulations(folder, neg_policy, network_new_labels,
                                                                   neg_seeds_new_labels, methods, remaining_methods,
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
        if True:
            welfare_output_folder = f"{folder}/figs/"
            gg.generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df)
