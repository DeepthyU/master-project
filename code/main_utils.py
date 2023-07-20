import pandas as pd
from tqdm import tqdm
import util
import os
import random
import timeit
from scipy.stats import pearsonr
from datetime import datetime
from collections import Counter
import multiprocessing as multip
import pickle
import numpy as np
import diffusionmodels as dm

from functools import partial

def run_simulations(folder, neg_policy, network_new_labels, neg_seeds_new_labels, methods, remaining_methods,
                    df_columns_added=True, num_simulations=100):
    print("Starting Running simulations...")
    start = timeit.default_timer()
    if os.path.exists(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl"):
        with open(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl", 'rb') as f:
            vul = pickle.load(f)
    else:
        # compute vulneribility of each node
        pool = multip.Pool(multip.cpu_count())
        hits = np.zeros(network_new_labels.number_of_nodes())

        inf_vuln_partial_func = partial(dm.icm, network_new_labels, neg_seeds_new_labels)
        for misinfo_impressed in tqdm(pool.imap_unordered(inf_vuln_partial_func, list(range(num_simulations))),
                                      total=num_simulations):
            for node in misinfo_impressed:
                hits[node] += 1
        pool.close()
        pool.join()
        vul = hits / num_simulations
        # write to file
        with open(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl", 'wb') as f:
            pickle.dump(vul, f)
    pp_dict_dict = {}
    if os.path.exists(f"{folder}/pp_dict_dict1.pkl"):
        with open(f"{folder}/pp_dict_dict1.pkl", 'rb') as f:
            pp_dict_dict = pickle.load(f)
    welfare_analysis_df = \
        pd.DataFrame(columns=["policy", "seed count", "impressed", "min_non_zero_prob", "min_prob", "avg_prob",
                              "zero_prob_count", "consistency", "corr"])

    if os.path.exists(f"{folder}/welfare_analysis_df1.csv") and not df_columns_added:
        welfare_analysis_df = pd.read_csv(f"{folder}/welfare_analysis_df1.csv", converters={'avg_sp_at_distance': eval,
                                                                                            'avg_prox_at_distance': eval,
                                                                                            'avg_vul_at_distance': eval,
                                                                                            'std_dev_sp_at_distance': eval,
                                                                                            'improved_sp_at_distance': eval,
                                                                                            'gini_coeff_list': eval
                                                                                            })
    # get distance of each node from center
    non_seed_nodes = list(set(network_new_labels.nodes()) - set(neg_seeds_new_labels))
    proximity = util.compute_proximity_arr(network_new_labels, neg_seeds_new_labels)
    reachable_nodes = np.where(proximity != np.inf)[0]
    if os.path.exists(f"{folder}/partition1.pkl"):
        with open(f"{folder}/partition1.pkl", 'rb') as f:
            partition1 = pickle.load(f)
    else:
        partition1 = list(set(random.sample(non_seed_nodes, int(len(non_seed_nodes) / 2))))
        with open(f"{folder}/partition1.pkl", 'wb') as f:
            pickle.dump(partition1, f)
    partition2 = list(set(non_seed_nodes) - set(partition1))
    for k in tqdm([0, 5, 10, 15, 20, 25]):
        for method in tqdm(methods):
            if not df_columns_added:
                if ((welfare_analysis_df['policy'] == method) & (welfare_analysis_df['seed count'] == k)).any():
                    if method not in remaining_methods:
                        continue
                    else:
                        # remove that rows from welfare_analysis_df
                        welfare_analysis_df = welfare_analysis_df[
                            ~((welfare_analysis_df['policy'] == method) & (welfare_analysis_df['seed count'] == k))]
            print("Running simulations...", method, datetime.now())
            start = timeit.default_timer()
            if k == 0:
                average_impressed = np.sum(vul)
                sp = 1 - vul
                min_non_zero_prob = np.min(sp[sp > 0])
                min_prob = np.min(sp[non_seed_nodes])
                average_sp = np.mean(sp)
                zero_nodes_count = np.count_nonzero(sp[non_seed_nodes] == 0)
                vulnerable_nodes = list(set(reachable_nodes) - set(neg_seeds_new_labels))
                consistency = util.compute_consistency(vul[vulnerable_nodes], sp[vulnerable_nodes])
                consistency2 = util.compute_consistency(proximity[vulnerable_nodes], sp[vulnerable_nodes])
                consistency3 = 0
                corr, _ = pearsonr(vul[vulnerable_nodes], sp[vulnerable_nodes])
                corr2, _ = pearsonr(proximity[vulnerable_nodes], sp[vulnerable_nodes])
                corr3 = 0
                corr4 = 0
                partition1_sp = np.mean(sp[partition1])
                partition2_sp = np.mean(sp[partition2])
                partition_diff = partition2_sp - partition1_sp
                partition1_sp_min = np.min(sp[partition1])
                partition2_sp_min = np.min(sp[partition2])
                partition_min_diff = partition2_sp_min - partition1_sp_min

                partition1_prox = np.inf
                partition2_prox = np.inf
                partition_prox_diff = 0
                partition1_prox_min = np.inf
                partition2_prox_min = np.inf
                partition_prox_min_diff = 0
                num_nodes_closer_to_T = 0
                saved_fraction = 0
                avg_sp_at_distance = []
                avg_prox_at_distance = []
                avg_vul_at_distance = []
                std_dev_sp_at_distance = []
                improved_sp_at_distance = []
                gini_coeff_list = []
                avg_proximity_of_T = np.inf
                gini_coef = util.compute_gini_coefficient(sp[vulnerable_nodes])
            else:
                pos_seeds = util.read_pos_seeds(network_new_labels, method, k, folder)
                pos_seeds = util.get_new_labels(network_new_labels, pos_seeds)
                print("Positive seed set is", pos_seeds)
                pool = multip.Pool(multip.cpu_count())
                pos_seeds_list = [pos_seeds] * num_simulations
                pos_seeds_list = [(i, pos_seeds) for i, pos_seeds in enumerate(pos_seeds_list)]
                neg_impressed_counts = []
                hits = np.zeros(network_new_labels.number_of_nodes())
                inf_partial_func = partial(util.coicm, network_new_labels, neg_seeds_new_labels)
                for neg_impressed_nodes in pool.imap_unordered(inf_partial_func, pos_seeds_list):
                    hits[neg_impressed_nodes] += 1
                    neg_impressed_counts.append(len(neg_impressed_nodes))
                mp = hits / num_simulations
                sp = 1 - mp
                pool.close()
                pool.join()

                partition1_sp = np.mean(sp[partition1])
                partition2_sp = np.mean(sp[partition2])
                partition_diff = partition2_sp - partition1_sp
                partition1_sp_min = np.min(sp[partition1])
                partition2_sp_min = np.min(sp[partition2])
                partition_min_diff = partition2_sp_min - partition1_sp_min
                proximity_from_T = util.compute_proximity_arr(network_new_labels, pos_seeds)
                partition1_prox = np.mean(proximity_from_T[partition1])
                partition2_prox = np.mean(proximity_from_T[partition2])
                partition_prox_diff = partition2_prox - partition1_prox
                partition1_prox_min = np.min(proximity_from_T[partition1])
                partition2_prox_min = np.min(proximity_from_T[partition2])
                partition_prox_min_diff = partition2_prox_min - partition1_prox_min
                zero_nodes_count = np.count_nonzero(sp[non_seed_nodes] == 0)
                average_sp = np.mean(sp)
                average_impressed = np.mean(neg_impressed_counts)
                min_non_zero_prob = np.min(sp[sp > 0])
                avg_neg_inf = np.sum(vul)
                saved_fraction = (avg_neg_inf - average_impressed) / avg_neg_inf

                avg_sp_at_distance = []
                avg_prox_at_distance = []
                avg_vul_at_distance = []
                improved_sp_at_distance = []
                std_dev_sp_at_distance = []
                vulnerable_nodes = list(set(reachable_nodes) - set(neg_seeds_new_labels) - set(pos_seeds))
                max_proximity = int(np.max(proximity[vulnerable_nodes]))
                proximity_of_vulnerable_nodes = proximity_from_T[vulnerable_nodes]
                sp_of_vulnerable_nodes = sp[vulnerable_nodes]
                vul_of_vulnerable_nodes = vul[vulnerable_nodes]
                gini_coeff_list = []
                for distance in range(1, max_proximity + 1, 1):
                    nodes_at_distance = np.where(proximity_of_vulnerable_nodes == distance)[0]
                    g_val = util.compute_gini_coefficient(sp_of_vulnerable_nodes[nodes_at_distance])
                    if np.isnan(g_val):
                        g_val = 0
                    gini_coeff_list.append(g_val)
                    avg_val = np.mean(sp_of_vulnerable_nodes[nodes_at_distance])
                    if np.isnan(avg_val):
                        avg_val = 0
                    avg_sp_at_distance.append(avg_val)
                    avg_val = np.mean(proximity_of_vulnerable_nodes[nodes_at_distance])
                    if np.isnan(avg_val):
                        avg_val = 0
                    avg_prox_at_distance.append(avg_val)
                    avg_val = np.mean(sp_of_vulnerable_nodes[nodes_at_distance])
                    if np.isnan(avg_val):
                        avg_val = 0
                    avg_vul_at_distance.append(avg_val)
                    improved_sp = sp_of_vulnerable_nodes[nodes_at_distance] - (
                                1 - vul_of_vulnerable_nodes[nodes_at_distance])
                    avg_val = np.mean(improved_sp)
                    if np.isnan(avg_val):
                        avg_val = 0
                    improved_sp_at_distance.append(avg_val)
                    std_val = np.std(sp_of_vulnerable_nodes[nodes_at_distance])
                    if np.isnan(std_val):
                        std_val = 0
                    std_dev_sp_at_distance.append(std_val)
                # number of nodes closer to T than M
                num_nodes_closer_to_T = np.sum(proximity_from_T[vulnerable_nodes] < proximity[vulnerable_nodes])
                min_prob = np.min(sp[non_seed_nodes])
                corr, _ = pearsonr(vul[vulnerable_nodes], sp[vulnerable_nodes])
                corr2, _ = pearsonr(proximity[vulnerable_nodes], sp[vulnerable_nodes])

                corr3, _ = pearsonr(proximity[vulnerable_nodes], proximity_from_T[vulnerable_nodes])
                corr4, _ = pearsonr(proximity_from_T[vulnerable_nodes], sp[vulnerable_nodes])
                consistency = util.compute_consistency(vul[vulnerable_nodes], sp[vulnerable_nodes])
                consistency2 = util.compute_consistency(proximity[vulnerable_nodes], sp[vulnerable_nodes])
                consistency3 = util.compute_consistency(proximity[vulnerable_nodes], proximity_from_T[vulnerable_nodes])
                avg_proximity_of_T = np.mean(proximity[pos_seeds])
                gini_coef = util.compute_gini_coefficient(sp[vulnerable_nodes])
            welfare_analysis_df = welfare_analysis_df.append(
                {"policy": method, "seed count": k, "impressed": average_impressed,
                 "min_non_zero_prob": min_non_zero_prob, "min_prob": min_prob, "avg_prob": average_sp,
                 "zero_prob_count": zero_nodes_count, "consistency": consistency, "consistency2": consistency2,
                 "consistency3": consistency3, "corr": corr, "corr2": corr2, "corr3": corr3, "corr4": corr4,
                 "partition1_sp": partition1_sp, "partition2_sp": partition2_sp, "partition_diff": partition_diff,
                 "partition1_sp_min": partition1_sp_min, "partition2_sp_min": partition2_sp_min,
                 "partition_min_diff": partition_min_diff, "partition1_prox": partition1_prox,
                 "partition2_prox": partition2_prox, "partition_prox_diff": partition_prox_diff,
                 "partition1_prox_min": partition1_prox_min, "partition2_prox_min": partition2_prox_min,
                 "partition_prox_min_diff": partition_prox_min_diff, "num_nodes_closer_to_T": num_nodes_closer_to_T,
                 "saved_fraction": saved_fraction, "avg_sp_at_distance": avg_sp_at_distance,
                 "avg_prox_at_distance": avg_prox_at_distance,
                 "avg_vul_at_distance": avg_vul_at_distance, "std_dev_sp_at_distance": std_dev_sp_at_distance,
                 "improved_sp_at_distance": improved_sp_at_distance,
                 "avg_proximity_of_T": avg_proximity_of_T, "gini_coef": gini_coef, "gini_coeff_list": gini_coeff_list},
                ignore_index=True)
            rounded_prob = [round(p, 1) for p in sp]
            pp_dict = dict(Counter(rounded_prob))
            plot_dict = {k: pp_dict[k] for k in sorted(pp_dict)}
            if k not in pp_dict_dict:
                pp_dict_dict[k] = {}
            pp_dict_dict[k][method] = plot_dict

            print("Simulations completed", method, "for ", k, " seeds")
            stop = timeit.default_timer()
            print('Time: ', stop - start)

            # save pp dict
            with open(f"{folder}/pp_dict_dict1.pkl", 'wb') as f:
                pickle.dump(pp_dict_dict, f)
            # save welfare analysis df
            welfare_analysis_df.to_csv(f"{folder}/welfare_analysis_df1.csv", index=False)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print("simulations completed")
    return pp_dict_dict, welfare_analysis_df


def get_methods(dataset, neg_policy, weight):
    if dataset == "twitter_combined":
        methods = ["degree", "pagerank", "cmia-o", "biog", "fair-cmia-o", "myopic_maximin",
                   "protect", "naive_protect"]
    elif dataset == "socfb-Caltech36":
        methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "rps", "fwrrs", "fair-cmia-o", "greedy_maximin",
                   "myopic_maximin", "protect", "naive_protect"]
    elif weight == "value":
        if neg_policy == "degree":
            if dataset == "email":
                methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "tib", "tibmm", "rps",
                           "fwrrs", "fair-cmia-o", "greedy_maximin", "myopic_maximin", "protect",
                           "naive_protect"]
            else:
                methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "fwrrs", "fair-cmia-o",
                           "myopic_maximin", "protect", "naive_protect"]
        else:
            if dataset == "email":
                methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "rps", "fwrrs", "fair-cmia-o", "greedy_maximin",
                       "myopic_maximin", "protect", "naive_protect"]
            else:
                methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "fair-cmia-o",
                           "myopic_maximin", "protect", "naive_protect"]
    elif weight == "1_value":
        if dataset == "email":
            methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "rps", "fwrrs", "fair-cmia-o", "greedy_maximin",
                       "myopic_maximin", "protect", "naive_protect"]
        else:
            methods = ["degree", "pagerank", "cmia-o", "biog", "rbf", "fair-cmia-o",
                       "myopic_maximin", "protect", "naive_protect"]

    return methods