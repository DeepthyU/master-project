"""
This file contains the seed selection methods.
"""
import sketch_based_methods as sbm
import networkx as nx
import rbf
import welfare_nim
import tibmm
import fair_baselines
import rps
import proximity_based_methods as proximity_method


def select_seeds(g, policy, seeds_number, neg_seeds, min_prob, dataset_name=None):
    # Seed selection
    if policy == 'degree':
        node_deg_list = sorted(g.out_degree, key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_deg_list if x not in neg_seeds][:seeds_number]
        print("seeds from degree:", seeds)
    elif policy == 'closeness':
        g_rev = nx.DiGraph.reverse(g)
        node_clo_list = sorted(nx.closeness_centrality(g_rev).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_clo_list if x not in neg_seeds][:seeds_number]
        print("seeds from closeness:", seeds)
    elif policy == 'betweenness':
        node_bet_list = sorted(nx.betweenness_centrality(g).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_bet_list if x not in neg_seeds][:seeds_number]
        print("seeds from betweenness:", seeds)
    elif policy == "pagerank":
        node_pr_list = sorted(nx.pagerank(g.reverse()).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_pr_list if x not in neg_seeds][:seeds_number]
        print("seeds from pagerank_reverse:", seeds)
    elif policy == 'cmia-o':
        seeds = sbm.CMIA_O(g, neg_seeds, seeds_number, min_prob)
        print("seeds from cmia-o:", seeds)
    elif policy == 'biog':
        seeds = sbm.BIOG(g, neg_seeds, seeds_number, min_prob)
        print("seeds from biog:", seeds)
    elif policy == 'tib':
        seeds = sbm.TIB_Solver(g, neg_seeds, seeds_number)
        print("seeds from tib:", seeds)
    elif policy == 'rbf':
        seeds = rbf.rbf(g, neg_seeds, seeds_number)
        print("seeds from rbf:", seeds)
    elif policy == 'cdd':
        seeds = rbf.cdd(g, neg_seeds, seeds_number)
        print("seeds from cdd:", seeds)
    elif policy == 'tibmm':
        seeds = tibmm.tibmm(g, neg_seeds, seeds_number)
        print("seeds from tibmm:", seeds)
    elif policy == 'rps':
        seeds = rps.rps(g, neg_seeds, seeds_number)
        print("seeds from tibmm:", seeds)
    elif policy == 'myopic_maximin':
        seeds = welfare_nim.myopic_maximin(g, neg_seeds, seeds_number)
        print("seeds from welfare_myopic_maximin:", seeds)
    elif policy == 'naive_protect':
        seeds = proximity_method.myopic_distance_count_method(g, neg_seeds, seeds_number)
        print("seeds from myopic_distance_count_method:", seeds)
    elif policy == 'protect':
        seeds = proximity_method.myopic_sim_distance_method(g, neg_seeds, seeds_number)
        print("seeds from sp_maximin_using_M_distance_v2:", seeds)
    elif policy == 'naive_myopic':
        seeds = welfare_nim.naive_myopic(g, neg_seeds, seeds_number)
        print("seeds from naive_myopic:", seeds)
    elif policy == 'greedy_maximin':
        seeds = welfare_nim.greedy_maximin(g, neg_seeds, seeds_number)
        print("seeds from greedy_maximin:", seeds)
    elif policy == 'fwrrs':
        seeds = fair_baselines.FWRRS(g, neg_seeds, seeds_number)
        print("seeds from fwrrs:", seeds)
    elif policy == 'fair-cmia-o':
        seeds = fair_baselines.CMIA_O_fair(g, neg_seeds, seeds_number, min_prob)
        print("seeds from fair-cmia-o:", seeds)
    elif policy == 'fair-cmia-o-sn':
        seeds = fair_baselines.CMIA_O_syn_group_fair(dataset_name, neg_seeds, seeds_number, min_prob)
        print("seeds from fair-cmia-o-sn:", seeds)
    else:
        raise NameError("Unknown policy")
    print(f'Number of Seeds: {len(seeds)}')
    return seeds

