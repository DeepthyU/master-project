"""
This file contains utility functions for the project
"""
import pandas as pd
import networkx as nx
import numpy as np

import diffusionmodels as dm
from tqdm import tqdm
from matplotlib import pyplot as plt

def read_pos_seeds(network, policy, num_seeds, folder):
    if policy == 'random':
        pos_seeds = np.random.choice(network.nodes(), size=num_seeds, replace=False)
    else:
        pos_seeds = []
        pos_seed_file = f"{folder}/pos_seeds_{policy}.txt"
        with open(pos_seed_file, 'r') as f:
             for line in f:
                 pos_seeds.append(int(line[:-1]))

        if len(pos_seeds) < num_seeds:
            print("Not enough seeds in the file")
            return []
        pos_seeds = pos_seeds[:num_seeds]
    return pos_seeds

def compute_proximity(g, M):
    proximity = {}
    queue = []
    for m in M:
        proximity[m] = 0
        queue.append(m)
    while len(queue) > 0:
        node = queue.pop(0)
        neighbors = g.neighbors(node)
        for neighbor in neighbors:
            if neighbor in proximity:
                oldproximity = proximity[neighbor]
                newproximity = proximity[node] + 1
                if newproximity < oldproximity:
                    proximity[neighbor] = newproximity
                    queue.append(neighbor)
            else:
                proximity[neighbor] = proximity[node] + 1
                queue.append(neighbor)
    return proximity


def compute_proximity_arr(g, M):
    proximity = np.ones(g.number_of_nodes()) * np.inf
    queue = []
    for m in M:
        proximity[m] = 0
        queue.append(m)
    while len(queue) > 0:
        node = queue.pop(0)
        neighbors = g.neighbors(node)
        for neighbor in neighbors:
            if proximity[neighbor] != np.inf:
                old_proximity = proximity[neighbor]
                new_proximity = proximity[node] + 1
                if new_proximity < old_proximity:
                    proximity[neighbor] = new_proximity
                    queue.append(neighbor)
            else:
                proximity[neighbor] = proximity[node] + 1
                queue.append(neighbor)
    return proximity

def get_neg_seeds(network, policy, num_seeds):
    if policy == 'random':
        np.random.seed(100)
        neg_seeds = list(np.random.choice(network.nodes(), size=num_seeds, replace=False))
    elif policy == 'degree':
        # highest degree nodes as neg seeds
        node_deg_list = sorted(network.out_degree, key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_deg_list][:num_seeds]
    elif policy == 'page_rank_reverse':
        node_pr_list = sorted(nx.pagerank(network.reverse()).items(), key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_pr_list][:num_seeds]
    elif policy == 'betweenness':
        node_bet_list = sorted(nx.betweenness_centrality(network).items(), key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_bet_list][:num_seeds]
    return neg_seeds




def load_graph(file):
    graph_df = pd.read_csv(file, sep="\t", header=None)
    graph_df.columns = ['s', 't']

    edges = []

    for index, row in tqdm(graph_df.iterrows()):
        if row.s == row.t:
            continue
        edge_cur = (row.s, row.t)
        edges.append(edge_cur)

    g = nx.DiGraph(edges)
    return g


def coicm(network, neg_seeds, pos_seeds):
    iter, pos_seeds = pos_seeds
    neg_impressed = dm.coicm(network, neg_seeds, pos_seeds, iter)
    return neg_impressed

def load_bidirectional_graph(file):
    graph_df = pd.read_csv(file, sep="\t", header=None)
    graph_df.columns = ['s', 't']

    edges = []

    for index, row in tqdm(graph_df.iterrows()):
        if row.s == row.t:
            continue
        edge_cur = (row.s, row.t)
        edges.append(edge_cur)
        edge_cur = (row.t, row.s)
        edges.append(edge_cur)

    g = nx.DiGraph(edges)
    return g




def set_weight_value(G,val,p_n):
    for u,v in G.edges():
        G[u][v][p_n]= round(val,3)
    return G



def set_weight_degree(G,p_n):
    for u,v in G.edges():
        G[u][v][p_n]= 1/G.in_degree(v)
    return G



def get_old_labels(g, M):
    return [g.nodes[node]["old_label"] for node in M]


def compute_consistency(vul, sp):
    vul = vul/np.max(vul)
    vul_mat = np.abs(vul[:, None] - vul[None, :])
    similarity_mat = 1 - (vul_mat)
    sp_mat = np.abs(sp[:, None] - sp[None, :])
    pdt = np.multiply(similarity_mat, sp_mat)
    num = np.sum(pdt)
    den = np.sum(similarity_mat)
    consistency = 1 - (num / den)
    return consistency


def get_new_labels(g, M):
    new_to_old_labels = nx.get_node_attributes(g, "old_label")
    old_to_new_labels = {v:k for k,v in new_to_old_labels.items()}
    new_labels = [old_to_new_labels[node] for node in M]
    return new_labels




def compute_gini_coefficient(sp):
    # sp = np.array([0.1, 0.2, 0.3, 0.4])
    total = 0
    for i, xi in enumerate(sp[:-1], 1):
        total += np.sum(np.abs(xi - sp[i:]))
    return total / (len(sp) ** 2 * np.mean(sp))


def init_settings(dataset_file, edge_weight_policy, imp_prob, neg_policy="degree", k_m=10):
    """
    Initialize the settings for the experiment
    :param dataset_file:
    :param edge_weight_policy:
    :param imp_prob:
    :param neg_policy:
    :param k_m:
    :return:
    """
    # if filename contains "facebook" then it is a bidirectional graph
    if "facebook" in dataset_file or "fb" in dataset_file:
        g = load_bidirectional_graph(f"../data/{dataset_file}.txt")
    else:
        g = load_graph(f"../data/{dataset_file}.txt")
    strongly_connected = nx.is_strongly_connected(g)
    print("is strongly connected: " + str(strongly_connected))
    if not strongly_connected:
        largest_component = max(nx.strongly_connected_components(g), key=len)
        g = g.subgraph(largest_component)
        print(f"largest_subgraph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    neg_seeds = get_neg_seeds(g, neg_policy, k_m)
    if edge_weight_policy == "value":
        g = set_weight_value(g, imp_prob, "positive")
        g = set_weight_value(g, imp_prob, "negative")
    elif edge_weight_policy == "degree":
        g = set_weight_value(g, 1, "positive")
        g = set_weight_degree(g, "negative")
    elif edge_weight_policy == "degree_both":
        g = set_weight_degree(g, "positive")
        g = set_weight_degree(g, "negative")
    elif edge_weight_policy == "1_value":
        g = set_weight_value(g, 1, "positive")
        g = set_weight_value(g, imp_prob, "negative")
    network_new_labels = nx.DiGraph(g)
    network_new_labels = nx.convert_node_labels_to_integers(network_new_labels, label_attribute="old_label")
    neg_seeds_new_labels = get_new_labels(network_new_labels, neg_seeds)
    return g, neg_seeds, network_new_labels, neg_seeds_new_labels



def plot_k_vs_df_columns(welfare_analysis_df,column_name,methods, y_label,filename,k_p=25):
    """
    Plots the k vs the column_name for each method in methods and saves the plot as filename
    :param welfare_analysis_df:
    :param column_name:
    :param methods:
    :param y_label:
    :param filename:
    :return:
    """
    plt.figure(figsize=(8, 6))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    markers = ["D", "s", "o", "P", "h", "v",  "<", ">", "^", "*", "p", "X", "d", ".","H","8","1","2","3","4","|","_","+"]
    colors = ["hotpink", "dodgerblue", "limegreen", "orange", "mediumseagreen", "orangered", "green", "olive", "darkred", "darkblue",
              "purple", "mediumvioletred", "red", "blue","steelblue","gray","darkgray","darkolive","peru","crimson"]
    all_methods = ["degree", "pagerank", "cmia-o","biog","rbf","tib","tibmm","rps","fwrrs","fair-cmia-o","greedy_maximin",
           "myopic_maximin","protect","naive_protect"]
    addtl_counter = -1
    for policy in methods:
        if policy not in all_methods:
            addtl_counter += 1
            idx = len(all_methods) + addtl_counter
        else:
            idx = all_methods.index(policy)
        df = welfare_analysis_df[welfare_analysis_df["policy"] == policy]
        df = df[df["seed count"] <= k_p]
        xs, ys = zip(*sorted(zip(df["seed count"], df[column_name])))
        plt.plot(xs, ys, label=policy, linestyle='dashed', marker=markers[idx], color=colors[idx], markersize=14)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("k", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 1, 1, 0.25), ncols = 4)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_fairness_bars(welfare_analysis_df, column_name,methods, y_label,filename,k_list=[5, 10, 25]):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1,3, sharey=True, figsize=(30,15))

    for ax_id, k in enumerate(k_list):
        df = welfare_analysis_df[welfare_analysis_df["seed count"] == k][["policy", column_name]]
        #filter by policy in methods
        df = df[df["policy"].isin(methods)]
        df = df.explode(column_name).reset_index().rename(columns={'index': 'distance_from_M'})
        df['distance_from_M'] = df.groupby('distance_from_M').cumcount()
        df['distance'] = df['distance_from_M']+1
        df = df.pivot(index='policy', columns='distance', values=column_name)
        # plot bar
        df.plot.bar(ax=axs[ax_id], rot=90, figsize=(10, 7) , label='distance')
        axs[ax_id].set_xlabel("Method", fontsize=30)
        axs[ax_id].set_ylabel(y_label, fontsize=30)
        axs[ax_id].legend(loc='upper center', bbox_to_anchor=(0, 1, 1, 0.1), ncols = 3)


    #save figure
    fig.savefig(f"{filename}{k}.pdf", bbox_inches="tight")
    plt.close()


def plot_bar_df(welfare_analysis_df, column_name,methods, y_label,filename, k_p=25):
    df = welfare_analysis_df[welfare_analysis_df["seed count"] <= k_p]
    df = df[df["policy"].isin(methods)]
    pd.pivot_table(df, index="policy", columns="seed count", values=column_name).plot(kind="bar", rot=90,
                                                                                           figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("k", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 1, 1, 0.1), ncols = 3)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()



