import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import util


def generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df):
    print("Generating graph...")
    # plot pp
    for t_k in tqdm([0, 5, 10, 15, 20, 25]):
        if t_k not in pp_dict_dict:
            continue
        plt.figure(figsize=(20, 10))
        temp_df = pd.DataFrame()
        prob_dict = {}
        for policy in methods:
            if policy not in pp_dict_dict[t_k]:
                    continue
            for prob, count in pp_dict_dict[t_k][policy].items():
                if prob not in prob_dict:
                    prob_dict[prob] = {}
                prob_dict[prob][policy] = count
        for prob, policy_dict in prob_dict.items():
            temp_df = temp_df.append({"prob": prob, **policy_dict}, ignore_index=True)
        ax = plt.gca()
        temp_df.sort_values(by="prob", ascending=True, inplace=True)
        temp_df.plot.bar(x="prob", y=methods, ax=ax)
        plt.xlabel("save probability")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"{welfare_output_folder}pp_distribution_t_k_{t_k}.pdf")
        plt.close()
    util.plot_k_vs_df_columns(welfare_analysis_df, "impressed", methods,
                              "Number of Negatively Influenced", f"{welfare_output_folder}welfare_analysis.pdf")
    util.plot_k_vs_df_columns(welfare_analysis_df, "saved_fraction", methods,
                              "Fraction of saved nodes", f"{welfare_output_folder}saved_fraction.pdf")
    util.plot_k_vs_df_columns(welfare_analysis_df, "min_prob", methods,
                              "Maximin sp", f"{welfare_output_folder}maximin_sp.pdf")
    util.plot_k_vs_df_columns(welfare_analysis_df, "avg_prob", methods,
                              "Average sp", f"{welfare_output_folder}avg_sp.pdf")

    # plot Average Probability Analysis across partitions
    util.plot_bar_df(welfare_analysis_df, "partition_diff", methods, "Difference in Average Save Probability",
                     f"{welfare_output_folder}welfare_analysis_utility_gap.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency", methods,
                              "Consistency", f"{welfare_output_folder}consistency1.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency2", methods,
                              "Consistency", f"{welfare_output_folder}consistency2.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency3", methods,
                              "Consistency", f"{welfare_output_folder}consistency3.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "zero_prob_count", methods,
                              "Number of nodes with zero sp", f"{welfare_output_folder}zero_prob_count.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr", methods,
                              "Correlation", f"{welfare_output_folder}corr1.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr2", methods,
                              "Correlation", f"{welfare_output_folder}corr2.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr3", methods,
                              "Correlation", f"{welfare_output_folder}corr3.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr4", methods,
                              "Correlation", f"{welfare_output_folder}corr4.pdf")

    util.plot_bar_df(welfare_analysis_df, "partition_min_diff", methods, "Difference in Maximin sp",
                     f"{welfare_output_folder}welfare_analysis_welfare_gap.pdf")

    util.plot_bar_df(welfare_analysis_df, "partition_prox_min_diff", methods, "Difference in Minimum distance from T",
                     f"{welfare_output_folder}welfare_analysis_min_prox_gap.pdf")

    util.plot_k_vs_df_columns(welfare_analysis_df, "num_nodes_closer_to_T", methods,
                              "Number of vulnerable nodes closer to T than M", f"{welfare_output_folder}closerToT.pdf")

    util.plot_bar_df(welfare_analysis_df, "avg_proximity_of_T", methods, "Average distance of T from M",
                     f"{welfare_output_folder}avg_proximity_of_T.pdf")
    # plot welfare_analysis_df avg_sp_at_distance
    util.plot_fairness_bars(welfare_analysis_df, "avg_sp_at_distance", methods, "Average save probability",
                            f"{welfare_output_folder}avg_sp_at_distance_k_")
    util.plot_fairness_bars(welfare_analysis_df, "std_dev_sp_at_distance", methods,
                            "Standard deviation of save probability",
                            f"{welfare_output_folder}std_dev_sp_at_distance_k_")
    util.plot_fairness_bars(welfare_analysis_df, "avg_prox_at_distance", methods, "Average proximity from T",
                            f"{welfare_output_folder}avg_prox_at_distance_k_")
    util.plot_fairness_bars(welfare_analysis_df, "improved_sp_at_distance", methods, "Improvement in save probability",
                            f"{welfare_output_folder}improved_sp_at_distance_k_")

    util.plot_fairness_bars(welfare_analysis_df, "gini_coeff_list", methods, "Gini Coefficient",
                            f"{welfare_output_folder}gini_coef_at_distance_k_")

    util.plot_k_vs_df_columns(welfare_analysis_df, "gini_coef", methods, "Gini Coefficient",
                              f"{welfare_output_folder}gini_coef.pdf")
    print("Graphs plotted")

