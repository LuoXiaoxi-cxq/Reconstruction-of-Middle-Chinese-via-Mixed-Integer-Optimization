"""
Cluster the phonetic values, and evaluation the result by the comparison with
ground-truth phonological categories.
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import k_means
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

from eval.variation_info import calculate_variation_of_information
from eval.self_sound import eval_sound
import pickle


def get_IPA(arr: np.array, center_IPA_save_pth, center_IPA_file_name=None, top_k=8) -> pd.DataFrame:
    """
    Each line in the result array represents the computed feature vector of a character in QieYun.
    For each line, return 8 IPA symbols that have the highest similarity with the current initial, and their similarity.
    :param arr: the calculated feature vectors, to be compared with IPA symbols. shape: (# of characters, 14)
    :param arr_IPA: the feature vectors of known IPA symbols. shape: (# of IPA symbols, 14)
    :param idx2IPA: convert the index (which indicates)
    :return: Dataframe. Find 8 most similar IPA for each line.
    """
    # Load IPA information
    df_all_feature = pd.read_excel("../data/IPA/MyFeatures.xlsx", sheet_name="add_diacritics")
    arr_IPA = np.array(df_all_feature.iloc[:, 1:].astype(int))[:, :14]
    idx2IPA = df_all_feature.reset_index()[['sound', 'index']].set_index(['index']).to_dict()['sound']

    arr_ = arr.reshape((arr.shape[0], 1, arr.shape[1]))
    arr_IPA_ = arr_IPA.reshape((1, arr_IPA.shape[0], arr_IPA.shape[1]))
    sim_arr = np.linalg.norm(arr_IPA_ - arr_, ord=1, axis=2)  # sim_arr: (instance_number, feature_number)
    assert sim_arr.shape[1] == len(idx2IPA)

    sim_index = np.argpartition(sim_arr, top_k, axis=1)[:, 0:top_k]
    print("shape: ", sim_index.shape)

    h, w = sim_index.shape
    res = []
    for i in range(h):
        tmp_res = []
        for j in range(w):
            tmp_res.append([idx2IPA[sim_index[i, j]], sim_arr[i][sim_index[i, j]]])
        tmp_res2 = sorted(tmp_res, key=lambda x: x[1])
        tmp_res2 = [j for i in tmp_res2 for j in i]
        res.append([i] + tmp_res2)

    col_ls = []
    for j in range(w):
        col_ls += [f"ans_{j}", f"dis_{j}"]
    df_res = pd.DataFrame(res, columns=['index'] + col_ls)

    if center_IPA_save_pth:
        if not center_IPA_save_pth.exists():
            center_IPA_save_pth.mkdir(exist_ok=True, parents=True)
        df_res.to_excel(f"{center_IPA_save_pth}/IPA_{center_IPA_file_name}.xlsx", index=False)


def kmeans_cluster(label_true: list, ini_res_: np.array, get_center_IPA=False,
                   center_IPA_save_pth=None, center_IPA_file_name=None, get_labels=False):
    """
    Cluster evaluation 1:
    Use K-Means to cluster the result array into 38 initials, then convert their center into IPA
    """
    cluster_num = len(set(label_true))
    print(f"there are {cluster_num} categories in ground truth")

    max_AMI = 0
    best_labels = None
    total_AMI, total_homo_score, total_comp_score, total_v_score = 0, 0, 0, 0
    variation_info_1, variation_info_2 = 0, 0

    for rand_seed in range(2010, 2030):
        cluster_res = k_means(ini_res_, n_clusters=cluster_num, random_state=rand_seed)
        centers, labels = cluster_res[0], cluster_res[1]
        AMI_metric = metrics.adjusted_mutual_info_score(labels_true=label_true, labels_pred=labels)
        total_AMI += AMI_metric
        if AMI_metric > max_AMI:
            max_AMI = AMI_metric
            best_labels = labels
        if get_center_IPA:
            assert (center_IPA_file_name and center_IPA_save_pth), "save path and file name cannot be none!"
            get_res(centers, center_IPA_save_pth, center_IPA_file_name)
            get_center_IPA = False

        total_homo_score += homogeneity_score(label_true, labels)
        total_comp_score += completeness_score(label_true, labels)
        total_v_score += v_measure_score(label_true, labels)
        variation_info = calculate_variation_of_information(U=label_true, V=labels)
        variation_info_1 += variation_info[0]
        variation_info_2 += variation_info[1]

    return best_labels, {"AMI": round(total_AMI / 20, 4), "homo": round(total_homo_score / 20, 4),
                         "comp": round(total_comp_score / 20, 4), "v_score": round(total_v_score / 20, 4),
                         "variation_info_1": round(variation_info_1 / 20, 4),
                         "variation_info_2": round(variation_info_2 / 20, 4)}


def eval_clustering(df: pd.DataFrame, gt, feature_ls):
    df_ini_recon = np.array(df[feature_ls])
    best_labels, eval_metric = kmeans_cluster(label_true=gt, ini_res_=df_ini_recon)
    print(eval_metric)
    return best_labels
