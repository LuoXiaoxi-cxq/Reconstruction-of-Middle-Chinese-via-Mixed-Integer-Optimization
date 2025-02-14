"""
Run baseline (majority vote and single best) on real data.
For majority vote, implemented vote by feature and vote by IPA
"""

import random
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import k_means, AgglomerativeClustering
from pathlib import Path
from utils.globals import FEATURE_LS, PLACE_LS
from .self_sound import eval_sound
import warnings

warnings.filterwarnings("ignore")


def merge_gy_char(save=False, choice='IPA'):
    """
    Merge 'data/align_train_initial' and 'data/align_train_initial' into a single dict
    with format {廣韻字序: {char: 字頭}, {phon: 聲紐}, {dia: [list of IPA in 20 dialects] }}
    """
    assert choice in ['IPA', 'feature']
    if Path(f'data/baseline/gy_vote_dict_{choice}.npy').exists():
        return np.load(f'data/baseline/gy_vote_dict_{choice}.npy', allow_pickle=True).item()

    char_dict = dict()
    print("Merging data ......")
    for df_name in ['train', 'result']:
        for place in PLACE_LS:
            df_dialect = pd.read_excel("data/align_" + df_name + "_initial.xlsx", sheet_name=place)
            for i in range(len(df_dialect)):
                gy_idx = df_dialect['廣韻字序'].iloc[i]
                if char_dict.get(gy_idx) is None:
                    char_dict[gy_idx] = {'char': df_dialect.iloc[i]['字頭'], 'phon': df_dialect.iloc[i]['聲紐'],
                                         'dia': list()}
                if choice == 'IPA':
                    if pd.isnull(df_dialect['聲母音標'].iloc[i]):
                        char_dict[gy_idx]['dia'].append('0')
                    else:
                        char_dict[gy_idx]['dia'].append(df_dialect.iloc[i]['聲母音標'])
                elif choice == 'feature':
                    char_dict[gy_idx]['dia'].append(df_dialect.iloc[i][FEATURE_LS].tolist())
    if save:
        np.save(f'data/baseline/gy_vote_dict_{choice}.npy', char_dict)
    return char_dict


def major_vote(char_dict: dict, choice='IPA') -> list or (list, int):
    """
    IPA-level or feature-level majority vote.
    choice='IPA': choose the IPA with highest frequency across dialects.
    choice='feature': for each feature, choose value with highest frequency across dialects.
    """
    assert choice in ["IPA", "feature"]
    df_gt = pd.read_excel(f"data/1960_2661_gt.xlsx")
    gy_char_ls = list()

    for i in range(len(df_gt)):
        key = df_gt['廣韻字序'].iloc[i]
        item = char_dict.get(key)
        if item is None:
            continue
        assert df_gt.iloc[i]['聲紐'] == item['phon'] and df_gt.iloc[i]['字頭'] == item['char']
        tmp_ls = [key, item['char'], item['phon']]
        dia_ls = item['dia']
        if choice == 'IPA':
            random.shuffle(dia_ls)
            freq_ini = max(dia_ls, key=dia_ls.count)
            tmp_ls.append(freq_ini)
            gy_char_ls.append(tmp_ls)
        elif choice == 'feature':
            dia_ini_ls = list(map(list, zip(*dia_ls)))
            assert len(dia_ini_ls) == 14
            for ls in dia_ini_ls:
                random.shuffle(ls)
                freq_ini = max(ls, key=ls.count)
                tmp_ls.append(freq_ini)
            gy_char_ls.append(tmp_ls)

    print(f"{len(gy_char_ls)} characters are used for majority vote")
    if choice == 'IPA':
        return gy_char_ls
    elif choice == 'feature':
        return gy_char_ls, len(set(df_gt['聲紐']))


def major_vote_IPA():
    """
    IPA-level majority vote. Calculate AMI.
    """
    gy_char_dict = merge_gy_char(save=True, choice="IPA")
    # get voting results
    voted_ls = major_vote(gy_char_dict, choice="IPA")
    # calculate AMI
    df_char_voted = pd.DataFrame(voted_ls, columns=['gy_idx', 'char', 'initial', 'voted_IPA'])
    print(f"{len(voted_ls)} characters are used when calculating AMI")
    # print(df_char_voted.head())
    AMI = metrics.adjusted_mutual_info_score(labels_true=df_char_voted['initial'],
                                             labels_pred=df_char_voted['voted_IPA'])
    print('IPA level AMI is: ', AMI)


def major_vote_feature():
    """
    Feature-level majority vote. Cluster and calculate AMI.
    """
    char_dict = merge_gy_char(save=True, choice='feature')
    voted_ls, num_cate = major_vote(char_dict, choice='feature')
    df_voted = pd.DataFrame(voted_ls, columns=['gy_idx', 'char', 'initial'] + FEATURE_LS)

    # evaluate self-soundness
    eval_sound(df_voted)

    # cluster and AMI
    AMI = 0
    for rand_seed in range(2010, 2030):
        cluster_res = k_means(np.array(df_voted[FEATURE_LS]), n_clusters=num_cate, random_state=rand_seed)
        labels = cluster_res[1]
        AMI_tmp = metrics.adjusted_mutual_info_score(labels_true=df_voted['initial'], labels_pred=labels)
        AMI += AMI_tmp

    print('feature level AMI is: ', round(AMI / 20, 4))


def single_dia_IPA_AMI():
    """
    Calculate the AMI between single dialect and ground truth, based on IPA
    """
    for place in PLACE_LS:
        phon_ls, ipa_ls = list(), list()
        for df_name in ['train_initial', 'result_initial']:
            df_dialect = pd.read_excel("data/align_" + df_name + ".xlsx", sheet_name=place)
            phon_ls += list(df_dialect['聲紐'])
            ipa_ls += list(df_dialect['聲母音標'])
        AMI = metrics.adjusted_mutual_info_score(labels_true=phon_ls, labels_pred=ipa_ls)
        # RI = metrics.adjusted_rand_score(labels_true=phon_ls, labels_pred=ipa_ls)
        print(f'In dialect {place}, AMI is {round(AMI, 4)}')


def single_dia_feature_AMI():
    """
    Calculate the AMI between single dialect and ground truth, based on feature
    """
    for place in PLACE_LS:
        phon_ls, feat_ls = list(), list()
        for df_name in ['train', 'result']:
            df_dialect = pd.read_excel("data/align_" + df_name + "_initial.xlsx", sheet_name=place)
            phon_ls += list(df_dialect['聲紐'])
            feat_ls.append(np.array(df_dialect[FEATURE_LS]))
        feat_arr = np.concatenate((feat_ls[0], feat_ls[1]), axis=0)
        assert feat_arr.shape[1] == 14

        # cluster and AMI
        AMI, RI = 0, 0
        num_cate = len(set(phon_ls))
        for rand_seed in range(2010, 2030):
            cluster_res = k_means(feat_arr, n_clusters=num_cate, random_state=rand_seed)
            labels = cluster_res[1]
            AMI_tmp = metrics.adjusted_mutual_info_score(labels_true=phon_ls, labels_pred=labels)
            RI_tmp = metrics.adjusted_rand_score(labels_true=phon_ls, labels_pred=labels)

            AMI += AMI_tmp
            RI += RI_tmp

        print('feature level AMI is: ', round(AMI / 20, 4))
        print('feature level rand index is: ', round(RI / 20, 4))


if __name__ == "__main__":
    # IPA-level majority vote
    major_vote_IPA()
    # feature-level majority vote
    major_vote_feature()

    # calculate AMI between single dialect and ground truth
    single_dia_IPA_AMI()
    single_dia_feature_AMI()
