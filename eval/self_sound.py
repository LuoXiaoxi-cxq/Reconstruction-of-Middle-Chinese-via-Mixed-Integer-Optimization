import pandas as pd
import numpy as np
from utils.globals import D_FEATURE_DICT, FEATURE_LS
import os
import pickle


def check_dependent(indep_f: float, dep_f: float, feature: str):
    """
    indep_f: the value of independent feature
    dep_f: the value of dependent feature
    feature: the name of dependent feature
    """
    if feature in ['labiodental', 'anterior', 'distributed']:
        # if labial is -1, labiodental must be 0
        penalty_1 = abs(dep_f) + (indep_f + 1)
        # if labial is 1, labiodental can be 1 or -1
        penalty_2 = min(abs(dep_f - 1), abs(dep_f + 1)) + abs(indep_f - 1)
        penalty = min(penalty_1, penalty_2)

    elif feature in ['high', 'front']:
        # if dorsal is -1, high/front must be 0
        penalty_1 = abs(dep_f) + (indep_f + 1)
        # if dorsal is 1, high/front must >=1
        tmp = 10000
        for hf in range(1, 4):
            tmp = min(tmp, abs(hf - dep_f))
        penalty_2 = tmp + abs(indep_f - 1)
        penalty = min(penalty_1, penalty_2)

    elif feature == 'delayed_release':
        # sonority must be 1, delayed release can be 1 or -1
        penalty_1 = abs(indep_f - 1) + min(abs(dep_f - 1), abs(dep_f + 1))
        # if sonority = 2/3/4/5, delayed release must be 0
        tmp = 10000
        for sono in range(2, 6):
            tmp = min(tmp, abs(sono - indep_f))
        penalty_2 = tmp + abs(dep_f)
        penalty = min(penalty_1, penalty_2)
    return penalty


def penalty_zero_initial(df):
    # independent and dependent feature can both be 0 with zero initial
    penalty = 10000
    for dep, indep in D_FEATURE_DICT.items():
        penalty = min(penalty, abs(df[dep]) + abs(df[indep]))
    return penalty


def eval_sound(df_recon: pd.DataFrame):
    disagree = 0
    sound_num = 0
    for i in range(len(df_recon)):
        char_dis = 0
        for l2, l1 in D_FEATURE_DICT.items():
            dis = check_dependent(indep_f=df_recon[l1].iloc[i], dep_f=df_recon[l2].iloc[i], feature=l2)
            char_dis += dis
        char_dis = min(char_dis, penalty_zero_initial(df=df_recon.iloc[i]))
        disagree += char_dis
        if char_dis < 1e-2:
            sound_num += 1
    print('average disagreement is: ', round(disagree / len(df_recon), 4))
    print(f'{round(sound_num / len(df_recon), 4)} portion of characters have self-sound reconstruction')

