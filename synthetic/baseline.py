import argparse
import random

import numpy as np
import pandas as pd

from eval.self_sound import eval_sound
from utils.globals import FEATURE_LS


def get_gt_arr(pth_name: str):
    """
    Return the array of the ground truth
    """
    df_gt = pd.read_excel(f'synthetic/data/fq_{pth_name}.xlsx', sheet_name='char')
    gt_val = [ipa2val[df_gt['被切字声母'].iloc[i]] for i in range(len(df_gt))]
    gt_val = np.array(gt_val)
    assert gt_val.shape == (len(df_gt), 14)
    return gt_val


def major_vote(pth_name: str, choice='feature'):
    """
    equal rate and avg L1 distance derived by majority vote
    """
    gt_arr = get_gt_arr(pth_name)

    dia_ini_ls = list()

    if choice == 'ipa':
        for p in range(20):
            df_dialect = pd.read_excel(f'data/dia_{pth_name}.xlsx', sheet_name=f'{p}_dialect')
            dia_ini_ls.append(list(df_dialect['ini_dialect']))
        dia_ini_ls = list(map(list, zip(*dia_ini_ls)))

        char_num = len(dia_ini_ls)
        assert len(dia_ini_ls[0]) == 20

        voted_ini = list()
        for i in range(char_num):
            random.shuffle(dia_ini_ls[i])
            freq_ini = max(dia_ini_ls[i], key=dia_ini_ls[i].count)
            # print(dia_ini_ls[i], freq_ini)
            voted_ini.append(freq_ini)

        voted_val = np.array([ipa2val[i] for i in voted_ini])
        assert voted_val.shape == (char_num, 14)

    elif choice == 'feature':
        for p in range(20):
            df_dialect = pd.read_excel(f'synthetic/data/dia_{pth_name}.xlsx', sheet_name=f'{p}_dialect')
            dia_ini_ls.append(np.array(df_dialect[FEATURE_LS]))
        dia_ini = np.array(dia_ini_ls)  # eg: (20, 2033, 14)
        dia_num, char_num, feat_num = dia_ini.shape

        voted_val = np.zeros((char_num, feat_num))
        for i in range(char_num):
            for j in range(feat_num):
                tmp_ls = dia_ini[:, i, j].tolist()
                random.shuffle(tmp_ls)
                freq_ini = max(tmp_ls, key=tmp_ls.count)
                voted_val[i][j] = freq_ini
        df_voted = pd.DataFrame(voted_val, columns=FEATURE_LS)
        eval_sound(df_voted)
    else:
        raise ValueError('undefined choice')

    sim = np.linalg.norm(gt_arr - voted_val, axis=1, ord=1)
    assert sim.size == char_num
    res = np.where(sim < 1e-4, 1, 0)
    ER = round(np.mean(res), 4)
    avg_l1 = round(np.mean(sim), 4)

    print(f"****** {pth_name} ******")
    print(f'{ER} portion of answer is equal to ground truth')
    print(f'The average distance between answer and authentic phonology is {avg_l1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', '-pth', type=str, default='fq_0.1_dia_0.7_char_0.7_phon_Latin',
                        help='the place where phonology is generated')
    parser.add_argument('--vote_choice', type=str, choices=['ipa', 'feature'], default='feature',
                        help='the place where phonology is generated')
    args = vars(parser.parse_args())

    df_phon = pd.read_excel('data/IPA/MyFeatures.xlsx', sheet_name='add_diacritics')
    ipa2val = {df_phon['sound'].iloc[i]: np.array(df_phon[FEATURE_LS].iloc[i]) for i in range(len(df_phon))}

    pth = args['pth']
    print(f"Majority vote baseline on setting {pth}, with choice {args['vote_choice']}")
    major_vote(pth_name=pth, choice=args['vote_choice'])
