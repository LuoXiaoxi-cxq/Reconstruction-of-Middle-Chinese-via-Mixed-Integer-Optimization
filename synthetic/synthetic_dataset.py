"""
Generate an synthetic phonology, including fanqie and dialect materials.
The initial set of consonants can be derived from real phonology, eg: Latin.
"""
import argparse
import random
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from utils.globals import FEATURE_LS
from utils.util import arg2filename


def gaussian(sim_val):
    """
    for a n-dim vector p, which represents similarity, the function generates a
    n-dim vector gaussian noise q, and return p+q
    """
    dim = sim_val.size
    q = np.random.multivariate_normal(mean=np.ones(dim) * args['gauss_mean'], cov=np.eye(dim) * args['gauss_var'],
                                      size=1).ravel()
    new_val = sim_val + q
    return new_val


class Sampler:
    def __init__(self, file_pth, ini_file_name=None):
        # self.all_IPA = pd.read_excel('MyFeatures2.xlsx', sheet_name='diacritics_no_duplicate')
        self.all_IPA = pd.read_excel('data/IPA/MyFeatures.xlsx', sheet_name='add_diacritics')
        self.file_pth = file_pth
        self.ini_file_name = ini_file_name  # the file where predefined phonology is saved, (file_name, sheet_name)
        self.initial_num = None  # number of initials
        self.medial_num = None  # number of medials

        self.ini_IPA = self.all_IPA.loc[(self.all_IPA['sonority'] <= 4) & (self.all_IPA['sonority'] >= 1)]
        self.ini_IPA.insert(0, column='con_index', value=range(len(self.ini_IPA)))

        self.selected_ini = None  # the df containing selected initials
        self.con2idx = self.ini_IPA[['sound', 'con_index']].set_index(['sound']).to_dict()[
            'con_index']  # map non-vowel sounds to index in ini_IPA
        self.rs_ini = None  # the index of chosen initials in ini_IPA

        self.med_IPA = None  # the df containing all the sounds that can be medials
        self.selected_med = None  # the df containing selected medials
        self.rs_med = None  # the index of chosen initials in ini_IPA

    def sample_initial(self):
        self.initial_num = random.randint(30, 40)
        ini_IPA = self.ini_IPA.drop_duplicates(subset=FEATURE_LS, keep='first', ignore_index=False)
        # phonemes that can be initial, with 'sonority'<=4 and non-zero
        self.rs_ini = random.sample(range(0, len(self.ini_IPA)), self.initial_num)
        self.selected_ini = self.ini_IPA.iloc[self.rs_ini]

    def sample_medial(self):
        self.medial_num = random.randint(4, 8)

        remain_idx = list(set(range(0, len(self.ini_IPA))) - set(self.rs_ini))
        remain_IPA = self.ini_IPA.iloc[remain_idx]
        remain_IPA = remain_IPA[remain_IPA['sonority'] >= 4]
        med_IPA_ = pd.concat([self.all_IPA[self.all_IPA['sonority'] == 5], remain_IPA])  # phonemes that can be medial
        self.med_IPA = pd.concat([med_IPA_, self.all_IPA[self.all_IPA['sonority'] == 0]])  # add zero medial
        tmp_ = range(0, len(self.med_IPA))
        self.med_IPA.insert(0, column='med_index', value=tmp_)
        self.rs_med = random.sample(tmp_, self.medial_num)
        self.selected_med = self.med_IPA.iloc[self.rs_med]


class Prechosen_Sampler(Sampler):
    """
    Use predefined phonology, eg: Latin/French.
    """

    def sample_initial(self):
        selected_ini = pd.read_excel(self.ini_file_name[0], sheet_name=self.ini_file_name[1])
        ini_set = set(selected_ini['sound'])
        self.selected_ini = self.ini_IPA[self.ini_IPA['sound'].isin(ini_set)]

        self.initial_num = len(self.selected_ini)
        self.rs_ini = list(self.ini_IPA[self.ini_IPA['sound'].isin(ini_set)]['con_index'])


class Char:
    def __init__(self, ini: str, med: str, idx: int):
        self.initial = ini
        self.med = med
        self.idx = idx


class CharSet:
    def __init__(self, sampler: Sampler, file_pth: str):
        self.all_char_ls = list()
        self.char_ls = dict()  # {ini_a: list, ini_b: list, ...}
        self.char_dict = dict()  # {ini_a: {med_a: list, med_b: list, ... }, ...}
        self.sampler = sampler
        self.feature_arr, self.con_sim_arr = self.cal_dist()
        self.file_pth = file_pth

    def cal_dist(self) -> np.array:
        """
        df_con: charset.sampler.ini_IPA
        arr[i]: the phonetic value of initial i
        con_sim_arr[i][j]: the distance between initial i and j
        """
        arr = np.array(self.sampler.ini_IPA[FEATURE_LS].astype(int))  # shape: (X, 14)
        arr_1 = arr.reshape((arr.shape[0], 1, arr.shape[1]))
        arr_2 = arr.reshape((1, arr.shape[0], arr.shape[1]))
        con_sim_arr = np.linalg.norm(arr_1 - arr_2, ord=1, axis=2)
        assert con_sim_arr.shape == (arr.shape[0], arr.shape[0])
        con_sim_arr += np.where(con_sim_arr < 1e-4, 10000, 0)
        return arr, con_sim_arr

    def dist_dict(self, ini=None, top_k=20) -> dict:
        top_k = min(top_k, self.con_sim_arr.shape[1])
        con_sim_dict = dict()
        if ini is None:
            # for each sound, the idx of top_k most similar sounds
            sim_index = np.argpartition(self.con_sim_arr, top_k, axis=1)[:, 0:top_k]
            h, w = sim_index.shape

            for i in range(h):
                tmp_ipa = self.sampler.ini_IPA['sound'].iloc[i]
                con_sim_dict[tmp_ipa] = dict()
                for j in range(w):
                    tmp_ = self.sampler.ini_IPA['sound'].iloc[sim_index[i, j]]
                    sim = self.con_sim_arr[i][sim_index[i, j]]
                    if sim == 0:
                        continue
                    con_sim_dict[tmp_ipa][tmp_] = sim

        else:
            ini_idx = self.sampler.con2idx[ini]
            # for each sound, the idx of top_k most similar sounds
            sim_index = np.argpartition(self.con_sim_arr[ini_idx], top_k, axis=1)[:, 0:top_k]
            for j in range(top_k):
                tmp_ = self.sampler.ini_IPA['sound'].iloc[sim_index[j]]
                sim = self.con_sim_arr[ini_idx][sim_index[j]]
                if sim == 0:
                    continue
                con_sim_dict[tmp_] = sim

        return con_sim_dict

    def generate_char(self, p=0.5):
        total_idx = 0
        for i in range(len(self.sampler.selected_ini)):
            ini = self.sampler.selected_ini['sound'].iloc[i]
            self.char_dict[ini] = dict()
            self.char_ls[ini] = list()
            med_ls = ['0']
            for j in range(len(self.sampler.selected_med)):
                # if larger than p, initial i cannot have medial j
                if random.uniform(0, 1) > p:
                    continue
                else:
                    med_ls.append(self.sampler.selected_med['sound'].iloc[j])
            for med in med_ls:
                self.char_dict[ini][med] = list()

            num_i = random.randint(20, 80)  # initial i has num_i characters
            for j in range(num_i):
                if med_ls:
                    med = random.choice(med_ls)
                else:
                    med = '0'
                tmp_char = Char(ini=ini, med=med, idx=total_idx)
                self.all_char_ls.append(tmp_char)
                self.char_dict[ini][med].append(total_idx)
                self.char_ls[ini].append(total_idx)
                total_idx += 1

    def make_fanqie(self):
        """
        generate fanqie data, p percent of fanqie notations violate rule
        """
        df_ls = list()
        column = ['被切字', '上字', '被切字声母', '被切字介音', '上字声母', '上字介音', 'note']
        core_char = dict()

        for ini, _ in self.char_dict.items():
            this_ini_num = len(self.char_ls[ini])
            key_char_num = random.randint(2, min(floor(this_ini_num / 5), 10))  # the number of core char
            tmp_sample = random.sample(range(this_ini_num), key_char_num)

            core_char[ini] = [self.char_ls[ini][k] for k in tmp_sample]

        for ini, dict2 in self.char_dict.items():
            new_rs_ini, sim_distri = self.fanqie_ini_change(ini)
            for med, ls in dict2.items():
                for idx in ls:  # character No. i with *ini* as initial and *med* as medial
                    p_violate = args['p_fq_violate']
                    if random.random() > p_violate:  # randomly choose one core char as fanqie upper speller
                        up_speller = random.choice(list(set(core_char[ini]) - {idx}))
                        df_ls.append([idx, up_speller, ini, med, ini, self.all_char_ls[up_speller].med, ""])
                    else:  # find another initial as fanqie upper speller
                        other_ini_idx = np.random.choice(new_rs_ini, p=sim_distri)
                        other_ini = self.sampler.ini_IPA['sound'].iloc[other_ini_idx]
                        up_speller = random.choice(core_char[other_ini])
                        assert other_ini != ini
                        df_ls.append(
                            [idx, up_speller, ini, med, other_ini, self.all_char_ls[up_speller].med, "fq violate"])

        fq_writer = pd.ExcelWriter(f'synthetic/data/fq{self.file_pth}.xlsx')
        df = pd.DataFrame(df_ls, columns=column)
        df.to_excel(fq_writer, sheet_name='fanqie', index=False)
        df2 = df.sort_values('被切字')
        df2.to_excel(fq_writer, sheet_name='char', index=False)
        fq_writer.close()

    def make_dialect(self):
        con_sim_dict = self.dist_dict()
        ini_changed_p = args['dia_ini_p']  # regular change of initials
        char_changed_p = args['char_ini_p']  # irregular change of chars

        df_ls = list()
        df_column = ['id', 'ini_MC', 'ini_dialect', 'note'] + FEATURE_LS
        for ini, ini_ls in self.char_ls.items():
            if random.random() < ini_changed_p:  # change the initial
                changed_ini = self.dialect_ini_change(ini, change_dict=con_sim_dict)
                note1 = f'initial category changed from {ini} to {changed_ini}, '
            else:
                changed_ini = ini
                note1 = ''

            for char_id in ini_ls:
                if random.random() < char_changed_p:
                    char_ini = self.dialect_ini_change(changed_ini, change_dict=con_sim_dict)
                    note2 = f'char initial changed from {changed_ini} to {char_ini}'
                else:
                    char_ini = changed_ini
                    note2 = ''
                char_ini_id = self.sampler.con2idx[char_ini]
                phonetic_val = list(self.feature_arr[char_ini_id].ravel())
                df_ls.append([char_id, ini, char_ini, note1 + note2] + phonetic_val)

        return pd.DataFrame(df_ls, columns=df_column)

    def generate_dialect(self):
        dia_writer = pd.ExcelWriter(f'synthetic/data/dia{file_name}.xlsx')
        for i in range(20):
            df = self.make_dialect()
            df.to_excel(excel_writer=dia_writer, sheet_name=f'{i}_dialect', index=False)
        dia_writer.close()

    def fanqie_ini_change(self, initial: str) -> np.array:
        """
        Calculate the probability distribution of choosing other initials when generating fanqie
        default arr: self.sampler.con_sim_arr
        """
        idx = self.sampler.con2idx[initial]  # Number among all initials
        new_rs_ini = list(set(self.sampler.rs_ini) - {idx})
        sim_line = self.con_sim_arr[idx][new_rs_ini]
        try:
            assert sim_line.all() > 0
        except AssertionError:
            print(idx, ' ', initial, '\n', new_rs_ini, '\n', sim_line)
            raise ValueError("has zero in sim_line")
        sim_line[np.where(sim_line <= 1e-2)] = 1e-2
        sim_prob = 1 / sim_line
        sim_distri = sim_prob / np.sum(sim_prob).ravel()

        return new_rs_ini, sim_distri

    def dialect_ini_change(self, initial: str, change_dict: dict) -> str:
        sim_ini, sim_distri = self.dialect_ini_sim(initial, change_dict, if_gauss=args['if_gauss'])
        changed_ini = np.random.choice(sim_ini, p=sim_distri)
        return changed_ini

    def dialect_ini_sim(self, initial: str, change_dict: dict, if_gauss=False) -> (list, np.array):
        """
        Calculate the probability distribution of sound change when generating dialects
        """
        sim_ini = list(change_dict[initial].keys())
        sim_val = np.array(list(change_dict[initial].values()))
        if if_gauss:
            sim_val = gaussian(sim_val)
            sim_val[np.where(sim_val <= 1e-2)] = 1e-2
        sim_prob = 1 / sim_val  # you can change it to other methods
        sim_distri = sim_prob / np.sum(sim_prob).ravel()

        return sim_ini, sim_distri


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_fq_violate', '-p_fq', type=float, default=0.05,
                        help='the portion of fanqie whose initial and upperspeller are in different categories')
    parser.add_argument('--dia_ini_p', '-p_dia', type=float, default=0.2,
                        help='the portion of initials that are regularly changed in dialects')
    parser.add_argument('--char_ini_p', '-p_char', type=float, default=0.2,
                        help='the portion of chars that are irregularly changed in dialects')
    parser.add_argument('--if_gauss', '-gauss', type=bool, default=True,
                        help='whether to add gaussian noise when generating dialects')
    parser.add_argument('--gauss_mean', '-mu', type=float, default=0,
                        help='the mean of gaussian noise on dialect initials')
    parser.add_argument('--gauss_var', '-sigma', type=float, default=1,
                        help='the deviance of gaussian noise on dialect initials (sqrt of variance)')
    parser.add_argument('--phon', '-phon', type=str, default='Latin',
                        help='Latin/Chinese/German/English')
    args = vars(parser.parse_args())
    print("args: ", args)

    file_name = arg2filename(key_dct={"phon": "phon", "p_fq_violate": "fq", "dia_ini_p": "dia",
                                      "char_ini_p": "char", "dia_num": "dia"},
                             file_name="", arg=args)
    print("file name = ", file_name)

    Path('synthetic/data/').mkdir(exist_ok=True, parents=True)

    if args['phon'] == 'random':
        print('random')
        sampler = Sampler(file_pth=file_name)
    else:
        print(args['phon'])
        sampler = Prechosen_Sampler(file_pth=file_name, ini_file_name=('data/synthetic/language.xlsx', args['phon']))
    sampler.sample_initial()
    sampler.sample_medial()

    charset = CharSet(sampler=sampler, file_pth=file_name)
    charset.generate_char()
    charset.make_fanqie()
    charset.generate_dialect()
