"""
Calculate sound values of initials with data from authentic phonology.
"""
import argparse
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import *

from eval.self_sound import eval_sound
from eval.cluster_eval import eval_clustering
from src.grb_func import compare_component, restrict
from utils.globals import FEATURE_LS
from utils.util import arg2filename, my_mkdir

place_ls = range(20)


class Character:
    """
    - A single character, as well as its features in different dialects
      and variables of different phonology in ancient times
    """

    def __init__(self, char, idx, ini, medial):
        """
        idx: index of the character in the selected set
        dialect_feature : features of characters in each dialect
            {dialect_name: {idx1: {feature1: xxx, ... }, ...} ...}
        dialect_pronounce: pronunciation of characters in each dialect
            {dialect_name: {idx1: xxx, ...}}
        Maybe add key idx1 in case that the character has different pronunciations (?)
        """
        self.char = char
        self.idx = idx
        self.dialect_pronounce = dict()
        self.dialect_feature = dict()
        self.ini_gt = ini
        self.medial = medial  # IPA
        self.ancient_variable = dict()

    def add_variable(self) -> dict:
        """
        add variables to instances of Character
        self.ancient_variable: {1I: {continuant: var,...}, 2M:{...}}
        """
        for feature in FEATURE_LS:
            tmp_var = self.get_variable(self.idx, feature)
            self.ancient_variable[feature] = tmp_var
        model.update()

    def get_variable(self, char_idx, feature_name) -> Var:
        """
        :param char_idx: index of the char
        :param feature_name: the name of the feature
        :return: variable of the feature in GUROBI
        """
        f_name = f"{char_idx}_{feature_name}"
        if feature_name == 'sonority':
            var = model.addVar(lb=0, ub=5, vtype=GRB.CONTINUOUS, name=f_name)
        elif feature_name == 'high':
            var = model.addVar(lb=0, ub=7, vtype=GRB.CONTINUOUS, name=f_name)
        elif feature_name == 'front':
            var = model.addVar(lb=0, ub=3, vtype=GRB.CONTINUOUS, name=f_name)
        else:
            var = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name=f_name)
        return var


class CharacterSet:
    """
    - self.char_list includes all instances of Character class.
    - Methods about known phonological systems (e.g.: dialects) will be included here.
    - Deal with inter-phonological-system-relationships.
    """

    def __init__(self, data_name: str):
        self.char_list = list()  # list of instances of class Character
        self.data_name = data_name

        self.df_fq = pd.read_excel(f"synthetic/data/fq_{data_name}.xlsx", sheet_name='char')
        print(f"{len(self.df_fq['被切字'])} characters, {len(set(self.df_fq['被切字声母']))} initials in total")

        for i in range(len(self.df_fq)):
            char_name = self.df_fq['被切字'].iloc[i]
            tmp_medial = self.df_fq['被切字介音'].iloc[i]  # add medial info

            tmp_char_instance = Character(char=char_name, idx=i, ini=self.df_fq['被切字声母'].iloc[i],
                                          medial=tmp_medial)
            tmp_char_instance.add_variable()
            self.char_list.append(tmp_char_instance)

    def add_dialect(self) -> None:
        """
        assign dialect_feature and dialect_pronounce to each character.
        dialect_feature : {dialect_name: {id1: {feature1: xxx, ... }, ...}}
        dialect_pronounce: {dialect_name: {id1: xxx, id2: xxx, ...}}
        """
        for place in place_ls:
            df_dialect = pd.read_excel(f'synthetic/data/dia_{self.data_name}.xlsx', sheet_name=f'{place}_dialect')
            for j in range(len(df_dialect)):
                char = df_dialect['id'].iloc[j]
                idx = int(char)
                assert j == idx and self.char_list[idx].idx == idx
                self.char_list[idx].dialect_feature[place] = dict()
                self.char_list[idx].dialect_pronounce[place] = dict()

                feature_dict = {s: df_dialect[s].iloc[j] for s in FEATURE_LS}
                self.char_list[idx].dialect_feature[place][idx] = feature_dict

                pronounce_1I = df_dialect['ini_dialect'].iloc[j]
                self.char_list[idx].dialect_pronounce[place][idx] = pronounce_1I

    def get_fanqie_obj(self, fanqie_idx: int, char_idx: int, var_name: str) -> QuadExpr:
        fanqie_obj = QuadExpr()

        var_name_fanqie = f"{var_name}_{char_idx}"
        medial_fq, medial_char = self.char_list[fanqie_idx].medial, self.char_list[char_idx].medial
        if medial_fq == medial_char:
            fq_w = args['fq_medial_weight']
        else:
            fq_w = 1

        var_name_fanqie = f"{var_name}_{char_idx}"
        fanqie_obj += fq_w * compare_component(comp_a=self.char_list[fanqie_idx].ancient_variable,
                                               comp_b=self.char_list[char_idx].ancient_variable,
                                               var_name=var_name_fanqie, char_a=self.char_list[fanqie_idx].char,
                                               char_b=self.char_list[char_idx].char, model=model)
        return fanqie_obj

    def calculate(self):
        # 反切
        fanqie_obj = QuadExpr(100)
        if args['fanqie']:
            print('Dealing with FanQie...')
            for j in range(len(self.df_fq)):
                assert self.char_list[j].char == self.df_fq['被切字'].iloc[j]
                shangzi = self.df_fq['上字'].iloc[j]
                sz_idx = int(shangzi)
                if sz_idx is not None:
                    fanqie_obj += self.get_fanqie_obj(fanqie_idx=sz_idx, char_idx=j, var_name='fanqie')

        # 方言音值
        # dialect_feature : {dialect_name: {zihui_idx: {1I: {feature1: xxx, ... }}, ...}}
        dialect_value_obj = QuadExpr(0)
        print('Dealing with sound values of dialects...')
        for place in place_ls:
            print(f"Dealing with dialect in {place}...")
            for char_obj in self.char_list:
                if char_obj.dialect_feature.get(place) is not None:
                    dia_dict = char_obj.dialect_feature[place]
                    # dia_value is a dict in the form of {feature1: xxx, ... }
                    for dia_value in dia_dict.values():
                        var_name = f"dialect_{place}_{char_obj.idx}"
                        dialect_value_obj += compare_component(comp_a=char_obj.ancient_variable, comp_b=dia_value,
                                                               var_name=var_name, char_a=char_obj.char,
                                                               char_b=char_obj.char, model=model)

        # restrictions to the values of L1 and L2 variables
        if args['restrict']:
            print('Adding restrictions to the values of L1 and L2 variables...')
            for char_obj in self.char_list:
                restrict(var_dict=char_obj.ancient_variable, idx=char_obj.idx, char=char_obj.char,
                         var_name=char_obj.idx, model=model)

        final_obj = fanqie_obj * args['fanqie_weight'] + dialect_value_obj
        return final_obj

    def verify(self, df_ans: pd.DataFrame, best_labels: list):
        arr_ans = np.array(df_ans[FEATURE_LS])

        df_phon = pd.read_excel('data/IPA/MyFeatures.xlsx', sheet_name='add_diacritics')
        ipa2val = {df_phon['sound'].iloc[i]: np.array(df_phon[FEATURE_LS].iloc[i]) for i in range(len(df_phon))}

        df_gt = pd.read_excel(f'synthetic/data/fq_{self.data_name}.xlsx', sheet_name='char')
        gt_val = [ipa2val[df_gt['被切字声母'].iloc[i]] for i in range(len(df_gt))]
        gt_val = np.array(gt_val)
        assert gt_val.shape == (len(df_gt), 14)

        sim = np.linalg.norm(gt_val - arr_ans, axis=1, ord=1)
        assert sim.size == len(df_ans)
        res = np.where(sim < 1e-4, 1, 0)
        print(f'{round(np.mean(res), 4)} portion of answer is equal to ground truth')
        print(f'The average distance between answer and authentic phonology is {np.mean(sim)}')

        fq_violate = check_fq(self.data_name, arr_ans, best_labels)

        return fq_violate.ravel(), sim.ravel()


def check_fq(data_name, arr_ans: np.array, best_labels: list):
    # check the violation of fanqie
    df_fq = pd.read_excel(f'synthetic/data/fq_{data_name}.xlsx', sheet_name=f'char')
    up_idx = np.array(df_fq['被切字'])
    low_idx = np.array(df_fq['上字'])
    up_arr = arr_ans[up_idx, :]
    low_arr = arr_ans[low_idx, :]
    assert up_arr.shape == (len(df_fq), 14)
    sim = np.linalg.norm(up_arr - low_arr, axis=1, ord=1)
    assert sim.size == len(df_fq)
    print(f'{round(np.mean(np.where(sim < 1e-4, 1, 0)), 4)} portion of fanqie spellings are satisfied')
    print(f'The average distance between fanqie upper and lower speller is {round(np.mean(sim), 4)}')

    best_labels = np.array(best_labels)
    up_label = best_labels[up_idx]
    low_label = best_labels[low_idx]
    unsatisfy_position = np.where(up_label == low_label, 0, 1)

    print(f'Clustering labels: {len(df_fq) - np.sum(unsatisfy_position)} pairs of fanqie are satisfied, '
          f'{len(df_fq)} pairs in total')

    return up_label == low_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_p', '-rand_p', type=float, default=0,
                        help='the portion of quadratic terms')
    parser.add_argument('--run_time', '-t', type=float, default=14400,
                        help='the time of experiment')
    parser.add_argument('--fq_medial_weight', '-fqmw', type=int, default=1,
                        help='the weight of fanqie notations with the same medial in the objective function')
    parser.add_argument('--fanqie_weight', '-w', type=int, default=20,
                        help='the weight of Fanqie in the objective function')
    parser.add_argument('--fanqie', '-fq', type=bool, default=True,
                        help='whether the constraints and objective terms of Fanqie will be included')
    parser.add_argument('--restrict', '-r', type=bool, default=True,
                        help='whether the constraints and objective terms about values of variables will be included')
    parser.add_argument('--data_name', '-pth', type=str, default='fq_0.05_dia_0.2_char_0.2_phon_Latin',
                        help='the place where phonology is generated')
    parser.add_argument('--sol', '-sol', type=str, default='',
                        help='the place where previous solution is saved')

    args = vars(parser.parse_args())
    print("args: ", args)
    file_name = arg2filename(key_dct={"fq_medial_weight": "fqmw", "fanqie_weight": "fqw"},
                             file_name=args["data_name"], arg=args)
    print("file name = ", file_name)

    data_dir = f'synthetic/result/{file_name}/'
    Path(data_dir).mkdir(exist_ok=True, parents=True)

    model = Model('Authentic_Phonology')
    model.Params.NonConvex = 2
    model.Params.PoolSolutions = 3
    model.Params.MIPGap = 0.0001
    model.Params.FeasibilityTol = 1e-4
    model.Params.TimeLimit = args['run_time']
    enable_scale = True
    model.Params.SolFiles = data_dir + "sol"

    Char_Set = CharacterSet(data_name=args['data_name'])
    Char_Set.add_dialect()
    final_obj = Char_Set.calculate()

    model.setObjective(final_obj, GRB.MINIMIZE)
    model.update()
    print(model)

    if args['sol']:
        model.read(args['sol'])

    model.optimize()
    if model.status == GRB.Status.INFEASIBLE:
        grb_infeasible(model)

    if model.Status == 2 or model.Status == GRB.Status.TIME_LIMIT or model.Status == 11:  # 11: interrupted
        print("Final MIP gap value: %f" % model.MIPGap)
        if model.Status == GRB.Status.TIME_LIMIT:
            print('tle!')
        elif model.Status == 2:
            print("Model is optimal")
        elif model.Status == 11:
            print("Model is interrupted!")

        writer_sol = pd.ExcelWriter(f"{data_dir}/syn_{file_name}.xlsx", engine='openpyxl')

        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            print(f"saving solution {i}: Obj_{i} = {model.PoolObjVal}")

            all_value = []
            for obj in Char_Set.char_list:
                char_res = [obj.char, obj.ini_gt]
                for feature in FEATURE_LS:
                    f_name = f"{obj.idx}_{feature}"
                    char_res.append(model.getVarByName(f_name).Xn)
                all_value.append(char_res)

            df_res = pd.DataFrame(all_value, columns=['char', 'ini_gt'] + FEATURE_LS)

            eval_sound(df_res)

            # eval clustering
            label_gt = list(df_res['ini_gt'])
            best_label = eval_clustering(df=df_res, gt=label_gt, feature_ls=FEATURE_LS)

            # compare results with ground truth
            fq_violate, sim = Char_Set.verify(df_res, best_label)

            df_res = pd.concat([df_res, pd.Series(fq_violate, name='fq_violate')
                                   , pd.Series(sim, name='dis_from_gt')], axis=1)
            df_res.to_excel(excel_writer=writer_sol, sheet_name=f"solution_{i}", index=False)

        writer_sol.close()
