"""
Modeling the diachronic change from Guangyun to modern Chinese dialects.
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import *

from eval.self_sound import eval_sound
from src.grb_func import compare_component, restrict
from utils.globals import FEATURE_LS, PLACE_LS
from utils.util import grb_infeasible, arg2filename, get_boolean
from eval.cluster_eval import eval_clustering


class Character:
    """
    - A single character, as well as its features in different dialects
      and variables of different phonology in ancient times
    """

    def __init__(self, char, idx, medial):
        """
        idx: index of the character in the selected set
        dialect_feature : features of characters in each dialect
            {dialect_name: {feature1: xxx, ... }, ...}
        dialect_pronounce: pronunciation of characters in each dialect
            {dialect_name: {zihui_idx: xxx, ...}}
        """
        self.char = char
        self.idx = idx
        self.dialect_pronounce = dict()
        self.dialect_feature = dict()
        self.ancient_variable = None
        self.medial = medial  # (开合, 等)

    def add_variable(self, char_idx) -> dict:
        """
        add variables to instances of Character
        self.ancient_variable: {1I: {continuant: var,...}, 2M:{...}}
        """
        self.ancient_variable = dict()
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

    def __init__(self):
        self.verify_df = None
        self.char_list = list()  # list of instances of class Character
        self.char2idx = dict()  # char to the index in char_list
        self.gy2idx = dict()  # '廣韻字序' to the index in char_list

        with open(f'data/char_1960_2661.pickle', 'rb') as f2:
            self.char_set = pickle.load(f2)

        df_gy = pd.read_excel("data/Guangyun.xlsx")
        self.df_gy = df_gy[df_gy['字頭'].isin(self.char_set)]
        self.verify_flag = [False] * len(self.df_gy)  # for evaluation

        cnt = 0
        for i in range(len(self.df_gy)):
            char_name = self.df_gy['字頭'].iloc[i]
            if char_name not in self.char_set:
                continue

            self.gy2idx[self.df_gy['廣韻字序'].iloc[i]] = cnt

            tmp_medial = (self.df_gy['呼'].iloc[i], self.df_gy['等'].iloc[i])  # add medial info

            tmp_char = Character(char=char_name, idx=cnt, medial=tmp_medial)
            tmp_char.add_variable(char_idx=cnt)
            self.char_list.append(tmp_char)

            if self.char2idx.get(char_name) is None:
                self.char2idx[char_name] = []
            self.char2idx[char_name].append(cnt)
            cnt += 1
        print(f"Added {cnt} characters in total")

        # convert 广韵字序 to index
        self.common_pron = np.load('data/fanqie_common.npy', allow_pickle=True).item()

    def add_dialect(self) -> None:
        """
        assign dialect_feature and dialect_pronounce to each character.
        dialect_feature : {dialect_name: {zh_id1: {feature1: xxx, ... }, ...}}
        dialect_pronounce: {dialect_name: {zh_id1: xxx, zh_id1: xxx, ...}}
        """
        for place in PLACE_LS:
            for df_name in ['train', 'result']:
                df_dialect = pd.read_excel("data/align_" + df_name + "_initial.xlsx", sheet_name=place)
                for i in range(len(df_dialect)):
                    char = df_dialect['字頭'].iloc[i]
                    if char not in self.char_set:
                        continue
                    idx = self.gy2idx[df_dialect['廣韻字序'].iloc[i]]
                    assert self.char_list[idx].char == char and self.char_list[idx].idx == idx
                    self.char_list[idx].dialect_feature[place] = dict()
                    self.char_list[idx].dialect_pronounce[place] = dict()

                    zihui_idx = df_dialect['總序號'].iloc[i]

                    feature_dict = {s: df_dialect[s].iloc[i] for s in FEATURE_LS}
                    self.char_list[idx].dialect_feature[place][zihui_idx] = feature_dict

                    pronounce_1I = df_dialect['1I'].iloc[i]
                    self.char_list[idx].dialect_pronounce[place][zihui_idx] = pronounce_1I

    def get_fanqie_idx(self, char) -> None or int:
        # if char == '葅':
        #     return
        if self.char2idx.get(char) is None:
            return
        tmp_ls = self.char2idx.get(char)
        if len(tmp_ls) == 1:
            return tmp_ls[0]
        idx = self.common_pron[char] + 1
        if self.gy2idx.get(idx) is not None:
            return self.gy2idx[idx]

    def get_fanqie_obj(self, fanqie_idx: int, char_idx: int, var_name: str) -> QuadExpr:
        fanqie_obj = QuadExpr()

        medial_fq, medial_char = self.char_list[fanqie_idx].medial, self.char_list[char_idx].medial
        fqmw = args['fq_medial_weight'] if medial_fq == medial_char else 1

        var_name_fanqie = f"{var_name}_{char_idx}"
        fanqie_obj += fqmw * compare_component(comp_a=self.char_list[fanqie_idx].ancient_variable,
                                               comp_b=self.char_list[char_idx].ancient_variable,
                                               var_name=var_name_fanqie, char_a=self.char_list[fanqie_idx].char,
                                               char_b=self.char_list[char_idx].char, model=model)
        return fanqie_obj

    def calculate(self):
        fanqie_obj = QuadExpr(0)
        if get_boolean(args['fanqie']):
            print('Adding FanQie information ...')
            for i in range(len(self.df_gy)):
                if random.random() < args['verify_p']:  # this part of information is used for verify
                    self.verify_flag[i] = True
                    continue
                char_id = self.gy2idx[int(self.df_gy['廣韻字序'].iloc[i])]
                assert self.char_list[char_id].char == self.df_gy['字頭'].iloc[i]
                shangzi = self.df_gy['上字'].iloc[i]
                sz_idx = self.get_fanqie_idx(shangzi)
                if sz_idx is not None:
                    assert self.char_list[sz_idx].char == shangzi
                    fanqie_obj += self.get_fanqie_obj(fanqie_idx=sz_idx, char_idx=char_id, var_name=f'fanqie')

        # 异读
        yidu_obj = QuadExpr(0)
        if get_boolean(args['yidu']):
            print('Adding YiDu information ...')
            for i in range(len(self.df_gy)):
                if pd.isna(self.df_gy['异读'].iloc[i]):
                    continue
                if self.verify_flag[i]:  # this part of information is used for held-out evaluation
                    continue
                char_id = self.gy2idx[int(self.df_gy['廣韻字序'].iloc[i])]
                assert self.char_list[char_id].char == self.df_gy['字頭'].iloc[i]
                yidu_ls = self.parse_yidu(self.df_gy['异读'].iloc[i])
                for yd, type in yidu_ls:
                    if type == "fq":
                        yidu_obj += self.get_fanqie_obj(fanqie_idx=yd, char_idx=char_id, var_name=f'yidu_{yd}')
                    else:  # Zhiyin
                        yidu_obj += compare_component(comp_a=self.char_list[yd].ancient_variable,
                                                      comp_b=self.char_list[char_id].ancient_variable,
                                                      var_name=f"yidu_{yd}_{i}", char_a=self.char_list[yd].char,
                                                      char_b=self.char_list[char_id].char, model=model)
        if args['verify_p'] > 0:
            self.verify_df = self.df_gy.iloc[self.verify_flag]

        # 方言音值
        dialect_value_obj = QuadExpr(0)
        print('Adding dialects...')
        for char_obj in self.char_list:
            assert char_obj.char in self.char_set
            for place in PLACE_LS:
                if char_obj.dialect_feature.get(place) is not None:
                    dia_dict = char_obj.dialect_feature[place]
                    for j_idx, dia_value in enumerate(dia_dict.values()):
                        var_name = f"dialect_{place}_{char_obj.idx}_pron{j_idx}"
                        dialect_value_obj += compare_component(comp_a=char_obj.ancient_variable, comp_b=dia_value,
                                                               var_name=var_name, char_a=char_obj.char,
                                                               char_b=char_obj.char, model=model)

        # restrictions to the values of L1 and L2 variables
        if args['restrict']:
            print('Adding restrictions to the values of L1 and L2 variables...')
            for char_obj in self.char_list:
                assert char_obj.char in self.char_set
                restrict(var_dict=char_obj.ancient_variable, var_name=f"{char_obj.idx}_restrict",
                         idx=char_obj.idx, char=char_obj.char, model=model)

        final_obj = fanqie_obj * args['fanqie_weight'] + yidu_obj + dialect_value_obj
        return final_obj

    def verify_similar(self, char1_idx: int, char1: str, char2_idx: int, char2: str, df_ans: pd.DataFrame):
        """
        compute the similarity of two initials give their idx in self.char_list
        """
        if char2_idx is None:
            print(f"in verify_df, upperspeller {char2} has no fanqie idx")
            return None, None
        assert self.char_list[char1_idx].char == char1 and self.char_list[char2_idx].char == char2
        value1 = np.array(df_ans[FEATURE_LS].iloc[char1_idx])
        value2 = np.array(df_ans[FEATURE_LS].iloc[char2_idx])
        eps = 1e-4
        dist = np.linalg.norm(value1 - value2, ord=1)
        if_same = (dist < eps)
        return dist, if_same

    def verify(self, df_ans: pd.DataFrame):
        """
        evaluate result with held-out data (fanqie and yidu not used in the model)
        df_ans: reconstruction result
        """
        l = len(self.verify_df)
        fq_ls, yidu_ls, tf_ls = list(), list(), list()
        fq_all_cnt, fq_all_diff, fq_cnt = 0, 0, 0
        yidu_all_cnt, yidu_all_diff, yidu_cnt = 0, 0, 0

        for i in range(l):
            tmp_df = self.verify_df.iloc[i]
            char = tmp_df['字頭']
            char_idx = self.gy2idx.get(tmp_df['廣韻字序'])
            if char_idx is None:
                continue

            # eval held-out fanqie
            up_char = tmp_df['上字']
            up_char_idx = self.get_fanqie_idx(char=up_char)
            diff, if_same = self.verify_similar(char1_idx=char_idx, char1=char, char2_idx=up_char_idx,
                                                char2=up_char, df_ans=df_ans)
            if diff is not None:
                fq_ls.append(diff)
                tf_ls.append(if_same)
                fq_all_cnt += 1
                fq_all_diff += diff
                fq_cnt += if_same
            else:
                fq_ls.append('None')
                tf_ls.append('None')

            # eval held-out yidu
            s = tmp_df['异读']
            if pd.isna(s) or s.strip() == '':
                yidu_ls.append('')
                continue
            ls = s.split('\n')
            yidu_tmp = list()
            for s_ in ls:
                if s_.strip() == '':
                    continue
                s_ls = s_.split(',')
                up_char, up_char_idx = s_ls[0][0], self.gy2idx.get(int(s_ls[1]))
                if up_char_idx is None:
                    continue
                diff, if_same = self.verify_similar(char1_idx=char_idx, char1=char, char2_idx=up_char_idx,
                                                    char2=up_char, df_ans=df_ans)
                if diff is None:
                    yidu_tmp.append('None')
                else:
                    yidu_tmp.append(diff)
                    yidu_all_diff += diff
                    yidu_all_cnt += 1
                    yidu_cnt += if_same
            yidu_ls.append(",".join(map(str, yidu_tmp)))

        fanqie_series = pd.Series(fq_ls, name='fanqie_sim')
        yidu_series = pd.Series(yidu_ls, name='yidu_sim')
        tf_series = pd.Series(tf_ls, name='fanqie_tf')
        assert len(fanqie_series) == len(tf_series) == len(self.verify_df)
        print(len(self.verify_df))
        verify_res_df = pd.concat([self.verify_df.reset_index(drop=True), fanqie_series, tf_series, yidu_series],
                                  axis=1)

        print(f'Fanqie: {fq_all_cnt} fanqie notations in total, '
              f'avg diff = {round(fq_all_diff / fq_all_cnt, 4)}')
        print(f'the number of positive example is {fq_cnt}, portion is {round(fq_cnt / fq_all_cnt, 4)}')
        print(f'Yidu: {yidu_all_cnt} Yidu notations in total,'
              f'avg diff = {round(yidu_all_diff / yidu_all_cnt, 4)}')
        print(f'the number of positive example is {yidu_cnt}, portion is {round(yidu_cnt / yidu_all_cnt, 4)}')
        return verify_res_df

    def parse_yidu(self, s: str):
        yidu_parsed = list()
        yidu_ls = s.split("\n")
        for j, yidu in enumerate(yidu_ls):
            if not yidu.strip():
                continue
            yidu_tmp = yidu.split(",")
            assert len(yidu_tmp) in [2, 3]
            type = "fq" if len(yidu_tmp) == 3 else "zy"  # Fanqie or Zhiyin
            if self.gy2idx.get(int(yidu_tmp[1])) is None:
                continue
            yidu = self.gy2idx[int(yidu_tmp[1])]
            assert self.char_list[yidu].char == yidu_tmp[0][0]
            yidu_parsed.append((yidu, type))
        return yidu_parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fq_medial_weight', '-fqmw', type=int, default=3,
                        help='weight assigned to Fanqie spellings (X, X_u) that satisfy X and X_u sharing the same medial')
    parser.add_argument('--fanqie_weight', '-w', type=int, default=20,
                        help='the weight of Fanqie in the objective function')
    parser.add_argument('--fanqie', '-fq', type=bool, default=True,
                        help='Whether to include Fanqie information in modeling.')
    parser.add_argument('--yidu', '-yd', type=bool, default=True,
                        help='Whether to include Yidu information in modeling.')
    parser.add_argument('--restrict', '-r', type=bool, default=True,
                        help='whether constraints designed to obtain a proper phonetic feature vector are incorporated into the model.')
    parser.add_argument('--verify_p', '-vp', type=float, default=0.3,
                        help='Portion of held-out data used for evaluation.')
    parser.add_argument('--sol', '-sol', type=str, default='',
                        help='Path to pre-solved solutions. If you have previously run experiments and saved the solutions,'
                             ' use this parameter to specify the path for loading them.')

    args = vars(parser.parse_args())
    file_name = arg2filename(key_dct={"fq_medial_weight": "fqmw", "fanqie_weight": "fqw",
                                      "verify_p": "vp"}, file_name="gy", arg=args)
    print("file name = ", file_name)
    file_dir = Path(f'result/{file_name}/')
    file_dir.mkdir(exist_ok=True, parents=True)

    # You can adjust the following parameters. For their meanings, please refer to
    # https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters.html#secparameters
    model = Model('Historical_Phonology')
    model.Params.NonConvex = 2
    model.Params.PoolSolutions = 1
    model.Params.MIPGap = 0.00001
    model.Params.TimeLimit = 36000
    # place to save solution file in the optimization process.
    # If you only want the final solution, comment the following line.
    model.Params.SolFiles = f'result/{file_name}/sol'

    Char_Set = CharacterSet()
    Char_Set.add_dialect()
    final_obj = Char_Set.calculate()

    model.setObjective(final_obj, GRB.MINIMIZE)
    model.update()

    print(model)
    if args['sol']:
        model.read(args['sol'])
        print(f'model read solution from path {args["sol"]}')

    model.optimize()
    if model.status == GRB.Status.INFEASIBLE:
        grb_infeasible(model)

    if model.Status == 2 or model.Status == GRB.Status.TIME_LIMIT or model.Status == 11:
        print("Final MIP gap value: %f" % model.MIPGap)
        if model.Status == GRB.Status.TIME_LIMIT:
            print("time limit exceeded!")
        elif model.Status == 2:
            print("Model is optimal")
        elif model.Status == 11:
            print("Model is interrupted!")

        model.Params.SolutionNumber = 0
        with pd.ExcelWriter(file_dir / f"{file_name}result.xlsx", engine='openpyxl') as writer_sol:
            print(f"----------saving solution: Obj = {model.PoolObjVal}----------")

            all_value = []
            for obj in Char_Set.char_list:
                char_res = [obj.char]
                for feature in FEATURE_LS:
                    f_name = f"{obj.idx}_{feature}"
                    char_res.append(model.getVarByName(f_name).Xn)
                all_value.append(char_res)

            df_res = pd.DataFrame(all_value, columns=['char'] + FEATURE_LS)
            df_res.to_excel(excel_writer=writer_sol, index=False)
            print("saved reconstruction result locally!")

            print(f"------ Solution 0, clustering evalution : ------")
            df_gt = pd.read_excel("data/Guangyun.xlsx")
            df_gt = df_gt[df_gt['字頭'].isin(Char_Set.char_set)]
            label_gt = list(df_gt['聲紐'])
            print(len(label_gt), len(df_res))
            eval_clustering(df=df_res, gt=label_gt, feature_ls=FEATURE_LS)
            eval_sound(df_res)

        if args['verify_p']:
            verify_res_df = Char_Set.verify(df_ans=df_res)
            verify_res_df.to_excel(file_dir / f"{file_name}verify.xlsx", index=False)
