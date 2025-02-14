from utils.globals import FEATURE_LS, I_FEATURE_LS, D_FEATURE_DICT, SCALE_DICT
from gurobipy import *


def qiuyi_component(comp_a: dict, comp_b: dict, model):
    qiuyi_diff = QuadExpr()
    for feature in feat_ls:
        qiuyi_diff += (comp_a[feature] - comp_b[feature]) * (comp_a[feature] - comp_b[feature])
    model.addConstr(qiuyi_diff >= 1)


def compare_L1_var(a1: Var, b1: Var or int, name: str, var_name: str, model) -> QuadExpr:
    """
    return the terms in objective function corresponding to the comparation of L1 variables
    """
    scale = SCALE_DICT[name] if SCALE_DICT.get(name) is not None else 2
    L1_0 = model.addVar(lb=-scale, ub=scale, name=f"{var_name}_L1_0")  # a1-b1
    L1_1 = model.addVar(lb=0, ub=scale, name=f"{var_name}_L1_1")  # |a1-b1|
    model.addConstr(L1_0 == a1 - b1, name=f"{var_name}_constr_L1_0")
    model.addConstr(L1_1 == abs_(L1_0), name=f"{var_name}_constr_L1_1")
    return QuadExpr(L1_1)


def compare_L2_var(a1: Var, a2: Var, b1: Var or int, b2: Var or int, name: str, var_name: str, model) -> QuadExpr:
    """
    a2, b2: L2 variables
    a1, b1: the L1 variables corresponding to them
    name: the name of L2 variable
    h(x) = min{x,1}, c = h(|a1-b1|), L += 2 * scale2 + (1-c) |a2-b2|
    return the terms in objective function corresponding to the comparison of L2 variables
    """
    scale1 = 5 if name == 'delayed_release' else 2

    if name in ['high', 'front']:
        scale2 = 3
    else:
        scale2 = 2

    L1_tmp1 = model.addVar(lb=-scale1, ub=scale1, name=f"{var_name}_L1_tmp1")  # a1-b1
    L1_tmp2 = model.addVar(lb=0, ub=scale1, name=f"{var_name}_L1_tmp2")  # |a1-b1|
    L1_tmp3 = model.addVar(lb=0, ub=1, name=f"{var_name}_L1_tmp3")  # c=min{|a1-b1|,1} \in (0,1)
    model.addConstr(L1_tmp1 == a1 - b1, name=f"{var_name}_constr1")
    model.addConstr(L1_tmp2 == abs_(L1_tmp1), name=f"{var_name}_constr3")
    model.addConstr(L1_tmp3 == min_(L1_tmp2, constant=1), name=f"{var_name}_constr5")

    L2_tmp1 = model.addVar(lb=-scale2, ub=scale2, name=f"{var_name}_L2_tmp1")  # a2-b2
    L2_tmp2 = model.addVar(lb=0, ub=scale2, name=f"{var_name}_L2_tmp2")  # |a2-b2|
    model.addConstr(L2_tmp1 == a2 - b2, name=f"{var_name}_constr2")
    model.addConstr(L2_tmp2 == abs_(L2_tmp1), name=f"{var_name}_constr4")
    return QuadExpr(L1_tmp3 * scale2 + (1 - L1_tmp3) * L2_tmp2)


def compare_component(comp_a: dict, comp_b: dict, var_name: str, char_a: str, char_b: str, model) -> QuadExpr:
    """
    compare a component of two syllables. Return objectives and add constraints to the model.
    :param comp_a: dict like {feature1: Var1, ...} in syllable a
    :param comp_b: dict like {feature1: Var1(or int), ...} in syllable b
    """
    comp_obj = QuadExpr(0)
    for feature in FEATURE_LS:
        var_name_new = f"{var_name}_{feature}"
        if feature in I_FEATURE_LS:
            comp_obj += compare_L1_var(comp_a[feature], comp_b[feature], name=feature, var_name=var_name_new,
                                       model=model)
        else:
            assert D_FEATURE_DICT.get(feature) is not None
            L1_feature = D_FEATURE_DICT[feature]
            comp_obj += compare_L2_var(a1=comp_a[L1_feature], a2=comp_a[feature], b1=comp_b[L1_feature],
                                       b2=comp_b[feature], name=feature, var_name=var_name_new, model=model)
    return comp_obj


def L2_restrict(a1: Var, a2: Var, name: str, var_name: str, model):
    """
    Add constraints and objectives of L2 variables according to the relationship between L1 and L2 variables.
    :param a1: L1 variable
    :param a2: L2 variable, |a2| <= max(a1,0)
    :return: max(a1^3,0)*(-a2^2+1)
    """
    if name == 'delayed_release':
        L1_tmp0 = model.addVar(lb=-3, ub=2, name=f"{var_name}_L1_tmp0")  # 2-sonority
        L1_tmp1 = model.addVar(lb=-3, ub=1, name=f"{var_name}_L1_tmp1")  # min(sonority, 2-sonority)
        L1_tmp2 = model.addVar(lb=0, ub=1, name=f"{var_name}_L1_tmp2")  # max(0, min(sonority, 2-sonority))
        L2_tmp1 = model.addVar(lb=0, ub=1, name=f"{var_name}_L2_tmp1")  # |delayed_release|

        model.addConstr(L1_tmp0 == 2 - a1, name=f"{var_name}_constr0")
        model.addConstr(L1_tmp1 == min_(a1, L1_tmp0), name=f"{var_name}_constr1")
        model.addConstr(L1_tmp2 == max_(L1_tmp1, constant=0), name=f"{var_name}_constr2")
        model.addConstr(L2_tmp1 == abs_(a2), name=f"{var_name}_constr3")
        model.addConstr(L2_tmp1 <= L1_tmp2, name=f"{var_name}_constr4")

    else:
        scale = SCALE_DICT[name] if SCALE_DICT.get(name) is not None else 1

        b = model.addVar(vtype=GRB.BINARY, name=f"{var_name}_b")
        M, eps = 2, 0.01
        model.addConstr(a1 >= 0.5 + eps - M * (1 - b), name=f"{var_name}_bigM_constr1")
        model.addConstr(a1 <= 0.5 + M * b, name=f"{var_name}_bigM_constr2")
        if name in ['high', 'front']:
            L2_tmp1 = model.addVar(lb=1 - scale, ub=1, name=f"{var_name}_L2_tmp1")  # 1-a2
            L2_tmp2 = model.addVar(lb=0, ub=1, name=f"{var_name}_L2_tmp2")  # max(0, 1-a2)
            model.addConstr(L2_tmp1 == 1 - a2, name=f"{var_name}_constr1")  # 1-a2
            model.addConstr(L2_tmp2 == max_(L2_tmp1, constant=0), name=f"{var_name}_constr2")  # max(0, 1-a2)
            model.addConstr(L2_tmp2 == 1 - b, name=f"{var_name}_constr3")  # max(0, 1-a2)
        else:
            L2_tmp1 = model.addVar(lb=0, ub=scale, name=f"{var_name}_L2_tmp1")  # |a2|
            model.addConstr(L2_tmp1 == abs_(a2), name=f"{var_name}_constr1")  # |a2|
            model.addConstr(L2_tmp1 == b, name=f"{var_name}_constr4")  # |a2|=b


def restrict(var_dict: dict, idx: int, char: str, var_name: str, model):
    """
    get constraints and objects in a component of syllable.
    :param var_dict: dict like {feature1: Var1, ...}, representing features in a component of syllable
    :param idx: index of the char in CharacterSet to be compared
    """
    for feature in FEATURE_LS:
        var_name_new = f"re_{var_name}_{feature}"
        if D_FEATURE_DICT.get(feature) is not None:
            L1_feature = D_FEATURE_DICT[feature]
            L2_restrict(a1=var_dict[L1_feature], a2=var_dict[feature], name=feature, var_name=var_name_new, model=model)
