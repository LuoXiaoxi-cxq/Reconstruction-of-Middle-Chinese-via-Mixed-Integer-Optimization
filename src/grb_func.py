from utils.globals import FEATURE_LS, I_FEATURE_LS, D_FEATURE_DICT, SCALE_DICT
from gurobipy import *


def qiuyi_component(comp_a: dict, comp_b: dict, model):
    qiuyi_diff = QuadExpr()
    for feature in feat_ls:
        qiuyi_diff += (comp_a[feature] - comp_b[feature]) * (comp_a[feature] - comp_b[feature])
    model.addConstr(qiuyi_diff >= 1)


def compare_I_var(a1: Var, b1: Var or int, name: str, var_name: str, model) -> QuadExpr:
    """
    return the terms in objective function corresponding to the comparison of independent variables
    """
    scale = SCALE_DICT[name] if SCALE_DICT.get(name) is not None else 2
    I0 = model.addVar(lb=-scale, ub=scale, name=f"{var_name}_I0")  # a1-b1
    I1 = model.addVar(lb=0, ub=scale, name=f"{var_name}_I1")  # |a1-b1|
    model.addConstr(I0 == a1 - b1, name=f"{var_name}_constr_I0")
    model.addConstr(I1 == abs_(I0), name=f"{var_name}_constr_I1")
    return QuadExpr(I1)


def compare_D_var(a1: Var, a2: Var, b1: Var or int, b2: Var or int, name: str, var_name: str, model) -> QuadExpr:
    """
    a2, b2: dependent variables
    a1, b1: the independent-variables corresponding to them
    name: the name of D-variable
    h(x) = min{x,1}, c = h(|a1-b1|), L += 2 * scale2 + (1-c) |a2-b2|
    return the terms in objective function corresponding to the comparison of dependent variables
    """
    scale1 = 5 if name == 'delayed_release' else 2

    if name in ['high', 'front']:
        scale2 = 3
    else:
        scale2 = 2

    I_tmp1 = model.addVar(lb=-scale1, ub=scale1, name=f"{var_name}_I_tmp1")  # a1-b1
    I_tmp2 = model.addVar(lb=0, ub=scale1, name=f"{var_name}_I_tmp2")  # |a1-b1|
    I_tmp3 = model.addVar(lb=0, ub=1, name=f"{var_name}_I_tmp3")  # c=min{|a1-b1|,1} \in (0,1)
    model.addConstr(I_tmp1 == a1 - b1, name=f"{var_name}_constr1")
    model.addConstr(I_tmp2 == abs_(I_tmp1), name=f"{var_name}_constr3")
    model.addConstr(I_tmp3 == min_(I_tmp2, constant=1), name=f"{var_name}_constr5")

    D_tmp1 = model.addVar(lb=-scale2, ub=scale2, name=f"{var_name}_D_tmp1")  # a2-b2
    D_tmp2 = model.addVar(lb=0, ub=scale2, name=f"{var_name}_D_tmp2")  # |a2-b2|
    model.addConstr(D_tmp1 == a2 - b2, name=f"{var_name}_constr2")
    model.addConstr(D_tmp2 == abs_(D_tmp1), name=f"{var_name}_constr4")
    return QuadExpr(I_tmp3 * scale2 + (1 - I_tmp3) * D_tmp2)


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
            comp_obj += compare_I_var(comp_a[feature], comp_b[feature], name=feature, var_name=var_name_new,
                                      model=model)
        else:
            assert D_FEATURE_DICT.get(feature) is not None
            I_feature = D_FEATURE_DICT[feature]
            comp_obj += compare_D_var(a1=comp_a[I_feature], a2=comp_a[feature], b1=comp_b[I_feature],
                                      b2=comp_b[feature], name=feature, var_name=var_name_new, model=model)
    return comp_obj


def D_restrict(a1: Var, a2: Var, name: str, var_name: str, model):
    """
    Add constraints and objectives of dependent variables according to the relationship between I- and D- variables.
    :param a1: I-variable
    :param a2: D-variable, |a2| <= max(a1,0)
    :return: max(a1^3,0)*(-a2^2+1)
    """
    if name == 'delayed_release':
        I_tmp0 = model.addVar(lb=-3, ub=2, name=f"{var_name}_I_tmp0")  # 2-sonority
        I_tmp1 = model.addVar(lb=-3, ub=1, name=f"{var_name}_I_tmp1")  # min(sonority, 2-sonority)
        I_tmp2 = model.addVar(lb=0, ub=1, name=f"{var_name}_I_tmp2")  # max(0, min(sonority, 2-sonority))
        D_tmp1 = model.addVar(lb=0, ub=1, name=f"{var_name}_D_tmp1")  # |delayed_release|

        model.addConstr(I_tmp0 == 2 - a1, name=f"{var_name}_constr0")
        model.addConstr(I_tmp1 == min_(a1, I_tmp0), name=f"{var_name}_constr1")
        model.addConstr(I_tmp2 == max_(I_tmp1, constant=0), name=f"{var_name}_constr2")
        model.addConstr(D_tmp1 == abs_(a2), name=f"{var_name}_constr3")
        model.addConstr(D_tmp1 <= I_tmp2, name=f"{var_name}_constr4")

    else:
        scale = SCALE_DICT[name] if SCALE_DICT.get(name) is not None else 1

        b = model.addVar(vtype=GRB.BINARY, name=f"{var_name}_b")
        M, eps = 2, 0.01
        model.addConstr(a1 >= 0.5 + eps - M * (1 - b), name=f"{var_name}_bigM_constr1")
        model.addConstr(a1 <= 0.5 + M * b, name=f"{var_name}_bigM_constr2")
        if name in ['high', 'front']:
            D_tmp1 = model.addVar(lb=1 - scale, ub=1, name=f"{var_name}_D_tmp1")  # 1-a2
            D_tmp2 = model.addVar(lb=0, ub=1, name=f"{var_name}_D_tmp2")  # max(0, 1-a2)
            model.addConstr(D_tmp1 == 1 - a2, name=f"{var_name}_constr1")  # 1-a2
            model.addConstr(D_tmp2 == max_(D_tmp1, constant=0), name=f"{var_name}_constr2")  # max(0, 1-a2)
            model.addConstr(D_tmp2 == 1 - b, name=f"{var_name}_constr3")  # max(0, 1-a2)
        else:
            D_tmp1 = model.addVar(lb=0, ub=scale, name=f"{var_name}_D_tmp1")  # |a2|
            model.addConstr(D_tmp1 == abs_(a2), name=f"{var_name}_constr1")  # |a2|
            model.addConstr(D_tmp1 == b, name=f"{var_name}_constr4")  # |a2|=b


def restrict(var_dict: dict, idx: int, char: str, var_name: str, model):
    """
    get constraints and objects in a component of syllable.
    :param var_dict: dict like {feature1: Var1, ...}, representing features in a component of syllable
    :param idx: index of the char in CharacterSet to be compared
    """
    for feature in FEATURE_LS:
        var_name_new = f"re_{var_name}_{feature}"
        if D_FEATURE_DICT.get(feature) is not None:
            I_feature = D_FEATURE_DICT[feature]
            D_restrict(a1=var_dict[I_feature], a2=var_dict[feature], name=feature, var_name=var_name_new, model=model)
