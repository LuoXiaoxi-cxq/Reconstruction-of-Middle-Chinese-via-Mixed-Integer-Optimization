import os

def my_mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f"------  Created new folder {path}  ------")
    else:
        print(f"------ Folder {path} existed ------")


def grb_infeasible(model):
    print('Optimization was stopped with status %d' % model.status)
    model.computeIIS()
    with open("constr.txt", 'w', encoding='UTF-8') as f:
        for c in model.getConstrs():
            if c.IISConstr:
                f.write(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')
        for v in model.getVars():
            if v.IISLB:
                f.write(f'\t{v.varname} ≥ {v.LB}\n')
            if v.IISUB:
                f.write(f'\t{v.varname} ≤ {v.UB}\n')


def arg2filename(arg, key_dct, file_name):
    for k in arg:
        if k in key_dct:
            v = arg[k]
            abbr_k = key_dct[k]
            file_name += f'_{abbr_k}_{v}'
    return file_name


def get_boolean(b):
    if b in ["True", "true", True, 1, "1"]:
        return True
    elif b in ["False", "false", False, 0, "0"]:
        return False
    else:
        raise ValueError("Unrecoginzed save model parameter")