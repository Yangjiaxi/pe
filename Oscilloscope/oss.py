import numpy as np


def exp1(g, o):
    c = {
        'T' : o['Dx'] * o['X'],
        'Vpp' : o['Dy'] * o['Y']
    }
    Ur = {
        'U' : round(abs(g['Vpp'] - c['Vpp']) / g['Vpp'], 4),
        'T' : round(abs(1.0 / g['f'] - c['T']) * g['f'], 4)
    }
    U = {
        'U' : round(Ur['U'] * c['Vpp'], 4),
        'T' : round(Ur['T'] * c['T'], 4)
    }
    return c, Ur, U

def U_A(_l):
    l = np.array(_l)
    n = l.shape[0]
    Ua = np.sqrt(((l - l.mean()) ** 2).sum() / (n * (n - 1)))
    return Ua

def exp2(o):
    k = np.array(o['Nx']) / np.array(o['Ny'])
    fy = np.round(k * np.array(o['fx']), 6)
    fy_bar = round(fy.mean(), 6)
    Ua = round(U_A(fy), 5)
    Ur = Ua / fy_bar
    res = {
        'f_y' : fy.tolist(),
        'f_y_bar' : fy_bar,
        'U_A' : Ua,
        'U_r' : str(round(Ur * 100, 5)) + '%'
    }
    return res

def exp3(o3):
    fy = np.array(o3['N']) * np.array(o3['fx'])
    fy_bar = round(fy.mean(), 6)
    Ua = round(U_A(fy), 6)
    Ur = Ua / fy_bar
    res = {
        'f_y' : fy.tolist(),
        'f_y_bar' : fy_bar,
        'U_A' : Ua,
        'U_r' : str(round(Ur * 100, 5)) + '%'
    }
    return res
    return fy