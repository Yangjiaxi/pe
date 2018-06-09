import numpy as np

def show_dict(dct):
    for ee in dct:
        print(ee, ":", dct[ee])
        
def run_1(d):
    T0 = 331.45
    Temp_K = 273.15
    v_theory = round(T0 * np.sqrt(1 + d['t'] / Temp_K), 2)
    n = len(d['L']) / 2
    data = np.array(d['L'])
    delta_l_list = (data[6:] - data[:6]) / 6
    delta_l = round(((data[6:] - data[:6]) / 6).mean(), 2)
    v = round(2 * d['f'] * delta_l * 10**-3, 2)
    delta_v = round(np.abs(v - v_theory), 2)
    U_r_v = delta_v / v_theory
    res = {
        '[Δl]' : delta_l_list.round(2),
        'v_空_理' :v_theory,
        'Δl' : delta_l,
        'v_空' : v,
        'Δv' : delta_v, 
        'Ur(v)' : str(round(100 * U_r_v, 4)) + '%'
    }
    return res

def run_2(d):
    f = np.array(d['f'])
    f_mean = round(f.mean(), 0)
    
    data = np.array(d['L'])
    delta_l_list = (data[6:] - data[:6]) / 6
    delta_l = (data[6:] - data[:6]) / 6
    delta_l_mean = round(delta_l.mean(), 2)
    
    n_dl = delta_l.shape[0]
    U_A_dl = np.sqrt(((delta_l - delta_l_mean) ** 2).sum() / n_dl / (n_dl - 1))
    U_B_dl = 0.02 / np.sqrt(3)
    U_dl = np.sqrt(U_A_dl**2 + U_B_dl**2)
    Ur_dl = U_dl / delta_l_mean
    
    v = 2 * f_mean * delta_l_mean * 10**-3
    n_f = f.shape[0]
    U_A_f = np.sqrt(((f - f_mean) ** 2).sum() / n_f / (n_f - 1))
    U_B_f = 1
    U_f = np.sqrt(U_A_f**2 + U_B_f**2)
    Ur_f = U_f / f_mean
    
    Ur_v = np.sqrt(Ur_dl**2 + Ur_f**2)
    U_v = v * Ur_v
    
    res = {
        '[Δl]' : delta_l_list.round(2),
        'Δl' : round(delta_l_mean, 2),
        'v水' : round(v, 2),
        'U_A(Δl)' : round(U_A_dl, 2),
        'U_B(Δl)' : round(U_B_dl, 3),
        'U(Δl)' : round(U_dl, 2),
        'Ur(Δl)' : str(round(100 * Ur_dl, 4)) + '%',
        'U_A(f)' : round(U_A_f, 2),
        'U_B(f)' : U_B_f,
        'U(f)' : round(U_f, 2),
        'Ur(f)' : str(round(100 * Ur_f, 4)) + '%',
        'Ur(v水)' : str(round(100 * Ur_v, 4)) + '%',
        'U(v水)' : round(U_v, 2)
    }
    return res