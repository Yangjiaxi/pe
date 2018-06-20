import matplotlib.pyplot as plt
import numpy as np

e = 1.6 * 10**-19
d = 0.500  # mm
U_d = 0.004  # mm
b = 4.00  # mm
U_b = 0.02  # mm
AC = 3.00  # mm
U_AC = 0.02  # mm
kB = 1.80 * 1000
I_M_1 = 0.450
U_I_M_1 = 0.001
I_S_1 = np.linspace(1.00, 4.50, 8)
I_S_2 = 4.50  # mA
U_I_S_2 = 0.01  # mA
I_M_2 = np.linspace(0.1, 0.45, 8)


def il(str):
    res = []
    for ele in str.split('/'):
        res.append([float(e) for e in ele.split('+')])
    if len(res) == 1:
        return res[0]
    else:
        return res


def sn(num, d):
    output_format = '%.' + str(d) + 'e'
    formatted = output_format % num
    component = formatted.split('e')
    mantissa = float(component[0])
    exponent = int(component[1])
    if exponent == 0:
        res = str(mantissa)
    else:
        res = str(mantissa) + ' * 10^[' + str(exponent) + ']'
    return res


def exp1(V1, V2, V3, V4):
    V_H = (V1 - V2 + V3 - V4) / 4
    plt.scatter(I_S_1, V_H, c='black')
    A = np.vstack([I_S_1, np.ones(len(I_S_1))]).T
    m, c = np.linalg.lstsq(A, np.array(V_H), rcond=1)[0]
    plt.plot(I_S_1, m * I_S_1 + c, c='black')
    plt.title("$V_H - I_s$关系图")
    plt.xlabel("$I_S/mV$")
    _ = plt.ylabel("$V_H/mA$")
    lbs = [
        '$V_H = {} * I_S {}{}$'.format(round(m, 4), '+' if c > 0 else '', round(c, 3))]
    plt.legend(lbs, frameon=True, fancybox=True, shadow=True)
    k = m
    R_H = k * d * 0.1 / (I_M_1 * kB) * 10**8
    Ur_R_H = np.linalg.norm([U_d / d, U_I_M_1 / I_M_1])
    U_R_H = R_H * Ur_R_H
    n = 1 / (R_H * e)  # cm^[-3]
    U_n = U_R_H / R_H * n
    # -------------------------------------------------------
    print("V_H = {}".format(V_H))
    print('k = {} '.format(round(k, 4)))
    print("*"*50)
    print('R_H = {} (cm^[3]/C)'.format(sn(R_H, 2)))
    print('Ur(R_H) = {}%'.format(round(100 * Ur_R_H, 2)))
    print('U(R_H) = {} (cm^[3]/C)'.format(round(U_R_H, 3)))
    print(
        'R_H = ({}±{})* 10^[3] (cm^[3]/C)'.format(round(R_H / 1000, 2), round(U_R_H / 1000, 2)))
    print("*"*50)
    print('n = {} (cm^[-3])'.format(sn(n, 2)))
    print('U(n) = {} (cm^[-3])'.format(sn(U_n, 2)))
    print('n = ({}±{}) * 10^[14] (cm^[-3])'.format(round(n /
                                                         10**14, 2), round(U_n / 10**14, 2)))


def exp2(V1, V2, V3, V4):
    V_H = (V1 - V2 + V3 - V4) / 4
    plt.scatter(I_M_2, V_H, c='black')
    A = np.vstack([I_M_2, np.ones(len(I_M_2))]).T
    m, c = np.linalg.lstsq(A, np.array(V_H), rcond=1)[0]
    plt.plot(I_M_2, m * I_M_2 + c, c='black')
    plt.title("$V_H - I_s$关系图")
    plt.xlabel("$I_S/mV$")
    _ = plt.ylabel("$V_H/mA$")
    lbs = [
        '$V_H = {} * I_S {}{}$'.format(round(m, 2), '+' if c > 0 else '', round(c, 3))]
    plt.legend(lbs, frameon=True, fancybox=True, shadow=True)
    k = m
    R_H = k * d * 0.1 / (I_S_2 * kB) * 10**8
    Ur_R_H = np.linalg.norm([U_d / d, U_I_S_2 / I_S_2])
    U_R_H = R_H * Ur_R_H
    n = 1 / (R_H * e)  # cm^[-3]
    U_n = U_R_H / R_H * n
    # -------------------------------------------------------
    print("V_H = {}".format(V_H))
    print('k = {} '.format(round(k, 2)))
    print("*"*50)
    print('R_H = {} (cm^[3]/C)'.format(sn(R_H, 2)))
    print('Ur(R_H) = {}%'.format(round(100 * Ur_R_H, 2)))
    print('U(R_H) = {} (cm^[3]/C)'.format(round(U_R_H, 3)))
    print(
        'R_H = ({}±{})* 10^[3] (cm^[3]/C)'.format(round(R_H / 1000, 2), round(U_R_H / 1000, 2)))
    print("*"*50)
    print('n = {} (cm^[-3])'.format(sn(n, 2)))
    print('U(n) = {} (cm^[-3])'.format(sn(U_n, 2)))
    print('n = ({}±{}) * 10^[14] (cm^[-3])'.format(round(n /
                                                         10**14, 2), round(U_n / 10**14, 2)))
    return R_H, U_R_H


def exp3(I_S, V_AC, R_H, U_R_H):
    plt.scatter(I_S, V_AC, c='black')
    A = np.vstack([I_S, np.ones(len(I_S))]).T
    m, c = np.linalg.lstsq(A, np.array(V_AC), rcond=1)[0]
    plt.plot(I_S, m * I_S + c, c='black')
    plt.title("$V_{AC} - I_S$关系图")
    plt.xlabel("$I_S/mV$")
    _ = plt.ylabel("$V_{AC}/mA$")
    lbs = [
        '$V_'+'{AC}'+'= {} * I_S {}{}$'.format(round(m, 2), '+' if c > 0 else '', round(c, 3))]
    plt.legend(lbs, frameon=True, fancybox=True, shadow=True)
    k = m
    sigma = (1 / k) * AC * 0.1 / (b * 0.1 * d * 0.1)
    Ur_sigma = np.linalg.norm([U_d / d, U_AC / AC, U_b / b])
    U_sigma = Ur_sigma * sigma
    miu = R_H * sigma
    Ur_miu = np.linalg.norm([U_R_H / R_H, U_sigma / sigma])
    U_miu = Ur_miu * miu
    n = 1 / (R_H * e)
    U_n = U_R_H / R_H * n
    # -------------------------------------------------------
    print('k = {} '.format(round(k, 2)))
    print("*"*50)
    print("sigma = {} (Ω^[-1]/cm)".format(sn(sigma, 2)))
    print("Ur(sigma) = {}%".format(round(Ur_sigma * 100, 2)))
    print("U(sigma) = {} (Ω^[-1]/cm)".format(sn(U_sigma, 2)))
    print("sigma = ({}±{}) * 10^[-1] (Ω^[-1]/cm)".format(
        round(sigma * 10, 2), round(U_sigma * 10, 2)))
    print("*"*50)
    print("miu = {} (cm^[3]Ω^[-1]C^[-1])".format(sn(miu, 2)))
    print("Ur(miu) = {}%".format(round(Ur_miu * 100, 2)))
    print("U(miu) = {} (cm^[3]Ω^[-1]C^[-1])".format(sn(U_miu, 2)))
    print("miu = ({}±{}) * 10^[3] (cm^[3]Ω^[-1]C^[-1])".format(
        round(miu / 10**3, 2), round(U_miu / 10**3, 2)))
    print("*"*50)
    print("n = {} (cm^[-3])".format(sn(n, 2)))
    print("U(n) = {} (cm^[-3])".format(sn(U_n, 2)))
    print("n = ({}±{}) * 10^[14] (cm^[-3])".format(round(n /
                                                         10**14, 2), round(U_n / 10**14, 2)))
