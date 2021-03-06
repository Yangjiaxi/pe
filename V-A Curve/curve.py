import numpy as np
import matplotlib.pyplot as plt


def il(str):
    res = []
    for ele in str.split('/'):
        res.append([float(e) for e in ele.split('+')])
    if len(res) == 1:
        return res[0]
    else:
        return res


def rd(mr, mn, K, _data):
    data = np.array(_data)
    r = mr * K / 100
    if r < 1:
        s = str(r)
        for i in range(len(s)):
            if float(s[0:2+i]) > 0:
                r_d = i
                break
    else:
        raise ValueError("not less than 1!")
    return np.round(data / mn * mr, r_d)


def show_data(data, name=None):
    if name is None:
        name = [''] * len(data)
    for (e, n) in zip(data, name):
        print(n, end=': ')
        for ee in e:
            print(ee, end=' ')
        print()


def exp1(_iU, _iI, _oU, _oI):
    iU = np.append(_iU, 0)
    iI = np.append(_iI, 0)
    oU = np.append(_oU, 0)
    oI = np.append(_oI, 0)
    Ul = [iU, oU]
    Il = [iI, oI]
    ml = ['^', 's']
    lbs = ['内接法', '外接法']
    ms = []
    cs = []
    for (a, b, s, n) in zip(Ul, Il, ml, lbs):
        A = np.vstack([a, np.ones(len(a))]).T
        m, c = np.linalg.lstsq(A, np.array(b), rcond=1)[0]
        ms.append(round(m, 2))
        cs.append(round(c, 2))
        plt.scatter(a[:-1], b[:-1], c='black', s=40, marker=s, label=n)

    iR = round(1000 / ms[0], 2)
    oR = round(1000 / ms[1], 2)
    lbs = ["内接法 $y={}x{}{},R={}$".format(ms[0], '+' if cs[0] > 0 else '', cs[0], iR),
           "外接法 $y={}x{}{},R={}$".format(ms[1], '+' if cs[1] > 0 else '', cs[1], oR)]
    plt.legend(lbs, frameon=True, fancybox=True, shadow=True)

    for (a, b) in zip(Ul, Il):
        A = np.vstack([a, np.ones(len(a))]).T
        m, c = np.linalg.lstsq(A, np.array(b), rcond=1)[0]
        plt.plot(a, m * a + c, c='black')

    plt.title("定值电阻伏安特性曲线")
    plt.xlabel("$U/V$")
    _ = plt.ylabel("$I/mA$")


def exp2(_Ua, _Ia, _Ub, _Ib):
    Ua = np.array(_Ua)
    Ia = np.array(_Ia)
    Ub = 0 - np.array(_Ub)
    Ib = 0 - np.array(_Ib)
    U = np.sort(np.concatenate((Ua, Ub)))
    I = np.sort(np.concatenate((Ia, Ib)))
    plt.scatter(U, I, color='black', s=30)
    plt.plot(U, I, color='black')
    plt.title("半导体二极管2CW52的正、反向伏安特征曲线")
    plt.xlabel("$U/V$")
    _ = plt.ylabel("$I/mA$")


def exp3(_U40, _I40, _U60, _I60, _U80, _I80):
    U40 = np.array(_U40)
    I40 = np.array(_I40)
    U60 = np.array(_U60)
    I60 = np.array(_I60)
    U80 = np.array(_U80)
    I80 = np.array(_I80)
    Ul = [U40, U60, U80]
    Il = [I40, I60, I80]
    mks = ['o', 's', '^']
    lbs = ["$40μA$", "$60μA$", "$80μA$"]
    for (a, b, s, n) in zip(Ul, Il, mks, lbs):
        # plt.plot(a, b, color='black')
        plt.scatter(a, b, color='black', s=30, marker=s, label=n)
    plt.legend(lbs, frameon=True, fancybox=True, shadow=True)
    Umax = int(np.max([U40.max(), U60.max(), U80.max()])) + 1
    Imax = int(np.max([I40.max(), I60.max(), I80.max()])) + 1
    plt.xlabel("$U/V$")
    plt.ylabel("$I/mA$")
    plt.xticks(np.linspace(0, Umax, 1+int(Umax/0.5)))
    plt.yticks(np.linspace(0, Imax, 1+int(Imax)))
    _ = plt.title("晶体三极管的输出特征曲线")
