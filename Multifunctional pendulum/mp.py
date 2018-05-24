import numpy as np


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


def C_d(_d_l, parameter):
    kp, C, tp = parameter['kp'], parameter['C'], parameter['C']
    # 游标卡尺测量小球直径d
    d_l = np.array(_d_l)
    d_mean = d_l.mean()
    n = len(_d_l)
    delta_equipment = 0.002  # cm
    U_A = np.sqrt(np.sum((d_l - d_mean)**2) / n / (n - 1))
    U_B = kp * delta_equipment / C
    U_d = np.sqrt((tp * U_A)**2 + U_B**2)
    res = {
        'U_A': U_A * 0.01,
        'U_B': U_B * 0.01,
        'U_d': U_d * 0.01,
        'unit': 'm',
        'd_mean': d_mean * 0.01
    }
    return res


def C_l(_l_l, parameter):
    kp, C, tp = parameter['kp'], parameter['C'], parameter['C']
    # 米尺测量摆线长度
    l_l = np.array(_l_l)
    l_mean = l_l.mean()
    n = len(_l_l)
    delta_equipment = 0.05  # cm
    U_A = np.sqrt(np.sum((l_l - l_mean)**2) / n / (n - 1))
    U_B = kp * delta_equipment / C
    U_l = np.sqrt((tp * U_A)**2 + U_B**2)
    res = {
        'U_A': U_A * 0.01,
        'U_B': U_B * 0.01,
        'U_l': U_l * 0.01,
        'unit': 'm',
        'l_mean': l_mean * 0.01
    }
    return res


def C_L(d_data, l_data):
    L = d_data['d_mean'] / 2 + l_data['l_mean']
    U_L = np.sqrt(d_data['U_d']**2 / 4 + l_data['U_l']**2)
    res = {
        'L': L,
        'U_L': U_L,
        'unit': 'm'
    }
    return res


def C_T60(_T60_l, parameter):
    kp, C, tp = parameter['kp'], parameter['C'], parameter['C']
    T60_l = np.array(_T60_l)
    T60_mean = T60_l.mean()
    n = len(_T60_l)
    equipment_min_value = 0.02  # s
    delta_equipment = equipment_min_value / 2
    U_A_60 = U_A = np.sqrt(np.sum((T60_l - T60_mean)**2) / n / (n - 1))
    U_equipment = kp * delta_equipment / C
    U_estimate = 0.2  # s
    U_B_60 = np.sqrt(U_equipment ** 2 + U_estimate ** 2)
    T_mean = T60_mean / 60
    U_A = U_A_60 / 60
    U_B = U_B_60 / 60
    U_T = np.sqrt((tp * U_A)**2 + U_B**2)
    res = {
        'U_A': U_A,
        'U_B': U_B,
        'U_T': U_T,
        'unit': 's',
        'T_mean': T_mean,
        'T60_mean': T60_mean,
        'U_A_60': U_A_60,
        'U_B_60': U_B_60
    }
    return res


def C_g(l_data, d_data, L_data, T_data):
    # g_0 = 9.79338  # m/s^2
    g = 4 * np.pi ** 2 * L_data['L'] * T_data['T_mean'] ** -2
    dl = 1 / (l_data['l_mean'] + d_data['d_mean'] / 2)
    dd = dl / 2
    dT = -2 / T_data['T_mean']
    dri_l = [dl*l_data['U_l'], dd*d_data['U_d'], dT*T_data['U_T']]
    Ur_g = np.linalg.norm(dri_l)

    U_g = g * Ur_g
    res = {
        'g': g,
        'Ur_g': Ur_g,
        'U_g': U_g,
        'unit': 'm/s^2',
        'flag': '<' if Ur_g < 0.01 else '>='
    }
    return res


def make(n, parameter, d_data, l_data, L_data, T_data, g_data):
    print('〇、参数')
    print('\t重复测量次数 n = %d' % n)
    print("\t取置信概率 P = %.1f%%, t_p = %.2lf, k_p = %d\n\t取C = %.3lf"
          % (parameter['P'] * 100, parameter['tp'], parameter['kp'], parameter['C']))
    print("*********************************************************************")
    print('一、球直径')
    print('\t平均 d_bar = %s m' % sn(d_data['d_mean'], 3))
    print('\tA类不确定度 U_A = %s m' % sn(d_data['U_A'], 1))
    print('\tB类不确定度 U_B = %s m' % sn(d_data['U_B'], 3))
    print('\t合成不确定度 U_d = %s m' % sn(d_data['U_d'], 3))
    print('\t所以 d = d_bar ± U_d = (%s ± %s) m' %
          (sn(d_data['d_mean'], 3), sn(d_data['U_d'], 3)))
    print("*********************************************************************")
    print('二、摆线长度')
    print('\t平均 l_bar = %s m' % sn(l_data['l_mean'], 2))
    print('\tA类不确定度 U_A = %s m' % sn(l_data['U_A'], 3))
    print('\tB类不确定度 U_B = %s m' % sn(l_data['U_B'], 3))
    print('\t合成不确定度 U_l = %s m' % sn(l_data['U_l'], 3))
    print('\t所以 l = l_bar ± U_l = (%s ± %s) m' %
          (sn(l_data['l_mean'], 2), sn(l_data['U_l'], 3)))
    print("*********************************************************************")
    print('三、有效摆长')
    print('\t有效摆长L_bar = l_bar + d_bar / 2 = (%s + %s / 2) m = %s m'
          % (sn(l_data['l_mean'], 2), sn(d_data['d_mean'], 3), sn(L_data['L'], 2)))
    print('\t由误差传递公式U_L = %s m' % sn(L_data['U_L'], 2))
    print("*********************************************************************")
    print('四、周期')
    print('\t平均 T_60_bar = %s s' % sn(T_data['T60_mean'], 3))
    print('\tA类不确定度 U_A_60 = %s s' % sn(T_data['U_A_60'], 4))
    print('\tB类不确定度 U_B_60 = %s s' % sn(T_data['U_B_60'], 3))
    print('\t平均到每一个周期:')
    print('\t平均 T_bar = %s s' % sn(T_data['T_mean'], 3))
    print('\tA类不确定度 U_A = %s s' % sn(T_data['U_A'], 4))
    print('\tB类不确定度 U_B = %s s' % sn(T_data['U_B'], 3))
    print('\t合成不确定度 U_T = %s s' % sn(T_data['U_T'], 3))
    print('\t所以 T = T_bar ± U_T = (%s ± %s) s' %
          (sn(T_data['T_mean'], 3), sn(T_data['U_T'], 3)))
    print("*********************************************************************")
    print('五、重力加速度')
    print('\t计算可得:g = 4 * π^[2] * %.2f * 10^[-2] * %.3f^[-2] = %.4f m/s^[2]'
          % (L_data['L'], T_data['T_mean'], g_data['g']))
    print('\tUr_g = %.4f = %.2f%% %s 1%%' %
          (g_data['Ur_g'], g_data['Ur_g'] * 100, g_data['flag']))
    print('\t所设计的实验方案[%s]达到预期的要求' % ('能' if g_data['flag'] == '<' else '不能'))
    print('\tU_g = g * Ur_g = (%.4f * %.2f%%) m/s^[2] = %.2f m/s^[2]'
          % (g_data['g'], 100 * g_data['Ur_g'], g_data['U_g']))
    print(
        '\t综上, g = g_bar ± U_g = (%.2f ± %.2f) m/s^[2]' % (g_data['g'], g_data['U_g']))


def show_data(obj):
    for item in obj:
        for ee in item:
            if not isinstance(item[ee], str):
                print(ee, ' : %e' % item[ee])
            else:
                print(ee, ' : ', item[ee])
        print("**************************************")

def C_d2(d2_lower, d2_upper):
    L = np.array(d2_lower)
    H = np.array(d2_upper)
    d = np.round((L + H) / 2, 3).tolist()
    return d

class Wiggler:
    def __init__(self, _T5, _d2, _D, M, L):
        self.G_0 = 7.7 * 10**10
        self.T5 = np.array(_T5)
        self.d2 = np.array(_d2)
        self.D = np.array(_D)
        self.D_mean = self.D.mean()
        self.fixed_part = 128 * np.pi * L * 10**- \
            2 * M * 10**-3 / (self.D_mean * 10**-2)**4
        self.para_l = [(1, 2), (1, 3), (2, 3)]

        self.T5_mean = self.T5.mean(axis=1)
        self.d2_mean = self.d2.mean(axis=1)

        self.T_mean = self.T5_mean / 5
        self.d_mean = self.d2_mean / 2

    def c_G(self, pair):
        i = pair[0] - 1
        j = pair[1] - 1
        i, j = (i, j) if i < j else (j, i)
        calc_part = ((self.d_mean[i] * 10**-2)**2 - (self.d_mean[j] * 10**-2)**2) \
            / (self.T_mean[i]**2 - self.T_mean[j]**2)
        return self.fixed_part * calc_part

    def run(self):
        G_l = []
        for ee in self.para_l:
            ij = self.c_G(ee)
            G_l.append(ij)

        G_bar = np.array(G_l).mean()
        Ur_G = np.abs(G_bar - self.G_0) / self.G_0
        res = {
            'G_l': G_l,
            'G_bar': G_bar,
            'Ur_G': Ur_G
        }
        return res

    def out(self, res):
        print("一、周期")
        print("\t五倍周期5T:")
        for ee in self.T5_mean:
            print("\t   %.4f s" % ee)
        print("\t周期T:")
        for ee in self.T_mean:
            print("\t   %.4f s" % ee)
        print("*********************************************")
        print("二、球距")
        print("\t2d:")
        for ee in self.d2_mean:
            print("\t   %.3f cm" % ee)
        print("\td:")
        for ee in self.d_mean:
            print("\t   %.3f cm" % ee)
        print("*********************************************")
        print("三、钢丝直径")
        print('\tD = %.4f cm' % self.D_mean)
        print("*********************************************")
        print("四、切变模量")
        for i in range(len(res['G_l'])):
            print("\tG_[%d%d] = %s Pa"
                  % (self.para_l[i][0],
                     self.para_l[i][1],
                     sn(res['G_l'][i], 3)))
        print("\tG_bar = mean(G_[ij]) = %s Pa" % sn(res['G_bar'], 2))
        print("*********************************************")
        print("五、相对误差")
        print("\tUr_G = %.4f = %.2f%%" % (res['Ur_G'], res['Ur_G'] * 100))
