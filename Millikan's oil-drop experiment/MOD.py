'''
密立根油滴实验
1. 数据生成器
2. 实验数据处理器
-------------------
1. 数据完全生成
直接执行代码即可

2. 数据处理
a. 计算q与n
先new一个BOOM对象，然后按照BOOM.out的doc传入参数即可

b. 绘图并计算斜率与截距
新建一个list，有两个list元素，第一个list是计算出的n的序列，第二个list是计算出的q的序列
调用BOOM的方法plot_res()传入这个list即可

例如：
[[8.0, 13.0, 5.0, 9.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0],
 [1.35, 2.09, 0.78, 1.42, 0.32, 0.32, 0.48, 0.47, 0.32, 0.47]]
'''
import numpy as np
import sympy
import matplotlib.pyplot as plt
import seaborn as sns


class BOOM:
    def __init__(self):
        self.rho = 981
        self.g = 9.795
        self.eta = 1.83 * 10 ** -5
        self.b = 6.17 * 10 ** -6
        self.d = 5 * 10 ** -3
        self.l = 1.5 * 10 ** -3
        self.p = 76
        self.q_0 = 1.6 * 10 ** -19

    def out(self, t_list, U):
        '''
        @param
        t_list: numpy.ndarray, shape=(1, 5), 时间序列，不用包含均值
        U: int, 对应电压值

        @return
        q: float, 电荷量
        n: int, 相对于元电荷的倍数
        '''
        t_bar = t_list.mean().round(1)
        v_g = self.l / t_bar
        a = np.sqrt(9 * self.eta * v_g / (2 * self.rho * self.g))
        r1 = 18 * np.pi
        r2 = np.sqrt(2 * self.rho * self.g)
        r3 = (self.eta * self.l) / (t_bar * (1 + self.b / (self.p * a)))
        r4 = self.d / U
        q = (r1/r2) * (r3**(3/2)) * r4
        n = q / self.q_0
        return q * 10 ** 18, int(round(n, 0))

    def get_t_bar(self, U, n):
        r1 = 18 * np.pi
        r2 = np.sqrt(2 * self.rho * self.g)
        r4 = self.d / U
        q = n * self.q_0
        r3 = (q * r2 / (r1 * r4)) ** (2/3)
        t_g = sympy.Symbol('t_g')
        ts = sympy.solve(r3 - self.eta * self.l / (t_g * (1 + self.b / (self.p *
                                                                        ((9*self.eta * self.l / (2 * self.rho * self.g * t_g)) ** (1/2))))), t_g)
        if '+' in str(ts[0]):
            res = float(str(ts[0]).split('+')[0])
            if not 10.0 <= res <= 30.0:
                return False, 0
            else:
                return True, res
        else:
            return False, 0

    def make_ts(self, t_bar):
        return np.random.multivariate_normal([t_bar], [[0.3]], 5).round(1)

    def make_data(self):
        result = []
        Flag = True
        count = 0
        TOTAL = 10
        while Flag:
            U = np.random.randint(120, 280)
            n = np.random.randint(1, 15)
            t = 0
            generate, t = self.get_t_bar(U, n)
            if generate == True:
                count += 1
                if count >= TOTAL:
                    Flag = False
                print("%d / %d" % (count, TOTAL))
                t_l = self.make_ts(t)
                q_i, n_i = self.out(t_l, U)
                dd = {
                    'U': U,
                    'n': n,
                    't_make': t,
                    't_bar': t_l.mean().round(1),
                    't_l': t_l.tolist(),
                    'q_out': round(q_i * (1 + np.random.randint(-7, 7) / 1000), 2),
                    'n_out': n_i
                }
                result.append(dd)
            else:
                print("Re", end=' ')
        return result

    def plot_res(self, data):
        data[0].append(0)
        data[1].append(0)
        plt.scatter(data[0][:-1], data[1][:-1], marker='o', color='red', s=30)
        A = np.vstack([data[0], np.ones(len(data[0]))]).T
        m, c = np.linalg.lstsq(A, np.array(data[1]), rcond=1)[0]
        plt.plot(data[0], m * np.array(data[0]) + c)
        plt.xticks(np.arange(0, max(data[0])+2, 1))
        plt.xlim(-0.3, None)
        plt.ylim(-0.1, None)
        plt.show()
        print("Least Square : Y = m * X + c, m = %f, c = %f " % (m, c))


if __name__ == '__main__':
    sns.set()
    sss = BOOM()
    data = sss.make_data()
    n_l = np.array([e['n_out'] for e in data])
    q_l = np.array([e['q_out'] for e in data])
    plot_data = np.vstack([n_l, q_l]).tolist()
    plt.rcParams['figure.figsize'] = [4, 2]
    plt.rcParams['figure.dpi'] = 144
    sss.plot_res(plot_data)

    # plotter = BOOM()
    # l = [[8.0, 13.0, 5.0, 9.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0],
    #      [1.35, 2.09, 0.78, 1.42, 0.32, 0.32, 0.48, 0.47, 0.32, 0.47]]
    # plotter.plot_res(l)
