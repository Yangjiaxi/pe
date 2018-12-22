import numpy as np

import matplotlib.pyplot as plt

def il(str=None):
    if str is not None:
        res = []
        for ele in str.split('/'):
            res.append([float(e) for e in ele.split('+')])
        if len(res) == 1:
            return res[0]
        else:
            return res
    else:
        return None
    
def exp2(x, y2, y4, y8):
    assert len(y2) == len(x)
    assert len(y4) == len(x)
    assert len(y8) == len(x)
    pref = {
        "c" : "black",
        "markersize" : 4,
        "linewidth" : 1.5
    }
    plt.plot(x, y2, marker="s", linestyle="-", label="$2mm$", **pref)
    plt.plot(x, y4, marker="^", linestyle="--",label="$4mm$", **pref)
    plt.plot(x, y8, marker="o", linestyle=":",label="$8mm$", **pref)
    plt.legend(shadow=True)
    plt.title("光电管的伏安特性曲线")
    plt.xlabel("$U/V$")
    plt.ylabel("$I/10^{-12}A$")
    
def exp3(gl_size, gl_436, gl_546):
    assert len(gl_size) == len(gl_436)
    assert len(gl_size) == len(gl_546)
    gl_fs = np.linspace(0, np.array(gl_size).max(), 100)
    pref = {
        "c" : "black",
        "markersize" : 4,
        "linewidth" : 1.5
    }
    fp_436 = np.polyfit(gl_size, gl_436, 2)
    fp_546 = np.polyfit(gl_size, gl_546, 2)
    
    plt.scatter(gl_size, gl_436, label="435.8nm", marker="o", c="black")
    plt.plot(gl_fs, fp_436[0] * 1.07 * gl_fs ** 2 + fp_436[1] + gl_fs + fp_436[2], 
             label="435.8nm 拟合曲线", **pref, linestyle="-")
    plt.scatter(gl_size, gl_546, label="546.1nm", marker="^", c="black")
    plt.plot(gl_fs, fp_546[0] * 1 * gl_fs ** 2 + fp_546[1] + gl_fs + fp_546[2], 
             label="546.1nm 拟合曲线", **pref, linestyle="--")
    plt.legend(shadow=True)
    plt.title("饱和光电流与光阑孔径的关系-1")
    plt.xlabel("光阑孔径 φ/mm")
    plt.ylabel("饱和光电流 $I/10^{-11}A$")
    plt.show()
    
    plt.plot(gl_size ** 2, gl_436, label="435.8nm", marker="o", c="black", linestyle="-")
    plt.plot(gl_size ** 2, gl_546, label="546.1nm", marker="^", c="black", linestyle="--")
    plt.legend(shadow=True)
    plt.title("饱和光电流与光阑孔径的关系-2")
    plt.xlabel("光阑面积 $S/mm^2$")
    plt.ylabel("饱和光电流 $I/10^{-11}A$")