from mpl_toolkits import mplot3d
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

color=['blue','red','green','black']
labels=['batch_size=1','batch_size=4','batch_size=6','batch_size=8']
def iter_loss(file='loss_batch_size_1.txt',index=0):
    f = open(file, 'r')
    line = f.readline().strip()
    x = []
    y = []
    i = 0
    loss = []
    avg = []
    cur = []
    while line:
        cur.append(float(line))
        if i % 20 == 0:
            x.append(i)
            loss.append(float(line))
        if i % 50 == 0:
            y.append(i)
            avg.append(sum(cur) / len(cur))
            cur = []

        line = f.readline().strip()
        i = i + 1
    f.close()

    # plt.plot(x,loss,color='blue',marker = "x",linestyle='--',label='lr=0.1')
    avg_smooth = savgol_filter(avg, 9, 6)
    # plt.plot(x, loss, color='red', label='lr=0.001')
    plt.plot(y, avg_smooth, color=color[index], label=labels[index])

for i,f in enumerate(['loss_batch_size_1.txt','loss_batch_size_4.txt','loss_batch_size_6.txt','loss_batch_size_8.txt']):
    iter_loss(f,i)
plt.title('设置不同的batch_size下，目标函数随迭代步数变化')
plt.xlabel("迭代步数")
plt.ylabel("目标函数")
plt.legend()
plt.show()