from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
f=open('lr0.1','r')
line=f.readline().strip()
x=[]
loss=[]
acuraccy=[]
while line:
    line=line.split()
    x.append(int(line[0]))
    loss.append(float(line[1]))
    acuraccy.append(float(line[2]))
    line=f.readline()
f.close()

# plt.plot(x,loss,color='blue',marker = "x",linestyle='--',label='lr=0.1')
plt.plot(x,acuraccy,color='blue',marker = "x",linestyle='--',label='lr=0.1')

f=open('lr0.01','r')
line=f.readline().strip()
x=[]
loss=[]
acuraccy=[]
while line:
    line=line.split()
    x.append(int(line[0]))
    loss.append(float(line[1]))
    acuraccy.append(float(line[2]))
    line=f.readline()
f.close()
# plt.plot(x,loss,color='orange',marker = "x",linestyle='--',label='lr=0.01')
plt.plot(x,acuraccy,color='orange',marker = "x",linestyle='--',label='lr=0.01')

f=open('lr0.001','r')
line=f.readline().strip()
x=[]
loss=[]
acuraccy=[]
while line:
    line=line.split()
    x.append(int(line[0]))
    loss.append(float(line[1]))
    acuraccy.append(float(line[2]))
    line=f.readline()
f.close()

# plt.plot(x,loss,color='black',marker = "x",linestyle='--',label='lr=0.001')
plt.plot(x,acuraccy,color='black',marker = "x",linestyle='--',label='lr=0.001')
plt.title('不同学习率下，准确率随隐层神经元个数变化折线图')
plt.xlabel("隐层神经元个数")
plt.ylabel("准确率")

# plt.title('不同学习率下，测试样本平均损失随隐层神经元个数变化折线图')
# plt.xlabel("隐层神经元个数")
# plt.ylabel("平均损失")
plt.legend()
plt.show()





