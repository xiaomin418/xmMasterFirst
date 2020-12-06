import numpy as np
import random
import matplotlib.pyplot as plt
def Kashyap(sample1,sample2,lr,iter,b_min):
    sample2 = -1 * sample2
    Y = np.vstack((sample1, sample2))
    a=np.array([[1,1,1]])
    # a = np.array([[1, 1.727, -1.0]])
    b=np.array([[0.005 for i in range(Y.shape[0])]])
    # YY=np.matmul(np.linalg.pinv(np.dot(Y.T, Y)), Y.T)
    # a=np.dot(np.linalg.pinv(Y),b.T).T
    # import pdb
    # pdb.set_trace()
    # temp=np.dot(Y,np.linalg.pinv(Y))
    # print(temp)
    e=np.array([0 for i in range(sample1.shape[0])])
    x=[]
    y=[]
    for i in range(iter):
        # if i%50==0:
        #   print("第{}次迭代，样本总体损失:{}".format(i, abs(sum(e*e))))
        e = np.dot(Y, a.T) - b.T #20x1
        e_plus = 0.5* (e + abs(e))#20x1
        b = b + 2 * lr * e_plus.T #1x20
        a = np.dot(np.linalg.pinv(Y),b.T).T
        x.append(i)
        y.append(abs(sum(e*e)))
        if abs(sum(e*e))<=b_min:
            break
    # plt.plot(x[1:],y[1:])
    # plt.show()
    return a,b,abs(sum(e*e)),i

sample1=np.array([[1,0.1,1.1],
                  [1,6.8,7.1],
                  [1,-3.5,-4.1],
                  [1,2.0,2.7],
                  [1,4.1,2.8],
                  [1,3.1,5.0],
                  [1,-0.8,-1.3],
                  [1,0.9,1.2],
                  [1,5.0,6.4],
                  [1,3.9,4.0]])
sample2=np.array([[1,7.1,4.2],
                  [1,-1.4,-4.3],
                  [1,4.5,0.0],
                  [1,6.3,1.6],
                  [1,4.2,1.9],
                  [1,1.4,-3.2],
                  [1,2.4,-4.0],
                  [1,2.5,-6.1],
                  [1,8.4,3.7],
                  [1,4.1,-2.2]
                  ])
sample3=np.array([[1,-3.0,-2.9],
                  [1,0.5,8.7],
                  [1,2.9,2.1],
                  [1,-0.1,5.2],
                  [1,-4.0,2.2],
                  [1,-1.3,3.7],
                  [1,-3.4,6.2],
                  [1,-4.1,3.4],
                  [1,-5.1,1.6],
                  [1,1.9,5.1]
                  ])
sample4=np.array([[1,-2.0,-8.4],
                  [1,-8.9,-0.2],
                  [1,-4.2,-7.7],
                  [1,-8.5,-3.2],
                  [1,-6.7,-4.0],
                  [1,-0.5,-9.2],
                  [1,-5.3,-6.7],
                  [1,-8.7,-6.4],
                  [1,-7.1,-9.7],
                  [1,-8.0,-6.3]
                  ])

a1=np.sum(sample1,axis=1)
a2=np.sum(sample2,axis=1)
max_iteration=1000
b_min=0.01

# a,b,loss,iters=Kashyap(sample1,sample3,lr=0.01,iter=max_iteration,b_min=b_min)
# print("(a) w1-w3训练结果：")
# print("--a:",a)
# print("--样本总体损失:",loss[0])
# if loss[0]>b_min:
#     print("w1-w3算法不收敛！")
# else:
#     print("分类达到收敛，且迭代次数为:", iters)


a,b,loss,iters=Kashyap(sample2,sample4,lr=0.5,iter=max_iteration,b_min=b_min)
print("(b) w2-w4训练结果：")
print("--a:",a)
print("--样本总体损失:",loss[0])
print("--迭代次数:",iters)
if loss[0]>b_min:
    print("w2-w4算法不收敛！")
else:
    print("分类达到收敛，且迭代次数为:", iters)

#图片展示分类结果
# plt.scatter(sample2[:, 1], sample2[:, 2])
# plt.scatter(sample4[:, 1], sample4[:, 2])
# boudary_x = np.array([i for i in range(-4, 6)])
# boudary_y = -a[0][1] / a[0][2] * boudary_x - a[0][0] / a[0][2]
# plt.plot(boudary_x, boudary_y)
# plt.show()
