import numpy as np
import matplotlib.pyplot as plt

def train(sample1,sample2,lr):
    sample2=-1*sample2
    data=np.vstack((sample1,sample2))
    w=np.array([0,0,0])
    iteration=0
    while True:
        y = np.dot(w, data.T)
        y = (y <= 0)
        y = np.where(y)
        if y[0].shape[0] == 0:
            break

        e_data = data[y]
        w = w + lr * sum(e_data)
        print("第{}次迭代法向量为:{}".format(iteration+1,w))
        iteration=iteration+1

    return w,iteration



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
w,iteration=train(sample1,sample2,lr=1)
print("最终分界面法向量为{},迭代轮次为{}".format(w,iteration))

plt.scatter(sample1[:, 1], sample1[:, 2])
plt.scatter(sample2[:, 1], sample2[:, 2])
boudary_x = np.array([i for i in range(-4, 6)])
boudary_y = -w[1] / w[2] * boudary_x - w[0] / w[2]
plt.plot(boudary_x, boudary_y)
plt.show()

# test_w=np.array([ 34.,-30.4,34.1])
# print(np.dot(test_w,sample1.T))
# print(np.dot(test_w,sample2.T))
