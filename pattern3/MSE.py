import numpy as np

def Multi_MSE(sample1,sample2,sample3,sample4):
    X = np.vstack((sample1, sample2,sample3,sample4)).T
    y1=np.array([[1,0,0,0] for _ in range(len(sample1))])
    y2=np.array([[0,1,0,0] for _ in range(len(sample2))])
    y3=np.array([[0,0,1,0] for _ in range(len(sample3))])
    y4=np.array([[0,0,0,1] for _ in range(len(sample4))])
    Y=np.vstack((y1,y2,y3,y4)).T
    W=np.ones((X.shape[0],4))
    e=np.linalg.norm(np.dot(W.T,X)-Y,'fro')#求该矩阵的F范数损失
    print("总损失：",e)
    # import pdb
    # pdb.set_trace()
    temp=np.dot(X,X.T)
    reverse_temp=np.linalg.pinv(temp)
    new_W=np.dot(reverse_temp,X)
    new_W=np.dot(new_W,Y.T)
    return new_W

def test(W,sample_test):
    """
    输出0代表第一类，1代表第二类，2代表第三类，3代表第四类。
    :param W:
    :param sample_test:
    :return:
    """
    out=np.dot(W.T, sample_test.T)
    index = np.argmax(out, axis=0)
    return index

sample1=np.array([[0.1,1.1,1],
                  [6.8,7.1,1],
                  [-3.5, -4.1,1],
                  [2.0, 2.7,1],
                  [4.1, 2.8,1],
                  [3.1, 5.0,1],
                  [-0.8, -1.3,1],
                  [0.9, 1.2,1],
                  [5.0, 6.4,1],
                  [3.9, 4.0,1]
                  ])
sample2=np.array([[7.1,4.2,1],
                  [-1.4,-4.3,1],
                  [4.5,0.0,1],
                  [6.3,1.6,1],
                  [4.2,1.9,1],
                  [1.4,-3.2,1],
                  [2.4,-4.0,1],
                  [2.5,-6.1,1],
                  [8.4,3.7,1],
                  [4.1,-2.2,1]
                  ])
sample3=np.array([[-3.0,-2.9,1],
                  [0.5,8.7,1],
                  [2.9,2.1,1],
                  [-0.1,5.2,1],
                  [-4.0,2.2,1],
                  [-1.3,3.7,1],
                  [-3.4,6.2,1],
                  [-4.1,3.4,1],
                  [-5.1,1.6,1],
                  [1.9,5.1,1]
                  ])
sample4=np.array([[-2.0,-8.4,1],
                  [-8.9,-0.2,1],
                  [-4.2,-7.7,1],
                  [-8.5,-3.2,1],
                  [-6.7,-4.0,1],
                  [-0.5,-9.2,1],
                  [-5.3,-6.7,1],
                  [-8.7,-6.4,1],
                  [-7.1,-9.7,1],
                  [-8.0,-6.3,1]
                  ])
#训练过程
new_W=Multi_MSE(sample1[:8],sample2[:8],sample3[:8],sample4[:8])
#测试过程
#第一类数据预测
table=""
y1_test=test(new_W,sample1[8:])
y2_test=test(new_W,sample2[8:])
y3_test=test(new_W,sample3[8:])
y4_test=test(new_W,sample4[8:])

print("数据".ljust(17)+'\t原类别\t'+"预测类别")
for i in range(len(y1_test)):
    print(str(sample1[8:][i]).ljust(17)+'\t1类\t\t'+str(y1_test[i]+1))
for i in range(len(y2_test)):
    print(str(sample2[8:][i]).ljust(17)+'\t2类\t\t'+str(y2_test[i]+1))
for i in range(len(y3_test)):
    print(str(sample3[8:][i]).ljust(17)+'\t3类\t\t'+str(y3_test[i]+1))
for i in range(len(y4_test)):
    print(str(sample4[8:][i]).ljust(17)+'\t4类\t\t'+str(y4_test[i]+1))

print(table)