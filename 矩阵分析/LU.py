# -*- coding: utf-8 -*-
"""
1. Requiremnt:
python>=3.5
numpy=1.18.1
2.How to run?
a) 在本目录下新建input_matrix文件，输入矩阵，文件一行代表矩阵一行元素，同一行元素之间用空格分隔。(本代码只分析矩阵为NxN的情况)以作业题11题数据为例，如：
1 2 4 17
3 6 -12 3
2 3 -3 2
0 2 -2 6
b) 运行python LU.py
c) LU分解结果以命令行打印输出
"""
import numpy as np

def LU_decompose(A):
    """
    对矩阵A进行LU分解
    :param A: 待分解矩阵
    :return:
    """
    n = len(A)
    L = np.zeros(shape=(n, n))
    U = np.zeros(shape=(n, n))

    for base in range(n - 1):
        for i in range(base + 1, n):
            L[i, base] = A[i, base] / A[base, base]
            A[i] = A[i] - L[i, base] * A[base]
    for i in range(n):  # range(n) 范围：[0，n-1]
        L[i, i] = 1
    U = np.array(A)
    print("L:")
    print(L)
    print("U:")
    print(U)
    return L,U

def PA_LU_decompose(A):
    """
    对矩阵A进行PA-LU分解
    :param A:待分解矩阵
    :return:
    """
    n = len(A)
    L = np.zeros(shape=(n, n))
    U = np.zeros(shape=(n, n))
    P = np.eye(n,dtype=int)


    for base in range(n - 1):#以第base元素为主元，更新L,U,P矩阵。
        max_pos = np.argwhere(A[:, base] == max(A[base+1:, base]))#获取当前列的最大元素
        if max_pos.tolist()[0][0] != base:#若最大主元不在[base,base]位置，则需要交换base行和最大主元所在行max_pos
            #保存交换操作于P置换矩阵
            b = P[base, :].copy()
            P[base, :] = P[max_pos.tolist()[0][0], :]
            P[max_pos.tolist()[0][0], :] = b

            # 更新A矩阵
            b = A[base, :].copy()
            A[base, :] = A[max_pos.tolist()[0][0], :]
            A[max_pos.tolist()[0][0], :] = b

            # 更新L矩阵
            b = L[base, :].copy()
            L[base, :] = L[max_pos.tolist()[0][0], :]
            L[max_pos.tolist()[0][0], :] = b

        for i in range(base + 1, n):
            L[i, base] = A[i, base] / A[base, base]#确定L矩阵消去系数
            A[i] = A[i] - L[i, base] * A[base]#更新A矩阵
    for i in range(n):  # range(n) 范围：[0，n-1]
        L[i, i] = 1
    U = np.array(A)
    return L,U,P



if __name__=='__main__':
    with open('../矩阵分析第二次/input_matrix', 'r') as file:
        A=[]
        line=file.readline().strip()
        i=0
        while line:
            line=line.split()
            line=[float(x) for x in line]
            if i>0 and len(line)!=len(A[-1]):
                print("Err: 矩阵格式错误")
                exit(-1)
            A.append(line)
            line=file.readline().strip()
            i=i+1
        file.close()
    if  len(A)!=len(A[0]):
        print("Err: 矩阵格式错误")
        exit(-1)
    A=np.array(A)
    print("----------待LU分解矩阵：----------")
    print(A)
    L,U,P=PA_LU_decompose(A)
    print("----------分解结果为：----------")
    print("L:\n",L)
    print("U:\n",U)
    print("P:\n",P)
