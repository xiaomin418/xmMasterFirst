import numpy as np
import argparse
"""
作者：肖敏
学号：202018014628034
"""
def print_matrix(P,name,i=-1,j=-1):
    P1 = P.copy()
    m=P1.shape[0]
    n=P1.shape[1]
    P1[P1 < 10*-10] = 0
    for t1 in range(m):
        for t2 in range(n):
            P1[t1, t2] = round(P1[t1, t2], 3)
    if i==-1 and j==-1:
        print('{}:'.format(name), '\n', P1)
    elif i!=-1 and j==-1:
        print('R_{}:'.format(i+1), '\n', P1)
    else:
        print('P_{}{}:'.format(i+1,j+1), '\n', P1)

def input_test(file=''):
    with open(file,'r') as file:
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
    A = np.array(A)
    return A

def householder(A):
    m = A.shape[0]
    n = A.shape[1]
    R=np.eye(m)
    for i in range(n):
        if m-i<2:
            break
        e1=np.zeros((m-i,1))
        e1[0][0]=1.0
        u=A[i:,i:i+1]-np.sqrt(np.sum(np.multiply(A[i:,i], A[i:,i])))*e1
        Ri=np.eye(m-i)-2*np.dot(u,u.T)/(np.dot(u.T,u))
        complete_Ri=np.eye(m)
        complete_Ri[i:,i:]=Ri
        R=np.dot(complete_Ri,R)
        A[i:,i:]=np.dot(Ri,A[i:,i:])
        # print_matrix(A,'A')
        # print_matrix(complete_Ri,'Ri',i=i)
    return R.T,A

def givens(A):
    m = A.shape[0]
    n = A.shape[1]
    P = np.mat(np.eye(max(m,m)))
    R = np.mat(np.eye(max(m,m)))
    for i in range(0, n):
        # print("-"*28,'约简第{}列'.format(i),'-'*28)
        for j in range(i+1, m):
            if A[j, i] == 0:
                # print('a[{},{}]=0,跳过'.format(j,i))
                continue
            else:
                # print('a[{},{}]={}'.format(j,i,A[j, i]))
                d = np.sqrt(np.sum(np.multiply(A[i:j+1, i], A[i:j+1, i])))
                c1 = A[i, i]
                s1 = A[j, i]
                c = c1/d
                s = s1/d
                P[i, i] = c
                P[i, j] = s
                P[j, i] = -s
                P[j, j] = c

                A = P*A
                R = P * R

                # print_matrix(P, 'P', i, j)
                # print_matrix(A, 'A')
                P = np.mat(np.eye(max(m,m)))

    # print("-" * 66)
    return A,R.T

def gram_schmidt(A):
    # print(A)
    m = A.shape[0]
    n = A.shape[1]
    Q=np.zeros((m,m))
    if m<n:
        Q = A.copy()[:m,:m]
    else:
        Q[:m,:n]=A.copy()
    R=np.zeros((m,n))
    # print(np.sqrt(Q[:,0].T*Q[:,0]))

    R[0,0]=np.sqrt(np.dot(Q[:,0].T,Q[:,0]))
    Q[:,0:1]=Q[:,0:1]/np.sqrt(np.dot(Q[:,0].T,Q[:,0]))
    # print(Q,R)
    # import pdb
    # pdb.set_trace()
    for i in range(1,n):
        if i>m-1:
            break
        compo=np.zeros((m,1))
        for j in range(i):
            R[j,i]=np.dot(Q[:,j].T,Q[:,i])
            compo=compo+R[j,i]*Q[:,j].reshape(-1,1)
        # import pdb
        # pdb.set_trace()
        Q[:, i:i+1]=Q[:, i:i+1].reshape(-1,1)-compo
        R[i, i] = np.sqrt(np.dot(Q[:, i].T,Q[:, i]))
        Q[:, i:i+1] = Q[:, i:i+1] / np.sqrt(np.dot(Q[:, i].T,Q[:, i]))
    if n>m:
        for i in range(m,n):
            for j in range(m):
                R[j,i] = np.dot(A[:, i].T,Q[:, j])
    return Q,R

def URV(A):
    m = A.shape[0]
    n = A.shape[1]
    print("URV分解如下：")
    P, B = householder(A)
    if m>n:
        Q, T = householder(B[:n,:n].T)
    elif m<n:
        Q, T = householder(B[:m,:m].T)
    else:
        Q, T = householder(B[:n,:n].T)

    print_matrix(P,'P')
    print_matrix(T.T,'T.T')
    print_matrix(Q.T,'Q.T')

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


def main(methods,A):
    if methods=='givens':
        print_matrix(A,'待分解矩阵')
        aa, po = givens(A)
        print('基于givens变换的QR分解:')
        print_matrix(po, 'Q')
        print_matrix(aa, 'R')
    elif methods=='householder':
        print_matrix(A,'待分解矩阵')
        Q, R = householder(A)
        print('基于Householder的QR分解:')
        print_matrix(Q, 'Q')
        print_matrix(R, 'R')
    elif methods=='gram_schmidt':
        print_matrix(A,'待分解矩阵')
        Q,R=gram_schmidt(A)
        print('基于Gram-Schmidt的QR分解:')
        print_matrix(Q, 'Q')
        print_matrix(R, 'R')
    elif methods=='URV':
        print_matrix(A,'待分解矩阵')
        URV(A)
    elif methods=='LU':
        print_matrix(A,'待分解LU矩阵')
        if len(A) != len(A[0]):
            print("Err: 矩阵格式错误，LU分解仅支持方阵分解")
            exit(-1)
        L, U, P = PA_LU_decompose(A)
        print("LU分解结果为：")
        print("L:\n", L)
        print("U:\n", U)
        print("P:\n", P)
    else:
        pass

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,default='input_matrix')
    parser.add_argument('--method',type=str,default='givens')
    args=parser.parse_args()
    ai1=input_test(args.filename)
    main(args.method,ai1.copy())

    #一次完整的测试:
    methods=['givens','householder','gram_schmidt','URV','LU']
    main(methods[0],ai1.copy())
    main(methods[1],ai1.copy())
    main(methods[2],ai1.copy())
    main(methods[3],ai1.copy())
    main(methods[4],ai1)
