#  基于吉文斯变换的QR分解

import numpy as np

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
    else:
        print('P_{}{}:'.format(i+1,j+1), '\n', P1)

def givens(A):
    m = A.shape[0]
    n = A.shape[1]
    # import pdb
    # pdb.set_trace()
    P = np.mat(np.eye(max(m,n)))
    R = np.mat(np.eye(max(m,n)))
    for i in range(0, n):
        print("-"*28,'约简第{}列'.format(i),'-'*28)
        for j in range(i+1, m):
            if A[j, i] == 0:
                print('a[{},{}]=0,跳过'.format(j,i))
                continue
            else:
                print('a[{},{}]={}'.format(j,i,A[j, i]))
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

                print_matrix(P, 'P', i, j)
                print_matrix(A, 'A')
                P = np.mat(np.eye(max(m,n)))

    print("-" * 66)
    return A, R.T


if __name__ == "__main__":
    # ai = np.mat([[3, 5, 5],
    #             [0, 3, 4],
    #             [4, 0, 5]], dtype=float)

    ai = np.mat([[1, 19, -34],
                [-2, -5, 20],
                [2, 8, 37]], dtype=float)
    # ai = np.mat([[-4, -2, -4, -2],
    #              [2, -2, 2, 1],
    #              [-4, 1, -3, -2]], dtype=float)
    ai = np.mat([[4, -3, 4],
                [2, -14, -3],
                [-2, 14, 0],
                [1, -7, 15]], dtype=float)
    aa, po = givens(ai)
    print('基于吉文斯变换的QR分解:')
    print_matrix(po,'Q')
    print_matrix(aa,'R')

