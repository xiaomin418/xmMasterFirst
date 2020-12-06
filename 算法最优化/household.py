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
    elif i!=-1 and j==-1:
        print('R_{}:'.format(i+1), '\n', P1)
    else:
        print('P_{}{}:'.format(i+1,j+1), '\n', P1)

def household(A):
    m = A.shape[0]
    n = A.shape[1]
    R=np.eye(m)
    for i in range(n):
        if m-i<2:
            break
        # if i==2:
        import pdb
        pdb.set_trace()
        e1=np.zeros((m-i,1))
        e1[0][0]=1.0
        u=A[i:,i]-np.sqrt(np.sum(np.multiply(A[i:,i], A[i:,i])))*e1
        print("u: ",u)
        # import pdb
        # pdb.set_trace()
        Ri=np.eye(m-i)-2*np.dot(u,u.T)/(np.dot(u.T,u))
        complete_Ri=np.eye(m)
        complete_Ri[i:,i:]=Ri
        R=np.dot(complete_Ri,R)
        A[i:,i:]=np.dot(Ri,A[i:,i:])
        print_matrix(A,'A')
        print_matrix(complete_Ri,'Ri',i=i)
    return A,R.T


if __name__ == "__main__":
    ai = np.mat([[4, -3, 4],
                [2, -14, -3] ,
                [-2, 14, 0],
                [1, -7, 15]], dtype=float)
    # ai = np.mat([[1, 0, -1],
    #              [1, 2, 1],
    #              [1, 1, -3],
    #              [0, 1, 1]], dtype=float)

    # ai = np.mat([[1, 19, -34],
    #              [-2, -5, 20],
    #              [2, 8, 37]], dtype=float)
    A,R=household(ai)

    print('基于Household的QR分解:')
    print_matrix(A,'Q')
    print_matrix(R,'R')
    # b=np.array([[5,-15,0,30]]).T
    # print(np.dot(R.T,b))