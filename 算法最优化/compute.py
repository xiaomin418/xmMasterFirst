import numpy as np
def fx(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-4*x[0]-2*x[1]


def Sherman(Ainv,c,dT):
    # import pdb
    # pdb.set_trace()
    fenzi=np.dot(Ainv,c)
    fenzi=np.dot(fenzi,dT)
    fenzi=np.dot(fenzi,Ainv)
    print("fenzi",fenzi)

    fenmu=np.dot(dT,Ainv)
    fenmu=np.dot(fenmu,c)
    fenmu=fenmu+1
    print("fenmu",fenmu)

    return Ainv-fenzi/fenmu
# while True:
#     x=input("输入：").split()
#     x=[int(n) for n in x]
#     x = np.array([x[0], x[1]])
#     print("输出：",fx(x))
Ainv=np.array([[1,0,1],[0,1,-1],[1,0,2]])
c=np.array([[0],[0],[2]])
dT=np.array([[0,1,0]])
print(Sherman(Ainv,c,dT))
