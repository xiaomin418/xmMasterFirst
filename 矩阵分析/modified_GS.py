import numpy as np

x=np.array([[1,0.001,0.001],[1,0.001,0],[1,0,0.001]])
u=x.copy()
I=np.eye(3)
E2=I-np.matmul(u[0].reshape(3,-1),u[0].reshape(-1,3))
u[1:]=np.dot(E2,u[1:].T).T
u[1]=u[1]/np.sqrt(np.sum(np.dot(u[1],u[1])))
# print(u)

E3=I-np.matmul(u[1].reshape(3,-1),u[0].reshape(-1,3))
u[2:]=np.dot(E2,u[2:].T).T
u[2]=u[2]/np.sqrt(np.sum(np.dot(u[2],u[2])))
print(u)
