# coding=utf-8
# 单纯形法的实现，只支持最简单的实现方法
# 且我们假设约束矩阵A的最后m列是可逆的
# 这样就必须满足A是行满秩的（m*n的矩阵）

import numpy as np

# column_sign=np.array(['x1','x2','y1','y2','λ1','λ2','μ1','μ2','u1','u2','u3','u4'])#3.4问题
column_sign=np.array(['x1','x2','x3','x4','λ','μ1','μ2','μ3','μ4','u1','u2','u3','u4','u5'])#3.2问题
# column_sign=np.array(['x1','x2','x3','x4','x5'])
# column_sign=np.array(['x1+','x1-','x2+','x2-','x3+','x3-','μ1','μ2'])#4p

class Simplex(object):
    def __init__(self, c, A, b):
        # 形式 minf(x)=c.Tx
        # s.t. Ax=b
        self.c = c
        self.A = A
        self.b = b

    def run(self):
        c_shape = self.c.shape
        A_shape = self.A.shape
        b_shape = self.b.shape
        assert c_shape[0] == A_shape[1], "Not Aligned A with C shape"
        assert b_shape[0] == A_shape[0], "Not Aligned A with b shape"

        # 找到初始的B，N等值
        # import pdb
        # pdb.set_trace()
        end_index = A_shape[1] - A_shape[0]
        N = self.A[:, 0:end_index]
        N_columns = np.arange(0, end_index)
        c_N = self.c[N_columns, :]
        # 第一个B必须是可逆的矩阵，其实这里应该用算法寻找，但此处省略
        B = self.A[:, end_index:]
        B_columns = np.arange(end_index, A_shape[1])
        c_B = self.c[B_columns, :]

        steps = 0
        while True:
            steps += 1
            print("Steps is {}".format(steps))
            # if steps==2:
            #     import pdb
            #     pdb.set_trace()
            is_optim, B_columns, N_columns = self.main_simplex(B, N, c_B, c_N, self.b, B_columns, N_columns)
            if is_optim:
                break
            else:
                B = self.A[:, B_columns]
                N = self.A[:, N_columns]
                c_B = self.c[B_columns, :]
                c_N = self.c[N_columns, :]

    def main_simplex(self, B, N, c_B, c_N, b, B_columns, N_columns):
        B_inverse = np.linalg.inv(B)
        # import pdb
        # pdb.set_trace()
        P = (c_N.T - np.matmul(np.matmul(c_B.T, B_inverse), N)).flatten()

        #计算x.T*\mu=0约束
        x_B = np.matmul(B_inverse, b)
        x0=np.zeros(B.shape[0]+N.shape[1])
        x0[B_columns]=x_B.T
        xT=x0[:4]
        mu=x0[5:9]

        if P.min() >= 0 and np.sum(xT*mu.T)==0:
            is_optim = True
            print("Reach Optimization.")
            print("B_columns is {}".format(B_columns+1))
            print("N_columns is {}".format(sorted(N_columns+1)))
            best_solution_point = np.matmul(B_inverse, b)
            print("Best Solution Point is {}".format(best_solution_point.flatten()))
            print("Best Value is {}".format(np.matmul(c_B.T, best_solution_point).flatten()[0]))
            print("\n")
            return is_optim, B_columns, N_columns
        else:
            # 入基
            print("Not Reach Optimization")
            N_i_in = np.argmin(P)
            N_i = N[:, N_i_in].reshape(-1, 1)
            # By=Ni， 求出基
            y = np.matmul(B_inverse, N_i)
            x_B = np.matmul(B_inverse, b)
            print("current x_B",x_B.T)
            N_i_out = self.find_out_base(y, x_B)
            tmp = N_columns[N_i_in]
            N_columns[N_i_in] = B_columns[N_i_out]
            B_columns[N_i_out] = tmp
            is_optim = False

            print("In Base is {},{}".format(tmp,column_sign[tmp]))
            print("Out Base is {},{}".format(N_columns[N_i_in],column_sign[N_columns[N_i_in]]))   # 此时已经被换过去了
            print("B_columns is {}".format(sorted(B_columns)))
            print("B_columns is {}".format(column_sign[B_columns.astype(int)]))
            print("N_columns is {}".format(sorted(N_columns)))
            print("\n")
            return is_optim, B_columns, N_columns

    def find_out_base(self, y, x_B):
        # 找到x_B/y最小且y>0的位置
        index = []
        min_value = []
        for i, value in enumerate(y):
            if value <= 0:
                continue
            else:
                index.append(i)
                min_value.append(x_B[i]/float(value))

        actual_index = index[np.argmin(min_value)]
        return actual_index


if __name__ == "__main__":
    '''
    c = np.array([-20, -30, 0, 0]).reshape(-1, 1)
    A = np.array([[1, 1, 1, 0], [0.1, 0.2, 0, 1]])
    b = np.array([100, 14]).reshape(-1, 1)
    '''
    # c = np.array([-4,-1,0, 0, 0]).reshape(-1, 1)
    # A = np.array([[-1, 2, 1, 0, 0], [2, 3, 0, 1, 0], [1, -1, 0, 0, 1]])
    # b = np.array([4,12,3]).reshape(-1, 1)


    # c = np.array([-12, -15, 0, 0, 0]).reshape(-1, 1)
    # A = np.array([[0.25, 0.5, 1, 0,0], [0.5, 0.5, 0, 1,0], [0.25, 0, 0, 0, 1]])
    # b = np.array([120,150,50]).reshape(-1, 1)

    # c = np.array([0, 0,20, 30]).reshape(-1, 1)#4x1
    # A = np.array([[1,1,1,0], [0.1,0.2,0,1]])
    # b = np.array([100,14]).reshape(-1, 1)
    #
    c = np.array([0, 0, 0,0, 0, 0,0, 0, 0,1,1,1,1,1]).reshape(-1, 1)
    A = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [1, -4, 2, 1, 1, -1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [-4, 16, -8, -4, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0],
                  [2, -8, 4, 2, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0],
                  [1, -4, 2, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 1]])
    b = np.array([4, 1,0,-7,-4]).reshape(-1, 1)


    #x1+,x1-,x2+,x2-,x3+,x3-,u1,u2
    # c = np.array([-1,1,1,-1,0,0,0,0]).reshape(-1, 1)
    # A = np.array([[-2,2,-3,3,5,-5,-1,0],
    #               [2,-2,3,-3,-5,5,0,-1]])
    # b = np.array([-1,-1]).reshape(-1, 1)

    # c = np.array([-1,1,0,0,0]).reshape(-1, 1)
    # A = np.array([[2,3,-5,1,0],
    #               [-2,-3,5,0,1]])
    # b = np.array([1,1]).reshape(-1, 1)


    # c = np.array([0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1]).reshape(-1, 1)
    # A = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #               [1,0,0,1,0,0,0,0,0,1,0,0],
    #               [2,0,0,0,1,1,-1,0,0,0,1,0],
    #               [0,8,0,0,1,0,0,-1,0,0,0,1]])
    # b = np.array([5,3,8,16]).reshape(-1, 1)


    # f = open('data.txt', 'w')
    # # f.write(str(A.shape[1])+'\n')
    # # f.write(str(A.shape[0])+'\n')
    # f.write(" ".join([str(x) for x in list(c.reshape(1, -1).tolist()[0])]) + '\n')
    # Ab = np.concatenate((A, b), axis=1)
    # for row in Ab:
    #     row = list(row)
    #     row = [str(x) for x in row]
    #     # import pdb
    #     # pdb.set_trace()
    #     f.write(" ".join(row) + '\n')
    # # f.write(" ".join([str(x) for x in list(b.reshape(1, -1).tolist()[0])]) + '\n')
    # f.close()

    simplex = Simplex(c, A, b)
    simplex.run()

