import numpy as np


def fx(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-4*x[0]-2*x[1]
def FNeld(x0):
    step=0
    while True:
        print("-------step: {}-------".format(step))
        y = fx(x0)
        high = np.argsort(-y)
        print("H:",x0[:,high[0].T],"-->",y[high[0]])
        print("M:",x0[:,high[1].T],"-->",y[high[1]])
        print("L:",x0[:,high[2].T],"-->",y[high[2]])

        if y[high[0]] != y[high[1]] and y[high[1]] != y[high[2]]:
            xr = x0[:,high[2]] + x0[:,high[1]] - x0[:,high[0]]
            ry = fx(xr)
            print("f(xr): ",ry)
            if ry < y[high[0]]:
                x0[:,high[0]] = xr
                print("替换为点:",xr.T)
            else:
                return x0[:,high[2]]
        else:
            xr1 = x0[:,high[2]] + x0[:,high[1]] - x0[:,high[0]]
            xr2 = x0[:,high[2]] + x0[:,high[0]] - x0[:,high[1]]
            ry1 = fx(xr1)
            ry2 = fx(xr2)
            if ry1 < ry2 and ry1 < y[high[0]]:
                x0[:,high[0]] = xr1
                print("f(xr): ",ry1)
                print("替换为点:",xr1.T)
            elif ry2 < ry1 and ry2 < y[high[0]]:
                x0[:,high[0]] = xr2
                print("f(xr): ",ry2)
                print("替换为点:",xr2.T)
            else:
                return x0[:,high[2]]
        step=step+1

x_opt=np.array([[0,0],[0,1],[1,0]]).T
print(x_opt)
opt=FNeld(x_opt)
print(opt)
