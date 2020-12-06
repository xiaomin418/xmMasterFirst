#随机快速排序
#coding: utf-8
import random
distance = lambda x: x[0] ** 2 + x[1] ** 2
class point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.dis=distance([x,y])


def random_quicksort(a,left,right,k):
    if(left<right):
        mid = random_partition(a,left,right)
        if k<mid-left:
            return random_quicksort(a,left,mid-1,k)
        elif k>mid-left:
            return random_quicksort(a,mid+1,right,k - (mid - left + 1))
        else:
            print(a[mid].x,a[mid].y)
            return
    else:
        print(a[left].x,a[right].y)
        return


def random_partition(a,left,right):
    # if (right - left) > 8:
    #     index = [left,int((right-left)/4),int((right-left)/2),int((right-left)*3/4),right]
    #     sample_dis = [[i, distance(a[i])] for i in index]
    #     sample_dis = bubble(sample_dis)
    #     t = sample_dis[2][0]
    # else:
    #     t = random.randint(left,right)     #生成[left,right]之间的一个随机数
    # t = random.randint(left, right)  # 生成[left,right]之间的一个随机数
    t=int((left+right)/2)
    a[t],a[right] = a[right],a[t]
    x = a[right]
    i = left-1                         #初始i指向一个空，保证0到i都小于等于 x
    for j in range(left,right):        #j用来寻找比x小的，找到就和i+1交换，保证i之前的都小于等于x
        if(a[j].dis<=x.dis):
            i = i+1
            a[i],a[j] = a[j],a[i]
    a[i+1],a[right] = a[right],a[i+1]  #0到i 都小于等于x ,所以x的最终位置就是i+1
    return i+1

N, K = input().split()
N, K = int(N), int(K)
points = []
for i in range(N):
    a, b = input().split()
    a, b = int(a), int(b)
    points.append(point(a, b))
    # point=[[1,3],[-2,2],[1,1],[3,4]]
    # K=3
# import pdb
# pdb.set_trace()
tgt=random_quicksort(points,0,len(points)-1,K-1)
# print(tgt)