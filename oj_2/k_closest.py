
comp_dis=lambda x,y: x ** 2 + y ** 2
import random
class point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.dis=comp_dis(x,y)

def kClosest(points, K):
    #points: List[List[int]], K: int
    # 计算欧几里得距离
    def work(i, j, K):
        import pdb
        pdb.set_trace()
        if i > j:
            return
        # 记录初始值
        oi, oj = i, j
        # 取最左边为哨兵值
        curmax=max(points[i:j+1],key=lambda x:x.dis)
        curmin=min(points[i:j+1],key=lambda x:x.dis)
        # randp=int((i+j)/2)
        pivot = int((curmax.dis+curmin.dis)/2)
        # points[i], points[randp] = points[randp], points[i]
        while i != j:
            while i < j and points[j].dis > pivot:
                j -= 1
            if i < j:
                points[i],points[j] = points[j],points[i]
                i=i+1
            while i < j and points[i].dis < pivot:
                i += 1
            if i < j:
                # 交换值
                points[j],points[i]=points[i],points[j]
                j=j-1

                # 交换哨兵
        # points[i], points[oi] = points[oi], points[i]
        # 递归
        # import pdb
        # pdb.set_trace()
        if K <i - oi:
            # 左半边排序
            work(oi, i - 1, K)
        elif K >i - oi:
            # 右半边排序
            work(i + 1, oj, K - (i - oi + 1))
        else:
            # 右半边排序
            print(points[K+oi].x,points[K+oi].y)
            return points[K]

    work(0, len(points) - 1, K)
    return points[:K]
N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append(point(a,b))
# point=[[1,3],[-2,2],[1,1],[3,4]]
# K=3
kClosest(points,K-1)