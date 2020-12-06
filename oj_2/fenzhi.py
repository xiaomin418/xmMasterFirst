import random

def kClosest(points, K):
    #points: List[List[int]], K: int
    # 计算欧几里得距离
    distance = lambda i: points[i][0] ** 2 + points[i][1] ** 2
    def work(i, j, K):
        # import pdb
        # pdb.set_trace()
        if i > j:
            return
        # 记录初始值
        oi, oj = i, j
        # 取最左边为哨兵值
        # randp = random.randint(i, j)
        if (j-i)>9:
            index = random.sample(range(i, j), 9)
            sample_dis = [[i, distance(i)] for i in index]
            sample_dis = sorted(sample_dis, key=lambda x: x[1])
            randp = sample_dis[5][0]
        else:
            randp = int((i + j) / 2)
        pivot = distance(randp)
        points[i], points[randp] = points[randp], points[i]
        while i != j:
            while i < j and distance(j) > pivot:
                j -= 1
            if i < j:
                points[i],points[j] = points[j],points[i]
                i=i+1
            while i < j and distance(i) < pivot:
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
            print(points[K+oi][0],points[K+oi][1])
            return points[K]

    work(0, len(points) - 1, K)
    return points[:K]
N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append([a,b])
# point=[[1,3],[-2,2],[1,1],[3,4]]
# K=3
kClosest(points,K-1)