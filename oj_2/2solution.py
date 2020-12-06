
class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        # 思路：将points中每一组坐标点的平方和保存到第三个位置，根据第三列值进行排序，取出前K个进行处理输出
        for i in range(len(points)):
            points[i].append(points[i][0] ** 2 + points[i][1] ** 2)

        points = list(sorted(points, key=lambda x: x[2]))
        res = []
        res=[points[K-1][0], points[K-1][1]]
        # for i in range(K):
        #     res.append([points[i][0], points[i][1]])

        return res
ss=Solution()
N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append([a,b])
# points = [[3,3],[5,-1],[-2,4]]
# K = 2
res=ss.kClosest(points,K)
print(res[0],res[1])