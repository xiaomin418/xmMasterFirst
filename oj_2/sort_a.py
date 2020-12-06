N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append([a,b])
distance = lambda x: x[0] ** 2 + x[1] ** 2
a=sorted(points,key=lambda x:distance(x))
print(a[K-1][0],a[K-1][1])