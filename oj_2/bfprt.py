distance = lambda x: x[0] ** 2 + x[1] ** 2

def partition(arr, pivotIndex):
    index = 0

    r = len(arr) - 1

    pivotValue = distance(arr[pivotIndex])

    arr[pivotIndex], arr[r] = arr[r], arr[pivotIndex]

    for i in range(0, r + 1):

        if distance(arr[i]) > pivotValue:
            arr[index], arr[i] = arr[i], arr[index]

            index += 1

    arr[index], arr[r] = arr[r], arr[index]

    return index


def pivot_median(arr):
    n = len(arr)

    while n > 5:

        cols = n / 5

        m = []
        cols=int(cols)
        for i in range(0, cols):
            s = sorted(arr[5 * i:(5 * i + 5)])

            m.append(s[2])

        arr = m

        n = len(arr)

    arr.sort()

    return arr[int(n / 2)]


def bfprt(arr, k):
    pivot = pivot_median(arr)

    pivotIndex = arr.index(pivot)

    index = partition(arr, pivotIndex)

    n = len(arr)
    import pdb
    pdb.set_trace()

    if k < n - index:

        return bfprt(arr[index + 1:n], k)

    elif k == n - index:

        return pivot

    elif k > n - index:

        return bfprt(arr[0:index], k - (n - index))


K = 3
points=[[1,3],[-2,2],[1,1],[3,4]]
# N,K=input().split()
# N,K=int(N),int(K)
# points=[]
# for i in range(N):
#     a,b=input().split()
#     a,b=int(a),int(b)
#     points.append([a,b])
res=bfprt(points, K)
print(res[0],res[1])
