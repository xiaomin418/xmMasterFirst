distance = lambda x: x[0] ** 2 + x[1] ** 2
def heap_build(parent, heap):
    child = 2 * parent + 1

    while child < len(heap):

        if child + 1 < len(heap) and distance(heap[child + 1]) < distance(heap[child]):
            child = child + 1

        if distance(heap[parent]) <= distance(heap[child]):
            break

        heap[parent], heap[child] = heap[child], heap[parent]

        parent, child = child, 2 * child + 1

    return heap


def Find_heap_kth(array, k):
    if k > len(array):
        return None

    heap = array[:k]

    for i in range(k, -1, -1):
        heap_build(i, heap)

    for j in range(k, len(array)):

        if distance(array[j]) > distance(heap[0]):
            heap[0] = array[j]

            heap_build(0, heap)

    return heap[0]


# print(Find_heap_kth([2, 1, 4, 3, 5, 9, 8, 0, 1, 3, 2, 5], 6))
N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append([a,b])

a=Find_heap_kth(points,N-K+1)
print(a[0],a[1])