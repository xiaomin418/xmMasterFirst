distance = lambda x: x[0] ** 2 + x[1] ** 2
def partition(data_list, begin, end):
    # 选择最后一个元素作为分区键

    partition_key = distance(data_list[end - 1])

    # index为分区键的最终位置

    index = begin

    for i in range(begin, end):

        if distance(data_list[i]) < partition_key:
            data_list[i], data_list[index] = data_list[index], data_list[i]  # 交换

            index += 1

    data_list[index], data_list[end] = data_list[end], data_list[index]  # 交换

    return index

def find_top_k(data_list,K):

    length = len(data_list)

    begin = 0

    end = length-1

    index = partition(data_list,begin,end)

    while index != length - K:

        if index >length - K:

            end = index-1

            index = partition(data_list,begin,index-1)

        else:

            begin = index+1

            index = partition(data_list,index+1,end)

    return data_list[index]

N,K=input().split()
N,K=int(N),int(K)
points=[]
for i in range(N):
    a,b=input().split()
    a,b=int(a),int(b)
    points.append([a,b])
# import pdb
# pdb.set_trace()
a=find_top_k(points,K)
print(a[0],a[1])