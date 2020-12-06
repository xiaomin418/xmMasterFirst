def maximumSum(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    # f[i]以arr[i]结尾的删除0个数的子数组最大和
    # f1[i]以arr[i]结尾的删除1个数的子数组最大和
    f, f1 = [0] * n, [0] * n
    f[0], f[1] = arr[0], max(arr[1], arr[0] + arr[1])
    f1[0], f1[1] = 0, max(arr[0], arr[1])
    # res必须保留一个数
    res = max(f[0], f[1], f1[1])
    for i in range(2, n):
        f[i] = max(arr[i], f[i - 1] + arr[i])
        f1[i] = arr[i] + max(f1[i - 1], f[i - 2])
        res = max(res, f[i], f1[i])
    return res

arr=input().split()
arr=[int(x) for x in arr]
print(maximumSum(arr))