# a=[57, 66,0, 1, 2, 3, 4, 7, 8, 9, 31, 32]
# a=[4,2,3,1]
# a=[6,3,2,8,7]
a=[3,9,1,8,12,11]
def merge_sort(alist):
    n = len(alist)  # 获取当前传入的列表的长度
    if n <= 1:  # 如果列表长度为1，即已经拆分到最后一步了，就直接返回当前列表，不再往下继续执行了
        return 0,alist
    mid = n // 2  # 取中间值，把当前输入的列表从中间拆分
    left_cnt ,left_li= merge_sort(alist[:mid])  # 取左半部分
    right_cnt ,right_li= merge_sort(alist[mid:])  # 取右半部分
    left_pointer = 0  # 设定左半部分的指针，从0开始
    right_pointer = 0  # 设定右半部分的指针，从0开始
    result = []  # 定义一个空列表result用于存储每次递归产生的排好序的列表
    cur_cnt=0
    while left_pointer < len(left_li) and right_pointer < len(right_li):  # 当各部分指针还没走到末尾时
        if left_li[left_pointer] <= 3*right_li[right_pointer]:  # 把较小的值存入result并让相应的指针+1
            result.append(left_li[left_pointer])
            left_pointer += 1
        else:
            result.append(right_li[right_pointer])
            right_pointer += 1
            cur_cnt=cur_cnt+(len(left_li)-left_pointer)
    result += left_li[left_pointer:]  # 如果是奇数个元素，最后单个的元素也添加到result里
    result += right_li[right_pointer:]
    return left_cnt+right_cnt+cur_cnt,result  # 最终返回的是一个新的排好序的列表result，因此空间复杂度要多一倍

# import pdb
# pdb.set_trace()
print(merge_sort(a))