#随机快速排序
#coding: utf-8
import random
def random_quicksort(a,left,right,k):
    if(left<right):
        mid = random_partition(a,left,right)
        if k<mid-left:
            return random_quicksort(a,left,mid-1,k)
        elif k>mid-left:
            return random_quicksort(a,mid+1,right,k - (mid - left + 1))
        else:
            return a[mid]


def random_partition(a,left,right):
    t = random.randint(left,right)     #生成[left,right]之间的一个随机数
    a[t],a[right] = a[right],a[t]
    x = a[right]
    i = left-1                         #初始i指向一个空，保证0到i都小于等于 x
    for j in range(left,right):        #j用来寻找比x小的，找到就和i+1交换，保证i之前的都小于等于x
        if(a[j]<=x):
            i = i+1
            a[i],a[j] = a[j],a[i]
    a[i+1],a[right] = a[right],a[i+1]  #0到i 都小于等于x ,所以x的最终位置就是i+1
    return i+1

while(True):
    try:
        k=int(input("输入k: \n"))
        s = input("输入待排序数组：\n")             #待排数组
        l =s.split()
        a = [int(t) for t in l]
        # import pdb
        # pdb.set_trace()
        tgt=random_quicksort(a,0,len(a)-1,k)
        print(tgt)
        # print ("排序后：")
        # for item in a:
        #     print(item,end=' ')
        # print("\n")
    except:
        break