a=  [5, 6, 7, 0, 1, 2,4]
# a=[57, 66,0, 1, 2, 3, 4, 7, 8, 9, 31, 32]
def find_mini(rotate,left,right):
    # import pdb
    # pdb.set_trace()
    if right-left==0:
        return rotate[left]
    elif right-left==1:
        if rotate[left]<rotate[right]:
            return rotate[left]
        else:
            return rotate[right]
    mid=int((left+right)/2)
    minimum=min([rotate[left],rotate[mid],rotate[mid+1],rotate[right]])
    if rotate.index(minimum)==left or rotate.index(minimum)==mid:
        return find_mini(rotate,left,mid)
    else:
        return find_mini(rotate,mid+1,right)
    # if rotate[mid]<rotate[right] or rotate[left]<rotate[right]:
    #     return find_mini(rotate,left,mid)
    # else:
    #     return find_mini(rotate,mid+1,right)

print(find_mini(a,0,len(a)-1))
