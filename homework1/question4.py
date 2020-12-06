a= [5, 7, 7, 8, 8, 10]
def find_pos(nums,key,left,right):
    if right-left==0:
        if nums[left]==key:
            return left,right
        else:
            return -1,-1
    else:
        mid=int((left+right)/2)
        l_left,l_right=find_pos(nums,key,left,mid)
        r_left,r_right=find_pos(nums,key,mid+1,right)
        if l_left==-1 and l_right==-1:
            return r_left,r_right
        elif r_left==-1 and r_right==-1:
            return l_left,l_right
        else:
            return l_left,r_right
# import pdb
# pdb.set_trace()
print(find_pos(a,11,0,len(a)-1))
