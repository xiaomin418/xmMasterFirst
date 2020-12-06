a=[57, 66,0, 1, 2, 3, 4, 7, 8, 9, 31, 32]

def find_max_sub(nums):
    subarrays=[]
    temp=[]
    for i in range(len(nums)):
        if len(temp)==0:
            temp.append(nums[i])
        else:
            if nums[i]==temp[-1]+1:
                temp.append(nums[i])
            elif len(temp)==1:
                temp[0]=nums[i]
            else:
                subarrays.append(temp)
                temp=[]
                temp.append(nums[i])
    if len(temp)>1:
        subarrays.append(temp)
    print("subarray:",subarrays)
    subarrays=[sum(x) for x in subarrays]
    print("sum of subarrays: ",subarrays)
    return max(subarrays)

print(find_max_sub(a))