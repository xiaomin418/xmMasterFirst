



def Methods(total,values,nums):
    if total==0:
        return 1
    elif total<0:
        return 0
    else:
        if len(values)<=0:
            return 0
        ways=0
        for i in range(nums[0]+1):
            ways=Methods(total-i*values[0],values[1:],nums[1:])+ways
        return ways



values=[1,2,5,10,20,50,100]
nums=[10,8,5,4,7,3,10]
total=9
print(Methods(total,values,nums))