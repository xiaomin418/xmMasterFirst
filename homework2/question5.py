
def Balance(stones,index,left,right):
    if index<0:
        return sum(left)-sum(right)
    else:
        left_diff=Balance(stones,index-1,left+[stones[index-1]],right)
        right_diff=Balance(stones,index-1,left,right+[stones[index-1]])
        if abs(left_diff)>abs(right_diff):
            return right_diff
        else:
            return left_diff



# stones=[1,5,8,4,10,6]
stones=[1,5,8,4,10,6]

print(Balance(stones,len(stones)-1,[],[]))