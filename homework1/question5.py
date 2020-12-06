
def decompose(n):
    if n==2:
        return 1
    elif n==3:
        return 1
    else:
        sum=0
        for i in range(2,n):
            sum=sum+decompose(i)*decompose(n-i+1)
        return sum
n=6
# import pdb
# pdb.set_trace()
print(decompose(n))