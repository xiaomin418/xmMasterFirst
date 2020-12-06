

def BST(n):
    if n==0 or n==1:
        return 1
    else:
        sum=0
        for i in range(n):
            sum=sum+BST(i)*BST(n-1-i)
        return sum

n=2
print(BST(n))