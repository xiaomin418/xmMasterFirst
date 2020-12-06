m,n=input().split()
if len(m)>3:
    m=m[-3:]
if m[0]=='-':
    m = int(m[1:])
else:
    m = int(m)
# m,n=int(m),int(n)
def compute(m,n):
    if n==1:
        return m
    if n%2==0:
        half=compute(m,int(n/2))
        return half*half%1000
    else:
        half=compute(m,int((n-1)/2))
        temp=half*half%1000
        temp=temp*m%1000
        return temp
def bigcompute(m,n,base,i):
    if i==-1:
        return 1

    temp=1
    cur_res=1
    # import pdb
    # pdb.set_trace()
    for k in range(10):
        temp=temp*base%1000
        if k==int(n[i])-1:
            cur_res=temp

    return cur_res*bigcompute(m,n,temp,i-1)%1000

# result=1
# if m > 1000:
#     m = m % 1000
# for i in range(n):
#     result=result*m%1000
result=bigcompute(m,n,m,len(n)-1)
print(result)