
def triangle(segs):
    segs.sort()
    for i in range(1,len(segs)-1):
        if segs[i-1]+segs[i]>segs[i+1]:
            return 'YES'
    return 'NO'

n=input()
segs=input().split()
segs=[int(x) for x in segs]
print(triangle(segs))