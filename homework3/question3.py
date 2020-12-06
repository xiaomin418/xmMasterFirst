def Boats(weights,M):
    weights.sort()
    numboats=0
    i=0
    j=len(weights)-1
    while i<j:
        if weights[i]+weights[j]<=M:
            numboats=numboats+1
            i=i+1
            j=j-1
        else:
            numboats=numboats+1
            j=j-1
    if i==j:
        numboats=numboats+1
    return numboats

weights=[3,2,2,1]
M=3
print(Boats(weights,M))


