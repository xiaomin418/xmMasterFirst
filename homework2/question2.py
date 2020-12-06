
def LargetSubset(set):
    if len(set)==0 or len(set)==1:
        return set
    else:
        newsub=[]
        for i in range(len(set)-1):
            if set[i]%set[-1]==0 or set[-1]%set[i]==0:
                newsub.append(set[i])
            else:
                continue
        haslast=LargetSubset(newsub)+[set[-1]]
        nolast=LargetSubset(set[:-1])
        if len(haslast)>len(nolast):
            return haslast
        else:
            return nolast


set=[24, 46, 3, 47, 49, 29, 12, 21, 33, 17]
# set=[3,4,8,9,10,15]
subset=LargetSubset(set)
print(len(subset),": ",subset)
