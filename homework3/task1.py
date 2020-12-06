

def deleteSub(s):
    i=0
    while i<len(s):
        if i+1>=len(s):
            break
        if s[i]=='A' and s[i+1]=='B':
            s.pop(i)
            s.pop(i)
            if i>0:
                i=i-1
        elif s[i]=='B' and s[i+1]=='B':
            s.pop(i)
            s.pop(i)
            if i>0:
                i=i-1
        else:
            i=i+1
    return len(s)




s=input()
s=list(s)
print(deleteSub(s))