

def Step(k_set,cur_pos,unbrokens_steps):
    if cur_pos==unbrokens_steps[-1]:
        return 1
    else:
        ways=0
        for k in k_set:
            if cur_pos+k in unbrokens_steps:
                if k<=1:
                    ways=Step([1,2],cur_pos+k,unbrokens_steps)+ways
                else:
                    ways=Step([k-1, k,k+1], cur_pos + k, unbrokens_steps)+ways
        return ways


# unbrokens_steps=[1,2,3,4,5,7,8,11,14]
# unbrokens_steps=[0,1,2,4,5,7,11,16]
n=input()
unbrokens_steps=input().split()
unbrokens_steps=[int(x) for x in unbrokens_steps]
# unbrokens_steps=[0,1,2,3,7,8,12]
ways=Step([1],1,unbrokens_steps)
if ways>0:
    print("true")
else:
    print("false")
