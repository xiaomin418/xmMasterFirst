class Job(object):
    def __init__(self,id,p,f):
        self.id=id
        self.p=p
        self.f=f


def Schedule(jobs):
    jobs.sort(key=lambda x:x.f,reverse=True)
    start_of_p=[0]
    for i in range(len(jobs)):
        start_of_p.append(jobs[i].p+start_of_p[-1])
    end_of_f=[]
    start_of_p=start_of_p[1:]
    for i in range(len(jobs)):
        end_of_f.append(start_of_p[i]+jobs[i].f)
    return max(end_of_f)

ptime=[1000,2000,3000]
ftime=[3000,1000,2000]
jobs=[]
for i in range(len(ptime)):
    jobs.append(Job(i+1,ptime[i],ftime[i]))
print(Schedule(jobs))