
def most_money(m_array,n):
    if n==0:
        return m_array[0]
    elif n<0:
        return 0
    else:
        # print("迭代{}".format(n))
        choice_1=most_money(m_array,n-1)
        choice_2=most_money(m_array,n-2)+m_array[n]
        if choice_1>choice_2:
            return choice_1
        else:
            return choice_2

def circle_house(m_array,start,end):
    if start==end:
        return m_array[start]
    elif end<start:
        return 0
    elif start+end==len(m_array)-1:
        choice_1=circle_house(m_array,start+1,end-2)+m_array[end]
        choice_2=circle_house(m_array,start,end-1)
        if choice_1>choice_2:
            return choice_1
        else:
            return choice_2
    else:
        choice_1 = circle_house(m_array, start,end-1)
        choice_2 = circle_house(m_array, start,end-2) + m_array[end]
        if choice_1>choice_2:
            return choice_1
        else:
            return choice_2

money=[4,2,7,8,9,35,19,22,8,3,15]
print("If houses are along a street")
most=most_money(money,len(money)-1)
print(most)
print("If houses are a circle:")
most=circle_house(money,0,len(money)-1)
print(most)
