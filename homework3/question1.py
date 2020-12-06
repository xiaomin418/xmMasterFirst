def Monkey_Banana(monkeys,bananas):
    monkeys.sort()
    bananas.sort()
    time=[abs(bananas[i]-monkeys[i])for i in range(len(monkeys))]
    return max(time)
monkeys=[1,3,6]
bananas=[2,4,6]
print(Monkey_Banana(monkeys,bananas))