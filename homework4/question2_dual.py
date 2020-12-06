from pulp import *

# 1. 建立问题
prob = LpProblem("Bleding Problem", LpMinimize)

# 2. 建立变量
y1 = LpVariable("y1", 0, None, LpContinuous)
y2 = LpVariable("y2", 0, None, LpContinuous)

# 3. 设置目标函数
prob += 100*y1 + 200*y2, "Total Cost of Profits per can"

# 4. 施加约束
prob += 3*y1 + 4*y2 >= 10, "A"
prob += 3*y1 + 2*y2 >= 5, "B"
prob += y1+y2>= 15, "C"

# 5. 求解
prob.solve()

# 6. 打印求解状态
print("Status:", LpStatus[prob.status])

# 7. 打印出每个变量的最优值
for v in prob.variables():
    print(v.name, "=", v.varValue)

# 8. 打印最优解的目标函数值
print("Total Cost of Ingredients per can = ", value(prob.objective))