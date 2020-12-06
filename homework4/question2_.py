from pulp import *

# 1. 建立问题
prob = LpProblem("Bleding Problem", LpMaximize)

# 2. 建立变量
x1 = LpVariable("x1", 0, None, LpContinuous)
x2 = LpVariable("x2", 0, None, LpContinuous)
x3 = LpVariable("x3", 0, None, LpContinuous)

# 3. 设置目标函数
prob += 10*x1 + 5*x2+ 15*x3, "Total Cost of Profits per can"

# 4. 施加约束
prob += 3*x1 + 3*x2 +x3<= 100, "Nickel"
prob += 4*x1 + 2*x2 +x3<= 200, "Aluminum"

# 5. 求解
prob.solve()

# 6. 打印求解状态
print("Status:", LpStatus[prob.status])

# 7. 打印出每个变量的最优值
for v in prob.variables():
    print(v.name, "=", v.varValue)

# 8. 打印最优解的目标函数值
print("Total Cost of Ingredients per can = ", value(prob.objective))