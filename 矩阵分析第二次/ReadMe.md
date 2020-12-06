#### 1. Requiremnt:
python>=3.5
numpy=1.18.1

#### 2.How to run?

a) 在本目录下新建input_matrix文件，输入矩阵，文件一行代表矩阵一行元素，同一行元素之间用空格分隔。输入数据格式举例，如下：

```
1 2 4 17
3 6 -12 3
2 3 -3 2
0 2 -2 6
```

b) 运行python decompostion.py --filename input_matrix --method givens

其中--filename为测试文件，--method为采用的分解方法，--method参数仅可选择：'givens', 'householder', 'gram_schmidt', 'URV', 'LU'中的一种

c) 分解结果以命令行打印输出

#### 3.运行示例

命令行运行示例如下所示：

##### 运行示例1：



![运行示例1](D:\document\研一课程\pythonProject\矩阵分析第二次\运行示例1.png)

##### 运行示例2：![运行示例2](D:\document\研一课程\pythonProject\矩阵分析第二次\运行示例2.png)

##### 运行示例3：

可以对decompostion.py中246行到251行代码取消注释，运行命令python decomposion.py，同时测试所有分解方法。![运行示例3_1](D:\document\研一课程\pythonProject\矩阵分析第二次\运行示例3_1.png)

![运行示例3_2](D:\document\研一课程\pythonProject\矩阵分析第二次\运行示例3_2.png)