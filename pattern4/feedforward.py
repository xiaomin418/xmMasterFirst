from numpy import exp, array, random, dot
import numpy as np
import argparse


class Dataset(object):
    def __init__(self,trainset,batch_size):
        self.trainset=trainset
        random.shuffle(self.trainset)
        self.batch_size=batch_size
        self.len=len(trainset)
        self.cur=0

    def __len__(self):
        return len(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur*self.batch_size>=self.len:
            raise StopIteration()
        if self.batch_size * (self.cur + 1)<self.len:
            trains=self.trainset[self.cur*self.batch_size:self.batch_size * (self.cur + 1)]
        else:
            trains=self.trainset[self.cur*self.batch_size:]
        self.cur=self.cur+1
        train_input=[x['x'] for x in trains]
        train_input=array(train_input)
        train_output=[x['y'] for x in trains]
        train_output=array(train_output)
        return [train_input,train_output]


class NeuralNetwork():
    def __init__(self,input_size,hidden_size,output_size,batch_size,lr):
        # 设置随机数种子，使每次运行生成的随机数相同
        # 便于调试
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.lr=lr
        random.seed(1)

        # 我们对单个神经元进行建模，其中有3个输入连接和1个输出连接
        # 我们把随机的权值分配给一个3x1矩阵，值在-1到1之间，均值为0。
        self.synaptic_weights_in_hide=2 * random.random((input_size, hidden_size)) - 1
        self.synaptic_weights = 2 * random.random((hidden_size, output_size)) - 1
        # self.example_inputs = array([[0,0,0,0,0,0, 0, 1], [0,1,0,0,0,1, 1, 1], [0,0,0,1,0,1, 0, 1], [0,0,0,0,0,0, 1, 1],
        #                          [0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 1]])
        # self.training_outputs = array([[0, 1, 1, 0, 1, 1, 1, 0]])
        # print("synaptic_weights",self.synaptic_weights)

    # Sigmoid函数, 图像为S型曲线.
    # 我们把输入的加权和通过这个函数标准化在0和1之间。
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def __tanh(self, x):
        return (exp(x)-exp(-x)) / (exp(x) + exp(-x))

    # Sigmoid函数的导函数.
    # 即使Sigmoid函数的梯度
    # 它同样可以理解为当前的权重的可信度大小
    # 梯度决定了我们对调整权重的大小，并且指明了调整的方向
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1-x*x

    # 我们通过不断的试验和试错的过程来训练神经网络
    # 每一次都对权重进行调整
    def train(self, trainset,number_of_training_iterations):

        # training_set_outputs=training_set_outputs.T
        # import pdb
        # pdb.set_trace()
        #training_set_inputs,training_set_inputs
        losstgt=open('loss.txt','w')
        for iteration in range(number_of_training_iterations):
            # 把训练集传入神经网络.
            # print("interation",training_set_inputs)
            dataset=Dataset(trainset,self.batch_size)
            for bid,batch_train in enumerate(dataset):
                training_set_inputs, training_set_outputs=batch_train[0],batch_train[1]
                hide_output, output = self.think(training_set_inputs)

                # 计算损失值(期望输出与实际输出之间的差。
                error = training_set_outputs - output
                # print(error)
                if iteration % 10 == 0 and bid==0:
                    print("------iteration:{} Loss:{}------".format(iteration,np.sum(error * error * 0.5)))
                    losstgt.write(str(np.sum(error * error * 0.5))+'\n')
                # print("training_set_outputs",training_set_outputs)
                # print("output",output)
                # print("error",error)
                # print("error_len",len(error))

                # 损失值乘上sigmid曲线的梯度，结果点乘输入矩阵的转置
                # 这意味着越不可信的权重值，我们会做更多的调整
                # 如果为零的话，则误区调制
                adjustment = -dot(hide_output.T, error * self.__sigmoid_derivative(output))

                # import pdb
                # pdb.set_trace()
                nextlayer = error * self.__sigmoid_derivative(output)  # Nxoutput_size
                nexterror = dot(self.synaptic_weights, nextlayer.T)  # hidden_sizexoutput_size
                adjustment_in_hide = -dot(training_set_inputs.T,
                                          nexterror.T * self.__tanh_derivative(hide_output))
                # print("training_set_inputs",training_set_inputs)
                # print("error",error)
                # print("output",self.__sigmoid_derivative(output))
                # print("adjustment",adjustment)
                # 调制权值

                self.synaptic_weights -= adjustment*self.lr
                self.synaptic_weights_in_hide -= adjustment_in_hide*self.lr
                if iteration % 100 == 0:
                    self.gradient_check(training_set_inputs, training_set_outputs, adjustment, adjustment_in_hide)
                # 神经网络的“思考”过程
        losstgt.close()
    def test(self,testset):
        print("test:")
        test_loss=0
        count_correct=0
        for batch in testset:
            training_set_inputs, training_set_outputs = array(batch['x']),array(batch['y'])
            hide_output, output = self.think(training_set_inputs)

            # 计算损失值(期望输出与实际输出之间的差。
            error = training_set_outputs - output
            # print("x:",training_set_inputs)
            # print("y:",training_set_outputs)
            # print("y':",output)
            if np.argmax(output)==np.argmax(training_set_outputs):
                count_correct=count_correct+1
            # print(error)
            # print(output)
            # print("Loss", np.sum(error * error * 0.5))
            test_loss=test_loss+np.sum(error * error * 0.5)
        print("平均损失为：",test_loss/len(testset))
        print("分类正确率：",count_correct/len(testset))

    def think(self, inputs):
        # 把输入数据传入神经网络
        # print("think",dot(inputs,self.synaptic_weights))
        # hide_output=self.__sigmoid(dot(inputs, self.synaptic_weights_in_hide))
        hide_output=self.__tanh(dot(inputs, self.synaptic_weights_in_hide))
        output=self.__sigmoid(dot(hide_output, self.synaptic_weights))
        return hide_output,output

    def gradient_check(self,training_set_inputs,training_set_outputs,adjustment,adjustment_in_hide):
        # 隐层到输出层
        gradapprox_in_to_hide=np.zeros([self.input_size,self.hidden_size])
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                before = self.synaptic_weights_in_hide[i][j]
                self.synaptic_weights_in_hide[i][j] = before + 1e-7
                hide_output1, output1 = self.think(training_set_inputs)
                #计算loss
                error = training_set_outputs - output1
                loss1 = np.sum(error*error*0.5)
                self.synaptic_weights_in_hide[i][j] = before - 1e-7
                hide_output2, output2 = self.think(training_set_inputs)
                error = training_set_outputs - output2
                loss2 = np.sum(error * error * 0.5)
                #计算梯度
                gradapprox_in_to_hide[i][j] = (loss1-loss2)/(2*1e-7)
                self.synaptic_weights_in_hide[i][j] = before
        numerator = np.linalg.norm(gradapprox_in_to_hide - adjustment_in_hide)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(gradapprox_in_to_hide) + np.linalg.norm(adjustment_in_hide)  # Step 2'
        difference = numerator / denominator  # Step 3'
        # print("in_to_hide_difference", difference)

        hide_to_output = np.zeros([self.hidden_size, self.output_size])
        for i in range(self.hidden_size):
            before = self.synaptic_weights[i][0]
            self.synaptic_weights[i][0] = before + 1e-7
            hide_output1, output1 = self.think(training_set_inputs)
            # 计算loss
            error = training_set_outputs - output1
            loss1 = np.sum(error * error * 0.5)
            self.synaptic_weights[i][0] = before - 1e-7
            hide_output2, output2 = self.think(training_set_inputs)
            error = training_set_outputs - output2
            loss2 = np.sum(error * error * 0.5)
            # 计算梯度
            hide_to_output[i][0] = (loss1 - loss2) / (2 * 1e-7)
            self.synaptic_weights[i][0] = before

        numerator = np.linalg.norm(hide_to_output - adjustment)  # Step 1'
        # print("numerator",numerator)
        # print("hide_to_output",hide_to_output)
        # print("adjustment",adjustment)
        denominator = np.linalg.norm(hide_to_output) + np.linalg.norm(adjustment)  # Step 2'
        difference = numerator / denominator  # Step 3'
        # print("hide_to_output_difference",difference)

def getdata(file='data.txt'):
    f=open(file,'r')
    line=f.readline().strip()
    dataset=[]
    while line:
        line=line.split()
        line=[float(x) for x in line]
        data={}
        data['x']=line[:3]
        if int(line[-1])==0:
            data['y']=[1,0,0]
        elif int(line[-1])==1:
            data['y']=[0,1,0]
        elif int(line[-1])==2:
            data['y']=[0,0,1]
        line=f.readline().strip()
        dataset.append(data)
    f.close()
    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=6)
    parser.add_argument('--output_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--iter', type=int, default=10000)
    args = parser.parse_args()

    # 初始化一个单神经元的神经网络
    trainset=getdata()
    neural_network = NeuralNetwork(args.input_size,args.hidden_size,args.output_size,args.batch_size,args.lr)

    # 输出随机初始的参数作为参照
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # 训练集共有四个样本，每个样本包括三个输入一个输出
    # 用训练集对神经网络进行训练
    # 迭代10000次，每次迭代对权重进行微调.
    neural_network.train(trainset, args.iter)

    # 输出训练后的参数值，作为对照。
    # print("New synaptic weights after training: ")
    # print(neural_network.synaptic_weights)

    # 用新样本测试神经网络.
    testset=getdata('test.txt')
    neural_network.test(testset)
    # print(neural_network.synaptic_weights)
    # print(neural_network.synaptic_weights_in_hide)
