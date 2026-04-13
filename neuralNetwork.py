import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        self.lr=learningrate

        self.activation_function=lambda x:scipy.special.expit(x)
        pass

    def train(self,inputs_list,targets_list):

        #把输入列表和输出目标列表转换成可以进行矩阵运算的二维数组
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets_list=numpy.array(targets_list,ndmin=2).T

        #计算隐藏层的输入
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        #计算输出层的输入
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        #计算输出层的误差
        outputs_errors=targets_list-final_outputs

        #计算隐藏层的误差
        hidden_errors=numpy.dot(self.who.T,outputs_errors)

        #更新权重
        self.who+=self.lr*numpy.dot((outputs_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))

        pass

    def query(self,inputs_list):

        inputs=numpy.array(inputs_list,ndmin=2).T

        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

        
    pass

input_nodes=784
hidden_nodes=200
output_nodes=10

learning_rate=0.3

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

data_file=open("mnist_data.csv","r")
data_list=data_file.readlines()
data_file.close()

training_data_list=data_list[:60000]

#训练神经网络
for record in training_data_list:
    all_values=record.split()
    inputs=(numpy.asarray(all_values[1:],dtype=float)/255.0*0.99)+0.01
    targets=numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])]=0.99
    n.train(inputs,targets)
    pass



test_data_list=data_list[60000:70000]


each_number_correct_rate=[0,0,0,0,0,0,0,0,0,0]
each_number=[0,0,0,0,0,0,0,0,0,0]


scorecard=[]
correct_num=0

#测试神经网络
for record in test_data_list:
    all_values=record.split()
    correct_label=int(all_values[0])
    each_number[correct_label]+=1

    inputs=(numpy.asarray(all_values[1:],dtype=float)/255.0*0.99)+0.01
    outputs=n.query(inputs)
    label=numpy.argmax(outputs)
    #print("正确标签=",correct_label,"网络输出=",label)
    if(label==correct_label):
        scorecard.append(1)
        correct_num+=1

        for i in range(10):
            if(correct_label==i):
                each_number_correct_rate[i]+=1


    else:
        scorecard.append(0)
        pass
    pass


print("正确率=",correct_num/len(test_data_list))
print("各类数字的正确率:")
for i in range(10):
    if(each_number[i] > 0):
        print(f"数字 {i}: {each_number_correct_rate[i] / each_number[i]:.4f}")
    else:
        print(f"数字 {i}: 无样本")
