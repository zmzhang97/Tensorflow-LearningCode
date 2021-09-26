

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data   #手写数字相关的数据包
```


```python
# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)    #载入数据，{数据集包路径，把标签转化为只有0和1的形式}

#定义变量，即每个批次的大小
batch_size = 100    #一次放100章图片进去
n_batch = mnist.train.num_examples // batch_size   #计算一共有多少个批次；训练集数量（整除）一个批次大小

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])    #[行不确定，列为784]
y = tf.placeholder(tf.float32,[None,10])    #数字为0-9，则为10

#创建简单的神经网络
W = tf.Variable(tf.zeros([784,10]))   #权重
b = tf.Variable(tf.zeros([10]))     #偏置
prediction = tf.nn.softmax(tf.matmul(x,W)+b)    #预测

#定义二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#准确数，结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))   #比较两个参数大小是否相同，同则返回为true，不同则返回为false；argmax()：返回张量中最大的值所在的位置

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #cast()：将布尔型转换为32位的浮点型；（比方说9个T和1个F，则为9个1，1个0，即准确率为90%）

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(101):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy" + str(acc))

```

    Extracting MNIST_data\train-images-idx3-ubyte.gz
    Extracting MNIST_data\train-labels-idx1-ubyte.gz
    Extracting MNIST_data\t10k-images-idx3-ubyte.gz
    Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
    Iter0,Testing Accuracy0.8318
    Iter1,Testing Accuracy0.8705
    Iter2,Testing Accuracy0.8817
    Iter3,Testing Accuracy0.8884
    Iter4,Testing Accuracy0.894
    Iter5,Testing Accuracy0.8966
    Iter6,Testing Accuracy0.9004
    Iter7,Testing Accuracy0.9012
    Iter8,Testing Accuracy0.9037
    Iter9,Testing Accuracy0.9055
    Iter10,Testing Accuracy0.9067
    Iter11,Testing Accuracy0.9072
    Iter12,Testing Accuracy0.9081
    Iter13,Testing Accuracy0.9095
    Iter14,Testing Accuracy0.9096
    Iter15,Testing Accuracy0.9109
    Iter16,Testing Accuracy0.9122
    Iter17,Testing Accuracy0.9121
    Iter18,Testing Accuracy0.9126
    Iter19,Testing Accuracy0.9128
    Iter20,Testing Accuracy0.9144
    Iter21,Testing Accuracy0.9151
    Iter22,Testing Accuracy0.9153
    Iter23,Testing Accuracy0.9162
    Iter24,Testing Accuracy0.9163
    Iter25,Testing Accuracy0.9161
    Iter26,Testing Accuracy0.9166
    Iter27,Testing Accuracy0.9169
    Iter28,Testing Accuracy0.9185
    Iter29,Testing Accuracy0.9176
    Iter30,Testing Accuracy0.9176
    Iter31,Testing Accuracy0.9184
    Iter32,Testing Accuracy0.9174
    Iter33,Testing Accuracy0.9186
    Iter34,Testing Accuracy0.9183
    Iter35,Testing Accuracy0.9187
    Iter36,Testing Accuracy0.919
    Iter37,Testing Accuracy0.9189
    Iter38,Testing Accuracy0.9189
    Iter39,Testing Accuracy0.9199
    Iter40,Testing Accuracy0.9196
    Iter41,Testing Accuracy0.9196
    Iter42,Testing Accuracy0.9198
    Iter43,Testing Accuracy0.9209
    Iter44,Testing Accuracy0.9201
    Iter45,Testing Accuracy0.9205
    Iter46,Testing Accuracy0.921
    Iter47,Testing Accuracy0.9213
    Iter48,Testing Accuracy0.9212
    Iter49,Testing Accuracy0.9209
    Iter50,Testing Accuracy0.9216
    Iter51,Testing Accuracy0.9215
    Iter52,Testing Accuracy0.9216
    Iter53,Testing Accuracy0.9219
    Iter54,Testing Accuracy0.9219
    Iter55,Testing Accuracy0.9222
    Iter56,Testing Accuracy0.9222
    Iter57,Testing Accuracy0.9228
    Iter58,Testing Accuracy0.9232
    Iter59,Testing Accuracy0.923
    Iter60,Testing Accuracy0.9232
    Iter61,Testing Accuracy0.9232
    Iter62,Testing Accuracy0.9231
    Iter63,Testing Accuracy0.9235
    Iter64,Testing Accuracy0.9237
    Iter65,Testing Accuracy0.9238
    Iter66,Testing Accuracy0.9239
    Iter67,Testing Accuracy0.9239
    Iter68,Testing Accuracy0.9236
    Iter69,Testing Accuracy0.924
    Iter70,Testing Accuracy0.9241
    Iter71,Testing Accuracy0.9242
    Iter72,Testing Accuracy0.9243
    Iter73,Testing Accuracy0.9245
    Iter74,Testing Accuracy0.9245
    Iter75,Testing Accuracy0.9245
    Iter76,Testing Accuracy0.9252
    Iter77,Testing Accuracy0.9242
    Iter78,Testing Accuracy0.924
    Iter79,Testing Accuracy0.9247
    Iter80,Testing Accuracy0.9243
    Iter81,Testing Accuracy0.9247
    Iter82,Testing Accuracy0.9254
    Iter83,Testing Accuracy0.9253
    Iter84,Testing Accuracy0.9253
    Iter85,Testing Accuracy0.9251
    Iter86,Testing Accuracy0.9245
    Iter87,Testing Accuracy0.9252
    Iter88,Testing Accuracy0.9254
    Iter89,Testing Accuracy0.9262
    Iter90,Testing Accuracy0.9259
    Iter91,Testing Accuracy0.9259
    Iter92,Testing Accuracy0.9263
    Iter93,Testing Accuracy0.926
    Iter94,Testing Accuracy0.9261
    Iter95,Testing Accuracy0.9261
    Iter96,Testing Accuracy0.9264
    Iter97,Testing Accuracy0.9262
    Iter98,Testing Accuracy0.9254
    Iter99,Testing Accuracy0.9258
    Iter100,Testing Accuracy0.9258
    


```python

```
