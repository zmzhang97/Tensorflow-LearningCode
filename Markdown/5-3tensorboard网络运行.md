

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

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean) #平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max',tf.reduce_max(var)) #最大值
        tf.summary.scalar('min',tf.reduce_min(var)) #最小值
        tf.summary.scalar('histogram',var) #直方图

#(在3-2基础上添加)命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x-input')    #[行不确定，列为784]
    y = tf.placeholder(tf.float32,[None,10],name='y-input')    #数字为0-9，则为10

with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')   #权重
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')     #偏置
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)    #预测

with tf.name_scope('loss'):
    #定义二次代价函数
    # loss = tf.reduce_mean(tf.square(y-prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #准确数，结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))   #比较两个参数大小是否相同，同则返回为true，不同则返回为false；argmax()：返回张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #cast()：将布尔型转换为32位的浮点型；（比方说9个T和1个F，则为9个1，1个0，即准确率为90%）
        tf.summary.scalar('accuracy',accuracy)

#合并所有的summary
merged = tf.summary.merge_all()

#在3-2基础上更改
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)    
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})   #边训练边统计将其反馈到summary中
            
        writer.add_summary(summary,epoch)  #将其记录下来
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy" + str(acc))

```

    WARNING:tensorflow:From <ipython-input-2-b21490ac10f8>:2: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From D:\anaconda\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From D:\anaconda\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data\train-images-idx3-ubyte.gz
    WARNING:tensorflow:From D:\anaconda\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data\train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From D:\anaconda\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting MNIST_data\t10k-images-idx3-ubyte.gz
    Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From D:\anaconda\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From <ipython-input-2-b21490ac10f8>:42: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1355     try:
    -> 1356       return fn(*args)
       1357     except errors.OpError as e:
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _run_fn(feed_dict, fetch_list, target_list, options, run_metadata)
       1340       return self._call_tf_sessionrun(
    -> 1341           options, feed_dict, fetch_list, target_list, run_metadata)
       1342 
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list, run_metadata)
       1428         self._session, options, feed_dict, fetch_list, target_list,
    -> 1429         run_metadata)
       1430 
    

    InvalidArgumentError: tags and values not the same shape: [] != [10] (tag 'layer/biases/summaries/histogram')
    	 [[{{node layer/biases/summaries/histogram}}]]

    
    During handling of the above exception, another exception occurred:
    

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-2-b21490ac10f8> in <module>
         69         for batch in range(n_batch):
         70             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    ---> 71             summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})   #边训练边统计将其反馈到summary中
         72 
         73         writer.add_summary(summary,epoch)  #将其记录下来
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
        948     try:
        949       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 950                          run_metadata_ptr)
        951       if run_metadata:
        952         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1171     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1172       results = self._do_run(handle, final_targets, final_fetches,
    -> 1173                              feed_dict_tensor, options, run_metadata)
       1174     else:
       1175       results = []
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1348     if handle is None:
       1349       return self._do_call(_run_fn, feeds, fetches, targets, options,
    -> 1350                            run_metadata)
       1351     else:
       1352       return self._do_call(_prun_fn, handle, feeds, fetches)
    

    D:\anaconda\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1368           pass
       1369       message = error_interpolation.interpolate(message, self._graph)
    -> 1370       raise type(e)(node_def, op, message)
       1371 
       1372   def _extend_graph(self):
    

    InvalidArgumentError: tags and values not the same shape: [] != [10] (tag 'layer/biases/summaries/histogram')
    	 [[node layer/biases/summaries/histogram (defined at <ipython-input-2-b21490ac10f8>:18) ]]
    
    Errors may have originated from an input operation.
    Input Source operations connected to node layer/biases/summaries/histogram:
     layer/biases/b/read (defined at <ipython-input-2-b21490ac10f8>:32)
    
    Original stack trace for 'layer/biases/summaries/histogram':
      File "D:\anaconda\lib\runpy.py", line 193, in _run_module_as_main
        "__main__", mod_spec)
      File "D:\anaconda\lib\runpy.py", line 85, in _run_code
        exec(code, run_globals)
      File "D:\anaconda\lib\site-packages\ipykernel_launcher.py", line 16, in <module>
        app.launch_new_instance()
      File "D:\anaconda\lib\site-packages\traitlets\config\application.py", line 658, in launch_instance
        app.start()
      File "D:\anaconda\lib\site-packages\ipykernel\kernelapp.py", line 505, in start
        self.io_loop.start()
      File "D:\anaconda\lib\site-packages\tornado\platform\asyncio.py", line 148, in start
        self.asyncio_loop.run_forever()
      File "D:\anaconda\lib\asyncio\base_events.py", line 539, in run_forever
        self._run_once()
      File "D:\anaconda\lib\asyncio\base_events.py", line 1775, in _run_once
        handle._run()
      File "D:\anaconda\lib\asyncio\events.py", line 88, in _run
        self._context.run(self._callback, *self._args)
      File "D:\anaconda\lib\site-packages\tornado\ioloop.py", line 690, in <lambda>
        lambda f: self._run_callback(functools.partial(callback, future))
      File "D:\anaconda\lib\site-packages\tornado\ioloop.py", line 743, in _run_callback
        ret = callback()
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 787, in inner
        self.run()
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 748, in run
        yielded = self.gen.send(value)
      File "D:\anaconda\lib\site-packages\ipykernel\kernelbase.py", line 378, in dispatch_queue
        yield self.process_one()
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 225, in wrapper
        runner = Runner(result, future, yielded)
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 714, in __init__
        self.run()
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 748, in run
        yielded = self.gen.send(value)
      File "D:\anaconda\lib\site-packages\ipykernel\kernelbase.py", line 365, in process_one
        yield gen.maybe_future(dispatch(*args))
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 209, in wrapper
        yielded = next(result)
      File "D:\anaconda\lib\site-packages\ipykernel\kernelbase.py", line 272, in dispatch_shell
        yield gen.maybe_future(handler(stream, idents, msg))
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 209, in wrapper
        yielded = next(result)
      File "D:\anaconda\lib\site-packages\ipykernel\kernelbase.py", line 542, in execute_request
        user_expressions, allow_stdin,
      File "D:\anaconda\lib\site-packages\tornado\gen.py", line 209, in wrapper
        yielded = next(result)
      File "D:\anaconda\lib\site-packages\ipykernel\ipkernel.py", line 294, in do_execute
        res = shell.run_cell(code, store_history=store_history, silent=silent)
      File "D:\anaconda\lib\site-packages\ipykernel\zmqshell.py", line 536, in run_cell
        return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
      File "D:\anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 2854, in run_cell
        raw_cell, store_history, silent, shell_futures)
      File "D:\anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 2880, in _run_cell
        return runner(coro)
      File "D:\anaconda\lib\site-packages\IPython\core\async_helpers.py", line 68, in _pseudo_sync_runner
        coro.send(None)
      File "D:\anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 3057, in run_cell_async
        interactivity=interactivity, compiler=compiler, result=result)
      File "D:\anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 3248, in run_ast_nodes
        if (await self.run_code(code, result,  async_=asy)):
      File "D:\anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 3325, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-2-b21490ac10f8>", line 33, in <module>
        variable_summaries(b)
      File "<ipython-input-2-b21490ac10f8>", line 18, in variable_summaries
        tf.summary.scalar('histogram',var) #直方图
      File "D:\anaconda\lib\site-packages\tensorflow\python\summary\summary.py", line 82, in scalar
        val = _gen_logging_ops.scalar_summary(tags=tag, values=tensor, name=scope)
      File "D:\anaconda\lib\site-packages\tensorflow\python\ops\gen_logging_ops.py", line 776, in scalar_summary
        "ScalarSummary", tags=tags, values=values, name=name)
      File "D:\anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 788, in _apply_op_helper
        op_def=op_def)
      File "D:\anaconda\lib\site-packages\tensorflow\python\util\deprecation.py", line 507, in new_func
        return func(*args, **kwargs)
      File "D:\anaconda\lib\site-packages\tensorflow\python\framework\ops.py", line 3616, in create_op
        op_def=op_def)
      File "D:\anaconda\lib\site-packages\tensorflow\python\framework\ops.py", line 2005, in __init__
        self._traceback = tf_stack.extract_stack()
    

