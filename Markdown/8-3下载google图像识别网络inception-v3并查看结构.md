

```python
import tensorflow as tf
import os
import tarfile
import requests
```


```python
# inception-v3 是googlenet的第三个版本
#inception模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#这里采用手动下载后直接放入下述模型存放地址中

#模型存放地址,
inception_pretrain_model_dir = "inception_model"   #此文件夹如果不存在会自动创建
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)
    
#获取文件名，以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

#下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)
#解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)
 
#模型结构存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
#'classify_image_graph_def.pb'为inception-v3中训练好的一个模型
with tf.Session() as sess:
    #创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    #保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()

```

    finish:  inception-2015-12-05.tgz
    WARNING:tensorflow:From <ipython-input-5-b2c8da65579d>:37: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.gfile.GFile.
    


```python

```
