

```python
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
```


```python
class NodeLookup(object):
    def __init__(self):
        label_lookup_path='inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path='inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup=self.load(label_lookup_path, uid_lookup_path)
        
    def load(self, label_lookup_path, uid_lookup_path):
        #加载分类字符串n************对应分类名称的文件
        proto_as_ascii_lines=tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human={}
        #一行一行读取数据
        for line in proto_as_ascii_lines:
            #去掉换行符
            line=line.strip('\n')
            #按照‘\t’分割
            parsed_items=line.split('\t')
            #获取分类编号
            uid=parsed_items[0]
            #获取分类名称
            human_string=parsed_items[1]
            #保存编号字符串n********与分类名称的映射关系
            uid_to_human[uid]=human_string
        #加载分类字符串n**********对应分类编号1-1000的文件
        proto_as_ascii=tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid={}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class=int(line.split(':')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n*********
                target_class_string=line.split(':')[1]
                #保存分类编号1-1000与编号字符串n*******映射关系
                node_id_to_uid[target_class]=target_class_string[2:-2]
        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name={}
        for key,val in node_id_to_uid.items():
            #获取分类名称
            name=uid_to_human[val]
            node_id_to_name[key]=name
        return node_id_to_name

    #传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

#创建一个图来放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor=sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data=tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions=sess.run(softmax_tensor, {'DecodeJpeg/contents:0' : image_data}) #图片格式是jpg格式
            predictions=np.squeeze(predictions) #把结果转换为1维数据

            #打印图片路径及名称
            print()
            image_path=os.path.join(root,file)
            print(image_path)
            #显示图片
            '''
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            '''

            #排序
            top_k=predictions.argsort()[-5:][::-1]
            node_lookup=NodeLookup()
            for node_id in top_k:
                #获取分类名称
                human_string=node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score=predictions[node_id]
                print('%s (score=%.5f)' % (human_string, score))
```

    WARNING:tensorflow:From <ipython-input-2-a8777ddaca0b>:50: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.gfile.GFile.
    
    images/5-1.png
    paper towel (score=0.20438)
    safety pin (score=0.11638)
    hook, claw (score=0.03058)
    nematode, nematode worm, roundworm (score=0.01845)
    strainer (score=0.01654)
    
    images/5-2-1.png
    rule, ruler (score=0.26179)
    rubber eraser, rubber, pencil eraser (score=0.12402)
    jigsaw puzzle (score=0.08158)
    pencil sharpener (score=0.03118)
    paper towel (score=0.02716)
    
    images/5-2-2.jpg
    menu (score=0.41445)
    web site, website, internet site, site (score=0.31569)
    crossword puzzle, crossword (score=0.01099)
    envelope (score=0.01059)
    power drill (score=0.00265)
    
    images/5-2-3.jpg
    menu (score=0.33410)
    envelope (score=0.17916)
    web site, website, internet site, site (score=0.13433)
    screw (score=0.01664)
    carton (score=0.01045)
    
    images/5-2-4.jpg
    menu (score=0.56554)
    web site, website, internet site, site (score=0.17386)
    crossword puzzle, crossword (score=0.06795)
    slide rule, slipstick (score=0.01034)
    hard disc, hard disk, fixed disk (score=0.00467)
    
    images/5-2-5.jpg
    menu (score=0.53163)
    crossword puzzle, crossword (score=0.20259)
    web site, website, internet site, site (score=0.06094)
    hard disc, hard disk, fixed disk (score=0.00684)
    envelope (score=0.00445)
    
    images/5-2-6.jpg
    menu (score=0.52707)
    slide rule, slipstick (score=0.06766)
    crossword puzzle, crossword (score=0.05489)
    envelope (score=0.02662)
    scale, weighing machine (score=0.01740)
    
    images/5-2-7.jpg
    menu (score=0.40245)
    web site, website, internet site, site (score=0.19769)
    crossword puzzle, crossword (score=0.05024)
    envelope (score=0.04827)
    slide rule, slipstick (score=0.00737)
    


```python

```
