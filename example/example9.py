#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# using BertClient in sync way

import sys
import time

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from bert_serving.client import BertClient

labels = ['耳聋', '耳聋遗传疾病', '耳聋遗传', '耳聋怎么办', '今天星期五', '谢谢', '明天星期几', '你好', '早上好']

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.seismic):
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    bc = BertClient(ip='172.16.10.46',port=int(5555), port_out=int(5556))
    # encode a list of strings
    
    vec = bc.encode(labels)
    d = cosine_similarity(vec)
    plot_confusion_matrix(d)
    plt.show()

