#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import sys
import datetime
import json
from numpy import linalg as LA
from sklearn.decomposition import PCA
from ctypes import *

f = h5py.File('dataset_2048_label.hdf5','r')
feature=f['dataset'][:]
print "读取原特征文件成功"
f.close()

print type(feature[0][0])

feature_num=feature.shape[0]
dim=feature.shape[1]

#************PCA****************
pca=PCA(n_components=512)
feature=pca.fit_transform(feature)
print "PCA降维后",feature.shape
f = h5py.File('dataset_512_PCA.hdf5','w')
f['dataset']=feature

print "完成PCA降维"

'''
#************均值hash****************
total_frame = feature.shape[0]
dim_feature = feature.shape[1]

print "feature shape",feature.shape

mean_feature = np.mean(feature, axis=0)

for i in range(0,dim_feature):
    temp_feature=feature[:,i]
    hash_col=temp_feature>mean_feature[i]    
    feature[:,i]=hash_col
    print "正在处理第"+str(i)+"列"

feature=feature.astype(np.int8)

print "hash成功"

rr = h5py.File('dataset_32_hash_rank2.hdf5','w')
rr['dataset']=feature
rr['name']=name
rr.close()'''

#************中值hash****************
'''
f = h5py.File('rerank_512_float.hdf5','r')
_feature1 = f['dataset'][:]
_feature2 = f['dataset'][:]
name=f['name'][:]
print "读取h5文件成功"
f.close()


row_feature = _feature1.shape[0]
col_feature = _feature1.shape[1]

middle=row_feature/2
middle_feature=np.zeros([col_feature,])

for i in range(0,col_feature):
    print i
    _feature1[:,i].sort()
    _temp1=_feature1[:,i]
    _feature2[:,i]=(_feature2[:,i]>_temp1[middle])

_feature2=_feature2.astype(np.int8)


rr = h5py.File('dataset_512_hash_mid.hdf5','w')
rr['dataset']=_feature2
rr['name']=name
rr.close()'''


#************归一化****************
'''total_frame = feature.shape[0]
dim_feature = feature.shape[1]

norm_Feature=np.zeros([total_frame,dim_feature])

for i in range(0,total_frame):
    print i
    norm_Feature[i]=feature[i]/LA.norm(feature[i])

print norm_Feature.dtype'''

'''
norm=h5py.File('dataset_512_FLANN_2.hdf5','w')
norm['dataset']=feature
norm['name']=name
norm.close()

print "写入",feature.dtype,name.dtype

print "成功写入文件"'''

#**********分成8段十进制整型,每段64位***********
'''
seg=8
bit_seg=dim/seg
feature_int=np.zeros([feature_num,seg],dtype=c_ulong) #uint64,需要调用c,因为python里最大可表示整数是signed int64

for n in range(0,feature_num):
    print n
    for i in range(0,seg): 
        s=''.join(str(k) for k in feature[n][bit_seg*i:bit_seg*i+bit_seg])  
        temp=int(s,2)
        feature_int[n][i]=temp
        if temp!=feature_int[n][i]:
            print "false"

norm=h5py.File('dataset_512_seg8.hdf5','w')
norm['dataset']=feature_int
norm['name']=name
norm.close()

print "写入",feature_int.dtype,name.dtype'''

