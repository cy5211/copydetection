#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import datetime
import json
import numba
import matplotlib.pyplot as plt

'''
f = h5py.File('rerank2_dataset_512_hash.hdf5', 'r')
feature = f['dataset'][:]
name = f['name'][:]
print "读取h5文件成功"
f.close()

item_num=feature.shape[0]
num_per_class=34
num_attack=33
#total_raw_frame=item_num/num_per_class
total_raw_frame=1000
dim=feature.shape[1]
num_group=32
dim_per_group=dim/num_group
stat=[]

@numba.jit
def numba_sum(a):
    _sum = 0
    _result = np.zeros([
        feature.shape[0],
    ])
    for i in range(0, a.shape[0]):
        _sum = 0
        for j in range(0, a.shape[1]):
            _sum += a[i][j]
        _result[i] = _sum
    return _result


def np_query_nor_lib(_query_feature, _lib_feature):
    lib_frame_index = 0
    one_num = []

    nor = np.bitwise_xor(_query_feature, _lib_feature)

    _sum = numba_sum(nor)

    return _sum

if __name__ == '__main__':
    #stat_num_copy=np.zeros([num_group,],dtype=np.float32)
    #stat_num_not_copy=np.zeros([num_group,],dtype=np.float32)
    #stat_percent_copy=np.zeros([num_group,],dtype=np.float32)
    #stat_percent_not_copy=np.zeros([num_group,],dtype=np.float32)
    #stat_attack=np.zeros([num_attack,],dtype=np.float32)
    #_temp=range(dim_per_group,dim+dim_per_group,dim_per_group)
    for i in range(0,total_raw_frame):
        print i
        raw_frame=feature[i*num_per_class]
        sum=np_query_nor_lib(raw_frame,feature)
        if i==0:
            stat_copy=sum[i*num_per_class+1:(i+1)*num_per_class].copy()
            stat_not_copy=sum[(i+1)*num_per_class:(i+1)*num_per_class+40].copy()
        else:
            stat_copy=np.append(stat_copy,sum[i*num_per_class+1:(i+1)*num_per_class])
            stat_not_copy=np.append(stat_not_copy,sum[(i+1)*num_per_class:(i+1)*num_per_class+40])  #这里的非拷贝没有包括前面的
        
        #mean = np.mean(sum[i*num_per_class+1:(i+1)*num_per_class])
        #stat_attack+=sum[i*num_per_class+1:(i+1)*num_per_class]
        for j in range(0,len(_temp)):
            if mean<_temp[j]:
                stat_num_copy[j]+=1
                break
        mean=np.mean(sum[(i+1)*num_per_class:])
        for k in range(0,len(_temp)):
            if mean<_temp[k]:
                stat_num_not_copy[k]+=1
                break

    for i in range(0,num_group):
        print i
        stat_percent_copy[i]=stat_num_copy[i]/total_raw_frame
        stat_percent_not_copy[i]=stat_num_not_copy[i]/total_raw_frame
    
    
    #stat_attack=stat_attack/total_raw_frame
    stat_copy=stat_copy.tolist()
    stat_not_copy=stat_not_copy.tolist()
    #stat_attack=stat_attack.tolist()
    print "len(copy)",len(stat_copy)
    print "len(not copy)",len(stat_not_copy)
    stat.append(stat_copy)
    stat.append(stat_not_copy)
    #stat.append(stat_attack)
    
    f=open('pro_hamming_distance.txt','w')
    f.write(json.dumps(stat))    #只能导入list
    f.close()
    print "已生成stat_num.txt"'''



with open('pro_hamming_distance.txt', 'r') as v:
    stat = json.load(v)
    stat_copy = stat[0]
    stat_not_copy = stat[1]

'''
stat_percent_copy = stat_percent_copy*total_raw_frame
stat_percent_not_copy = stat_percent_copy
num_group=32
_stat_percent_copy=np.zeros([num_group,],dtype=np.float32)
_stat_percent_not_copy=np.zeros([num_group,],dtype=np.float32)    
dim=512    
_temp=range(16,dim+16,16)

_stat_percent_copy[0]=stat_percent_copy[0]
_stat_percent_not_copy[0]=stat_percent_not_copy[0]
for i in range(1,num_group):
    print i
    for j in range(0,i+1):
        _stat_percent_copy[i]+=stat_percent_copy[j]
        _stat_percent_not_copy[i]+=stat_percent_not_copy[j]'''

copy_mean=np.array(stat_copy).mean()
print copy_mean

not_copy_mean=np.array(stat_not_copy).mean()
print not_copy_mean

copy_std=np.array(stat_copy).std()
#print copy_std
not_copy_std=np.array(stat_not_copy).std()


def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

temp=np.arange(0, 500,1)
copy_pdf=normfun(temp,copy_mean,copy_std)
not_copy_pdf=normfun(temp,not_copy_mean,not_copy_std)

plt.figure()  
plt.plot(temp,copy_pdf)  
plt.plot(temp,not_copy_pdf)  
plt.xlabel('Hamming Distance')
plt.ylabel('Probability')
plt.show() 


    
    










