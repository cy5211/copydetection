#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import datetime
import json
import numba
from matplotlib import pyplot as plt 
plt.switch_backend('agg')

'''
with open('stat_512_hash_mean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功"

f = h5py.File('rerank2_dataset_512_hash.hdf5', 'r')   #测试说明3.0里用的是中值hash
feature = f['dataset'][:]
name = f['name'][:]
print "读取h5文件成功"
f.close()



#查询视频
list_query_video = range(0, num_video)

#nor阈值
thre_hamming = range(180,255,5)

thre2 = 195
total_raw_frame=1700

l_total_num = 0
l_total_recall = 0
l_total_pre = 0
num_per_class=34

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


def np_query_nor_lib(_query_feature, _lib_feature, _thre):
    lib_frame_index = 0
    one_num = []

    nor = np.bitwise_xor(_query_feature, _lib_feature)

    _sum = numba_sum(nor)

    _index = np.where(_sum < _thre)

    one_num = _sum[_index]
    lib_frame_index = _index[0]
    lib_index_one = np.vstack((lib_frame_index, one_num))

    return lib_index_one

def pic_np_query_nor_lib(_query_feature, _lib_feature, _thre):
    lib_frame_index = 0
    one_num = []

    nor = np.bitwise_xor(_query_feature, _lib_feature)

    _sum = numba_sum(nor)

    _index = np.where(_sum < _thre)

    one_num = _sum[_index]
    lib_frame_index = _index[0]

    return lib_frame_index


def search_sort(_result_nor_index, _result_nor_one, _thre2):

    rank_dists = []
    _result = []
    num_right = 0
    _num = len(_result_nor_index)
    #将帧索引排序
    _temp_index = np.argsort(_result_nor_index)
    _result_nor_index = _result_nor_index[_temp_index]
    _result_nor_one = _result_nor_one[_temp_index]

    for i in range(0, num_video):
        if _num < 1:
            break
        stat_num_video_frame = stat[i + 2][0]
        #索引中包含原图索引
        for k in range(0, num_attack + 1):
            if _num < 1:
                break

            start_index = stat[i + 2][k + 1]
            end_index = start_index + stat_num_video_frame

            if end_index > _result_nor_index[0]:

                temp1 = np.where(_result_nor_index > (start_index - 1))
                temp2 = np.where(_result_nor_index < end_index)
                frame_index = np.intersect1d(temp1, temp2)

                _num -= len(frame_index)

                if len(frame_index) > 3:
                    #排除和原图做匹配
                    if k != num_attack:
                        wei_sum = np.sum(_result_nor_one[frame_index])
                        wei_avg = float(wei_sum) / len(frame_index)
                        name = str(i) + "_" + str(k)
                        dict_stat_num[name] = wei_avg

                _result_nor_index = _result_nor_index[len(frame_index):]
                _result_nor_one = _result_nor_one[len(frame_index):]

    rank_dists = [k for k, v in dict_stat_num.iteritems() if v < _thre2]

    for i in range(0, len(rank_dists)):
        if rank_dists[i].split('_')[0] == str(query_video):
            num_right += 1

    _result.append(num_right)
    _result.append(len(rank_dists))
    return _result


def pic_search_sort(_result_nor_index,_index_raw_frame):
    #将帧索引排序
    global stat_result
    for i in range(0,num_attack):
        if _index_raw_frame+i+1 in _result_nor_index:
            stat_result[j][i]+=1

if __name__ == '__main__':
    num=0
    startime= datetime.datetime.now()
    stat=[]
    stat_result=np.zeros([len(thre_hamming),num_attack],dtype=np.float64)
    total_num=np.zeros([len(thre_hamming),])
    for j in range(0,len(thre_hamming)):
        _thre_hamming=thre_hamming[j]
        for i in range(0,total_raw_frame):
            print num
            num+=1
            raw_frame=feature[i*num_per_class]
            result_index=pic_np_query_nor_lib(raw_frame,feature,_thre_hamming)
            #print len(result_index)
            total_num[j]+=len(result_index)
            pic_search_sort(result_index,i*num_per_class)

    stat_result=stat_result.tolist()
    total_num=total_num.tolist()
    stat.append(stat_result)
    stat.append(total_num)
    f=open('stat_result.txt','w')
    f.write(json.dumps(stat))    #只能导入list
    f.close()
    print "已生成stat_num.txt"
    print "用时",(datetime.datetime.now() - startime).total_seconds()'''


'''
num_attack=33
total_raw_frame=1700
thre_hamming = range(180,255,5)
stat_hard_attack=np.zeros([num_attack,])
precision_total_frame=np.zeros([len(thre_hamming),])
recall_total_frame=np.zeros([len(thre_hamming),])
with open('stat_result.txt', 'r') as v:
    stat = json.load(v)
    result_attack_num=stat[0]
    total_rusult=stat[1]
    print "txt文件成功"

result_attack_num=np.array(result_attack_num)
recall_attack=result_attack_num/total_raw_frame


recall_attack_mean=np.mean(recall_attack,axis=1)
for i in range(0,len(thre_hamming)):
    print "第",i,"个阈值"
    for j in range(0,num_attack):
        if recall_attack[i][j]<recall_attack_mean[i]:
            stat_hard_attack[j]+=1

temp=np.sum(result_attack_num,axis=1)
for i in range(0,len(thre_hamming)):
    precision_total_frame[i]=temp[i]/total_rusult[i]
    print "precision",precision_total_frame[i]
    recall_total_frame[i]=temp[i]/(total_raw_frame*33)
    print "recall",recall_total_frame[i]


plt.figure() 
plt.plot(precision_total_frame,recall_total_frame)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR')
plt.show()

recall_attack=recall_attack.tolist()
stat.append(recall_attack)
precision_total_frame=precision_total_frame.tolist()
recall_total_frame=recall_total_frame.tolist()
stat.append(precision_total_frame)
stat.append(recall_total_frame)
stat_hard_attack=stat_hard_attack.tolist()
stat.append(stat_hard_attack)
f=open('stat_result.txt','w')
f.write(json.dumps(stat))    #只能导入list
f.close()
'''

#************测试float型特征的PR值,用cosine***********
#f = h5py.File('dataset_30_class_ae.hdf5', 'r')
f = h5py.File('dataset_2048_label.hdf5', 'r') 
f_feature = f['dataset'][:]
print "读取h5文件成功"
f.close()

#归一化
for i in range(0,f_feature.shape[0]):
    f_feature[i]=f_feature[i]/np.linalg.norm(f_feature[i])

print "特征的shape",f_feature.shape
test_class=30
num_per_class=34
#thre_cos=np.array([0.7,0.75,0.8,0.85,0.9,0.95])
thre_cos=np.arange(0,1,0.05)
#构建查询数组
query_feature = f_feature[0]
for i in range(1, test_class):
    temp_feature = f_feature[i*num_per_class]
    query_feature = np.vstack((query_feature, temp_feature))

#计算余弦值
array_cos = np.dot(query_feature, f_feature.T)
print "余弦值计算完成"

#只有一张原图时,是array(340,)
if test_class==1:
    array_cos = array_cos.reshape(1, -1)


def stat_result(_array_cos,_thre,_test_class,_num_per_class):
    thre_cos_index = np.where(_array_cos > _thre)
    thre_row_index = thre_cos_index[0]
    thre_col_index = thre_cos_index[1]
    total_num=len(thre_row_index)
    right_num=0
    _temp1=0
    _result=[]
    _result.append(total_num)

    _temp=np.argsort(thre_row_index)
    thre_col_index=thre_col_index[_temp]
    thre_row_index.sort()
     
    for i in range(0,_test_class):
        #print "正在统计第",i,"类"
        thre_col_index=thre_col_index[_temp1:]
        thre_row_index=thre_row_index[_temp1:]
        _temp=np.where(thre_row_index==i)
        _temp1=len(_temp[0])
       
        _temp_class=thre_col_index[_temp]
        _temp_class_1=np.where(_temp_class>i*_num_per_class)
        _temp_class_2=np.where(_temp_class<(i+1)*_num_per_class)
        right_index = np.intersect1d(_temp_class_1, _temp_class_2)
        right_num+=len(right_index)

    _result.append(right_num)
    return _result

recall=np.zeros([len(thre_cos),])
precision=np.zeros([len(thre_cos),])
total_F1_score=0
for i in range(0,len(thre_cos)):  
    print "正在计算阈值",thre_cos[i]
    result=stat_result(array_cos,thre_cos[i],test_class,num_per_class)
    total_pre_num=result[0]    
    right_num=result[1]
    recall[i]=float(right_num)/(test_class*(num_per_class-1))
    precision[i]=float(right_num)/total_pre_num
    temp_F1_score=2*precision[i]*recall[i]/(recall[i]+precision[i])
    total_F1_score+=temp_F1_score
    print"total_num ",total_pre_num,"right_num " ,right_num,"precision ",precision[i],"recall ",recall[i]

print "avg_F1score",total_F1_score/len(thre_cos)

plt.figure() 
plt.plot(precision,recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR')
plt.show()
plt.savefig("1024-PR.jpg")


