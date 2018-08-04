#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import datetime
import json

with open('stat_num_1w_euclidean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功"

f = h5py.File('dataset_512_FLANN_norm.hdf5', 'r')

f_feature = f['feature'][:]
f_name = f['name'][:]
print "读取h5文件成功"
f.close()


#print "feature",f_feature.dtype,f_feature.shape,type(f_feature)
#print "测试用 ",sum(f_feature[0]==1)
#查询视频
list_query_video = range(1,2)

#测试阈值
list_thre = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

#cosine阈值
thre_cos = 0.65  

total_precision = np.zeros(len(list_thre))
total_recall = np.zeros(len(list_thre))
total_num = 0

for query_video in list_query_video:
    starttime = datetime.datetime.now()
    # 该video抽出的视频帧数
    num_video_frame = stat[query_video+2]
    total_num += 1
    print "正在查询第"+str(total_num)+"个video"
    #准确查询出的攻击视频数目
    num_right = 0
    dict_stat_num = {}

    #计算该查询视频在h5文件中的起始索引
    num_temp1 = 0
    if query_video == 0:
        num_temp1 = 0
    else:
        for i in range(0, query_video):
            num_temp1 += stat[i+2]*34  

    starttime1 = datetime.datetime.now()
    #统计一个video的所有原图
    query_feature = f_feature[num_temp1+num_video_frame*33]
    for i in range(1, num_video_frame):
        temp_feature = f_feature[num_temp1+num_video_frame*33+i]
        query_feature = np.vstack((query_feature, temp_feature))

    endtime1 = datetime.datetime.now()
    print "构建查询数组用时"+str((endtime1 - starttime1).seconds)

    starttime2 = datetime.datetime.now()
    #余弦值计算
    array_cos = np.dot(query_feature, f_feature.T)
    #print (type(array_cos),array_cos.shape,array_cos.dtype)
    endtime2 = datetime.datetime.now()
    print "余弦计算用时",(endtime2 - starttime2).total_seconds()
    print (type(array_cos),array_cos.shape,array_cos.dtype)
    #num_video_frame为1时是array(1500000,)
    if num_video_frame == 1:
        array_cos = array_cos.reshape(1, -1)

    thre_cos_index = np.where(array_cos > thre_cos)
    thre_row_index = thre_cos_index[0]
    thre_col_index = thre_cos_index[1]
    print "query ",thre_col_index.shape

    for i in range(0, num_video):      
        stat_num_video_frame = stat[i+2]
        print "正在统计第"+str(i)+"个video"
        #计算该视频在h5文件中的起始索引
        num_temp2 = 0
        if i == 0:
            num_temp2 = 0
        else:
            for j in range(0, i):
                num_temp2 += stat[j+2]*34

        for k in range(0, num_attack):
            wei_sum = 0
            start_index = num_temp2+k*stat_num_video_frame
            end_index = start_index+stat_num_video_frame

            temp1 = np.where(thre_col_index > (start_index-1))
            temp2 = np.where(thre_col_index < end_index)
            frame_index = np.intersect1d(temp1, temp2)

            if len(frame_index) == 0:
                wei_sum = 0
                wei_avg = 0
            else:
                for n in frame_index:
                    wei_sum += array_cos[thre_row_index[n]][thre_col_index[n]]
                wei_avg = wei_sum/num_video_frame

            name = str(i)+"_"+str(k)
            if wei_sum > 0:
                dict_stat_num[name] = wei_avg

    for t in range(0, len(list_thre)):
        rank_result = [k for k, v in dict_stat_num.iteritems() if v > list_thre[t]]
        num_right = 0

        for i in range(0, len(rank_result)):
            if rank_result[i].split('_')[0] == str(query_video):
                num_right += 1

        print "以下为"+str(list_thre[t])+"测试结果"
        print "检索到的record数目为 "+str(len(rank_result))
        
        recall = num_right/33.0

        if len(rank_result) == 0:
            precision = 0
        else:    
            precision = float(num_right)/len(rank_result)
       
        print "recall "+str(recall)
        print "precision "+str(precision)

        total_recall[t] += recall
        total_precision[t] += precision
        avg_recall = total_recall[t]/total_num
        avg_precision = total_precision[t]/total_num

        print "avg_recall "+str(avg_recall)
        print "avg_precision "+str(avg_precision)

    endtime = datetime.datetime.now()
    print "测试该视频用时"+str((endtime - starttime).seconds)
