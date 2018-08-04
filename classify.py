#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import json
import random

#*********1 生成简约版result.txt文件********** 生成{video:{attack:name},video:[],attack:[]}
'''
print "开始生成temp_stat.txt"
f = h5py.File('new_dataset_512_hash.hdf5','r')
Name=f['name'][:]
feature=f['feature'][:]
item_num = Name.shape[0]
print feature.shape[0],feature.dtype

dict_video={}
list_attack=[]
list_video=[]
for i in range(0,item_num):
    #print i
    a=len(Name[i].split('_'))
    video=Name[i].split('_')[0]
    #print a
    if a==2:    
        if video not in dict_video:
            list_video.append(video)
            dict_video_temp={video:{'0':[Name[i]]}} #无攻击
            dict_video=dict(dict_video,**dict_video_temp)
        elif '0' not in dict_video[video]:
            dict_video_temp={video:{'0':[Name[i]]}} #无攻击
            dict_video=dict(dict_video,**dict_video_temp)
        else:
            dict_video[video]['0'].append(Name[i])               
    else:
        attack=Name[i].split('_',2)[2].split('.')[0]
        if video not in dict_video:
            list_video.append(video)
            dict_video_temp={video:{attack:[Name[i]]}} #无攻击
            dict_video=dict(dict_video,**dict_video_temp)
        
        elif attack not in dict_video[video]:
            if attack not in list_attack:
                list_attack.append(attack)
            dict_attack_temp={attack:[Name[i]]}
            dict_video[video]=dict(dict_video[video],**dict_attack_temp)
        else:
            dict_video[video][attack].append(Name[i])

dict_temp={'video':list_video,'attack':list_attack}
dict_video=dict(dict_video,**dict_temp)

#写入txt文件
f=open('temp_stat_512hash.txt','w')
f.write(json.dumps(dict_video))
f.close()

print "已生成temp.txt"

#*******2 生成stat_num.txt文件,记录各个攻击数目********
print "开始生成stat_num.txt"
f=open('temp_stat_512hash.txt','r')
dict_video = json.load(f)
list_attack=dict_video['attack']
list_video=dict_video['video']
num_video=len(list_video)
num_attack=len(list_attack)
print "视频数目为",num_video
print "攻击数目为",num_attack

stat=[]  #前两位为list_video和list_attack,后面为从video抽取的帧数
stat.append([])
stat.append([])
#list前两位为attack和video
stat[0]=list_video 
stat[1]=list_attack

for i in range(0,num_video):  #某视频
    stat.append([])
    stat[i+2]=len(dict_video[list_video[i]]['0']) #记录原图数

f=open('stat_512hash.txt','w')
f.write(json.dumps(stat))    #只能导入list
f.close()
print "已生成stat_num.txt" '''

'''
#**********3 生成rerank.h5文件********* 同一类攻击的在一起
f = h5py.File('new_dataset_512_hash.hdf5','r')
f_name=f['name'][:]
feature=f['feature'][:]
print "读取h5文件成功"
f.close()


total_frame=feature.shape[0]


rr_Feature=np.zeros([total_frame,512],dtype=np.int8)
rr_Name=[]   #rr_Name=f_name 是传址操作

v=open('stat_512hash.txt','r')
stat = json.load(v)
list_video=stat[0]
list_attack=stat[1]
v.close()
print "读取stat_num.txt文件成功"


#统计已记录的每个视频中的每种攻击中的frame数
#最后一类为原图数目
num_record_frame=np.zeros([len(list_video),34],dtype='int32')

for i in range(0,total_frame):
    print i
    _video=f_name[i].split('_')[0]
    a=len(f_name[i].split('_'))
    index_video=list_video.index(_video)
    num_video_frame=stat[index_video+2]
    
    num_temp1=0
    for j in range(0,index_video):
        if index_video==0:
            num_temp1=0
            break
        else:
            num_temp1+=stat[j+2]*34          
    
    if a==2:
        #原图记录在所在视频的最后一条
        rr_Feature[num_temp1+num_video_frame*33+num_record_frame[index_video,33]]=feature[i]
        rr_Name.insert(num_temp1+num_video_frame*33+num_record_frame[index_video,33],f_name[i])
        num_record_frame[index_video,33]+=1
    
    else:
        _attack=f_name[i].split('_',2)[2].split('.')[0]
        #print _name,_video,_attack
        index_attack=list_attack.index(_attack)
        #计算该video中这个attack之前的record数目
        num_temp2=num_video_frame*index_attack               
        rr_Feature[num_temp1+num_temp2+num_record_frame[index_video,index_attack]]=feature[i]
        rr_Name.insert(num_temp1+num_temp2+num_record_frame[index_video,index_attack],f_name[i])
        num_record_frame[index_video,index_attack]+=1
        #print rr_Name[0:67]
    

print "重排序完成"

rr = h5py.File('_dataset_512_hash.hdf5','w')
rr['dataset']=rr_Feature
rr['name']=rr_Name
print "rank feature",rr_Feature.dtype,rr_Feature.shape
rr.close()'''


'''
#**********4 test rerank.h5文件*********
f = h5py.File('new_dataset_512_hash.hdf5','r')
new_name=f['name'][:]
new_feature=f['dataset'][:]
print "读取h5文件成功"
f.close()

f = h5py.File('feature_hash512.h5','r')
old_name=f['name'][:]
old_feature=f['feature'][:]
print "读取h5文件成功"
f.close()

with open('stat_512hash.txt', 'r') as v:  #new_dataset对应的new stat
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat文件成功"


for i in range(0,5):
    video=random.randint(0,num_video)
    attack=random.randint(0,num_attack)
    num_video_frame = stat[video+2]

    #计算该查询视频在h5文件中的起始索引
    num_temp1 = 0
    if video == 0:
        num_temp1 = 0
    else:
        for i in range(0, video):
            num_temp1 += stat[i+2]*34
    

    start_index = num_temp1+attack*num_video_frame
    end_index = start_index+num_video_frame

    for j in range(start_index,end_index):
        _new_name=new_name[j]
        _new_feature=new_feature[j]
        _old_index=np.where(old_name==_new_name)
        _old_index=_old_index[0][0]
        if _new_feature.all()!=old_feature[_old_index].all():
            print "ERROR!"        
'''
'''
#**********5 记录各攻击视频的起始索引********* 
#stat=[[list_video],[list_attack],[num_video_frame,index_attack1,index_attack2,,原图],,,,]
with open('stat_num_1w_euclidean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功"

_stat=[]  #前两位为list_video和list_attack
_stat.append([])
_stat.append([])
#list前两位为attack和video
_stat[0]=list_video 
_stat[1]=list_attack

for i in range(0, num_video):  
    num_video_frame = stat[i+2]
    _stat.append([])
    _stat[i+2].append(num_video_frame)

    #统计视频起始索引
    video_index=0
    if i == 0:
        video_index = 0
    else:
        for j in range(0, i):
            video_index = video_index+stat[j+2]*34
    
    if i==3012:
        print "video_index",video_index
    _stat[i+2].append(video_index)
    attack_index=0
    for k in range(1, num_attack+1):
        attack_index = video_index+k*num_video_frame
        
        _stat[i+2].append(attack_index)

    #print len(_stat[i+2])
 '''

'''
#**********6 生成原图和拷贝图片在一起的rerank********* [原图,原图_攻击1,原图_攻击2,原图_攻击3...]
with open('stat_512_hash_mean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功"

f = h5py.File('dataset_512_hash.hdf5', 'r')
name_512 = f['name'][:]
print "读取h5文件成功"
f.close()

f = h5py.File('feature_1w_euclidean.h5', 'r')
feature = f['feature'][:]
name = f['name'][:]
print "读取h5文件成功"
f.close()


#item_num = name.shape[0]
dim=feature.shape[1]
raw_frame=[]
num_raw_frame=0
num_not_raw_frame=0
num_per_class=34
total_raw_frame=item_num/num_per_class

rr_Feature=np.zeros([item_num,dim],dtype=np.int8)
rr_Name=name_512

for i in range(0,item_num):
    _name=name[i]
    a=len(_name.split('_'))
    if a==2:
        raw_frame.append(_name)
        rr_Feature[num_raw_frame*num_per_class]=feature[i]
        rr_Name[num_raw_frame*num_per_class]=name[i]
        num_raw_frame+=1

print num_raw_frame

for i in range(0,item_num):
    print i
    _name=name[i]
    a=len(_name.split('_'))
    if a!=2:
        _attack=_name.split('_',2)[2].split('.')[0]
        _index_attack=list_attack.index(_attack)
        num_not_raw_frame+=1
        _raw_frame=_name.split('_')[0]+str('_')+_name.split('_')[1]+str('.jpg')
        _index=raw_frame.index(_raw_frame)
        rr_Feature[_index*num_per_class+_index_attack+1]=feature[i]
        rr_Name[_index*num_per_class+_index_attack+1]=name[i]
        
print num_not_raw_frame



rr = h5py.File('rerank2_feature_2048.hdf5','w')
rr['dataset']=rr_Feature
rr['name']=rr_Name
print "rank feature",rr_Feature.dtype,rr_Feature.shape
rr.close()'''


#**********7 一个视频只选一帧的rerank,同时进行标记,2048维,用于t-SNE测试********* [原图,原图_攻击1,原图_攻击2,原图_攻击3...]
with open('stat_512_hash_mean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功","num_video ",num_video

f = h5py.File('feature_1w_euclidean.h5', 'r')
feature_2048 = f['feature'][:]
name_2048 = f['name'][:]
print "读取h5文件成功",name_2048.dtype
f.close()

'''
#print feature.dtype
dim=feature_2048.shape[1]
num_per_class=34
rr_Feature=np.zeros([num_video*num_per_class,dim],dtype=np.float64)
rr_Name=[]
rr_labels=np.zeros([num_video*num_per_class,],dtype=np.float64)
record_video=[]
num_frame=0
test_num_video=2

#帧数
for i in range(0,44359):
    _name=name_2048[i*num_per_class]
    #print "正在查询帧",_name
    _video=_name.split('_')[0]
    if (_video not in record_video):
        print "正在查询第",num_frame,"个video",_video
        record_video.append(_video)
        for j in range(0,num_per_class):
            _name=name_2048[i*num_per_class+j]
            #print "_name",_name
            rr_Name.append(_name)
            rr_Feature[num_frame*num_per_class+j]=feature_2048[i*num_per_class+j]
            rr_labels[num_frame*num_per_class+j]=num_frame
        num_frame+=1

rr = h5py.File('dataset_2048_label.hdf5','w')
rr['dataset']=rr_Feature
rr['name']=rr_Name
rr['label']=rr_labels
rr.close()'''





