#-*- coding:UTF-8 -*-
import h5py
import numpy as np
import datetime
import json
import numba

with open('stat_512_hash_mean.txt', 'r') as v:
    stat = json.load(v)
    list_video = stat[0]
    list_attack = stat[1]
    num_video = len(list_video)
    num_attack = len(list_attack)
    print "读取stat_num_euclidean.txt文件成功"

f = h5py.File('dataset_512_hash.hdf5', 'r')   #测试说明3.0里用的是中值hash
feature = f['dataset'][:]
name = f['name'][:]
print "读取h5文件成功"
f.close()

for i in range(0,100):
    print name[i]

#查询视频
list_query_video = range(0, num_video)

#nor阈值
thre_hamming = 200

thre2 = 195

l_total_num = 0
l_total_recall = 0
l_total_pre = 0


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


if __name__ == '__main__':
    for query_video in list_query_video:
        # 该video抽出的视频帧数
        num_video_frame = stat[query_video + 2][0]

        if num_video_frame > 9:
            l_total_num += 1
            print "正在查询第", l_total_num, "个长video", "帧数为", num_video_frame

            #准确查询出的攻击视频数目
            num_right = 0
            dict_stat_num = {}

            query_feature = feature[stat[query_video + 2][34]]
            result_nor = np_query_nor_lib(query_feature, feature, thre_hamming)
            result_nor_index = result_nor[0]
            result_nor_one = result_nor[1]

            if num_video_frame > 1:
                for i in range(1, num_video_frame):
                    query_feature = feature[stat[query_video + 2][34] + i]
                    _result_nor = np_query_nor_lib(query_feature, feature,
                                                   thre_hamming)
                    _index_result_nor = _result_nor[0]
                    _one_result_nor = _result_nor[1]

                    result_nor_index = np.append(
                        result_nor_index, _index_result_nor)  #两个一维数组直接串联
                    result_nor_one = np.append(result_nor_one,
                                               _one_result_nor)  #两个一维数组直接串联

            result = search_sort(result_nor_index, result_nor_one, thre2)

            num_right = result[0]
            num_len = result[1]

            recall = num_right / 33.0
            pre = num_right / float(num_len)
            l_total_recall += recall
            l_total_pre += pre
            l_avg_recall = l_total_recall / l_total_num
            l_avg_pre = l_total_pre / l_total_num

            print "recall", recall
            print "pre", pre
            print "l-avgrecall ", l_avg_recall
            print "l-avgpre ", l_avg_pre
