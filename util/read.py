# -*- coding: UTF-8 -*-

import os
import csv


def get_item_info(input_file):
    """获取每个电影的标题以及分类"""
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    with open(input_file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for item in data:
            if len(item) < 3:
                continue
            elif len(item) == 3:
                movieId, title, genres = item[0], item[1], item[2]
            elif len(item) > 3:
                movieId = item[0]
                genres = item[-1]
                # 有些电影名称中可能会出现逗号，
                title = ",".join(item[1:-1])
            item_info[movieId] = [title, genres]
    return item_info


def get_ave_score(input_file):
    """获取每部电影的平均分"""
    if not os.path.exists(input_file):
        return {}
    record_dict = {}
    score_dict = {}
    with open(input_file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for item in data:
            if len(item) < 4:
                continue
            userId, movieId, rating = item[0], item[1], float(item[2])
            if movieId not in record_dict:
                record_dict[movieId] = [0, 0]
            record_dict[movieId][0] += 1
            record_dict[movieId][1] += rating
    for movieId in record_dict:
        score_dict[movieId] = round(record_dict[movieId][1] / record_dict[movieId][0], 3)
    return score_dict


def get_train_data(input_file):
    """计算正负样本，得到训练集，训练集中的正负样本数量一致"""
    if not os.path.exists(input_file):
        return []
    score_dict = get_ave_score(input_file)
    pos_dict = {}
    neg_dict = {}
    train_data = []
    score_threshold = 4.0
    with open(input_file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for item in data:
            if len(item) < 4:
                continue
            userId, movieId, rating = item[0], item[1], float(item[2])
            if userId not in pos_dict:
                pos_dict[userId] = []
            if userId not in neg_dict:
                neg_dict[userId] = []
            # 评分大于等于指定阈值，即为正样本
            if rating >= score_threshold:
                pos_dict[userId].append((movieId, 1))
            else:
                score = score_dict.get(movieId, 0)
                # 这里不直接把label写成0，而是score，是因为之后需要根据score进行负采样
                # 这里的score是所有已评分用户对该电影的平均评分，而并不是当前用户对该电影的评分
                neg_dict[userId].append((movieId, score))
    # 正负样本均衡，负采样
    for userId in pos_dict:
        # 负样本数通常比正样本数大很多，因此这里的data_num基本可看做是正样本的数量
        data_num = min(len(pos_dict[userId]), len(neg_dict.get(userId, [])))
        # 若正负样本数均大于0
        if data_num > 0:
            # 添加正样本。确保正负样本的数量均为data_num
            train_data += [(userId, item[0], item[1]) for item in pos_dict[userId]][:data_num]
        else:
            continue
        # 对同一个用户评过的所有电影根据评分进行降序排列。reverse = True 降序 ， reverse = False 升序（默认）
        # neg_dict[userId]为(movieId,score)
        sorted_neg_dict = sorted(neg_dict[userId], key=lambda x: x[1], reverse=True)[:data_num]
        # 添加负样本。最后面的0为label，表示负样本。1表示正样本
        train_data += [(userId, item[0], 0) for item in sorted_neg_dict]
    return train_data


if __name__ == '__main__':
    # item_dict = get_item_info("../data/movies.csv")
    # print(len(item_dict))
    # print(item_dict["1"])
    # print(item_dict["11"])

    # score_dict = get_ave_score("../data/ratings.csv")
    # print(len(score_dict))
    # print(score_dict["31"])

    # userId、movieId、label（0或1）
    train_data = get_train_data("../data/ratings.csv")
    print(len(train_data))
    print(train_data[:50])
