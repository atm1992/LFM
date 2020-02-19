# -*- coding: UTF-8 -*-
import numpy as np
from util import read
import operator


def lfm_train(train_data, F, alpha, beta, step):
    """
    :param train_data: 训练集
    :param F: 用户向量的长度或维度、item向量的长度或维度，这两长度一致。隐分类个数
    :param alpha:L2正则化参数
    :param beta:学习率
    :param step:迭代次数
    :return:两个字典
        dict    key:user_id, value:list 用户向量的隐特征列表
        dict    key:item_id, value:list item向量的隐特征列表
    """
    user_vec = {}
    item_vec = {}

    for step_index in range(step):
        for userId, movieId, label in train_data:
            # 将userId对应的vector 和 movieId对应的vector分别初始化为长度为F的向量
            # 按照标准分布初始化向量
            if userId not in user_vec:
                user_vec[userId] = init_model(F)
            if movieId not in item_vec:
                item_vec[movieId] = init_model(F)
            # 模型参数迭代
            delta = label - model_predict(user_vec[userId], item_vec[movieId])
            for f in range(F):
                user_vec[userId][f] += beta * (delta * item_vec[movieId][f] - alpha * user_vec[userId][f])
                item_vec[movieId][f] += beta * (delta * user_vec[userId][f] - alpha * item_vec[movieId][f])
            # 学习率衰减
            beta *= 0.9
    return user_vec, item_vec


def init_model(F):
    # F为隐类个数，即 向量的长度
    return np.random.randn(F)


def model_predict(user_vec, item_vec):
    """返回user_vec与item_vec之间的距离，表示推荐的强度、权重。这里使用cosine距离，向量内积除以向量模的乘积"""
    res = np.dot(user_vec, item_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(item_vec))
    return res


def model_train_process():
    """test lfm model train"""
    userId = '24'
    train_data = read.get_train_data("../data/ratings.csv")
    user_vec, item_vec = lfm_train(train_data, 10, 0.02, 0.2, 20)
    recom_result = get_recom_result(user_vec, item_vec, userId)
    analysis_recom_result(train_data, userId, recom_result)


def get_recom_result(user_vec, item_vec, userId):
    """使用LFM模型获取指定用户userId的推荐结果，返回一个推荐列表"""
    # 推荐前10个item
    fix_num = 10
    if userId not in user_vec:
        return []
    record = {}
    recom_result = []
    the_user_vec = user_vec[userId]
    for itemId in item_vec:
        the_item_vec = item_vec[itemId]
        # 计算指定用户向量 与 指定物品向量之间的欧氏距离
        res = np.dot(the_user_vec, the_item_vec) / (np.linalg.norm(the_user_vec) * np.linalg.norm(the_item_vec))
        record[itemId] = res
    for item in sorted(record.items(), key=operator.itemgetter(1), reverse=True)[:fix_num]:
        itemId = item[0]
        # 得分保留3位小数
        score = round(item[1], 3)
        recom_result.append((itemId, score))
    return recom_result


def analysis_recom_result(train_data, userId, recom_result):
    """分析指定用户userId的推荐结果。train_data是用于获取该用户之前感兴趣的item"""
    item_info = read.get_item_info("../data/movies.csv")
    for user_id, movieId, label in train_data:
        if user_id == userId and label == 1:
            print(item_info[movieId])
    print("recom_result")
    for item in recom_result:
        print(item_info[item[0]])


if __name__ == '__main__':
    model_train_process()
