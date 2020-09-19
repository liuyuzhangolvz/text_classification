# -*- coding: utf-8 -*-
import os
import argparse
from data_load import load_list, get_type_list, TFIDF
from model import kNN, MultinomialNB, SVM
from sklearn.naive_bayes import MultinomialNB as NB  
from sklearn.linear_model.logistic import LogisticRegression as LR 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as kNN
import time


def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag):

    """分类器"""
    starttime = time.time()
    if flag == 'knn':
        #SVM分类器
        #classifier = kNN(train_feature_list, train_class_list)
        classifier = kNN(algorithm='auto').fit(train_feature_list, train_class_list)
    if flag == 'nb':
        # 朴素贝叶斯分类器 拟合 默认拉普拉斯平滑 不指定先验概率先验概率
        #classifier = MultinomialNB()
        #classifier.fit(train_feature_list, train_class_list)
        classifier = NB().fit(train_feature_list, train_class_list)
    if flag == 'svm':
        #SVM分类器
        #classifier = SVM()
        #classifier.fit(train_feature_list, train_class_list)
        classifier = LinearSVC().fit(train_feature_list, train_class_list)
    if flag == 'lg':
        # 逻辑回归分类器 指定liblinear为求解最优化问题的算法 最大迭代数 多分类问题策略
        classifier = LR(solver='liblinear',max_iter=5000, multi_class='auto')
        classifier.fit(train_feature_list, train_class_list)     
    if flag == 'rf':
        # 随机森林分类器
        classifier = RF(n_estimators=200)
        classifier.fit(train_feature_list, train_class_list)
    endtime = time.time()
    #print("train time>>", endtime - starttime)
    #print("start to test")
    atime = time.time()
    test_accuracy = classifier.score(test_feature_list, test_class_list)        # 测试准确率
    btime = time.time()
    #print("test time>>", btime - atime)
    return test_accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='.\corpus', help='划分数据集后生成的训练和测试语料地址')
    #parser.add_argument('--batch_size', type=int, default= 16, help='batch size')
    args = parser.parse_args()

    train_datas = load_list(os.path.join(args.corpus_dir, 'train.pk'))
    test_datas = load_list(os.path.join(args.corpus_dir, 'test.pk'))

    vocab_list = load_list(os.path.join(args.corpus_dir, 'vocab.pk'))
    train_types = get_type_list(train_datas)
    test_types = get_type_list(test_datas)
    print(len(test_types))

    types_list = list(set(train_types))
    types2idx = {types: idx for idx, types in enumerate(types_list)}
    train_types_vec = [types2idx[t] for t in train_types]
    test_types_vec = [types2idx[t] for t in test_types]

    print("data vector....")
    train_vec = TFIDF().get_tfidf(train_datas, vocab_list)
    print("train vector shape>>> {}".format(train_vec.shape))
    test_vec = TFIDF().get_tfidf(test_datas, vocab_list)
    print("test vector shape>>> {}".format(test_vec.shape))

    
    print("start to train svm...")
    test_accuracy = text_classifier(train_vec, test_vec, train_types, test_types, flag='svm')
    print("svm-accuracy: {}".format(test_accuracy))
    
    print("start to train nb...")
    test_accuracy = text_classifier(train_vec, test_vec, train_types_vec, test_types_vec, flag='nb')
    print("nb-accuracy: {}".format(test_accuracy))
    
    print("start to train lg...")
    test_accuracy = text_classifier(train_vec, test_vec, train_types, test_types, flag='lg')
    print("lg-accuracy: {}".format(test_accuracy))

    print("start to train rf...")
    test_accuracy = text_classifier(train_vec, test_vec, train_types, test_types, flag='rf')
    print("rf-accuracy: {}".format(test_accuracy))
    
    print("start to train knn...")
    test_accuracy = text_classifier(train_vec, test_vec, train_types, test_types, flag='knn')
    print("knn-accuracy: {}".format(test_accuracy))
    """