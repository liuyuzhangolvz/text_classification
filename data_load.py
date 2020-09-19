# -*- coding: utf-8 -*-
import os
import pickle
import random
import math
from collections import defaultdict

import numpy as np 


def load_list(data_file):
    with open(data_file, 'rb') as reader:
        datas = pickle.load(reader)
    return datas

def get_type_list(datas):
    types = []
    for data in datas:
        types.append(data['new_type'])
    return types

class TFIDF(object):
    """docstring for TFIDF"""
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.word2df = defaultdict(int)
        #self.type2idx = {}
        self.tf_dict = {}
        self.text_lists = []

    def process_data(self, datas, vocab_list):
        for word in vocab_list:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        for data in datas:
            self.text_lists.append(data['text'])
            for word in set(data['text']):
                if word in self.word2idx:
                    self.word2df[word] += 1
                else:
                    self.word2df["<unk>"] += 1

    def get_tf(self):
        """获得文档的 TF 向量"""
        tf_vec = []
        for text in self.text_lists:
            vec = [0] * len(self.word2idx)
            words = set(text)
            for word in words:
                if word in self.word2idx:
                    vec[self.word2idx[word]] = text.count(word) / len(text)
                else:
                    vec[self.word2idx["<unk>"]] = 0
            tf_vec.append(vec)
        return np.array(tf_vec)

    def get_idf(self):
        """获得文档的 idf 向量"""
        total = len(self.text_lists)
        idf_vec = []
        for text in self.text_lists:
            vec = [0] * len(self.word2idx)
            words = set(text)
            for word in words:
                if word in self.word2idx:
                    vec[self.word2idx[word]] = math.log(total + 1.0) - math.log(self.word2df[word] + 1.0)
                else:
                    vec[self.word2idx["<unk>"]] = 0
            idf_vec.append(vec)
        return np.array(idf_vec)

    def get_tfidf(self, datas, vocab_list):
        self.process_data(datas, vocab_list)
        tf = self.get_tf()
        idf = self.get_idf()
        tfidf = tf * idf
        return tfidf
