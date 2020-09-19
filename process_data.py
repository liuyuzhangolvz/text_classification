# -*- coding: utf-8 -*-
import os
import re
import json
import jieba
import random
import pickle
import argparse

def data_processor(data_dir, corpus_dir, stopwords, test_size=0.2):
    os.makedirs(corpus_dir, exist_ok=True)

    def save_list(datas, save_dir):
        """Save datas to dir"""
        with open(save_dir, 'wb') as writer:
            pickle.dump(datas, writer)
        print("Save to {}".format(save_dir))

    def build_vocab(datas):
        vocab = {}
        for data in datas:
            for word in data['text']:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return list(list(zip(*vocab))[0])

    def delete_stopwords(sentence_list, stop_words):
        delete_stopwords = []
        for word in sentence_list:
            if word not in stop_words and word != ' ' and word != '-' and not word.isdigit():
                delete_stopwords.append(word)
        return delete_stopwords

    def delete_puctuation(sentence):
        sentence = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", sentence)
        return sentence

    datas = []
    for fname in os.listdir(data_dir):
        print("Load {} file".format(fname.encode('utf-8')))
        new_dir = os.path.join(data_dir, fname)
        for text_dir in os.listdir(new_dir):
            if ".txt" not in text_dir:
                continue
            fpath = os.path.join(new_dir, text_dir)
            with open(fpath, 'r', encoding='utf-8') as reader:
                tmp = {}
                raw = reader.read().replace(' ', '')
                content = delete_puctuation(raw)
                #seg_content = jieba.lcut(content)
                tmp['text'] = delete_stopwords(content, stop_words)
                tmp['new_type'] = fname.encode('utf-8')
                #print(tmp)
                datas.append(tmp)
    """先打乱，再按9:1划分数据集"""
    random.shuffle(datas)
    index = int(len(datas) * test_size) + 1
    train_datas = datas[index:]
    test_datas = datas[:index]

    vocab = build_vocab(train_datas)

    save_list(train_datas, os.path.join(corpus_dir, 'train.pk'))
    save_list(test_datas, os.path.join(corpus_dir, 'test.pk'))
    save_list(vocab, os.path.join(corpus_dir, 'vocab.pk'))

def make_stopwords(stop_words_file):
    stop_words = set()
    with open(stop_words_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            word = line.strip()
            if len(word) > 0 and word not in stop_words:
                stop_words.add(word)
    return stop_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.\data', help='数据源地址')
    parser.add_argument('--corpus_dir', type=str, default='.\corpus', help='划分数据集后生成的训练和测试语料地址')
    parser.add_argument('--stopwords_dir', type=str, default='.\corpus', help='中文停用词地址')
    args = parser.parse_args()

    stop_words = make_stopwords(os.path.join(args.stopwords_dir, 'cn_stopwords.txt'))

    data_processor(args.data_dir, args.corpus_dir, stop_words)
