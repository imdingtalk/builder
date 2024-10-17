# -*- coding:utf-8 -*-
"""
@file name  : main.py
@author     : 建
@date       : 2024-10-15
@brief      : docker 部署文件
"""
import json
import argparse
import ast

import jieba
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset


def read_data_file(file_path1) -> list[dict]:
    """
    读取txt文件中的每条数据作为字典
    :param file_path1: txt文件文件路径
    :return: 字典列表
    """
    data_list = []
    with open(file_path1, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用 ast.literal_eval 将字符串转换为字典
            data_dict = ast.literal_eval(line.strip())
            data_dict['Label'] = int(data_dict['Label'])
            data_list.append(data_dict)
    return data_list


class WordToIndex(object):
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.OOV = 0

    def encode(self, sentence, vocab_dict, max_len=None):
        # 将句子中的单词根据词表 (vocab_dict) 映射为对应的索引。
        if max_len is not None:    # 补齐，切割 句子固定长度
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [vocab_dict.get(word, self.OOV) for word in sentence]

    @staticmethod
    def decode(ws_inverse, indices):
        # 将一组索引转换回对应的单词
        return [ws_inverse.get(idx) for idx in indices]


def text_split(content: str) -> list[str]:
    """
    对原始文本进行token化，包含一系列预处理清洗操作
    :param content: 输入的中文文本
    :return: 分词后的token列表
    """
    content = content.replace('\n', ' ').strip()  # 去除不必要的符号，例如换行符和多余的空格
    tokens = jieba.lcut(content)  # 使用 jieba 进行中文分词
    return tokens

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(LSTMTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 一个output向量长度是2倍的hidden size，有两个output拼接，所以是4倍

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


class AclImdbDataset(Dataset):
    def __init__(self, sentence_list, vocab_path, max_len=200):
        """
        :param sentence_list: 预处理好的数据列表，格式：[{'ID': 'xxx', 'Label': '0', 'Content': 'xxx'}, ...]
        :param vocab_path: 词表的路径
        :param max_len: 每个句子的最大长度
        """
        self.sentence_list = sentence_list
        self.max_len = max_len
        self.vocab_path = vocab_path
        self.word2index = WordToIndex()
        self._init_vocab()


    def __getitem__(self, item):
        # 获取当前样本
        sample = self.sentence_list[item]
        content = sample['Content']
        label = int(sample['Label'])  # 将标签转换为整数

        # tokenize & encode to index
        token_list = text_split(content)  # 对文本内容进行分词处理
        token_idx_list = self.word2index.encode(token_list, self.vocab, self.max_len)

        return np.array(token_idx_list), label

    def __len__(self):
        return len(self.sentence_list)

    def _init_vocab(self):
        # 加载词表字典
        self.vocab = np.load(self.vocab_path, allow_pickle=True).item()


def get_args_parser():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行分类")
    parser.add_argument("--test-data", "-r", type=str, required=True, help="需要识别的数据路径(test_data.txt)")
    parser.add_argument("--output", "-o", type=str, required=True, help="导出结果路径 (result.log)")
    return parser


def load_model(model_path, vocab_path, device):
    # 模型参数
    input_size = 300  # embedding size
    hidden_size = 128  # hidden state size
    num_layers = 2  # LSTM层数
    cls_num = 2  # 分类类别

    # 加载词表
    vocab = np.load(vocab_path, allow_pickle=True).item()

    # 创建模型
    model = LSTMTextClassifier(len(vocab), input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab


def calculate_metrics(pred_labels, true_labels):
    # 初始化计数器
    TP = FP = TN = FN = 0
    # 计算TP, FP, TN, FN
    for pred, true in zip(pred_labels, true_labels):
        if pred == 1 and true == 1:  # True Positive
            TP += 1
        elif pred == 1 and true == 0:  # False Positive
            FP += 1
        elif pred == 0 and true == 0:  # True Negative
            TN += 1
        elif pred == 0 and true == 1:  # False Negative
            FN += 1

    # 计算准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    # 计算精确率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # 计算F1值
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def main():
    args = get_args_parser().parse_args()

    # 加载测试数据
    test_data_path = args.test_data
    vocab_path = 'parameter_file/aclImdb_vocab.npy'  # 词汇表路径
    model_path = 'parameter_file/checkpoint_best.pth'  # 模型权重路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取测试数据
    sentence_list = read_data_file(test_data_path)

    # 提取ID
    ids = [item['ID'] for item in sentence_list]
    true_labels = [item['Label'] for item in sentence_list]

    # 构建测试数据集
    test_dataset = AclImdbDataset(sentence_list, vocab_path, max_len=500)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 加载模型
    model, vocab = load_model(model_path, vocab_path, device)

    results = []
    pred_labels = []  # 预测标签
    total_samples = len(sentence_list)  # 总样本数量
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            pred_labels = pred_labels + preds.tolist()
            for id_val, pred, prob in zip(ids[i*64:(i+1)*64], preds, probs):
                result = {"ID": id_val, "Label": str(pred.item()), "prob": str(prob[pred].item())}
                results.append(result)


    # 保存结果
    output_path = args.output
    with open(output_path, 'w') as f:
        for res in results:
            f.write(f"{json.dumps(res)}\n")

    print(f"结果已保存到文件：{output_path}")
    # 计算指标
    metrics = calculate_metrics(pred_labels, true_labels)
    T_count_1 = true_labels.count(1)
    T_count_0 = true_labels.count(0)
    P_count_1 = pred_labels.count(1)
    P_count_0 = pred_labels.count(0)
    # 输出结果
    print(f'\n实际为1的个数为{T_count_1}')
    print(f'实际为0的个数为{T_count_0}')
    print(f'预测为1的个数为{P_count_1}')
    print(f'预测为0的个数为{P_count_0}')
    print('\n针对标签为1的数据')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')


if __name__ == "__main__":
    main()




