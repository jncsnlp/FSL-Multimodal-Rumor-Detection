# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from dataloader import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def extract_emb(config, model, data_iter, featstype):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(data_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()

    # 存储训练集/验证集/测试集的embedding
    data_embs_con = []
    data_embs_com = []


    for i, (docs_comments, labels) in enumerate(data_iter):
        # print(docs)
        if featstype == "pooled":
            outputs_con, outputs_com = model.get_pooled(docs_comments)
        elif featstype == "emb_outs":
            outputs_con, outputs_com = model.get_emb(docs_comments)
        elif featstype == "enc_layer":
            outputs_con, outputs_com = model.get_enc(docs_comments)

        # 输出bert embedding
        # print(outputs.size())
        np_outputs_con = outputs_con.cpu().detach().numpy()
        np_outputs_com = outputs_com.cpu().detach().numpy()
        data_embs_con.append(np_outputs_con)
        data_embs_com.append(np_outputs_com)

    # print(len(data_embs))
    return data_embs_con, data_embs_com




# def test(config, model, test_iter):
#     # test
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score...")
#     print(test_report)
#     print("Confusion Matrix...")
#     print(test_confusion)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     labels_all = np.array([], dtype=int)
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss
#             labels = labels.data.cpu().numpy()
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             labels_all = np.append(labels_all, labels)
#             predict_all = np.append(predict_all, predic)

#     acc = metrics.accuracy_score(labels_all, predict_all)
#     if test:
#         labels = [i for i in range(0,28)]
#         report = metrics.classification_report(labels_all, predict_all, labels = labels, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(labels_all, predict_all)
#         return acc, loss_total / len(data_iter), report, confusion
#     return acc, loss_total / len(data_iter)