# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, data_path, doclabel_file, bert_path=None):
        self.model_name = 'bert'
        self.doc_path = doclabel_file                                # 文本类标数据路径
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            data_path + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = data_path + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 1                                             # epoch数
        self.batch_size = 1                                           # mini-batch大小
        # self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                       # 学习率
        # self.learning_rate = 5e-5                                       # 学习率\
        if bert_path is None:
            self.bert_path = './bert_pretrain'
        else:
            self.bert_path = bert_path
            print('Use bert path', self.bert_path)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        comment = x[1]  # 输入的评论
        # print(context,comment)
        mask_con = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask_com = x[5]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        embedding_output, encoder_layer, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # print(embedding_output.size())
        out = self.fc(pooled)
        return out

    def get_emb(self, x):
        context = x[0]  # 输入的句子
        comment = x[1]  # 输入的评论
        # print(context,comment)
        # print(context)
        mask_con = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask_com = x[5]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # print(mask_com)
        embedding_output_con, encoder_layer_con, pooled_con = self.bert(context, attention_mask=mask_con, output_all_encoded_layers=False)
        embedding_output_com, encoder_layer_com, pooled_com = self.bert(comment, attention_mask=mask_com, output_all_encoded_layers=False)
        return embedding_output_con, embedding_output_com

    def get_enc(self, x):
        context = x[0]  # 输入的句子
        comment = x[1]  # 输入的评论
        # print(context)
        mask_con = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask_com = x[5]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        embedding_output_con, encoder_layer_con, pooled_con = self.bert(context, attention_mask=mask_con, output_all_encoded_layers=False)
        embedding_output_com, encoder_layer_com, pooled_com = self.bert(comment, attention_mask=mask_com, output_all_encoded_layers=False)
        return encoder_layer_con, embedding_output_com

    def get_pooled(self, x):
        context = x[0]  # 输入的句子
        comment = x[1]  # 输入的评论
        # print(context)
        mask_con = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask_com = x[5]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        embedding_output_con, encoder_layer_con, pooled_con = self.bert(context, attention_mask=mask_con, output_all_encoded_layers=False)
        embedding_output_com, encoder_layer_com, pooled_com = self.bert(comment, attention_mask=mask_com, output_all_encoded_layers=False)
        return pooled_con, pooled_com
