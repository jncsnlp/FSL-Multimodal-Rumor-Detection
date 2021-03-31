# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32, com_pad_size=32):
        print(path)
        contents_comments = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                # print(line)
                lin = line.strip()
                if not lin:
                    continue
                content, comments, label = lin.split('\t')
                # tokenize content
                token_con = config.tokenizer.tokenize(content)
                token_con = [CLS] + token_con
                #print(token)
                seq_len = len(token_con)
                mask = []
                token_con_ids = config.tokenizer.convert_tokens_to_ids(token_con)

                if pad_size:
                    if len(token_con) < pad_size:
                        mask = [1] * len(token_con_ids) + [0] * (pad_size - len(token_con))
                        token_con_ids += ([0] * (pad_size - len(token_con)))
                    else:
                        mask = [1] * pad_size
                        token_con_ids = token_con_ids[:pad_size]
                        seq_len = pad_size

                # tokenize comment
                # com_pad_size = 32
                all_comments = comments.split(';')
                all_token_com_ids = []
                all_seq_len_com = []
                all_mask_com = []
                # print("=================",all_comments)
                for idx in range(3):
                    # print(idx, all_comments)

                    if (idx+1 > len(all_comments)):
                        comment = ""
                        # print(idx, comment)
                    elif (len(all_comments[idx]) > 0):
                        comment = all_comments[idx]
                        # print(idx, comment)
                    else:
                        comment = ""
                        # print(idx, comment)

                    token_com = config.tokenizer.tokenize(comment)
                    token_com = [CLS] + token_com
                    #print(token)
                    seq_len_com = len(token_com)
                    mask_com = []
                    token_com_ids = config.tokenizer.convert_tokens_to_ids(token_com)
                    # print(token_com_ids)

                    if com_pad_size:
                        if len(token_com) < com_pad_size:
                            mask_com = [1] * len(token_com_ids) + [0] * (com_pad_size - len(token_com))
                            token_com_ids += ([0] * (com_pad_size - len(token_com)))
                        else:
                            mask_com = [1] * com_pad_size
                            token_com_ids = token_com_ids[:com_pad_size]
                            seq_len_com = com_pad_size
                    # print(seq_len_com)
                    # print(mask_com)
                    all_token_com_ids.extend(token_com_ids)
                    all_seq_len_com.append(seq_len_com)
                    all_mask_com.extend(mask_com)
                # print(all_token_com_ids, all_seq_len_com, all_mask_com)

                contents_comments.append((token_con_ids, all_token_com_ids, int(label), seq_len, mask, all_seq_len_com, all_mask_com))

        return contents_comments
    data = load_dataset(config.doc_path, config.pad_size, config.com_pad_size)
    return data


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_con = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x_com = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len_con = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        seq_len_com = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        mask_con = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        mask_com = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        return (x_con, x_com, seq_len_con, mask_con, seq_len_com, mask_com), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
