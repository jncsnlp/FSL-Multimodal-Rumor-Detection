import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from dataloader import build_dataset, build_iterator, get_time_dif
from importlib import import_module
from learn_emb import extract_emb


class CategoriesSampler:

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        assert self.n_cls % 2 == 0, 'Should be pairwise rumor-nonrumor, so that should be even'

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
            print('Class %d has %d samples' % (i, len(ind)))

        self.num_labels = len(self.m_ind)
        assert self.num_labels % 2 == 0, 'Should be pairwise rumor-nonrumor, so that should be even'
        self.fixed_batches = []
        self.fixed_batches_classes = []
        self.mode = 'rand'  # 'probe', 'fix'

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        if self.mode == 'rand':
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(self.num_labels // 2)[
                          :self.n_cls // 2]  # random sample num_class indices,e.g. 5
                for c in classes:
                    c1 = c * 2
                    c2 = c1 + 1
                    for ci in [c1, c2]:
                        l = self.m_ind[ci]  # all data indexs of this class
                        assert len(l) >= self.n_per
                        pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                        batch.append(l[pos])
                batch = torch.stack(batch).t().reshape(-1)
                # .t() transpose,
                # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
                # instead of aaaabbbbccccdddd
                yield batch

        elif self.mode == 'probe':
            print('Probe to fix val set')
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(self.num_labels // 2)[
                          :self.n_cls // 2]  # random sample num_class indices,e.g. 5
                for c in classes:
                    c1 = c * 2
                    c2 = c1 + 1
                    for ci in [c1, c2]:
                        l = self.m_ind[ci]  # all data indexs of this class
                        assert len(l) >= self.n_per
                        pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                        batch.append(l[pos])

                self.fixed_batches.append(batch)
                batch_t = torch.stack(batch).t().reshape(-1)
                # .t() transpose,
                # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
                # instead of aaaabbbbccccdddd
                yield batch_t

        else:
            assert self.mode == 'fix'
            assert len(self.fixed_batches) == self.n_batch
            # print(self.fixed_batches)
            for ix, batch in enumerate(self.fixed_batches):
                batch_t = torch.stack(batch).t().reshape(-1)
                yield batch_t


class wb_rumor_fsl_dataset(Dataset):

    def __init__(self, split_name='train', split_no=0, model='bert', featstype='emb_outs', data_path='./DataSet_pair_comments/',
                 pad_size=None, com_pad_size=None, bert_path=None):

        assert split_name in ['train', 'dev', 'test']
        assert split_no in [0, 1, 2]

        self.split_name = split_name
        self.split_no = split_no
        self.data_path = data_path + "data"

        # doclabel_path:原始文件路径
        if split_name in ['train', 'dev']:
            # train or dev file: doc + "\t" + label
            # print(data_path)
            doclabel_file = os.path.join(self.data_path, '%s_%d.txt' % (split_name, split_no))
        else:
            # test file: doc + "\t" + label
            doclabel_file = os.path.join(self.data_path, '%s.txt' % (split_name,))

        # print(doclabel_file)
        assert os.path.isfile(doclabel_file)
        assert pad_size, 'Please set a valid pad size'

        print("Loading data...")

        model_name = model  # bert
        print("model name: %s" % (model_name))
        X = import_module('bertmodels.' + model_name)
        # print(X)
        config = X.Config(data_path, doclabel_file, bert_path)

        # load label
        # for x in open(doclabel_file, 'r').readlines():
        #     print(x.strip().split("\t"))
        #
        # print()
        # for x in open(doclabel_file, 'r', encoding='utf-8').readlines():
        #     print(x.strip().split("\t")[0])
        #     print(x.strip().split("\t")[2])
        lines = [int(x.strip().split("\t")[2]) for x in open(doclabel_file, 'r', encoding='utf-8').readlines()]
        # print(lines)
        print('%s split %d has %d samples' % (split_name, split_no, len(lines)))
        # print(lines,'---- dataset Line 112')

        # load doc and label
        # doc_com_label: tokenIDs_con, tokenIDs_com, labels, seq_len_con, masks_con, seq_len_com(list), masks_com
        config.pad_size = pad_size
        config.com_pad_size = com_pad_size
        doc_com_label = build_dataset(config)
        # print(doc_label[0][2])
        data_iter = build_iterator(doc_com_label, config)

        # learn embeddings

        # train
        bertmodel = X.Model(config).to(config.device)
        emb_con, emb_com = extract_emb(config, bertmodel, data_iter, featstype)
        # print(len(emb_con), len(emb_com))
        # print(len(emb_con[0]), len(emb_com[0]))
        # print(emb_con, emb_com)
        feat = np.array(emb_con)
        feat_com = np.array(emb_com)
        # print(feat.shape)
        # print(feat_com.shape)

        # load feature
        # feat = np.load(emd_file)
        print('Load numpy of shape', feat.shape)  # (N, 1, PAD, D)
        assert len(feat.shape) == 4
        assert feat.shape[0] == len(lines)
        # self.feat = feat[:, 0, :, :]
        self.feat = feat[:, :, :, :]
        print(self.feat.shape)
        self.feat_com = feat_com[:, :, :, :]
        print(self.feat_com.shape)

        self.num_data = feat.shape[0]

        # normalize labels to start from zero
        self.raw_label_2_real_label = {}
        self.real_label_2_raw_label = {}
        raw_class = sorted(np.unique(lines))
        print(raw_class)

        self.num_class = len(raw_class)  # should ensure it's pairwise
        # check the raw_class should contain pairwise label, e.g., class 0,1 class 5,6
        assert self.num_class % 2 == 0, 'Should be pairwise rumor-nonrumor, so that should be even'
        for i in range(self.num_class // 2):
            assert raw_class[i * 2] + 1 == raw_class[
                i * 2 + 1], 'Should be pairwise rumor-nonrumor, so that raw_class number should be pair'

        self.labels = []
        for i in range(self.num_class):
            j = raw_class[i]
            self.raw_label_2_real_label[j] = i
            self.real_label_2_raw_label[i] = j

        print('raw class -> real class', self.raw_label_2_real_label)
        print('real class -> raw class', self.real_label_2_raw_label)
        print("Make sure the raw class to real class is continuous")

        labels = []
        for i in range(len(lines)):
            cls = self.raw_label_2_real_label[lines[i]]
            labels.append(cls)
        # print('Real labels used in training', labels)
        self.all_labels = labels

        # 每个doc的seq len
        lens = []
        # 每个comment的seq len
        lens_com = []
        for doc_label_len_mask in doc_com_label:
            lens.append(doc_label_len_mask[3])
            lens_com.append(doc_label_len_mask[5])
        self.all_lens = lens
        self.all_lens_com = lens_com
        self.mode = 'norm'

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # for gaining fixed validation sets
        if self.mode == 'dummy':
            return 1

        # a single data in a batch
        if len(index) == 1:
            ft, ft_com, lb, ln, ln_com = self.feat[index], self.feat_com[index], self.all_labels[index], self.all_lens[index], self.all_lens_com[index]
            return ft, ft_com, lb, ln, ln_com

        feats = []
        feats_com = []
        lbs = []
        lns = []
        lns_com = []
        # print(index)
        for ind in index:
            ft, ft_com, lb, ln, ln_com = self.feat[ind:ind + 1, :], self.feat_com[ind:ind + 1, :], self.all_labels[ind], self.all_lens[ind], self.all_lens_com[ind]
            feats.append(ft)
            feats_com.append(ft_com)
            lbs.append(lb)
            lns.append(ln)
            lns_com.append(ln_com)
        feats = np.concatenate(feats, axis=0)
        feats_com = np.concatenate(feats_com, axis=0)
        # print(feats.shape)
        lbs = np.array(lbs, dtype=np.int)  # this lbs are the raw class labels, which will not be used in training
        lns = np.array(lns, dtype=np.int)
        lns_com = np.array(lns_com, dtype=np.int)
        # print(lbs)
        return feats, lbs, lns, feats_com, lns_com


if __name__ == '__main__':

    if 1 == 2:
        for sp in ['train', 'dev', 'test']:
            for split_no in [0, 1, 2]:
                ds = wb_rumor_fsl_dataset(sp, split_no)
                # print(len(ds), np.unique(ds.label), ds.accumulate)

        pass
    elif 2 == 2:
        for sp in ['test']:
            for split_no in [0, 1, 2]:
                ds = wb_rumor_fsl_dataset(sp, split_no)
                # print(len(ds), np.unique(ds.label), ds.accumulate)

        pass
