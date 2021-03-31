import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from learner import LearnerResNet
from copy import deepcopy


class MetaLSTM(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """
        :param args:
        """
        super(MetaLSTM, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.query
        self.task_num = args.batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.hidden_size = args.hidden_size

        print('Use MetaLSTM')
        # self.net = LearnerResNet(config, args.imgc, args.imgsz)

        self.feature_extractor = LearnerResNet(config[:-1])
        self.classifier = LearnerResNet(config[-1:])

        # print(self.feature_extractor, self.classifier)

        self.meta_optim = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                            {'params': self.classifier.parameters(), 'lr': self.meta_lr}],
                                           lr=self.meta_lr, weight_decay=0.0005)

        # self.meta_optim = torch.optim.SGD([{'params': self.feature_extractor.parameters()},
        #                                     {'params': self.classifier.parameters(), 'lr': self.meta_lr}],
        #                                    lr=self.meta_lr, weight_decay=0.0005)

    def forward(self, x_spt, y_spt, l_spt, x_qry, y_qry, l_qry):
        """
        setsz = n_way * k_shot
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        # task_num, setsz, len_sent, D_vec = x_spt.size()
        task_num, setsz, seq_, d_ = x_spt.size()
        # print(x_spt.shape)
        # print(l_spt.shape)
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        feats_shot = []
        feats_query = []
        # l_shot = []
        # l_query = []
        for i in range(task_num):
            fs = x_spt[i, ...]  # self.convs(x_spt[i, ...])
            feats_shot.append(fs)
            fq = x_qry[i, ...]  # self.convs(x_qry[i, ...])
            feats_query.append(fq)
            # ls = l_spt[i, ...]
            # l_shot.append(ls)
            # lq = l_qry[i, ...]
            # l_query.append(lq)

        # print(task_num)
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            feat_0 = feats_shot[i].detach()
            # print(feat_0.size())
            h_0 = torch.zeros((feat_0.size(0), self.hidden_size)).to(x_spt.device)

            feat = self.feature_extractor(feat_0, vars=None, bn_training=True, hidden=h_0.detach(), lens=l_spt[i])
            logits = self.classifier(feat, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # update parameters for the episode, and clone the params theta'

            grad_1 = torch.autograd.grad(loss, self.feature_extractor.parameters(), retain_graph=True)
            fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                      zip(grad_1, self.feature_extractor.parameters().parameters())))
            grad_2 = torch.autograd.grad(loss, self.classifier.parameters())
            fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                      zip(grad_2, self.classifier.parameters())))

            for k in range(1, self.update_step):
                feat_i = self.feature_extractor(feat_0, fast_weights_1, bn_training=True, hidden=h_0.detach(),
                                                lens=l_spt[i])
                logit_i = self.classifier(feat_i, vars=fast_weights_2, bn_training=True)
                loss = F.cross_entropy(logit_i, y_spt)
                grad_1 = torch.autograd.grad(loss, fast_weights_1, retain_graph=True)
                fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_1, fast_weights_1)))
                grad_2 = torch.autograd.grad(loss, fast_weights_2)
                fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_2, fast_weights_2)))

            h_q = torch.zeros((feats_query[i].size(0), self.hidden_size)).to(x_spt.device)
            feat_q = self.feature_extractor(feats_query[i], fast_weights_1, bn_training=True, hidden=h_q, lens=l_qry[i])
            logits_q = self.classifier(feat_q, fast_weights_2, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)

            # logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            # loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[k + 1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetuning(self, x_spt, y_spt, l_spt, x_qry, y_qry, l_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
        # convs = deepcopy(self.convs)
        feature_extractor = deepcopy(self.feature_extractor)
        classifier = deepcopy(self.classifier)

        # 1. run the i-th task and compute loss for k=0
        with torch.no_grad():
            feats_shot = x_spt  # convs(x_spt)
            feats_query = x_qry  # convs(x_qry)

        h_0 = torch.zeros((feats_shot.size(0), self.hidden_size)).to(x_spt.device)
        feat_0 = feature_extractor(feats_shot, hidden=h_0.detach(), lens=l_spt)
        logits = classifier(feat_0)
        loss = F.cross_entropy(logits, y_spt)
        grad_1 = torch.autograd.grad(loss, feature_extractor.parameters(), retain_graph=True)
        fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_1, feature_extractor.parameters().parameters())))
        grad_2 = torch.autograd.grad(loss, classifier.parameters())
        fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_2, classifier.parameters())))

        # grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        h_q = torch.zeros((feats_query.size(0), self.hidden_size)).to(x_spt.device)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            # logits_q = net(x_qry, net.parameters(), bn_training=True)
            feat_q = feature_extractor(feats_query, feature_extractor.parameters(), bn_training=True,
                                       hidden=h_q.detach(), lens=l_qry)
            logits_q = classifier(feat_q, classifier.parameters(), bn_training=True)

            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            feat_q = feature_extractor(feats_query, fast_weights_1, bn_training=True,
                                       hidden=h_q.detach(), lens=l_qry)
            logits_q = classifier(feat_q, fast_weights_2, bn_training=True)

            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            feat_i = feature_extractor(feats_shot, fast_weights_1, bn_training=True,
                                       hidden=h_0.detach(), lens=l_spt)
            logit_i = classifier(feat_i, fast_weights_2, bn_training=True)
            loss = F.cross_entropy(logit_i, y_spt)
            grad_1 = torch.autograd.grad(loss, fast_weights_1, retain_graph=True)
            fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_1, fast_weights_1)))
            grad_2 = torch.autograd.grad(loss, fast_weights_2)
            fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_2, fast_weights_2)))

            with torch.no_grad():
                feat_q = feature_extractor(feats_query, fast_weights_1, bn_training=True,
                                           hidden=h_q.detach(), lens=l_qry)
                logits_q = classifier(feat_q, fast_weights_2, bn_training=True)

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del feature_extractor, classifier

        accs = np.array(corrects) / querysz

        return accs


class MetaLSTMMultiTask(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """
        :param args:
        """
        super(MetaLSTMMultiTask, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.query
        self.task_num = args.batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.hidden_size = args.hidden_size
        self.eta = args.eta
        self.com_pad_size = args.com_pad_size
        # assert 0 < self.eta < 1

        print('Use MetaLSTMMultiTask')

        self.feature_extractor = LearnerResNet(config[:-2])
        self.com_feature_extractor = LearnerResNet(config[:-2])
        self.classifier_binary = LearnerResNet(config[-2:-1])
        self.classifier_topic = LearnerResNet(config[-1:])
        self.meta_optim = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                            {'params': self.com_feature_extractor.parameters()},
                                            {'params': self.classifier_binary.parameters()},
                                            {'params': self.classifier_topic.parameters()}],
                                           lr=self.meta_lr, weight_decay=0.0005)

    def forward(self, x_spt, x_com_spt, y_spt, y_spt_topic, l_spt, l_com_spt, x_qry, x_com_qry, y_qry, y_qry_topic, l_qry, l_com_qry):
        """
        setsz = n_way * k_shot
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        # task_num, setsz, len_sent, D_vec = x_spt.size()
        task_num, setsz, seq_, d_ = x_spt.size()
        # print(x_spt.shape)
        # print(l_spt.shape)
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        corrects_topic = [0 for _ in range(self.update_step + 1)]

        feats_shot = []
        feats_query = []
        feats_com_shot = []
        feats_com_query = []
        # l_shot = []
        # l_query = []
        for i in range(task_num):
            fs = x_spt[i, ...]  # self.convs(x_spt[i, ...])
            feats_shot.append(fs)
            fs_com = x_com_spt[i, ...]  # self.convs(x_spt[i, ...])
            feats_com_shot.append(fs_com)
            fq = x_qry[i, ...]  # self.convs(x_qry[i, ...])
            feats_query.append(fq)
            fq_com = x_com_qry[i, ...]  # self.convs(x_qry[i, ...])
            feats_com_query.append(fq_com)

        feats_shot = x_spt
        feats_query = x_qry
        feats_com_shot = x_com_spt
        feats_com_query = x_com_qry

        # print(task_num)
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            feat_0 = feats_shot[i].detach()
            feat_com_0 = feats_com_shot[i].detach()
            # print(feat_com_0.size())
            feat_com_0_0 = feat_com_0[:,:self.com_pad_size,:]
            feat_com_0_1 = feat_com_0[:,self.com_pad_size:2*self.com_pad_size,:]
            feat_com_0_2 = feat_com_0[:,2*self.com_pad_size:,:]
            # print(feat_com_0_0.shape, feat_com_0_1.shape, feat_com_0_2.shape)
            h_0 = torch.zeros((feat_0.size(0), self.hidden_size)).to(x_spt.device)
            h_com_0 = torch.zeros((feat_com_0_0.size(0), self.hidden_size)).to(x_com_spt.device)

            # # print(l_spt[i])
            # feat = self.feature_extractor(feat_0, vars=None, bn_training=True, hidden=h_0.detach(), lens=l_spt[i])

            feat_com_0 = self.com_feature_extractor(feat_com_0_0, vars=None, bn_training=True, hidden=h_com_0.detach(), lens=l_com_spt[i][:,0:1].reshape(l_com_spt[i].size(0)))
            feat_com_1 = self.com_feature_extractor(feat_com_0_1, vars=None, bn_training=True, hidden=h_com_0.detach(), lens=l_com_spt[i][:,1:2].reshape(l_com_spt[i].size(0)))
            feat_com_2 = self.com_feature_extractor(feat_com_0_2, vars=None, bn_training=True, hidden=h_com_0.detach(), lens=l_com_spt[i][:,2:].reshape(l_com_spt[i].size(0)))
            feat_com = 1.0/3*(feat_com_0+feat_com_1+feat_com_2)


            # feat = torch.cat((feat,feat_com),1)
            feat = feat_com
            # print(feat.size())
            logits_1 = self.classifier_binary(feat, vars=None, bn_training=True)
            logits_2 = self.classifier_topic(feat, vars=None, bn_training=True)
            # print(logits_1.size(), logits_2.size(), y_spt.size(), y_spt_topic.size())
            loss = F.cross_entropy(logits_1, y_spt) + self.eta * F.cross_entropy(logits_2, y_spt_topic)
            # update parameters for the episode, and clone the params theta'

            # grad_1 = torch.autograd.grad(loss, self.feature_extractor.parameters(), retain_graph=True)
            # fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
            #                           zip(grad_1, self.feature_extractor.parameters().parameters())))
            grad_com_1 = torch.autograd.grad(loss, self.com_feature_extractor.parameters(), retain_graph=True)
            fast_weights_com_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                      zip(grad_com_1, self.com_feature_extractor.parameters().parameters())))
            grad_2 = torch.autograd.grad(loss, self.classifier_binary.parameters())
            fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                      zip(grad_2, self.classifier_binary.parameters())))
            grad_3 = torch.autograd.grad(loss, self.classifier_topic.parameters())
            fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                      zip(grad_3, self.classifier_topic.parameters())))

            for k in range(1, self.update_step):
                # feat_i = self.feature_extractor(feat_0, fast_weights_1, bn_training=True, hidden=h_0.detach(),
                #                                 lens=l_spt[i])
                feat_com_i_0 = self.com_feature_extractor(feat_com_0_0, fast_weights_com_1, bn_training=True, hidden=h_com_0.detach(),
                                                lens=l_com_spt[i][:,0:1].reshape(l_com_spt[i].size(0)))
                feat_com_i_1 = self.com_feature_extractor(feat_com_0_1, fast_weights_com_1, bn_training=True, hidden=h_com_0.detach(),
                                                lens=l_com_spt[i][:,1:2].reshape(l_com_spt[i].size(0)))
                feat_com_i_2 = self.com_feature_extractor(feat_com_0_2, fast_weights_com_1, bn_training=True, hidden=h_com_0.detach(),
                                                lens=l_com_spt[i][:,2:].reshape(l_com_spt[i].size(0)))
                feat_com_i = 1.0/3*(feat_com_i_0+feat_com_i_1+feat_com_i_2)
                

                # feat_i = torch.cat((feat_i, feat_com_i),1)
                feat_i = feat_com_i
                logit_i_1 = self.classifier_binary(feat_i, vars=fast_weights_2, bn_training=True)
                logit_i_2 = self.classifier_topic(feat_i, vars=fast_weights_3, bn_training=True)
                loss = F.cross_entropy(logit_i_1, y_spt) + self.eta * F.cross_entropy(logit_i_2, y_spt_topic)
                # grad_1 = torch.autograd.grad(loss, fast_weights_1, retain_graph=True)
                # fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_1, fast_weights_1)))
                grad_com_1 = torch.autograd.grad(loss, fast_weights_com_1, retain_graph=True)
                fast_weights_com_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_com_1, fast_weights_com_1)))
                grad_2 = torch.autograd.grad(loss, fast_weights_2)
                fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_2, fast_weights_2)))
                grad_3 = torch.autograd.grad(loss, fast_weights_3)
                fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_3, fast_weights_3)))

            # h_q = torch.zeros((feats_query[i].size(0), self.hidden_size)).to(x_spt.device)
            # feat_q = self.feature_extractor(feats_query[i], fast_weights_1, bn_training=True, hidden=h_q, lens=l_qry[i])
            h_com_q = torch.zeros((feats_com_query[i].size(0), self.hidden_size)).to(x_com_spt.device)
            feat_com_q_0 = self.com_feature_extractor(feats_com_query[i][:,:self.com_pad_size,:], fast_weights_com_1, bn_training=True, hidden=h_com_q, lens=l_com_qry[i][:,0:1].reshape(l_com_qry[i].size(0)))
            feat_com_q_1 = self.com_feature_extractor(feats_com_query[i][:,self.com_pad_size:2*self.com_pad_size,:], fast_weights_com_1, bn_training=True, hidden=h_com_q, lens=l_com_qry[i][:,1:2].reshape(l_com_qry[i].size(0)))
            feat_com_q_2 = self.com_feature_extractor(feats_com_query[i][:,2*self.com_pad_size:,:], fast_weights_com_1, bn_training=True, hidden=h_com_q, lens=l_com_qry[i][:,2:].reshape(l_com_qry[i].size(0)))
            feat_com_q = 1.0/3*(feat_com_q_0+feat_com_q_1+feat_com_q_2)


            # feat_q = torch.cat((feat_q,feat_com_q),1)
            feat_q = feat_com_q
            logits_q = self.classifier_binary(feat_q, fast_weights_2, bn_training=True)
            logits_q_topic = self.classifier_topic(feat_q, fast_weights_3, bn_training=True)

            loss_q = F.cross_entropy(logits_q, y_qry) + self.eta * F.cross_entropy(logits_q_topic, y_qry_topic)

            # logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            # loss_q = F.cross_entropy(logits_q, y_qry)
            k = self.update_step-1
            losses_q[k + 1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

                pred_q_topic = F.softmax(logits_q_topic, dim=1).argmax(dim=1)
                correct_topic = torch.eq(pred_q_topic, y_qry_topic).sum().item()  # convert to numpy
                corrects_topic[k + 1] = corrects_topic[k + 1] + correct_topic

        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)
        accs_topic = np.array(corrects_topic) / (querysz * task_num)

        return accs, accs_topic

    def finetuning(self, x_spt, x_com_spt, y_spt, y_spt_topic, l_spt, l_com_spt, x_qry, x_com_qry, y_qry, y_qry_topic, l_qry, l_com_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]
        corrects_topic = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
        # convs = deepcopy(self.convs)

        # feature_extractor = deepcopy(self.feature_extractor)
        com_feature_extractor = deepcopy(self.com_feature_extractor)
        classifier_binary = deepcopy(self.classifier_binary)
        classifier_topic = deepcopy(self.classifier_topic)

        # 1. run the i-th task and compute loss for k=0
        # with torch.no_grad():
        feats_shot = x_spt  # convs(x_spt)
        feats_query = x_qry  # convs(x_qry)
        feats_com_shot = x_com_spt  # convs(x_spt)
        feats_com_query = x_com_qry  # convs(x_qry)

        # h_0 = torch.zeros((feats_shot.size(0), self.hidden_size)).to(x_spt.device)
        # feat_0 = feature_extractor(feats_shot, hidden=h_0.detach(), lens=l_spt)
        h_com_0 = torch.zeros((feats_com_shot.size(0), self.hidden_size)).to(x_com_spt.device)

        feat_com_0_0 = com_feature_extractor(feats_com_shot[:,:self.com_pad_size,:], hidden=h_com_0.detach(), lens=l_com_spt[:,0:1].reshape(l_com_spt.size(0)))
        feat_com_0_1 = com_feature_extractor(feats_com_shot[:,self.com_pad_size:2*self.com_pad_size,:], hidden=h_com_0.detach(), lens=l_com_spt[:,1:2].reshape(l_com_spt.size(0)))
        feat_com_0_2 = com_feature_extractor(feats_com_shot[:,2*self.com_pad_size:,:], hidden=h_com_0.detach(), lens=l_com_spt[:,2:].reshape(l_com_spt.size(0)))
        feat_com_0 = 1.0/3*(feat_com_0_0+feat_com_0_1+feat_com_0_2)


        # feat_0 = torch.cat((feat_0,feat_com_0),1)
        feat_0 = feat_com_0
        logits_1 = classifier_binary(feat_0)
        logits_2 = classifier_topic(feat_0)
        loss = F.cross_entropy(logits_1, y_spt) + 0.1 * F.cross_entropy(logits_2, y_spt_topic)
        # grad_1 = torch.autograd.grad(loss, feature_extractor.parameters(), retain_graph=True)
        # fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
        #                           zip(grad_1, feature_extractor.parameters().parameters())))
        grad_com_1 = torch.autograd.grad(loss, com_feature_extractor.parameters(), retain_graph=True)
        fast_weights_com_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_com_1, com_feature_extractor.parameters().parameters())))
        grad_2 = torch.autograd.grad(loss, classifier_binary.parameters())
        fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_2, classifier_binary.parameters())))
        grad_3 = torch.autograd.grad(loss, classifier_topic.parameters())
        fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_3, classifier_topic.parameters())))

        # grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # h_q = torch.zeros((feats_query.size(0), self.hidden_size)).to(x_spt.device)
        h_com_q = torch.zeros((feats_com_query.size(0), self.hidden_size)).to(x_com_spt.device)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            # logits_q = net(x_qry, net.parameters(), bn_training=True)
            # feat_q = feature_extractor(feats_query, feature_extractor.parameters(), bn_training=True,
            #                            hidden=h_q.detach(), lens=l_qry)
            feat_com_q_0 = com_feature_extractor(feats_com_query[:,:self.com_pad_size,:], com_feature_extractor.parameters(), bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,0:1].reshape(l_com_qry.size(0)))
            feat_com_q_1 = com_feature_extractor(feats_com_query[:,self.com_pad_size:2*self.com_pad_size,:], com_feature_extractor.parameters(), bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,1:2].reshape(l_com_qry.size(0)))
            feat_com_q_2 = com_feature_extractor(feats_com_query[:,2*self.com_pad_size:,:], com_feature_extractor.parameters(), bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,2:].reshape(l_com_qry.size(0)))
            feat_com_q = 1.0/3*(feat_com_q_0+feat_com_q_1+feat_com_q_2)


            # feat_q = torch.cat((feat_q,feat_com_q),1)
            feat_q = feat_com_q
            logits_q = classifier_binary(feat_q, classifier_binary.parameters(), bn_training=True)
            logits_q_topic = classifier_topic(feat_q, classifier_topic.parameters(), bn_training=True)

            # accuracy
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

            pred_q_topic = F.softmax(logits_q_topic, dim=1).argmax(dim=1)
            correct_topic = torch.eq(pred_q_topic, y_qry_topic).sum().item()  # convert to numpy
            corrects_topic[0] = corrects_topic[0] + correct_topic

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            # feat_q = feature_extractor(feats_query, fast_weights_1, bn_training=True,
            #                            hidden=h_q.detach(), lens=l_qry)

            feat_com_q_0 = com_feature_extractor(feats_com_query[:,:self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,0:1].reshape(l_com_qry.size(0)))
            feat_com_q_1 = com_feature_extractor(feats_com_query[:,self.com_pad_size:2*self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,1:2].reshape(l_com_qry.size(0)))
            feat_com_q_2 = com_feature_extractor(feats_com_query[:,2*self.com_pad_size:,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_q.detach(), lens=l_com_qry[:,2:].reshape(l_com_qry.size(0)))
            feat_com_q = 1.0/3*(feat_com_q_0+feat_com_q_1+feat_com_q_2)


            # feat_q = torch.cat((feat_q, feat_com_q),1)
            feat_q = feat_com_q
            logits_q = classifier_binary(feat_q, fast_weights_2, bn_training=True)
            logits_q_topic = classifier_topic(feat_q, fast_weights_3, bn_training=True)

            # accuracy
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

            pred_q_topic = F.softmax(logits_q_topic, dim=1).argmax(dim=1)
            correct_topic = torch.eq(pred_q_topic, y_qry_topic).sum().item()  # convert to numpy
            corrects_topic[1] = corrects_topic[1] + correct_topic

        for k in range(1, self.update_step_test):
            # feat_i = feature_extractor(feats_shot, fast_weights_1, bn_training=True,
            #                            hidden=h_0.detach(), lens=l_spt)
            feat_com_i_0 = com_feature_extractor(feats_com_shot[:,:self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_0.detach(), lens=l_com_spt[:,0:1].reshape(l_com_spt.size(0)))
            feat_com_i_1 = com_feature_extractor(feats_com_shot[:,self.com_pad_size:2*self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_0.detach(), lens=l_com_spt[:,1:2].reshape(l_com_spt.size(0)))
            feat_com_i_2 = com_feature_extractor(feats_com_shot[:,2*self.com_pad_size:,:], fast_weights_com_1, bn_training=True,
                                       hidden=h_com_0.detach(), lens=l_com_spt[:,2:].reshape(l_com_spt.size(0)))
            feat_com_i = 1.0/3*(feat_com_i_0+feat_com_i_1+feat_com_i_2)


            # feat_i = torch.cat((feat_i,feat_com_i),1)
            feat_i = feat_com_i
            logit_i_1 = classifier_binary(feat_i, fast_weights_2, bn_training=True)
            logit_i_2 = classifier_topic(feat_i, fast_weights_3, bn_training=True)
            loss = F.cross_entropy(logit_i_1, y_spt) + 0.1 * F.cross_entropy(logit_i_2, y_spt_topic)

            # grad_1 = torch.autograd.grad(loss, fast_weights_1, retain_graph=True)
            # fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_1, fast_weights_1)))
            grad_com_1 = torch.autograd.grad(loss, fast_weights_com_1, retain_graph=True)
            fast_weights_com_1 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_com_1, fast_weights_com_1)))
            grad_2 = torch.autograd.grad(loss, fast_weights_2)
            fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_2, fast_weights_2)))
            grad_3 = torch.autograd.grad(loss, fast_weights_3)
            fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_3, fast_weights_3)))

            with torch.no_grad():
                # feat_q = feature_extractor(feats_query, fast_weights_1, bn_training=True,
                #                            hidden=h_q.detach(), lens=l_qry)

                feat_com_q_0 = com_feature_extractor(feats_com_query[:,:self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                           hidden=h_com_q.detach(), lens=l_com_qry[:,0:1].reshape(l_com_qry.size(0)))
                feat_com_q_1 = com_feature_extractor(feats_com_query[:,self.com_pad_size:2*self.com_pad_size,:], fast_weights_com_1, bn_training=True,
                                           hidden=h_com_q.detach(), lens=l_com_qry[:,1:2].reshape(l_com_qry.size(0)))
                feat_com_q_2 = com_feature_extractor(feats_com_query[:,2*self.com_pad_size:,:], fast_weights_com_1, bn_training=True,
                                           hidden=h_com_q.detach(), lens=l_com_qry[:,2:].reshape(l_com_qry.size(0)))
                feat_com_q = 1.0/3*(feat_com_q_0+feat_com_q_1+feat_com_q_2)
                

                # feat_q = torch.cat((feat_q,feat_com_q),1)
                feat_q = feat_com_q
                logits_q = classifier_binary(feat_q, fast_weights_2, bn_training=True)
                logits_q_topic = classifier_topic(feat_q, fast_weights_3, bn_training=True)

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

                pred_q_topic = F.softmax(logits_q_topic, dim=1).argmax(dim=1)
                correct_topic = torch.eq(pred_q_topic, y_qry_topic).sum().item()  # convert to numpy
                corrects_topic[k + 1] = corrects_topic[k + 1] + correct_topic

        del com_feature_extractor, classifier_binary, classifier_topic
        # del feature_extractor, com_feature_extractor, classifier_binary, classifier_topic

        accs = np.array(corrects) / querysz
        accs_topic = np.array(corrects_topic) / querysz

        return accs, accs_topic


class BaseFCN(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """
        :param args:
        """
        super(BaseFCN, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.query
        self.task_num = args.batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.hidden_size = args.hidden_size
        self.base_type = args.base_type

        print('Use MetaLSTM')
        # self.net = LearnerResNet(config, args.imgc, args.imgsz)

        self.feature_extractor = LearnerResNet(config[:-1])
        self.classifier = LearnerResNet(config[-1:])

        # print(self.feature_extractor, self.classifier)

        self.meta_optim = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                            {'params': self.classifier.parameters(), 'lr': self.meta_lr}],
                                           lr=self.meta_lr, weight_decay=0.0005)

        # self.meta_optim = torch.optim.SGD([{'params': self.feature_extractor.parameters()},
        #                                     {'params': self.classifier.parameters(), 'lr': self.meta_lr}],
        #                                    lr=self.meta_lr, weight_decay=0.0005)

    def forward(self, x_spt, y_spt, l_spt, x_qry, y_qry, l_qry):
        """
        setsz = n_way * k_shot
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        # task_num, setsz, len_sent, D_vec = x_spt.size()
        task_num, setsz, seq_, d_ = x_spt.size()
        # print(x_spt.shape)
        # print(l_spt.shape)
        querysz = x_qry.size(1)

        losses_q = [0]  # losses_q[i] is the loss on step i
        corrects = [0]

        feats_shot = []
        feats_query = []
        # l_shot = []
        # l_query = []

        # make a mean

        if self.base_type == 'cnn':
            for i in range(task_num):
                fs = x_spt[i, ...]
                fs_new = []
                for j in range(fs.size(0)):
                    x = torch.mean(fs[j:j + 1, :l_spt[i, j], :], dim=1)
                    fs_new.append(x)

                fs_new = torch.cat(fs_new, dim=0)
                feats_shot.append(fs_new)

                fq = x_qry[i, ...]  # self.convs(x_qry[i, ...])
                fq_new = []
                for j in range(fq.size(0)):
                    x = torch.mean(fq[j:j + 1, :l_qry[i, j], :], dim=1)
                    fq_new.append(x)

                fq_new = torch.cat(fq_new, dim=0)
                feats_query.append(fq_new)
        else:
            for i in range(task_num):
                fs = x_spt[i, ...]  # self.convs(x_spt[i, ...])
                feats_shot.append(fs)
                fq = x_qry[i, ...]  # self.convs(x_qry[i, ...])
                feats_query.append(fq)

        # print(task_num)
        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            feat_0 = feats_shot[i].detach()
            # print(feat_0.size())
            h_0 = torch.zeros((feat_0.size(0), self.hidden_size)).to(x_spt.device)
            feat = self.feature_extractor(feat_0, vars=None, bn_training=True, hidden=h_0.detach(), lens=l_spt[i])
            logits = self.classifier(feat, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            losses_q[0] += loss

        # optimize theta parameters
        loss_q = losses_q[0] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        for i in range(task_num):
            h_q = torch.zeros((feats_query[i].size(0), self.hidden_size)).to(x_spt.device)
            feat_q = self.feature_extractor(feats_query[i], None, bn_training=True, hidden=h_q, lens=l_qry[i])
            logits_q = self.classifier(feat_q, None, bn_training=True)
            # loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[0] = corrects[0] + correct

        accs = np.array(corrects) / (querysz * task_num)

        return accs, accs

    def finetuning(self, x_spt, y_spt, l_spt, x_qry, y_qry, l_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = x_qry.size(0)
        corrects = [0]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
        # convs = deepcopy(self.convs)
        feature_extractor = deepcopy(self.feature_extractor)
        classifier = deepcopy(self.classifier)

        # 1. run the i-th task and compute loss for k=0
        if self.base_type == 'cnn':
            feats_shot = []
            feats_query = []

            for j in range(x_spt.size(0)):
                x = torch.mean(x_spt[j:j + 1, :l_spt[j], :], dim=1)
                feats_shot.append(x)
            feats_shot = torch.cat(feats_shot, dim=0)

            for j in range(x_qry.size(0)):
                x = torch.mean(x_qry[j:j + 1, :l_qry[j], :], dim=1)
                feats_query.append(x)
            feats_query = torch.cat(feats_query, dim=0)

        else:
            feats_shot = x_spt  # convs(x_spt)
            feats_query = x_qry  # convs(x_qry)

        h_0 = torch.zeros((feats_shot.size(0), self.hidden_size)).to(x_spt.device)
        feat_0 = feature_extractor(feats_shot, hidden=h_0.detach(), lens=l_spt)
        logits = classifier(feat_0)
        loss = F.cross_entropy(logits, y_spt)
        grad_1 = torch.autograd.grad(loss, feature_extractor.parameters(), retain_graph=True)
        fast_weights_1 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_1, feature_extractor.parameters().parameters())))
        grad_2 = torch.autograd.grad(loss, classifier.parameters())
        fast_weights_2 = list(map(lambda p: p[1] - self.update_lr * p[0],
                                  zip(grad_2, classifier.parameters())))

        # grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        h_q = torch.zeros((feats_query.size(0), self.hidden_size)).to(x_spt.device)
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            feat_q = feature_extractor(feats_query, fast_weights_1, bn_training=True,
                                       hidden=h_q.detach(), lens=l_qry)
            logits_q = classifier(feat_q, fast_weights_2, bn_training=True)

            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        del feature_extractor, classifier

        accs = np.array(corrects) / querysz

        return accs, accs


def main():
    pass


if __name__ == '__main__':
    main()