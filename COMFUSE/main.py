import os
import torch
import random
import argparse
import numpy as np
import datetime
from torch.utils.data import DataLoader
from selfdropout import rnddrop_2
from config import make_lstm_multi_task_config, make_bilstm_multi_task_config, make_bilstm_multi_layer_multi_task_config
from agent_base import MetaLSTMMultiTask
from rumor_dataset import CategoriesSampler, wb_rumor_fsl_dataset


def train(args):
    run_id = datetime.datetime.now().strftime('%m-%d-%H-%M')
    logdir = os.path.join("models", run_id)

    # avoid overriding
    os.makedirs(logdir, exist_ok=False)
    print("Model dir {}".format(logdir))

    # ============= Training =============

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(split_data_files)
    b_size = args.batch_size
    n_worker = 4 if torch.cuda.is_available() else 1
    task_num = args.batch_size
    n_topic = args.topic
    n_way = args.way

    # how many episodes to train/test
    total_epoch = 20
    num_batch_train = 200
    num_batch_test = 100

    # datasets
    print(n_way * n_topic, args.shot + args.query)
    # return: feats, lb, lns, feats_com, lns_com
    train_set = wb_rumor_fsl_dataset('train', args.split_number, args.model, args.featstype, args.data_path,
                                     args.pad_size, args.com_pad_size, args.bert_path)
    train_sampler = CategoriesSampler(train_set.all_labels,
                                      num_batch_train * b_size, n_way * n_topic, args.shot + args.query)
    # print(train_sampler)
    train_batchsampler = torch.utils.data.BatchSampler(train_sampler, task_num, drop_last=True)

    val_set = wb_rumor_fsl_dataset('dev', args.split_number, args.model, args.featstype, args.data_path, args.pad_size, args.com_pad_size,
                                   args.bert_path)
    val_sampler = CategoriesSampler(val_set.all_labels, num_batch_test, n_way * n_topic, args.shot + args.query)
    val_batchsampler = torch.utils.data.BatchSampler(val_sampler, 1, drop_last=True)

    # n_worker = 1  # --------------------- remove this after debugging
    # trainloader = data.DataLoader(train_set, batch_sampler=train_sampler, num_workers=n_worker, pin_memory=True)
    # valloader = data.DataLoader(val_set, batch_sampler=val_sampler, num_workers=n_worker, pin_memory=True)

    trainloader = DataLoader(train_set, batch_sampler=train_batchsampler, num_workers=n_worker, pin_memory=True)
    valloader = DataLoader(val_set, batch_sampler=val_batchsampler, num_workers=n_worker, pin_memory=True)

    print('fix val set for all epochs')
    val_sampler.mode = 'probe'
    val_set.mode = 'dummy'
    # make one pass of dataset, and internally keep the indices
    # see rumor_dataset.py -> CategoriesSampler -> __iter__() -> probe
    for x in valloader:
        pass
    # val_sampler will use this fixed set to evaluate model
    val_sampler.mode = 'fix'
    val_set.mode = 'norm'
    print('fixed val set has %d batches' % (len(val_sampler.fixed_batches),))

    ########################################
    # Setup Model in agent_base.py
    ########################################
    n_way = args.way
    k_shot = args.shot
    n_query = args.query

    if args.gru_type == "gru":
        config = make_lstm_multi_task_config(args.way, args.hidden_size, n_topic)  # input dim=784
    elif args.gru_type == "gru_bi":
        config = make_bilstm_multi_task_config(args.way, args.hidden_size, n_topic)
    elif args.gru_type == "gru_bi_mult":
        assert args.gru_num_layer > 1, "GRU_BI_MULT is a multilayer (>1) GRU, set num_layer to 2"
        config = make_bilstm_multi_layer_multi_task_config(args.way, args.hidden_size, args.gru_num_layer, n_topic)
    else:
        raise Exception("Not Implemented Error")

    model = MetaLSTMMultiTask(args, config)
    model = model.to(device)

    # check dropout arguments
    drop_type = args.droptype
    drop_rate = args.droprate
    assert drop_type in [0, 1, 2, 3], 'invalid dropout type'
    if drop_type > 0:
        assert 0 < drop_rate < 1, 'invalid dropout rate'

    ##################
    # resume training
    ##################

    if len(args.pretrain_dir) > 0:
        model_path = os.path.join(args.pretrain_dir, 'best_model.pt')
        print('--------------------------')
        print('Load pre-trained model %s' % (model_path,))
        print('--------------------------')
        model.load_state_dict(torch.load(model_path), strict=True)

    # Generate the labels for train set of the episodes
    # label_shot = torch.arange(n_way).repeat(n_topic).repeat(k_shot)

    t1 = []
    t2 = []

    for i in range(n_topic):
        for j in range(n_way):
            t1.append(j)

    for i in range(n_topic):
        for j in range(n_way):
            t2.append(i)

    # print(t1, t2)

    label_shot = torch.tensor(t1).repeat(k_shot)  # [(0,1),(0,1),(0,1)]... there are k_shots
    label_shot = label_shot.to(device).long()
    label_shot_topic = torch.tensor(t2).repeat(k_shot)  # [(0,0), (1,1), (2,2)] .... repeat k_shots
    label_shot_topic = label_shot_topic.to(device).long()
    p = n_topic * n_way * k_shot

    label_query = torch.tensor(t1).repeat(n_query)
    label_query = label_query.to(device).long()
    label_query_topic = torch.tensor(t2).repeat(n_query)
    label_query_topic = label_query_topic.to(device).long()

    print(label_shot, label_shot_topic)
    print(label_query, label_query_topic)

    # print('label_shot size', label_shot.size())
    # print('label_query size', label_query.size())

    #####################
    ### main function ###
    #####################
    best_val_acc = 0.0
    best_val_epoch = -1
    for epoch in range(total_epoch):

        print('Epoch %d/%d' % (epoch, total_epoch))

        acc_clients = []
        acc_topic_clients = []
        for ix, data in enumerate(trainloader):
            feat, raw_label, seq_len, feat_com, seq_len_com = data
            # print(feat.size())
            if drop_type > 0:
                feat = rnddrop_2(feat, drop_rate, drop_type, False)
                feat_com = rnddrop_2(feat_com, drop_rate, drop_type, False)
            feat = feat.to(device)
            feat_com = feat_com.to(device)
            x_shot, x_qry = feat[:, :p, 0, ...], feat[:, p:, 0, ...]  # split to adaptation data and meta-learning data
            x_com_shot, x_com_qry = feat_com[:, :p, 0, ...], feat_com[:, p:, 0, ...]  # split to adaptation data and meta-learning data
            # print(p)
            # print("x_shot", x_shot.shape)
            # print("x_qry", x_qry.shape)
            # Generate the labels for test set of the episodes during meta-train updates
            y_shot = label_shot.detach()
            y_shot_topic = label_shot_topic.detach()
            y_qry = label_query.detach()
            y_qry_topic = label_query_topic.detach()
            # print(y_qry.size())

            # split to get len of data and meta-learning data
            # print(seq_len.shape)
            l_spt = seq_len[:, :p]
            l_qry = seq_len[:, p:]
            l_com_spt = seq_len_com[:, :p]
            l_com_qry = seq_len_com[:, p:]
            # print(l_spt)
            # print(l_qry)
            # print("l_spt", l_spt.shape)
            # print("l_qry", l_qry.shape)

            accs, accs_topic = model(x_shot, x_com_shot, y_shot, y_shot_topic, l_spt, l_com_spt, x_qry, x_com_qry, y_qry, y_qry_topic, l_qry, l_com_qry)
            acc_clients.append(accs[-1])
            acc_topic_clients.append(accs_topic[-1])

            if ix % 20 == 0:
                print('Train step %d/%d' % (ix, len(trainloader)),
                      '  training acc: %.4f,  topic acc: %.4f' % (accs[-1], accs_topic[-1]))

        print('Training avg acc: %.4f, topic acc: %.4f' % (np.mean(acc_clients), np.mean(acc_topic_clients)))

        # evaluation
        if epoch % args.val_frequency == 1:
            all_accs = []
            all_topic_accs = []
            accs_all_test = []

            # print('Test dataloader', len(db_test), len(db_test.dataset)) # 100,100

            for ix, data in enumerate(valloader):
                feat, raw_label, seq_len, feat_com, seq_len_com = data
                feat = feat.to(device)
                feat_com = feat_com.to(device)

                # in validation and testing, batch size is one
                assert feat.size(0) == 1
                assert feat_com.size(0) == 1
                # label = label.to(device)  # ignore raw_label, see rumor_dataset.py, __getitem__ last line
                x_shot, x_qry = feat[0, :p, 0, ...], feat[0, p:, 0, ...]
                x_com_shot, x_com_qry = feat_com[0, :p, 0, ...], feat_com[0, p:, 0, ...]
                # Generate the labels for test set of the episodes during meta-train updates
                y_shot = label_shot.detach()
                y_shot_topic = label_shot_topic.detach()
                y_qry = label_query.detach()
                y_qry_topic = label_query_topic.detach()

                l_spt = seq_len[0, :p]
                l_qry = seq_len[0, p:]
                l_com_spt = seq_len_com[0, :p]
                l_com_qry = seq_len_com[0, p:]

                ###########################
                # finetuning on query set
                ###########################
                # print("l_spt",l_spt.shape)
                accs, accs_topic = model.finetuning(x_shot, x_com_shot, y_shot, y_shot_topic, l_spt, l_com_spt, x_qry, x_com_qry, y_qry, y_qry_topic,
                                                    l_qry, l_com_qry)
                accs_all_test.append(accs)
                all_accs.append(accs[-1])
                all_topic_accs.append(accs_topic[-1])

                if ix % 40 == 0:
                    print('  [Ep %d/%d] Test acc %.4f, topic acc %.4f' %
                          (ix, len(valloader), np.mean(all_accs), np.mean(all_topic_accs)),
                          np.array(accs_all_test).mean(axis=0).astype(np.float16)[::4])

            avg_test_acc = np.mean(all_accs)
            avg_test_topic_acc = np.mean(all_topic_accs)
            print('Testing avg acc: %.4f, topic acc: %.4f' % (avg_test_acc, avg_test_topic_acc))
            if best_val_acc < avg_test_acc:
                best_val_acc = avg_test_acc
                best_val_epoch = epoch
            print('Best testing acc: %.4f at epoch %d' % (best_val_acc, best_val_epoch))

            # Save model every 5 epochs
            model_path = os.path.join(logdir, 'model_%02d.pt' % (epoch,))
            torch.save(model.state_dict(), model_path)
            print('Save model to %s' % (model_path,))

            # save best model
            if best_val_epoch == epoch:
                model_path = os.path.join(logdir, 'best_model.pt')
                torch.save(model.state_dict(), model_path)


def test(args):
    model_dir = args.model_dir
    assert os.path.isdir(model_dir)
    model_path = os.path.join(model_dir, 'best_model.pt')
    assert os.path.isfile(model_path)
    print('Load model %s' % (model_path))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(split_data_files)
    # b_size = args.batch_size
    n_worker = 4 if torch.cuda.is_available() else 1
    num_batch_test = 100
    n_way = args.way
    k_shot = args.shot
    n_query = args.query
    n_topic = args.topic

    # datasets
    test_set = wb_rumor_fsl_dataset('test', args.split_number, args.model, args.featstype, args.data_path,
                                    args.pad_size, args.com_pad_size, args.bert_path)
    test_sampler = CategoriesSampler(test_set.all_labels, num_batch_test, n_way * n_topic, args.shot + args.query)
    test_batchsampler = torch.utils.data.BatchSampler(test_sampler, 1, drop_last=True)
    testloader = DataLoader(test_set, batch_sampler=test_batchsampler, num_workers=n_worker, pin_memory=True)

    print('Test set should be fixed by setting the same seed.')

    ########################################
    # Setup Model in agent_base.py
    ########################################

    if args.gru_type == "gru":
        config = make_lstm_multi_task_config(args.way, args.hidden_size, n_topic)  # input dim=784
    elif args.gru_type == "gru_bi":
        config = make_bilstm_multi_task_config(args.way, args.hidden_size, n_topic)
    elif args.gru_type == "gru_bi_mult":
        assert args.gru_num_layer > 1, "GRU_BI_MULT is a multilayer (>1) GRU, set num_layer to 2"
        config = make_bilstm_multi_layer_multi_task_config(args.way, args.hidden_size, args.gru_num_layer, n_topic)
    else:
        raise Exception("Not Implemented Error")

    model = MetaLSTMMultiTask(args, config)
    model = model.to(device)

    ##################
    # resume training
    ##################

    print('--------------------------')
    print('Load trained model %s' % (model_path,))
    print('--------------------------')
    model.load_state_dict(torch.load(model_path), strict=True)

    # Generate the labels for train set of the episodes
    t1 = []
    t2 = []

    for i in range(n_topic):
        for j in range(n_way):
            t1.append(j)

    for i in range(n_topic):
        for j in range(n_way):
            t2.append(i)

    label_shot = torch.tensor(t1).repeat(k_shot)  # [(0,1),(0,1),(0,1)]... there are k_shots
    label_shot = label_shot.to(device).long()
    label_shot_topic = torch.tensor(t2).repeat(k_shot)  # [(0,0), (1,1), (2,2)] .... repeat k_shots
    label_shot_topic = label_shot_topic.to(device).long()
    p = n_topic * n_way * k_shot

    label_query = torch.tensor(t1).repeat(n_query)
    label_query = label_query.to(device).long()
    label_query_topic = torch.tensor(t2).repeat(n_query)
    label_query_topic = label_query_topic.to(device).long()

    print('label_shot size', label_shot.size())
    print('label_query size', label_query.size())

    #####################
    ### main function ###
    #####################

    all_accs = []
    all_topic_accs = []
    accs_all_test = []

    # print('Test dataloader', len(db_test), len(db_test.dataset)) # 100,100

    for ix, data in enumerate(testloader):
        feat, raw_label, seq_len, feat_com, seq_len_com = data
        feat = feat.to(device)
        feat_com = feat_com.to(device)

        # in validation and testing, batch size is one
        assert feat.size(0) == 1
        # label = label.to(device)  # ignore raw_label, see rumor_dataset.py, __getitem__ last line
        x_shot, x_qry = feat[0, :p, 0, ...], feat[0, p:, 0, ...]
        x_com_shot, x_com_qry = feat_com[0, :p, 0, ...], feat_com[0, p:, 0, ...]
        # x_shot, x_qry = feat[0, :p, ...], feat[0, p:, ...]
        # print("x_shot:", x_shot.shape)
        # Generate the labels for test set of the episodes during meta-train updates
        y_shot = label_shot.detach()
        y_shot_topic = label_shot_topic.detach()
        y_qry = label_query.detach()
        y_qry_topic = label_query_topic.detach()
        # print(y_qry.size())

        l_spt = seq_len[0, :p]
        l_qry = seq_len[0, p:]
        l_com_spt = seq_len_com[0, :p]
        l_com_qry = seq_len_com[0, p:]

        ###########################
        # finetuning on query set
        ###########################
        acc, acc_topic = model.finetuning(x_shot, x_com_shot, y_shot, y_shot_topic, l_spt, l_com_spt, x_qry, x_com_qry, y_qry, y_qry_topic, l_qry, l_com_qry)
        accs_all_test.append(acc)
        all_accs.append(acc[-1])
        all_topic_accs.append(acc_topic[-1])

        if ix % 40 == 0:
            print('  [Ep %d/%d] Test acc %.4f, topic acc: %.4f' %
                  (ix, len(testloader), np.mean(all_accs), np.mean(all_topic_accs)),
                  np.array(accs_all_test).mean(axis=0).astype(np.float16)[::4])

    avg_test_acc = np.mean(all_accs)
    avg_topic_test_acc = np.mean(all_topic_accs)

    print('Testing avg acc: %.4f, topic acc: %.4f' % (avg_test_acc, avg_topic_test_acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    # about model
    parser.add_argument('--model', type=str, default="bert", help="bert")
    parser.add_argument('--dataset', type=str, default="weibo", choices="weibo|pheme")

    parser.add_argument('--gru_type', type=str, default="gru_bi_mult", choices=["gru", "gru_bi", "gru_bi_mult"],
                        help="gru|gru_bidirection")
    parser.add_argument('--gru_num_layer', type=int, default=2, help="num of gru hidden layers")

    # about feature type
    parser.add_argument('--featstype', type=str, default="emb_outs", help="emb_outs")
    parser.add_argument('--droptype', type=int, default=1, help="0-nodrop|1-drop word|2-drop dim|3-drop both")
    parser.add_argument('--droprate', type=float, default=0.3, help="dropout rate")

    # about path
    parser.add_argument('--ph', type=int, default=0, choices=[0, 1], help='train|test')
    parser.add_argument('--pretrain_dir', type=str, default='', help='path of models')
    parser.add_argument('--is_seg', type=int, default=0, choices=[0, 1], help='classification|segmentation')
    parser.add_argument('--model_dir', type=str, default='', help='path of models')

    # about training
    # parser.add_argument("--config", type=str, default="configs/mrms_fsl.yml", help="Configuration file to use", )
    parser.add_argument("--gpu", type=str, default="0", help="Used GPUs")
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of tasks')
    parser.add_argument('--val_frequency', type=int, default=5, help="Validate every 50 episodes")
    parser.add_argument('--split_number', type=int, default=0, help='Cross-validation split number 0,1,2')
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=20)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=30)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=128)

    # about task
    parser.add_argument('--way', type=int, default=2)
    parser.add_argument('--topic', type=int, default=3)  # how many topics to sample at a same time as multi-tasking
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=9, help='number of query per class')

    args = parser.parse_args()

    # Set the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # assert args.way == 6

    # Setup seeds
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == 'weibo':
        args.data_path = './DataSet_pair_comments/'
        args.pad_size = 100
        args.com_pad_size = 32
        args.eta = 0.1
        assert args.way == 2 and args.topic == 3
        args.bert_path = './bert_pretrain'
    elif args.dataset == 'pheme':
        args.data_path = './Pheme_DataSet_Pair_comments/'
        args.pad_size = 48
        args.com_pad_size = 48
        args.eta = 0.1
        assert args.way == 2 and args.topic == 2
        args.bert_path = 'bert-base-uncased'
        args.update_step = 10
        args.update_step_test = 10
    else:
        assert 1 == 2

    if args.ph == 0:
        train(args)
    else:
        # need to provide existing model path
        assert len(args.model_dir) > 0
        test(args)
