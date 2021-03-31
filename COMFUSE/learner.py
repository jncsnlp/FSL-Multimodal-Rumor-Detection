import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LearnerResNet(nn.Module):

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(LearnerResNet, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):

            if name == 'basicblock':
                v1, v2 = self.basic_block_vars(param)
                self.vars.extend(v1)
                self.vars_bn.extend(v2)

            elif name in ['conv2d', 'bn', 'linear']:
                v1, v2 = self.get_vars(name, param)
                self.vars.extend(v1)
                self.vars_bn.extend(v2)

            elif name == 'gru':
                v1, v2 = self.gru_vars(param)
                self.vars.extend(v1)
                self.vars_bn.extend(v2)

            elif name == 'gru_bi':
                # bidirectional GRU
                v1, v2 = self.gru_vars(param)  # forward  direction
                v3, v4 = self.gru_vars(param)  # backward direction
                self.vars.extend(v1)
                self.vars.extend(v3)
                self.vars_bn.extend(v2)
                self.vars_bn.extend(v4)

            elif name == 'gru_multilayer':
                v1, v2 = self.gru_multilayer_vars(param)
                self.vars.extend(v1)
                self.vars_bn.extend(v2)

            elif name == 'gru_bi_multilayer':
                v1, v2 = self.gru_bi_multilayer_vars(param)
                self.vars.extend(v1)
                self.vars_bn.extend(v2)

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def get_vars(self, name, param):

        if name == 'conv2d':
            # [ch_out, ch_in, kernelsz, kernelsz]
            w = nn.Parameter(torch.ones(*param[:4]))
            # gain=1 according to cbfin's implementation
            torch.nn.init.kaiming_normal_(w)
            return [w, nn.Parameter(torch.zeros(param[0]))], []

        elif name == 'linear':
            # [ch_out, ch_in]
            w = nn.Parameter(torch.ones(*param))
            # gain=1 according to cbfinn's implementation
            torch.nn.init.kaiming_normal_(w)
            return [w, nn.Parameter(torch.zeros(param[0]))], []

        elif name == 'bn':
            # print(param, '---bn')

            # [ch_out]
            w = nn.Parameter(torch.ones(param[0]))

            # must set requires_grad=False
            running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
            running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)

            return [w, nn.Parameter(torch.zeros(param[0]))], [running_mean, running_var]


        elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                      'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
            return [], []
        else:
            raise NotImplementedError

    def gru_vars(self, params):
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        hidden_size, input_size = params[:2]

        # rt
        wir = nn.Parameter(torch.ones(hidden_size, input_size))
        torch.nn.init.kaiming_normal_(wir)
        bir = nn.Parameter(torch.zeros(hidden_size))
        whr = nn.Parameter(torch.ones(hidden_size, hidden_size))
        torch.nn.init.kaiming_normal_(whr)
        bhr = nn.Parameter(torch.zeros(hidden_size))

        # zt
        wiz = nn.Parameter(torch.ones(hidden_size, input_size))
        torch.nn.init.kaiming_normal_(wir)
        biz = nn.Parameter(torch.zeros(hidden_size))
        whz = nn.Parameter(torch.ones(hidden_size, hidden_size))
        torch.nn.init.kaiming_normal_(whz)
        bhz = nn.Parameter(torch.zeros(hidden_size))

        # nt
        win = nn.Parameter(torch.ones(hidden_size, input_size))
        torch.nn.init.kaiming_normal_(win)
        bin = nn.Parameter(torch.zeros(hidden_size))
        whn = nn.Parameter(torch.ones(hidden_size, hidden_size))
        torch.nn.init.kaiming_normal_(whn)
        bhn = nn.Parameter(torch.zeros(hidden_size))

        return [wir, bir, whr, bhr, wiz, biz, whz, bhz, win, bin, whn, bhn], []

    def gru_multilayer_vars(self, params):
        hidden_size, input_size, num_layer = params[:3]
        vars, vars_bn = [], []
        for l in range(num_layer):
            if l == 0:
                # first layer have input size
                v1, v2 = self.gru_vars([hidden_size, input_size])
            else:
                # the rest of the layer have hidden size as input
                v1, v2 = self.gru_vars([hidden_size, hidden_size])

            vars.extend(v1)
            vars_bn.extend(v2)
        return vars, vars_bn

    def gru_bi_multilayer_vars(self, params):
        # TODO
        hidden_size, input_size, num_layer = params[:3]
        vars, vars_bn = [], []

        for _ in range(2):
            for l in range(num_layer):
                if l == 0:
                    # first layer have input size
                    v1, v2 = self.gru_vars([hidden_size, input_size])
                else:
                    # the rest of the layer have hidden size as input
                    v1, v2 = self.gru_vars([hidden_size, hidden_size])

                vars.extend(v1)
                vars_bn.extend(v2)

        return vars, vars_bn

    def basic_block_vars(self, params):
        vars, vars_bn = [], []
        planes, inplanes, ksize, ksize, stride, padding, downsample = params
        # print(params)

        v1, v2 = self.get_vars('conv2d', [planes, inplanes, 3, 3])
        vars.extend(v1)
        vars_bn.extend(v2)

        v1, v2 = self.get_vars('bn', [planes])
        vars.extend(v1)
        vars_bn.extend(v2)

        v1, v2 = self.get_vars('conv2d', [planes, planes, 3, 3])
        vars.extend(v1)
        vars_bn.extend(v2)

        v1, v2 = self.get_vars('bn', [planes])
        vars.extend(v1)
        vars_bn.extend(v2)

        if downsample:
            assert planes == 2 * inplanes
            v1, v2 = self.get_vars('conv2d', [planes, inplanes, 1, 1])
            vars.extend(v1)
            vars_bn.extend(v2)

            v1, v2 = self.get_vars('bn', [planes])
            vars.extend(v1)
            vars_bn.extend(v2)

        '''
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        '''
        return vars, vars_bn

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name == 'gru':
                tmp = 'gru:(in:%d, out:%d)' % (param[0], param[1])
                info += tmp + '\n'

            elif name == 'gru_bi':
                tmp = 'gru:(in:%d, out:%d)' % (param[0], param[1])
                info += tmp + '\n'

            elif name == 'gru_multilayer':
                tmp = 'gru_multilayer:(in:%d, layer:%d, out:%d)' % (param[0], param[2], param[1])
                info += tmp + '\n'

            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward_gru(self, x, vars, hidden):
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        wir, bir, whr, bhr, wiz, biz, whz, bhz, win, bin, whn, bhn = vars
        rt = F.linear(x, wir, bir) + F.linear(hidden, whr, bhr)
        rts = torch.sigmoid(rt)
        zt = F.linear(x, wiz, biz) + F.linear(hidden, whz, bhz)
        zts = torch.sigmoid(zt)
        yin = F.linear(x, win, bin)
        qin = F.linear(hidden, whn, bhn)
        nt = yin + rts * qin
        ntt = torch.tanh(nt)
        ht = (1. - zts) * ntt + zts * hidden
        return ht

    def forward_func(self, x, name, vars, vars_bn, params, bn_training=True, hidden=None):
        if name == 'conv2d':
            w, b = vars[0], vars[1]
            # remember to keep synchrozied of forward_encoder and forward_decoder!
            x = F.conv2d(x, w, b, stride=params[4], padding=params[5])
            # print(name, param, '\tout:', x.shape)
        elif name == 'linear':
            w, b = vars[0], vars[1]
            x = F.linear(x, w, b)
        elif name == 'gru':
            # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
            x = self.forward_gru(x, vars, hidden)
        elif name == 'bn':
            w, b = vars[0], vars[1]
            running_mean, running_var = vars_bn[0], vars_bn[1]
            x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
        elif name == 'flatten':
            # print(x.shape)
            x = x.view(x.size(0), -1)
        elif name == 'relu':
            x = F.relu(x, inplace=True)
        elif name == 'leakyrelu':
            x = F.leaky_relu(x, negative_slope=params[0], inplace=params[1])
        elif name == 'max_pool2d':
            x = F.max_pool2d(x, params[0], params[1], params[2])
        elif name == 'avg_pool2d':
            x = F.avg_pool2d(x, params[0], params[1], params[2])
        elif name == 'tanh':
            x = torch.tanh(x)
        elif name == 'sigmoid':
            x = torch.sigmoid(x)
        else:
            raise NotImplementedError
        return x

    def forward(self, x, vars=None, bn_training=True, hidden=None, lens=None):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            # print(name)

            if name == 'conv2d':
                # self.forward_func(name, vars, )
                # w, b = vars[idx], vars[idx + 1]
                # x = F.conv2d(x, w, b, stride=param[4], padding=param[5])

                w, b = vars[idx], vars[idx + 1]
                x = self.forward_func(x, name, [w, b], None, param)
                idx += 2

            elif name == 'linear':
                # w, b = vars[idx], vars[idx + 1]
                # x = F.linear(x, w, b)

                w, b = vars[idx], vars[idx + 1]
                x = self.forward_func(x, name, [w, b], None, param)
                idx += 2

            elif name == 'gru_mean':
                assert x.dim() == 3
                # print(x.size(), lens.size())
                assert x.size(0) == lens.size(0)
                assert hidden is not None and lens is not None
                wbs = vars[idx:idx + 12]

                h = hidden
                outputs = []
                L = x.size(1)
                for t in range(L):
                    h = self.forward_func(x[:, t, :], name, wbs, None, param, bn_training, hidden=h)
                    h2 = h.unsqueeze(0)
                    outputs.append(h2)

                outputs = torch.cat(outputs, dim=0)
                # print(outputs.size())

                outputs_new = []
                for k in range(lens.size(0)):
                    L = lens[k]
                    out_feat = torch.mean(outputs[:L, k:k + 1, :], dim=0)
                    outputs_new.append(out_feat)
                x = torch.cat(outputs_new, dim=0)
                idx += 12

            elif name == 'gru':
                # assert x.dim() == 3
                # print(x.size(), lens.size())
                # assert x.size(0)==lens.size(0)
                # assert hidden is not None and lens is not None
                wbs = vars[idx:idx + 12]

                h = hidden
                outputs = [h]
                # L = x.size(1)
                L = min(x.size(1), torch.max(lens).item())  # the longest sequence

                for t in range(L):
                    h = self.forward_func(x[:, t, :], name, wbs, None, param, bn_training, hidden=h)
                    outputs.append(h)

                outputs_new = []
                for k in range(lens.size(0)):
                    lk = lens[k]
                    outputs_new.append(outputs[lk][k:k + 1, :])
                x = torch.cat(outputs_new, dim=0)
                idx += 12

            elif name == 'gru_bi':
                # assert x.dim() == 3
                # print(x.size(), lens.size())
                # assert x.size(0)==lens.size(0)
                # assert hidden is not None and lens is not None

                ########################
                ####### forward ########
                ########################
                wbs = vars[idx:idx + 12]
                h = hidden.detach()
                outputs = [h]
                # L = x.size(1)
                L = min(x.size(1), torch.max(lens).item())  # the longest sequence

                for t in range(L):
                    h = self.forward_func(x[:, t, :], "gru", wbs, None, param, bn_training, hidden=h)
                    outputs.append(h)

                outputs_new = []
                for k in range(lens.size(0)):
                    lk = lens[k]
                    outputs_new.append(outputs[lk][k:k + 1, :])
                x1 = torch.cat(outputs_new, dim=0)
                idx += 12

                ########################
                ####### backward ########
                ########################
                wbs = vars[idx:idx + 12]
                h = hidden.detach()
                outputs = [h]
                # L = x.size(1)

                for t in range(L):
                    h = self.forward_func(x[:, L - t - 1, :], "gru", wbs, None, param, bn_training, hidden=h)
                    outputs.append(h)

                # outputs_new = []
                # for k in range(lens.size(0)):
                #     lk = lens[k]
                #     outputs_new.append(outputs[lk][k:k + 1, :])
                # x = torch.cat(outputs_new, dim=0)

                x2 = outputs[-1]
                # print(x1.size(), x2.size())
                idx += 12

                x = 0.5 * (x1 + x2)

            elif name == 'gru_multilayer':
                # multi-layer GRU

                hidden_size, input_size, num_layer = param[:3]
                # assert hidden.dim() == 2  # hidden should be (B, hidden_size)

                L = torch.max(lens).item()  # the longest sequence
                outputs_all = []

                for l in range(num_layer):
                    wbs = vars[idx:idx + 12]
                    idx += 12

                    if l == 0:
                        h = hidden
                    else:
                        h = torch.zeros_like(hidden).to(hidden.device)
                    outputs = [h]
                    for t in range(L):
                        if l == 0:
                            h = self.forward_func(x[:, t, :], 'gru', wbs, None, param, bn_training, hidden=outputs[-1])
                        else:
                            h = self.forward_func(outputs_all[-1][t + 1], 'gru', wbs, None, param, bn_training,
                                                  hidden=outputs[-1])
                        outputs.append(h)

                    if l < num_layer - 1:
                        outputs_all.append(outputs)
                    else:
                        outputs_new = []
                        for k in range(lens.size(0)):
                            lk = lens[k]
                            outputs_new.append(outputs[lk][k:k + 1, :])
                        x = torch.cat(outputs_new, dim=0)

            elif name == 'gru_bi_multilayer':
                hidden_size, input_size, num_layer = param[:3]

                ########################
                ####### forward ########
                ########################
                # L = x.size(1)
                L = torch.max(lens).item()  # the longest sequence
                outputs_all = []

                for l in range(num_layer):
                    wbs = vars[idx:idx + 12]
                    idx += 12
                    if l == 0:
                        h = hidden.detach()
                    else:
                        h = torch.zeros_like(hidden).to(hidden.device)

                    outputs = [h]
                    for t in range(L):
                        if l == 0:
                            h = self.forward_func(x[:, t, :], 'gru', wbs, None, param, bn_training, hidden=outputs[-1])
                        else:
                            h = self.forward_func(outputs_all[-1][t + 1], 'gru', wbs, None, param, bn_training,
                                                  hidden=outputs[-1])
                        outputs.append(h)

                    if l < num_layer - 1:
                        outputs_all.append(outputs)
                    else:
                        outputs_new = []
                        for k in range(lens.size(0)):
                            lk = lens[k]
                            outputs_new.append(outputs[lk][k:k + 1, :])
                        x1 = torch.cat(outputs_new, dim=0)

                ########################
                ####### backward ########
                ########################
                outputs_back = []

                for l in range(num_layer):
                    wbs = vars[idx:idx + 12]
                    idx += 12
                    if l == 0:
                        h = hidden.detach()
                    else:
                        h = torch.zeros_like(hidden).to(hidden.device)

                    outputs = [h]
                    for t in range(L):
                        if l == 0:
                            h = self.forward_func(x[:, L - t - 1, :], 'gru', wbs, None, param, bn_training,
                                                  hidden=outputs[-1])
                        else:
                            h = self.forward_func(outputs_back[-1][t + 1], 'gru', wbs, None, param, bn_training,
                                                  hidden=outputs[-1])
                        outputs.append(h)

                    if l < num_layer - 1:
                        outputs_back.append(outputs)
                    else:
                        x2 = outputs[-1]

                x = 0.5 * (x1 + x2)


            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = self.forward_func(x, name, [w, b], [running_mean, running_var], param, bn_training)
                # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'basicblock':
                # TODO, check param

                planes, inplanes, ksize, ksize, stride, padding, downsample = param
                if downsample:
                    stride0 = 2
                else:
                    stride0 = 1

                residual = x

                # conv1
                w, b = vars[idx], vars[idx + 1]
                out = self.forward_func(x, 'conv2d', [w, b], None, [planes, inplanes, ksize, ksize, stride0, padding])
                idx += 2

                # bn1
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                out = self.forward_func(out, 'bn', [w, b], [running_mean, running_var], None, bn_training)
                # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

                out = F.relu(out, inplace=True)

                # conv2
                w, b = vars[idx], vars[idx + 1]
                out = self.forward_func(out, 'conv2d', [w, b], None, param)
                idx += 2

                # bn2
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                out = self.forward_func(out, 'bn', [w, b], [running_mean, running_var], None, bn_training)
                # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

                # if downsample ??
                # TODO, check param
                if param[-1]:
                    # conv2
                    w, b = vars[idx], vars[idx + 1]
                    # no bias parameter
                    residual = self.forward_func(x, 'conv2d', [w, b], None, [planes, inplanes, 1, 1, 2, 0])
                    idx += 2

                    # bn2
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    residual = self.forward_func(residual, 'bn', [w, b], [running_mean, running_var], None, bn_training)
                    # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                # print(out.size(), residual.size())
                out += residual
                x = F.relu(out, inplace=True)

            else:
                x = self.forward_func(x, name, None, None, param)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
