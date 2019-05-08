from torch import nn
from torch.autograd import Variable
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,batch_size=32,out_='one',
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, train_type='bi',
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.hidden_dim = 512
        self.out_ = out_
        self.low_dim = 512
        self.type = train_type

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        self.feature_dim = self._prepare_tsn(num_class)

        self.drop = nn.Dropout(p=dropout)

        if self.type == 'bi':
            self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True, bidirectional=True, dropout=0.8).cuda()
        elif self.type == 'res':
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm1d(self.low_dim)
            self.lstm_residual = nn.LSTM(self.low_dim, self.low_dim, batch_first=True, bidirectional=True, dropout=0.8).cuda()
            self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.8).cuda()
        elif self.type == 'dense':
            self.lstm_d1 = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTM(self.feature_dim+self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.new_fc_ = nn.Linear(self.feature_dim+self.hidden_dim*2, num_class)
        elif self.type == 'dense_add':
            self.reduce_fc = nn.Linear(self.feature_dim, self.hidden_dim)
            self.lstm_d1 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.new_fc_ = nn.Linear(self.hidden_dim, num_class)
        elif self.type == 'bi_dense' or self.type == 'bi_dense_v2':
            self.lstm_d1 = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True,dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTM(self.feature_dim+self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.lstm_d1_reverse = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True,dropout=0.8).cuda()
            self.lstm_d2_reverse = nn.LSTM(self.feature_dim+self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.new_fc_ = nn.Linear(self.feature_dim+self.hidden_dim*4, num_class)
            #self.conv1d = nn.Conv1d(self.num_segments, 1, 1, stride=1)
        elif self.type == 'multi_dense':
            self.lstm_d1 = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True,dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTM(self.feature_dim+self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.lstm_d3 = nn.LSTM(self.feature_dim+2*self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.lstm_d4 = nn.LSTM(self.feature_dim+3*self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.new_fc_ = nn.Linear(self.feature_dim+self.hidden_dim*4, num_class)
        elif self.type == 'dense_v2':
            self.lstm_d1 = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True,dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTM(self.hidden_dim+self.feature_dim, self.hidden_dim, batch_first=True, dropout=0.8).cuda()
            self.new_fc_ = nn.Linear(self.hidden_dim*2+self.feature_dim, num_class)
        elif self.type == 'dense_attention':
            self.lstm_d1 = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True,dropout=0.8).cuda()
            self.lstm_d2 = nn.LSTMCell(self.feature_dim+self.hidden_dim, self.hidden_dim).cuda()
            self.new_fc_ = nn.Linear(self.feature_dim+self.hidden_dim*2, num_class)
            self.lstm_attention = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True, bidirectional=True).cuda()
            self.at_fc = nn.Linear(self.hidden_dim*2, 1)
            self.sigmoid = nn.Sigmoid()

        # normal(self.new_fc_.weight, 0, 0.001)
        # constant(self.new_fc_.bias, 0)
        #normal(self.reduce_fc.weight, 0, 0.001)
        #constant(self.reduce_fc.bias, 0)
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=0))
            self.new_fc = nn.Linear(feature_dim, num_class)
        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0) 
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model or 'densenet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            if 'densenet' in base_model:
                self.base_model.last_layer_name = 'classifier'
            else:
                self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            for para in self.base_model.parameters():
                para.requires_grad = False

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            for para in self.base_model.parameters():
                para.requires_grad = False

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    def get_train_param(self):
        para = []
        if self.type == 'bi':
            para.append({'params':self.lstm.parameters()})
            para.append({'params':self.new_fc_.parameters()})
        elif self.type == 'res':
            para.append({'params':self.lstm.parameters()})
            para.append({'params':self.new_fc_.parameters()})
            para.append({'params':self.lstm_residual.parameters()})
            para.append({'params':self.new_fc.parameters()})
            para.append({'params':self.bn.parameters()})
        elif self.type == 'dense' or self.type == 'dense_v2' or self.type == 'bi_dense' or self.type == 'bi_dense_v2':
            para.append({'params':self.new_fc_.parameters()})
            para.append({'params':self.lstm_d1.parameters()})
            para.append({'params':self.lstm_d2.parameters()})
            if self.type == 'bi_dense' or self.type == 'bi_dense_v2':
                #para.append({'params':self.conv1d.parameters()})
                para.append({'params':self.lstm_d1_reverse.parameters()})
                para.append({'params':self.lstm_d2_reverse.parameters()})
            if self.type == 'dense_v2':
                para.append({'params':self.lstm_d3.parameters()})
        elif self.type == 'dense_add':
            para.append({'params':self.new_fc_.parameters()})
            para.append({'params':self.lstm_d1.parameters()})
            para.append({'params':self.lstm_d2.parameters()})
        elif self.type == 'multi_dense' :
            para.append({'params':self.new_fc_.parameters()})
            para.append({'params':self.lstm_d1.parameters()})
            para.append({'params':self.lstm_d2.parameters()})
            para.append({'params':self.lstm_d3.parameters()})
            para.append({'params':self.lstm_d4.parameters()})

        elif self.type == 'dense_attention':
            para.append({'params':self.lstm_d1.parameters()})
            para.append({'params':self.new_fc_.parameters()})
            para.append({'params':self.lstm_d2.parameters()})
            para.append({'params':self.lstm_attention.parameters()})
            para.append({'params':self.at_fc.parameters()})
    
        return para

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):

                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        fc_normal_weight = []
        fc_normal_bias = []

        bn = []
        lstm = []

        normal_count = 0
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_count += 1
                fc_normal_weight.append(ps[0])
                if len(ps) == 2:
                    fc_normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            # elif isinstance(m, torch.nn.LSTM):
            #     normal_bias_.extend(list(m.parameters())[2:])
            #     normal_weight_.extend(list(m.parameters())[:2])
            # elif isinstance(m, torch.nn.LSTM):
            #     normal_bias_.extend(list(m.parameters())[1], list(m.parameters())[3])
            #     normal_weight_.extend(list(m.parameters())[0], list(m.parameters())[2])
            # elif len(m._modules) == 0:
            #     if len(list(m.parameters())) > 0:
            #         raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,'lr':0.0001,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0, 'lr':0.0001,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'lr':0.0001,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'lr':0.0001,
             'name': "normal_bias"},
            {'params': fc_normal_weight, 'lr_mult': 1, 'decay_mult': 1, 
             'name': "fc_normal_weight"},
            {'params': fc_normal_bias, 'lr_mult': 2, 'decay_mult': 0, 
             'name': "fc_normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'lr':0.0001,
             'name': "BN scale/shift"},
        ]
        # return [{'params': normal_weight_, 'lr_mult': 1, 'decay_mult': 1,
        #       'name': "normal_weight_"},
        #      {'params': normal_bias_, 'lr_mult': 2, 'decay_mult': 0,
        #       'name': "normal_bias_"},
        #       {'params': list(self.new_fc_.parameters())[0], 'lr_mult': 1, 'decay_mult': 1,
        #       'name': "normal_weight"},
        #      {'params': list(self.new_fc_.parameters())[1], 'lr_mult': 2, 'decay_mult': 0,
        #       'name': "normal_bias"},]


    def dense(self, base_out):
   
        base_out = base_out.view(-1, self.num_segments, self.feature_dim)

        output , _ = self.lstm_d1(base_out)
        dense1 = torch.cat((base_out, output), 2)

        output, _ = self.lstm_d2(dense1)
        output = torch.cat((dense1, output), 2)
        output = output[:, -1, :]

        #dropout
        # output = self.drop(output)
        out = self.new_fc_(output)
        
        return out

    def dense_add(self, base_out):

        base_out = base_out.view(-1, self.num_segments, self.feature_dim)
        base_out = self.reduce_fc(base_out)
        
        output, _ = self.lstm_d1(base_out)
        dense1 = base_out + output

        output, _ = self.lstm_d2(dense1)
        output = dense1 + output
        output = output[:, -1, :]

        out = self.new_fc_(output)
        return out

    def bi_dense(self, base_out):
   
        base_out = base_out.view(-1, self.num_segments, self.feature_dim)
        inv_idx = torch.arange(self.num_segments-1, -1, -1).long().cuda()
        base_out_reverse = base_out[:, inv_idx]

        output , _ = self.lstm_d1(base_out)
        dense1 = torch.cat((base_out, output), 2)

        output, _ = self.lstm_d2(dense1)

        output1_reverse , _ = self.lstm_d1_reverse(base_out_reverse)
        dense1_reverse = torch.cat((base_out_reverse, output1_reverse), 2)

        output2_reverse, _ = self.lstm_d2_reverse(dense1_reverse)

        # output = torch.cat((dense1, output, output1_reverse[:, inv_idx],
         # output2_reverse[:, inv_idx]), 2)
        output = torch.cat((dense1, output, output1_reverse,
         output2_reverse), 2)
        if self.out_ == 'one':
            out = self.new_fc_(output[:, -1, :])
        elif self.out_ == 'all':
            out = self.new_fc_(output.view(-1, output.size()[-1]))
            # out = self.conv1d(out.view(-1, self.num_segments, out.size()[-1]))
            out = self.consensus(out.view(-1, self.num_segments, out.size()[-1]))
        
        return out

    def multi_dense(self, base_out):
   
        base_out = base_out.view(-1, self.num_segments, self.feature_dim)

        output , _ = self.lstm_d1(base_out)
        dense1 = torch.cat((base_out, output), 2)

        output, _ = self.lstm_d2(dense1)
        dense2 = torch.cat((dense1, output), 2)

        output, _ = self.lstm_d3(dense2)
        dense3 = torch.cat((dense2, output), 2)

        output, _ = self.lstm_d4(dense3)
        output = torch.cat((dense3, output), 2)
        output = output[:, -1, :]
        out = self.new_fc_(output)
        
        return out

    def bi_dense_v2(self, base_out):
   
        base_out = base_out.view(-1, self.num_segments, self.feature_dim)
        inv_idx = torch.arange(self.num_segments-1, -1, -1).long().cuda()
        base_out_reverse = base_out[:, inv_idx]

        output1 , _ = self.lstm_d1(base_out)
        dense1 = torch.cat((base_out, output1), 2)

        output2, _ = self.lstm_d2(dense1)

        output1_reverse , _ = self.lstm_d1_reverse(base_out_reverse)
        dense1_reverse = torch.cat((base_out_reverse, output1_reverse), 2)

        output2_reverse, _ = self.lstm_d2_reverse(dense1_reverse)

        # output = torch.cat((dense1, output, output1_reverse[:, inv_idx],
        #  output2_reverse[:, inv_idx]), 2)
        output = torch.cat((output1, output2, output1_reverse,
         output2_reverse), 2)
        if self.out_ == 'one':
            out1 = self.new_fc_(output[:, -1, :])
            out2 = self.new_fc(base_out.view(-1, base_out.size()[-1]))
            out2 = self.consensus(out2.view(-1, self,num_segments, out2.size()[-1]))
            out = out1 + out2
        elif self.out_ == 'all':
            out = self.new_fc_(output.view(-1, output.size()[-1]))
            # out = self.conv1d(out.view(-1, self.num_segments, out.size()[-1]))
            out = self.consensus(out.view(-1, self.num_segments, out.size()[-1]))
        
        return out

    def dense_attention(self, base_out):
        base_out = base_out.view(-1, self.num_segments, self.feature_dim)
        attention, _ = self.lstm_attention(base_out)
        attention = self.sigmoid(self.at_fc(attention.contiguous().view(-1, attention.size()[-1])))
        attention = attention.view(-1, self.num_segments, 1)
        output1 , _ = self.lstm_d1(base_out)
        hx = Variable(torch.zeros(base_out.size()[0], self.hidden_dim)).cuda()
        cx = Variable(torch.zeros(base_out.size()[0], self.hidden_dim)).cuda()

        dense1 = torch.cat((base_out, output1), 2)
        for t in xrange(self.num_segments):
            h = hx
            hx, cx = self.lstm_d2(dense1[:,t], (hx, cx))
            hx = hx*attention[:, t]+(1.-attention[:,t])*h
        output = self.drop(hx)
        out = self.new_fc_(torch.cat((dense1[:,-1,:], output), 1))
        return out

    def forward(self, input):

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.type == 'dense':
            output = self.dense(base_out)
        elif self.type == 'bi_dense':
            output = self.bi_dense(base_out)
        elif self.type == 'dense_v2':
            output = self.dense_v2(base_out)
        elif self.type == 'multi_dense':
            output = self.multi_dense(base_out)
        elif self.type == 'dense_attention':
            output = self.dense_attention(base_out)
        elif self.type == 'dense_add':
            output = self.dense_add(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # if self.reshape:
        #     base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data
      
    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
