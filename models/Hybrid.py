import torch.nn as nn
import torch

class Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batchsize = config.batch_size
        self.num_station = config.num_station

        self.in_channels1 = config.conv.in_channels1
        self.in_channels2 = config.conv.in_channels2
        self.in_channels3 = config.conv.in_channels3

        self.out_channels1 = config.conv.out_channels1
        self.out_channels2 = config.conv.out_channels2
        self.out_channels3 = config.conv.out_channels3

        self.kernel_size1 = config.conv.kerner_size1
        self.kernel_size2 = config.conv.kerner_size2
        self.kernel_size3 = config.conv.kerner_size3

        self.input_size = config.bilstm.input_size
        self.hidden_size = config.bilstm.hidden_size
        self.num_layers = config.bilstm.num_layers

        self.in_features_fc = config.conv.in_features_fc
        self.out_features_fc = config.conv.out_features_fc

        self.in_features_fs = config.fusion.in_features_fs
        self.out_features_fs = config.fusion.out_features_fs
        self.output_dim = config.num_station

        self.p_dropout = config.dropout

        self.kernel_size_maxpool = config.conv.kernel_size_maxpool
        self.padding_maxpool = config.conv.padding_maxpool
        self.stride_maxpool = config.conv.stride_maxpool

        self.lookup_size = config.lookup_size

        # 3 1D-Conv layer
        self.conv1 = nn.Conv1d(in_channels=self.in_channels1, out_channels=self.out_channels1, kernel_size=self.kernel_size1)
        self.conv2 = nn.Conv1d(in_channels=self.in_channels2, out_channels=self.out_channels2, kernel_size=self.kernel_size2)
        self.conv3 = nn.Conv1d(in_channels=self.in_channels3, out_channels=self.out_channels3, kernel_size=self.kernel_size3)

        # BiLSTM layer
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.flatten = nn.Flatten()
        # FC layer after 3 1D-Conv layer
        self.fc_conv = nn.Linear(in_features=self.in_features_fc, out_features=self.out_features_fc)

        # fusion layer and linear layer at the end of model
        self.fusion = nn.Linear(in_features=(self.lookup_size + 1)*self.hidden_size*2, out_features=self.out_features_fs)
        self.linear = nn.Linear(in_features=self.out_features_fs, out_features=self.output_dim)

        self.batchnorm = nn.BatchNorm1d()
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size_maxpool, padding=self.padding_maxpool, stride=self.stride_maxpool)

    def forward(self, x):
        # list_x is list of multiple stations data
        # xi has shape(batch_size, features, seq_length)
        # list_L is list of multiple conv output
        hidden = self.init_hidden()

        list_L = []
        for i in range(self.num_station):
            xi = x[:, i*14 : (i+1)*14]
            out_conv1 = self.relu(self.batchnorm(self.conv1(xi)))
            out_conv2 = self.relu(self.batchnorm(self.conv2(out_conv1)))
            out_conv3 = self.relu(self.batchnorm(self.conv3(out_conv2)))
            out_flatten = self.flatten(out_conv3)
            out_conv_L = self.relu(self.dropout(self.fc_conv(out_flatten)))
            #reshape L (batch_size, seq_length, 1)
            L = out_conv_L.contigous().view(out_conv_L.size()[0], out_conv_L.size()[1],1)
            list_L.append(out_conv_L)
        
        # concat all conv output LC shape(batch_size, seq_length, features)
        LC = torch.cat(list_L, dim=2)

        # feed LC to bilstm
        out_bilstm, hidden = self.bilstm(LC, hidden)

        # get Ot -> Ot-l with l is lookup size, Ot is (hf_t,hb_t) of bilstm
        O = out_bilstm[:, -self.lookup_size:, :]
        # flatten O then feed into fusion layer
        O_flatten = self.flatten(O)
        out_fusion = self.fusion(O_flatten)
        predict = self.linear(out_fusion)

        return predict

    
    def init_hidden(self) :
        h0 = torch.zeros((2 * self.num_layers, self.batchsize,self.hidden_size))
        c0 = torch.zeros((2 * self.num_layers, self.batchsize,self.hidden_size))
        hidden = (h0,c0)
        return hidden





