import torch.nn as nn
import torch

class Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batchsize = config.get('batch_size')
        self.num_station = config.get('num_station')
        self.n_in = config.get('n_in')


        self.in_channels1 = config.get('conv').get('in_channels1')
        self.in_channels2 = config.get('conv').get('in_channels2')
        self.in_channels3 = config.get('conv').get('in_channels3')

        self.out_channels1 = config.get('conv').get('out_channels1')
        self.out_channels2 = config.get('conv').get('out_channels2')
        self.out_channels3 = config.get('conv').get('out_channels3')

        self.kernel_size1 = config.get('conv').get('kernel_size1')
        self.kernel_size2 = config.get('conv').get('kernel_size2')
        self.kernel_size3 = config.get('conv').get('kernel_size3')

        self.input_size = config.get('num_station')
        self.hidden_size = config.get('bilstm').get('hidden_size')
        self.num_layers = config.get('bilstm').get('num_layers')

        # self.in_features_fc = config.get('conv.in_features_fcs
        self.out_features_fc = config.get('conv').get('out_features_fc')

        self.in_features_fs = config.get('fusion').get('in_features_fs')
        self.out_features_fs = config.get('fusion').get('out_features_fs')
        self.output_dim = config.get('num_station')

        self.p_dropout = config.get('dropout')

        self.kernel_size_maxpool = config.get('conv').get('kernel_size_maxpool')
        self.padding_maxpool = config.get('conv').get('padding_maxpool')
        self.stride_maxpool = config.get('conv').get('stride_maxpool')

        self.lookup_size = config.get('lookup_size')

        # 3 1D-Conv layer
        self.conv1 = nn.Conv1d(in_channels=self.in_channels1, out_channels=self.out_channels1, kernel_size=self.kernel_size1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=self.in_channels2, out_channels=self.out_channels2, kernel_size=self.kernel_size2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=self.in_channels3, out_channels=self.out_channels3, kernel_size=self.kernel_size3, padding='same')

        # batchnorm
        self.batchnorm1 = nn.BatchNorm1d(self.out_channels1)
        self.batchnorm2 = nn.BatchNorm1d(self.out_channels2)
        self.batchnorm3 = nn.BatchNorm1d(self.out_channels3)

        # BiLSTM layer
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.flatten = nn.Flatten()
        # FC layer after 3 1D-Conv layer
        # self.fc_conv = nn.Linear(in_features=self.in_features_fc, out_features=self.out_features_fc)
        self.fc_conv = nn.Linear(in_features=self.n_in*14*self.out_channels3, out_features=self.out_features_fc)

        # fusion layer and linear layer at the end of model
        self.fusion = nn.Linear(in_features=(self.lookup_size + 1)*self.hidden_size*2, out_features=self.out_features_fs)
        self.linear = nn.Linear(in_features=self.out_features_fs, out_features=self.output_dim)

        self.dropout = nn.Dropout(p=self.p_dropout)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size_maxpool, padding=self.padding_maxpool, stride=self.stride_maxpool)

    def forward(self, x):
        # list_x is list of multiple stations data
        # xi has shape(batch_size, features, seq_length)
        # list_L is list of multiple conv output
        import pdb; pdb.set_trace()

        hidden = self.init_hidden(x)
        list_L = []
        for i in range(self.num_station):

            xi = x[ :, i*14*self.n_in : (i+1)*14*self.n_in].view(x.size()[0], 1, -1)
            out_conv1 = self.relu(self.batchnorm1(self.conv1(xi)))
            out_conv2 = self.relu(self.batchnorm2(self.conv2(out_conv1)))
            out_conv3 = self.relu(self.batchnorm3(self.conv3(out_conv2)))
            out_flatten = self.flatten(out_conv3)

            # self.in_features_fc = out_flatten.size()[1]
            
            out_conv_L = self.relu(self.dropout(self.fc_conv(out_flatten)))
            #reshape L (batch_size, seq_length, 1)
            L = out_conv_L.contiguous().view(out_conv_L.size()[0], out_conv_L.size()[1],1)
            list_L.append(L)
        
        # concat all conv output LC shape(batch_size, seq_length, features)
        LC = torch.cat(list_L, dim=2)

        # feed LC to bilstm
        out_bilstm, hidden = self.bilstm(LC, hidden)

        # get Ot -> Ot-l with l is lookup size, Ot is (hf_t,hb_t) of bilstm
        O = out_bilstm[:, -self.lookup_size - 1:, :]

        # flatten O then feed into fusion layer
        O_flatten = self.flatten(O)
        out_fusion = self.fusion(O_flatten)
        predict = self.linear(out_fusion)

        return predict

    
    def init_hidden(self, x) :
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros((2 * self.num_layers, x.size()[0],self.hidden_size)).to(device)
        c0 = torch.zeros((2 * self.num_layers, x.size()[0],self.hidden_size)).to(device)
        hidden = (h0,c0)
        return hidden





