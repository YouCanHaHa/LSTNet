import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.input_len= args.window;
        self.input_size = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.cnn_kernel_size = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = (self.input_len- self.cnn_kernel_size)//self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.cnn_kernel_size, self.input_size));
        ### batch_first=False (by default)
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            ### batch_first=False (by default)
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.input_size);
        else:
            self.linear1 = nn.Linear(self.hidR, self.input_size);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        ### x.size: [BS, input_len, input_size]
        batch_size = x.size(0);
        
        #CNN
        ### c.size: [BS, 1, input_len, input_size] because self.conv1.input_channel==1
        c = x.view(-1, 1, self.input_len, self.input_size);
        
        ''' c.size: [BS, hidC, cnn_output_len(input_len-cnn_kernel_size+1),1]
        cnn_output_len = (input_len - 2 * cnn_emb_params["padding"] -\
            cnn_emb_params["dilation"]*(cnn_emb_params["kernel_size"]-1)-1)\
            //cnn_emb_params["stride"] + 1
        
        in the case
            padding=0, dilation=1, stride=1
            cnn_output_len = input_len - cnn_kernel_size + 1
        '''
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        ### r.size: [cnn_output_len, BS, hidC]
        r = c.permute(2, 0, 1).contiguous();
        ### r: hidden state r.size: [num_layer*num_dir, BS, hidden_size] --> (1, BS, hidR)
        _, r = self.GRU1(r);
        ### r.size: [BS, hidR]
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            ### s.size: [BS, hidC, pt*skip] (pt*skip-->input_len of skipRNN, hidC-->input_size of skipRNN)
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            
            ### s.size: [pt, BS, skip, hidC]
            s = s.permute(2,0,3,1).contiguous();
            '''
            Most important and tricky step
            
            Fuse the skip dimension and BS dimension together 
                to use more BS and accelerate forwarding
            s.size: [pt, BS*skip, hidC]
            '''
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        '''
        Here, use x[j][T-hw],..., x[j][T] to approximate x[j][T+output_horizon]
            where j is the index of input_size
        Therefore, the highway component is used as a univariate AR model
        '''
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            ### z.size: [BS, input_size, hw(highway_window_size)] --> (BS*input_size, highway_window_size)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            ### z.size [BS*input_size, 1]
            z = self.highway(z);
            ### z.size: [BS, input_size]
            z = z.view(-1,self.input_size);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
    
        
        
        
