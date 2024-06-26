import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, LBO_MATRIX, LBO_INVERSE):
        
        super(Approximation_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = LBO_MATRIX.shape[1]
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
                        
        ################################################################
        # Encode
        ################################################################
        x = x = x.permute(0, 2, 1)
        x = self.LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)
        
        ################################################################
        # Approximator
        ################################################################
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        
        ################################################################
        # Decode
        ################################################################
        x =  x @ self.LBO_MATRIX.T
        
        return x
    
class NORM_Net(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(2, self.width) 
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        time_1 = time.perf_counter()

        grid = self.get_grid(x.shape, x.device) # B, N, 1
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        time_2 = time.perf_counter()
        # print('self.fc0(x) : %.6f' % (time_2 - time_1))
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        time_3 = time.perf_counter()
        # print('self.conv0(x) : %.6f' % (time_3 - time_2))

        x2 = self.w0(x)
        time_4 = time.perf_counter()
        # print('self.w0(x) : %.6f' % (time_4 - time_3))

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     

class NORM_Net_DeltaPhi(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net_DeltaPhi, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        # self.fc0 = nn.Linear(1 + 2, self.width) 
        self.fc0 = nn.Linear(2 + 3, self.width) 
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
        ref_score = ref_score.reshape( -1, 1, 1 ) * torch.ones_like( x ) # B, N, 1
        # x = torch.cat((x, ref_score, ref_x), dim=-1)

        grid = self.get_grid(x.shape, x.device) # B, N, 1

        x = torch.cat((x, ref_score, ref_x, ref_y.reshape(x.shape), grid), dim=-1)

        time_1 = time.perf_counter()

        x = self.fc0(x)
        time_2 = time.perf_counter()
        # print('self.fc0(x) : %.6f' % (time_2 - time_1))
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        time_3 = time.perf_counter()
        # print('self.conv0(x) : %.6f' % (time_3 - time_2))

        x2 = self.w0(x)
        time_4 = time.perf_counter()
        # print('self.w0(x) : %.6f' % (time_4 - time_3))

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x + ref_y.reshape(x.shape)
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)    