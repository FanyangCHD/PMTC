import torch.nn as nn
import torch
from swin_transformer import *

def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)

    def conv_block(self, input_channels, k):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.LeakyReLU(),
            nn.Conv2d(input_channels, k, kernel_size=3, padding=1))
            
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X

class Conv_path(nn.Module):
    def __init__(self, in_channel=32,k=8):
        super(Conv_path, self).__init__()
        
        self.Dense = DenseBlock(in_channel=in_channel, k=k)
        self.final_conv = nn.Conv2d(4*k+ in_channel, 32, 1)

    def forward(self, x):
        x1 = self.Dense(x)     
        x2 = self.final_conv(x1)
        return x2

class Fuse_block(nn.Module):
    def __init__(self, shallow_dim=32):
        super(Fuse_block,self).__init__()
     
        #########   SwinT Path   ########## 
        self.SwinT_path = SwinTransformer()
        #########   Conv Path   ########## 
        self.Conv_path = Conv_path()
        
        self.fuse = nn.Sequential(nn.Conv2d(shallow_dim*2, shallow_dim, kernel_size=1))
    
    def forward(self, x):
        x1 = self.SwinT_path(x)
        x_swin = x1 + x

        x2 = self.Conv_path(x)
        x_conv = x2 + x

        x3 = torch.cat((x_swin,x_conv), dim=1)      
        x4 = self.fuse(x3)
        return x4

class Generator(nn.Module):
    def __init__(self, in_channel=1, shallow_dim=32, num_layers=6):
        super(Generator,self).__init__()

        #########   first downsample layer   ##########
      
        self.shallow_feature = nn.Sequential(nn.Conv2d(in_channel, shallow_dim, 3, 1, 1),
                                        nn.BatchNorm2d(shallow_dim), nn.LeakyReLU())     #  padding = (kernel_size - 1) // 2        

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = Fuse_block()
            self.layers.append(layer)

        #########   out Layer   ########## 
        self.out_layer = nn.Sequential(nn.BatchNorm2d(shallow_dim), nn.LeakyReLU(),nn.Conv2d(shallow_dim, in_channel, 3, 1, 1))

    def forward(self, x):
      
        x1 = self.shallow_feature(x)

        for layer in self.layers:
            x1 = layer(x1)

        out = self.out_layer(x1)

        mask = torch.zeros_like(x)
        mask[x == 0] = 1
        output = torch.mul(mask, out) + x
           
        return output


if __name__ == "__main__":
    
    X = torch.rand(size=(32, 1, 20, 1024))
    net =Generator()
    out = net(X)
    print(out.shape)