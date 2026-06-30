""" Parts of the map conditioning model """

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Adopted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # [B, in_channels, H, W]  -> [B, mid_channels, H, W]
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True),
            # [B, mid_channels, H, W] -> [B, out_channels, H, W]
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)      # [B, out_channels, H, W]
        )

    def forward(self, x):
        # x: Input tensor of shape [B, in_channels, H, W]
        return self.double_conv(x) # [B, out_channels, H, W]

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # [B, in_channels, H, W] -> [B, in_channels, H//2, W//2] 
            # [B, in_channels, H//2, W//2] -> [B, out_channels, H//2, W//2]
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # x: Input tensor of shape [B, in_channels, H, W]
        return self.maxpool_conv(x) # [B, out_channels, H//2, W//2]

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class MapEncoder(nn.Module):
    """
    A flexible encoder that, when in training mode (for autoencoder pretraining),
    returns intermediate features for skip connections.
    When in inference mode, it returns a fixed 128-dimensional vector.
    """
    def __init__(self, in_channels=1, base_channels=16, bilinear=False):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)       # x1: [B, 16, H, W]
        self.down1 = Down(base_channels, base_channels * 2)     # x2: [B, 32, H/2, W/2]
        self.down2 = Down(base_channels*2, base_channels*4) # x3: [B, 64, H/4, W/4]
        self.down3 = Down(base_channels*4, base_channels*8) # x4: [B, 128, H/4, W/4] 
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels*8, base_channels*16// factor)    # x5: [B, 256//factor, H/16, W/16]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                # reduces spatial dims to 1x1
        self.bilinear = bilinear

    def forward(self, x, return_skip):
        # x: SDF map tensor of shape [B, 1, H, W]
        x1 = self.inc(x)        # [B, 1, H, W] -> [B, b_c, H, W] # b_c = base_channels
        x2 = self.down1(x1)     # [B, b_c, H, W] -> [B, b_c*2, H//2, W//2]
        x3 = self.down2(x2)     # [B, b_c*2, H//2, W//2] -> [B, b_c*4, H//4, W//4]
        x4 = self.down3(x3)     # [B, b_c*4, H//4, W//4] -> [B, b_c*8, H//8, W//8]
        x5 = self.down4(x4)     # [B, b_c*8, H//8, W//8] -> [B, b_c*16//factor, H//16, W//16]
        if return_skip: # Return all intermediate features for the decoder (skip connections)
            return x1, x2, x3, x4, x5
        else:           # For downstream tasks: pool and flatten x5 to get a fixed vector.
            pooled = self.pool(x5)              # [B, b_c*16//factor, H//16, W//16] -> [B, b_c*16//factor, 1, 1]
            vector = torch.flatten(pooled, 1)   # [B, b_c*16//factor, 1, 1] -> [B, 1024//factor]
            return vector

class MapAct(nn.Module):
    #NOTE: I forget why I have this name MapAct, it's just a simple MLP without map conditioning
    def __init__(self, state_dim, num_actions, hidden_dim): # normally state_dim=4, num_actions=31, hidden_dim=512 or 1024
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=True)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions, bias=True)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, map_embedding=None): # map_embedding is not used for vanilla MapAct
        x = self.bn_1(self.act(self.fc1(state)))
        x = self.bn_2(self.act(self.fc2(x)))
        logits = self.out(x)
        return self.softmax(logits)

class MapPixelFeature(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, bilinear=False, feature_dim=64):
        super().__init__()
        self.encoder = MapEncoder(in_channels=in_channels, base_channels=base_channels, bilinear=bilinear)

        factor = 2 if bilinear else 1
        # Decoder to restore spatial details (using skip connections)
        self.up1 = Up(base_channels*16, base_channels*8 // factor, bilinear)
        self.up2 = Up(base_channels*8, base_channels*4 // factor, bilinear)
        self.up3 = Up(base_channels*4, base_channels*2 // factor, bilinear)
        self.up4 = Up(base_channels*2, base_channels, bilinear)
        self.out_conv = nn.Conv2d(base_channels, feature_dim, kernel_size=1) # dense pixel-level embeddings

    def forward(self, x):
        # Encoder pathway
        x1, x2, x3, x4, x5 = self.encoder(x, return_skip=True)

        # Decoder pathway with skip connections
        d = self.up1(x5, x4)    # Up-sample and fuse with encoder feature x4
        d = self.up2(d, x3)     # Up-sample and fuse with x3
        d = self.up3(d, x2)     # Up-sample and fuse with x2
        d = self.up4(d, x1)     # Up-sample and fuse with x1

        # Pixel-level embedding map [B, feature_dim, H, W]
        pixel_embedding = self.out_conv(d)

        return pixel_embedding
class MapAct_PixelInterpolated(nn.Module):
    def __init__(self, feature_channels=64, state_dim=4, num_actions=31, hidden_dim=2048, dropout_rate=0.0):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, feature_channels)
        self.fc1 = nn.Linear(feature_channels * 2, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, interpolated_features):
        state_embedding = self.state_proj(state)
        fused = torch.cat([state_embedding, interpolated_features], dim=-1)
        # x = self.bn_1(self.dropout(self.act(self.fc1(fused))))
        # x = self.bn_2(self.dropout(self.act(self.fc2(x))))
        # x = self.dropout(self.bn_1(self.act(self.fc1(fused))))
        # x = self.dropout(self.bn_2(self.act(self.fc2(x))))
        x = self.bn_1(self.act(self.fc1(fused)))
        x = self.bn_2(self.act(self.fc2(x)))
        logits = self.out(x)
        return self.softmax(logits)
