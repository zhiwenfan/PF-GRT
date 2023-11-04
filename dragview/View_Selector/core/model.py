import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalNorm2d(nn.Module):
    ## borrowed from https://github.com/DagnyT/hardnet/tree/deab7e892468a07fb2cf77d41e38714fa96a6e99
    def __init__(self, kernel_size = 32):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size = self.ks, stride = 1, padding = 0)
        self.eps = 1e-10
        return
    def forward(self,x):
        pd = int(self.ks/2)
        mean = self.pool(F.pad(x, (pd,pd,pd,pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.pad(x*x, (pd,pd,pd,pd), 'reflect')) - mean*mean)) + self.eps), min = -6.0, max = 6.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, stride=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if stride != 1 or in_planes != planes:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if stride != 1 or in_planes != planes:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if stride != 1 or in_planes != planes:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=bias), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)


class ResNet_encoder(nn.Module):
    def __init__(self, norm_fn='none', dropout=0.5):
        super().__init__()
        if norm_fn == 'group':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=64)

        elif norm_fn == 'batch':
            self.norm = nn.BatchNorm2d(64)

        elif norm_fn == 'instance':
            self.norm = nn.InstanceNorm2d(64)

        elif norm_fn == 'none':
            self.norm = nn.Sequential()

        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.resblock_1 = nn.Sequential(
            ResidualBlock(32, 32, norm_fn=norm_fn, kernel_size=3, stride=1, padding=1, bias=False),
            ResidualBlock(32, 64, norm_fn=norm_fn, kernel_size=3, stride=2, padding=1, bias=False),
            ResidualBlock(64, 64, norm_fn=norm_fn, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.resblock_2 = nn.Sequential(
            ResidualBlock(64, 128, norm_fn=norm_fn, kernel_size=3, stride=2, padding=1, bias=False),
            ResidualBlock(128, 128, norm_fn=norm_fn, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.resblock_3 = nn.Sequential(
            ResidualBlock(128, 128, norm_fn=norm_fn, kernel_size=3, stride=2, padding=1, bias=False),
            ResidualBlock(128, 128, norm_fn=norm_fn, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out1 = self.resblock_1(out) # /2
        out2 = self.resblock_2(out1) # /4
        out3 = self.resblock_3(out2) # /8
        out4 = self.conv_4(out3) # /8 -3

        return [out1, out2, out3, out4]

class ResNet_decoder(nn.Module):
    def __init__(self, dropout=0.5, feature_dim=16):
        super().__init__()
        self.feature_dim = feature_dim

        self.head_4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, self.feature_dim, kernel_size=1, bias=False)
        )

        self.up_sample_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, bias=False)
        )
        self.conv_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),

        )
        self.head_3 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, self.feature_dim, kernel_size=1, bias=False)
        )

        self.up_sample_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),

        )
        self.head_2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, self.feature_dim, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2, x3, x4):
        out4 = self.head_4(x4)

        out = self.up_sample_3(x4)
        out = torch.cat([out, x3], dim=1)
        out = self.conv_3(out)
        out3 = self.head_3(out)

        out = self.up_sample_2(out)
        out = torch.cat([out, x2], dim=1)
        out = self.conv_2(out)
        out2 = self.head_2(out)

        return [out2, out3, out4]

class RetrievalNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg["MODEL"]["HIDDEN_DIM"]**2
        self.feature_dim = cfg["MODEL"]["FEATURE_DIM"]
        self.input_dim = cfg["DATA"]["CROP_SIZE"]

        self.input_norm = LocalNorm2d(17)

        self.encoder = ResNet_encoder(norm_fn='none', dropout=cfg["TRAIN"]["DROP"])
        self.decoder = ResNet_decoder(dropout=self.cfg["TRAIN"]["DROP"], feature_dim=cfg["MODEL"]["FEATURE_DIM"])

    def forward(self, img):
        B, _, H, W = img.shape

        if img.size(1) > 1:
            img = img.mean(dim=1, keepdim=True)

        if self.cfg["MODEL"]["LOCALNORM"] is True:
            img = self.input_norm(img)

        [out1, out2, out3, out4] = self.encoder(img)

        out = self.decoder(out1, out2, out3, out4)

        out = [out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True).clamp(min=1e-8) for i in range(len(out))]
        return out

class Sim_predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = self.cfg["TRAIN"]["DROP"]
        self.feature_dim = cfg["MODEL"]["FEATURE_DIM"]

        self.scales = cfg["MODEL"]["SCALES"]

        self.fc_1 = nn.Sequential(
            nn.Conv1d(self.feature_dim, 2, 1),
        )
        self.fc_2 = nn.Sequential(
            nn.Conv1d(self.feature_dim, 2, 1),
        )
        self.fc_3 = nn.Sequential(
            nn.Conv1d(self.feature_dim, 2, 1),
        )

        self.fcs = [self.fc_1, self.fc_2, self.fc_3]

        self.fc_finial = nn.Sequential(
            nn.Linear(self.feature_dim*len(self.scales), self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, 1)
        )

    def fusion(self, src_f, ref_f):
        out = []
        for i in range(len(self.scales)):
            fuse_f = (src_f[i][:, None]*ref_f[i][None]).view(-1, self.feature_dim, self.scales[i]**2)
            weights = self.fcs[i](fuse_f)

            local_mask = weights[:, 0, :]
            global_mask = weights[:, 1, :]

            local_mask = torch.sigmoid(local_mask)

            weights = torch.exp(global_mask) * local_mask
            weights = weights / torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-8)

            out.append((fuse_f * weights[:, None]).sum(dim=-1))

        out = torch.cat(out, dim=-1)
        return out

    def forward(self, src_f, ref_f):
        B_src = src_f[0].shape[0]
        out = self.fusion(src_f, ref_f)
        sim = torch.tanh(self.fc_finial(out)).view(B_src, -1)
        return sim
