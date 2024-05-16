import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchsummary import summary  # Statistical parameter


class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S

        self.convs = nn.ModuleList([])
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        self.se_blocks = nn.ModuleList([])
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1:SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs,s,ci,h,w
        SPC_out_clone = SPC_out.clone()  # clone a copy
        for idx, conv in enumerate(self.convs):
            SPC_out_clone[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out_clone[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.view(b, -1, 1, 1)
        SPC_out_clone = SPC_out_clone.view(b, -1, h, w)

        return SE_out, SPC_out_clone


class LaplacianHighPass(nn.Module):
    def __init__(self, num_channels):
        super(LaplacianHighPass, self).__init__()

        self.num_channels = num_channels
        self.high_pass_filters = nn.ModuleList()

        for _ in range(num_channels):
            laplacian_kernel = torch.tensor([[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]], dtype=torch.float32)

            filter_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            filter_conv.weight.data.copy_(laplacian_kernel.view(1, 1, 3, 3))

            self.high_pass_filters.append(filter_conv)

    def forward(self, x):
        high_pass_outputs = []

        for i in range(self.num_channels):

            x_channel = x[:, i, :, :].unsqueeze(1)

            high_pass_output = self.high_pass_filters[i](x_channel)
            high_pass_outputs.append(high_pass_output)

        high_pass_outputs = torch.cat(high_pass_outputs, dim=1)

        return high_pass_outputs


class RGB_HighFilter(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=192):
        super(RGB_HighFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.convfirst = nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv_activate = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.highpass = LaplacianHighPass(self.hidden_channels)

        self.conv_hiddenlayer_lowpass = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv_hiddenlayer_highpass = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv_recover = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.conv1x1 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, rgb):
        x1 = self.lrelu(self.convfirst(rgb))
        x2 = self.conv_activate(x1)
        high_pass = self.highpass(x2)
        low_pass = x2 - high_pass

        high_pass_feature = self.conv_hiddenlayer_highpass(high_pass)
        low_pass_feature = self.conv_hiddenlayer_lowpass(low_pass)

        x3 = self.lrelu(self.conv1x1(low_pass_feature + high_pass_feature))
        rgb_recover = self.conv_recover(x3)

        return rgb_recover, high_pass_feature


class SpectralNet(nn.Module):
    def __init__(self, in_channels=31, out_channels=31, hidden_channels=192, reduction=4, a='learnable', device='cuda:0', layer_index=0, level=3):
        super(SpectralNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.reduction = reduction
        self.device = device
        self.layer_index = layer_index
        self.level = level

        self.convfirst_SR = nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv_activate_SR = nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.convfirst_LR = nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv_activate_LR = nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)

        self.conv_recover = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

        self.psa1 = PSA(channel=self.hidden_channels)
        self.conv_spe1 = nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.rgb_feature = RGB_HighFilter(hidden_channels=self.hidden_channels)
        if a == 'learnable':
            self.a = nn.Parameter(torch.randn(1).clamp(min=0.00001, max=1).requires_grad_())
        else:
            self.a = float(a)
        self.fuse_rgb_srhsi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0)

    def forward(self, srhsi, lrhsi, rgb):
        sr1 = self.lrelu(self.convfirst_SR(srhsi))
        sr2 = self.lrelu(self.conv_activate_SR(sr1))

        lr1 = self.lrelu(self.convfirst_LR(lrhsi))
        lr2 = self.lrelu(self.conv_activate_LR(lr1))

        b, c, h, w = sr2.size()
        sr_se, sr_spc = self.psa1(sr2)
        lr_se, lr_spc = self.psa1(lr2)

        spec_recover = self.conv_spe1(torch.cat((sr_se, lr_se), dim=1))
        sr_se_out, lr_se_out = spec_recover, spec_recover
        sr_se_out = sr_se_out.expand_as(sr_spc)
        lr_se_out = lr_se_out.expand_as(lr_spc)

        sr_softmax_out = self.softmax(sr_se_out)
        lr_softmax_out = self.softmax(lr_se_out)

        sr_out = sr_spc * sr_softmax_out
        lr_out = lr_spc * lr_softmax_out

        rgb_recover, high_pass_feature = self.rgb_feature(rgb)

        se_feature = self.lrelu(self.fuse_rgb_srhsi(sr_out + self.a * high_pass_feature) + srhsi)

        hr_out = self.conv_recover(se_feature)

        return hr_out, lr_out, rgb_recover, self.a, high_pass_feature

    def clamp_a(self):
        if self.a.requires_grad:
            self.a.data.clamp_(min=0.00001, max=1)


class Backbone(nn.Module):
    def __init__(self, in_channels=31, out_channels=31, hidden_channels=64, reduction=4, level=3, a='learnable'):
        super(Backbone, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels in range(1, 64):
            self.hidden_channels = 64
        elif in_channels in range(64, 128):
            self.hidden_channels = 128
        else:
            self.hidden_channels = 256

        self.scale = reduction
        self.a = a
        self.level = level

        self.SR_first_activate = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.LR_first_activate = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.spectral_modules = nn.ModuleList([
            SpectralNet(in_channels=self.hidden_channels, out_channels=self.hidden_channels, hidden_channels=self.hidden_channels, a=self.a, layer_index=i + 1, level=self.level)
            for i in range(self.level)
        ])

        self.feature_2_HR = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.feature_2_LR = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.hidden_channels // 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, lrhsi, hrrgb):

        srhsi = F.interpolate(lrhsi, scale_factor=2, mode='bilinear')
        if self.scale == 4:
            srhsi = F.interpolate(srhsi, scale_factor=2, mode='bilinear')

        srhsi_feature = self.SR_first_activate(srhsi)
        lrhsi_feature = self.LR_first_activate(lrhsi)

        rgb_recovers = []
        para_a = []
        high_pass_features = []
        for module in self.spectral_modules:
            hr_out, lr_out, rgb_recover, a, high_pass_feature = module(srhsi_feature, lrhsi_feature, hrrgb)
            srhsi_feature = hr_out
            lrhsi_feature = lr_out
            hrrgb = rgb_recover

            rgb_recovers.append(rgb_recover)
            if self.a == 'learnable':
                para_a.append(a.cpu().item())
            else:
                para_a.append(a)
            high_pass_features.append(high_pass_feature)

        hr_result = self.feature_2_HR(hr_out)
        lr_result = self.feature_2_LR(lr_out)

        return hr_result, lr_result, rgb_recovers, para_a, high_pass_features

    def clamp_a(self):
        for module in self.spectral_modules:
            module.clamp_a()


# device = torch.device('cuda:0')
# model = Backbone(in_channels=31, out_channels=31, hidden_channels=64, a='learnable').to(device=device, dtype=torch.float32)

# sr_hsi = torch.randn(1, 31, 16, 16).to(device=device)
# lr_hsi = torch.randn(1, 31, 16, 16).to(device=device)
# hr_rgb = torch.randn(1, 1, 64, 64).to(device=device)

# hr_result, lr_result, rgb_recovers, para_a, high_pass_features = model(sr_hsi, lr_hsi, hr_rgb)

# print(f'hr_result\'s shape:{hr_result.shape}\n',
#       f'lr_result\'s shape:{lr_result.shape}\n',
#       f'rgb_recover1\'s shape:{rgb_recover1.shape}\n',
#       f'rgb_recover2\'s shape:{rgb_recover2.shape}\n',
#       f'rgb_recover3\'s shape:{rgb_recover3.shape}\n',
#       f'high_pass_feature_l1\'s shape:{high_pass_feature_l1.shape}\n',
#       f'high_pass_feature_l2\'s shape:{high_pass_feature_l2.shape}\n',
#       f'high_pass_feature_l3\'s shape:{high_pass_feature_l3.shape}'
#       )
# summary(model, [(31, 64, 64), (31, 16, 16), (3, 64, 64)])
