import torch
import torch.nn as nn


def build_conv_block(in_c, out_c1, out_c2):
    conv_block = []
    conv_block += [
        nn.Conv3d(in_channels=in_c, out_channels=out_c1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
                  padding_mode='circular')]
    conv_block += [nn.GroupNorm(32, out_c1)]
    conv_block += [nn.ReLU()]
    conv_block += [
        nn.Conv3d(in_channels=out_c1, out_channels=out_c2, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
                  padding_mode='circular')]
    conv_block += [nn.GroupNorm(32, out_c2)]
    conv_block += [nn.ReLU()]

    return nn.Sequential(*conv_block)


class VINet(nn.Module):

    def __init__(self):
        super(VINet, self).__init__()

        # encoder
        self.conv1 = build_conv_block(1, 64, 64)
        self.conv2 = build_conv_block(64, 128, 128)
        self.conv3 = build_conv_block(128, 256, 256)
        self.conv4 = build_conv_block(256, 512, 256)

        # decoder
        self.conv5 = build_conv_block(512, 256, 128)
        self.conv6 = build_conv_block(256, 128, 64)
        self.conv7 = build_conv_block(128, 64, 64)

        # prediction layer
        self.pred = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        # downsample layer
        self.downSample = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True)

        # upsample layer
        self.upSample = nn.MaxUnpool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

    def forward(self, x):
        # downSample
        C1 = self.conv1(x)
        D1, Indices1 = self.downSample(C1)
        C2 = self.conv2(D1)
        D2, Indices2 = self.downSample(C2)
        C3 = self.conv3(D2)
        D3, Indices3 = self.downSample(C3)
        C4 = self.conv4(D3)

        # upSample
        U3 = self.upSample(C4, Indices3)
        concat3 = torch.cat((C3, U3), dim=1)
        C5 = self.conv5(concat3)
        U2 = self.upSample(C5, Indices2)
        concat2 = torch.cat((C2, U2), dim=1)
        C6 = self.conv6(concat2)
        U1 = self.upSample(C6, Indices1)
        concat1 = torch.cat((C1, U1), dim=1)
        C7 = self.conv7(concat1)

        # pred
        output = self.pred(C7)

        return output


class ASNet(nn.Module):

    def __init__(self):
        super(ASNet, self).__init__()

        # encoder
        self.conv1 = build_conv_block(2, 64, 64)
        self.conv2 = build_conv_block(64, 128, 128)
        self.conv3 = build_conv_block(128, 256, 256)
        self.conv4 = build_conv_block(256, 512, 256)

        # decoder
        self.conv5 = build_conv_block(512, 256, 128)
        self.conv6 = build_conv_block(256, 128, 64)
        self.conv7 = build_conv_block(128, 64, 64)

        # prediction layer
        self.pred = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

        # downsample layer
        self.downSample = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True)

        # upsample layer
        self.upSample = nn.MaxUnpool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

    def forward(self, x):
        # downSample
        C1 = self.conv1(x)
        D1, Indices1 = self.downSample(C1)
        C2 = self.conv2(D1)
        D2, Indices2 = self.downSample(C2)
        C3 = self.conv3(D2)
        D3, Indices3 = self.downSample(C3)
        C4 = self.conv4(D3)

        # upSample
        U3 = self.upSample(C4, Indices3)
        concat3 = torch.cat((C3, U3), dim=1)
        C5 = self.conv5(concat3)
        U2 = self.upSample(C5, Indices2)
        concat2 = torch.cat((C2, U2), dim=1)
        C6 = self.conv6(concat2)
        U1 = self.upSample(C6, Indices1)
        concat1 = torch.cat((C1, U1), dim=1)
        C7 = self.conv7(concat1)

        # pred
        output = self.pred(C7)

        return output
