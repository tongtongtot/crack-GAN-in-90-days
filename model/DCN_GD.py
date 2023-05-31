import torch.nn as nn
# 定义生成器网络G
class Generator(nn.Module):
    def __init__(self, ngf, nz):
        super(Generator, self).__init__()
        # layer1输入的是一个100x7x7的随机噪声, 输出尺寸(ngf*8)x4x4
        # self.layer0 = nn.Linear(nz,16)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x14x14
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x28x28
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸 3x96x96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        # print("this is x:" , x.shape)
        # out = self.layer0(x)
        # print("layer0:", out.shape)
        out = self.layer1(x)
        # print("layer1:" , out.shape)
        out = self.layer2(out)
        # print("layer2:" ,out.shape)
        out = self.layer3(out)
        # print("layer3:" ,out.shape)
        out = self.layer4(out)
        # print("layer4:" ,out.shape)
        out = self.layer5(out)
        # print("??layer5:" ,out.shape)
        return out


# 定义鉴别器网络D
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        # layer1 输入 3 x 28 x 28, 输出 (ndf) x 14 x 14
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 7 x 7
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 1, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 7 x 7
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 7, 7, 0, bias=False),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self,x):
        # print("this is x", x.shape)
        out = self.layer1(x)
        # print("layer1:", out.shape)
        out = self.layer2(out)
        # print("layer2:", out.shape)
        out = self.layer3(out)
        # print("layer3:", out.shape)
        out = self.layer4(out)
        # print("layer4:", out.shape)
        out = self.layer5(out)
        # print("layer5:", out.shape)
        return out
       
        