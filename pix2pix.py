import torch
from torch import nn

features=[64,128,256,512]
class Discriminator(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channel*2,features[0],kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2))
        layers=[]
        in_features=features[0]
        for feature in features[1:]:
            layers.append(nn.Conv2d(in_features,feature,kernel_size=4,stride=2 if feature!=features[-1] else 1 ,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2))
            in_features=feature
        layers.append(nn.Conv2d(features[-1],1,kernel_size=4,stride=1,padding=1))
        self.model=nn.Sequential(*layers)
    
    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)
        x=self.initial(x)
        x=self.model(x)
        return x

class Down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.model(x)

class Up(nn.Module):
    def __init__(self,in_channel,up_channel,drop=False):
        super().__init__()
        self.model=nn.Sequential(
            nn.ConvTranspose2d(in_channel,up_channel,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(),
            nn.Dropout() if drop else nn.Identity()
        )
    def forward(self,x):
        return self.model(x)

class Generator(nn.Module):
    def __init(self,in_channels=3,features=64):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels,features,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2)
        )
        self.down1=Down(features,features*2)
        self.down2=Down(features*2,features*4)
        self.down3=Down(features*4,features*8)
        self.down4=Down(features*8,features*8)
        self.down5=Down(features*8,features*8)
        self.down6=Down(features*8,features*8)
        self.down7=Down(features*8,features*8)
        self.up1=Up(features*8,features*8,drop=True)
        self.up2=Up(features*8*2,features*8,drop=True)
        self.up3=Up(features*8*2,features*8,drop=True)
        self.up4=Up(features*8*2,features*8)
        self.up5=Up(features*8*2,features*4)
        self.up6=Up(features*4*2,features*2)
        self.up7=Up(features*2*2,features)
        self.final=nn.Sequential(
            nn.ConvTranspose2d(features*2,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def forward(self,x):
        d1=self.initial(x)
        d2=self.down1(d1)
        d3=self.down2(d2)
        d4=self.down3(d3)
        d5=self.down4(d4)
        d6=self.down5(d5)
        d7=self.down6(d6)
        d8=self.down7(d7)
        up1=self.up1(d8)
        up2=self.up2(torch.cat([up1,d7],dim=1))
        up3=self.up3(torch.cat([up2,d6],dim=1))
        up4=self.up4(torch.cat([up3,d5],dim=1))
        up5=self.up5(torch.cat([up4,d4],dim=1))
        up6=self.up6(torch.cat([up5,d3],dim=1))
        up7=self.up7(torch.cat([up6,d2],dim=1))
        return self.final(torch.cat([up7,d1],dim=1))
        