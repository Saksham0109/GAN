from torch import nn 
class Generator(nn.Module):
    def __init__(self,z_dim=100,im_channel=3,hidden_dim=128):
        super().__init__()
        self.z_dim=z_dim
        self.im_channel=im_channel
        self.hidden_dim=hidden_dim
        self.gen=nn.Sequential(
            self.make_block(),
            nn.ConvTranspose2d(128,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh())
    
    def forward(self,z):
        z = z.view(len(z), self.z_dim, 1, 1)
        return self.gen(z)

    def make_block(self):
        layers=[nn.ConvTranspose2d(self.z_dim,self.hidden_dim*8,kernel_size=4,stride=1,padding=0),nn.BatchNorm2d(self.hidden_dim*8),nn.ReLU(inplace=True),nn.ReLU(inplace=True)]
        x=self.hidden_dim*8
        while x!=128:
            y=int(x/2)
            layers.append(nn.ConvTranspose2d(x,y,kernel_size=4,stride=2,padding=1))
            layers.append(nn.BatchNorm2d(int(x/2)))
            layers.append(nn.ReLU(inplace=True))
            x=y
        return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self,im_channel=3,hidden_dim=128):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(im_channel,hidden_dim,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2,inplace=True),
            self.make_block(hidden_dim,hidden_dim*2),
            self.make_block(hidden_dim*2,hidden_dim*4),
            self.make_block(hidden_dim*4,hidden_dim*8),
            nn.Conv2d(hidden_dim*8,1,kernel_size=4,stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc(x)

    def make_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
