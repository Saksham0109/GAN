from torch import nn
class Generator(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.gen=nn.Sequential(
            nn.Linear(in_features,256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(256,out_features),
            nn.Tanh()
        )

    def forward(self,z):
        return self.gen(z)

class Discriminator(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Linear(in_features,256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,img):
        return self.disc(img)
