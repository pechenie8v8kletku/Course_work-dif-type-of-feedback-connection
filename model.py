import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
'''
импортирум encoder От resnet18,добавляем слой Think_layer в котором мы будем анализровать разные виды обратной связи,и добавляем финальный слой классификации
Основные параметры:
num_laythink-глубина думающего блока,
percent-dropout_rate
num_think-число итераций сходимости думающего блока,
connection- функция возвращающая значение обратной связи от x,f(x) на вход



'''

Size=128
class RoflNET(nn.Module):
    def __init__(self,out_channels=120,num_lay1=1,num_laythink=4,percent=0.04,num_think=3,connection=None):
        super(RoflNET,self).__init__()
        weights= ResNet18_Weights.IMAGENET1K_V1
        modelres = resnet18(weights=weights)
        encoder = nn.Sequential(
        modelres.conv1,
        modelres.bn1,
        modelres.relu,
        modelres.maxpool,
        modelres.layer1,
        modelres.layer2,
        modelres.layer3,
        modelres.layer4
        )
        self.encoder=encoder
        self.Layers=nn.ModuleList()
        self.ThinkLayers=nn.ModuleList()
        self.out_channels=out_channels 
        self.num_laythink=num_laythink
        cur_channels=512
        self.connection=connection
        self.num_think = num_think
        '''self.conv4=nn.Sequential(
            nn.Conv2d(cur_channels,cur_channels*2,kernel_size=3,stride=1,padding=int((3-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cur_channels*2),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(cur_channels*2,cur_channels*2,kernel_size=3,stride=1,padding=int((3-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cur_channels*2)
        )'''
        '''for i in range(num_lay1):
            conv=nn.Sequential(
            nn.Conv2d(cur_channels,cur_channels*2,kernel_size=3,stride=1,padding=int((3-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cur_channels*2),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(cur_channels*2,cur_channels*4,kernel_size=3,stride=1,padding=int((3-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cur_channels*4),
            nn.AvgPool2d(kernel_size=5,stride=2,padding=int((5-1)//2))
            )
            cur_channels*=4
            self.Layers.append(conv)'''
        for i in range(num_laythink):
            conv=nn.Sequential(
            nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=1,padding=int((3-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=percent))
            self.ThinkLayers.append(conv)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Linear(cur_channels, out_channels)
    def forward(self,x):
        '''for layer in self.Layers:
            x=layer(x)'''
        
        x=self.encoder(x)
        
        z=x.clone()
        for _ in range(self.num_think):
            # Here I place type of conection
            '''основной блок итерационных вычислений для анализа модели
            '''
            
            for layer in self.ThinkLayers:
                x=x+layer(x)
            if self.connection is not None:
                x=self.connection(x,z)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
                
                
        
        return x

