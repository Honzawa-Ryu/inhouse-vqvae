import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class lpips_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True).features.eval()
        for param in vgg16.parameters():
            param.requires_grad = False

        self.features_1 = nn.Sequential(*list(vgg16.children())[:4])   # relu1_2
        self.features_2 = nn.Sequential(*list(vgg16.children())[4:9])  # relu2_2
        self.features_3 = nn.Sequential(*list(vgg16.children())[9:16]) # relu3_3
        self.features_4 = nn.Sequential(*list(vgg16.children())[16:23])# relu4_3
        self.features_5 = nn.Sequential(*list(vgg16.children())[23:30])# relu5_3

        del vgg16

        self.lin0 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.lin1 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, bias=False))
        self.lin2 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False))
        self.lin3 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, bias=False))
        self.lin4 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, bias=False))
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        
        try:
            pretrained_dict = torch.load('/opt/testenv/lib/python3.11/site-packages/lpips/weights/v0.1/vgg16.pth', map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {k: v.replace('model.1', '0') for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
    
        for lin in self.lins:
            for param in lin.parameters():
                param.requires_grad = False
        
        self.eval()

    def forward(self, x, y):
        x = self.features_1(x)
        y = self.features_1(y)
        diff = (F.normalize(x, p=2, dim=1) - F.normalize(y, p=2, dim=1)) ** 2
        val = torch.mean(self.lins[0](diff))

        x = self.features_2(x)
        y = self.features_2(y)
        diff = (F.normalize(x, p=2, dim=1) - F.normalize(y, p=2, dim=1)) ** 2
        val += torch.mean(self.lins[1](diff))

        x = self.features_3(x)
        y = self.features_3(y)
        diff = (F.normalize(x, p=2, dim=1) - F.normalize(y, p=2, dim=1)) ** 2
        val += torch.mean(self.lins[2](diff))

        x = self.features_4(x)
        y = self.features_4(y)
        diff = (F.normalize(x, p=2, dim=1) - F.normalize(y, p=2, dim=1)) ** 2
        val += torch.mean(self.lins[3](diff))

        x = self.features_5(x)
        y = self.features_5(y)
        diff = (F.normalize(x, p=2, dim=1) - F.normalize(y, p=2, dim=1)) ** 2
        val += torch.mean(self.lins[4](diff))

        return val