from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class PretrainedResNext50_32(nn.Module):
    def __init__(self, num_classes, pretrained=True, size=224):
        super(PretrainedResNext50_32, self).__init__()
        self.size = size
        resnext50_32 = ptcv_get_model("resnext50_32x4d", pretrained=pretrained)

        if self.size == 224:
            self.model = nn.Sequential(*(list(resnext50_32.children())[0]))
        else:
            self.model = nn.Sequential(*(list(resnext50_32.children())[0][:-1]))
            self.avgpool = nn.AvgPool2d(int(self.size / 32), stride=1)

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.model(x)
        if self.size != 224:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.last_fc(out)
        return out


class PretrainedResNext101_32(nn.Module):
    def __init__(self, num_classes, pretrained=True, size=224):
        super(PretrainedResNext101_32, self).__init__()
        self.size = size
        resnext101_32 = ptcv_get_model("resnext101_32x4d", pretrained=pretrained)

        if self.size == 224:
            self.model = nn.Sequential(*(list(resnext101_32.children())[0]))
        else:
            self.model = nn.Sequential(*(list(resnext101_32.children())[0][:-1]))
            self.avgpool = nn.AvgPool2d(int(self.size / 32), stride=1)

        self.last_fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        out = self.model(x)
        if self.size != 224:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.last_fc(out)
        return out