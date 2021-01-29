from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class PretrainedSeresNext50(nn.Module):
    def __init__(self, num_classes, pretrained=True, size=224):
        super(PretrainedSeresNext50, self).__init__()
        self.size = size
        seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=pretrained)

        if self.size == 224:
            self.model = nn.Sequential(*(list(seresnext50.children())[0]))
        else:
            self.model = nn.Sequential(*(list(seresnext50.children())[0][:-1]))
            self.avgpool = nn.AvgPool2d(int(self.size / 32), stride=1)

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.model(x)
        if self.size != 224:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.last_fc(out)
        return out


class PretrainedSeresNext101(nn.Module):
    def __init__(self, num_classes, pretrained=True, size=224):
        super(PretrainedSeresNext101, self).__init__()
        self.size = size
        seresnext101 = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)

        if self.size == 224:
            self.model = nn.Sequential(*(list(seresnext101.children())[0]))
        else:
            self.model = nn.Sequential(*(list(seresnext101.children())[0][:-1]))
            self.avgpool = nn.AvgPool2d(int(self.size / 32), stride=1)

        self.last_fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        out = self.model(x)
        if self.size != 224:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.last_fc(out)
        return out