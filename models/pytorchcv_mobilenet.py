from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class PretrainedMobilenetWD4(nn.Module):
    """
    MobileNet x0.25	-> https://github.com/osmr/imgclsmob/tree/master/pytorch
    """
    def __init__(self, num_classes, pretrained=True, size=224):
        super(PretrainedMobilenetWD4, self).__init__()
        self.size = size
        mobilenet = ptcv_get_model("mobilenet_wd4", pretrained=pretrained)

        if self.size == 224:
            self.model = nn.Sequential(*(list(mobilenet.children())[0]))
        else:
            self.model = nn.Sequential(*(list(mobilenet.children())[0][:-1]))
            self.avgpool = nn.AvgPool2d(int(self.size / 32), stride=1)

        self.last_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.model(x)
        if self.size != 224:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.last_fc(out)
        return out

