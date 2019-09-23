import torchvision.models as models
import torch.nn as nn
from .resnet import *


def model_selector(model_name, num_classes=9, pretrained=False):
    if pretrained:
        print("Pretrained-> Remember at end: {}".format("transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"))

    if model_name == "resnet18":
        if not pretrained:
            return ResNet18(num_classes=num_classes).cuda()
        else:
            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
            for param in resnet18.parameters(): # Defrost model
                param.requires_grad = True
            print(resnet18)
            return resnet18.cuda()
    if model_name == "resnet34":
        return ResNet34(num_classes=num_classes).cuda()
    if model_name == "resnet50":
        return ResNet50(num_classes=num_classes).cuda()
    if model_name == "resnet101":
        return ResNet101(num_classes=num_classes).cuda()
    if model_name == "resnet152":
        return ResNet152(num_classes=num_classes).cuda()
    else:
        assert False, "Uknown model selected!"

def test():
    net = model_selector("resnet18", num_classes=9)
    y = net(torch.randn(1, 3, 32, 32).cuda())
    print(y.size())

#test()