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
            for param in resnet18.parameters():  # Defrost model
                param.requires_grad = True
            print(resnet18)
            return resnet18.cuda()
    if model_name == "resnet34":
        if not pretrained:
            return ResNet34(num_classes=num_classes).cuda()
        else:
            resnet34 = models.resnet34(pretrained=True)
            resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
            for param in resnet34.parameters():  # Defrost model
                param.requires_grad = True
            print(resnet34)
            return resnet34.cuda()
    if model_name == "resnet50":
        if not pretrained:
            return ResNet50(num_classes=num_classes).cuda()
        else:
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
            for param in resnet50.parameters():  # Defrost model
                param.requires_grad = True
            print(resnet50)
            return resnet50.cuda()
    if model_name == "resnet101":
        if not pretrained:
            return ResNet101(num_classes=num_classes).cuda()
        else:
            resnet101 = models.resnet101(pretrained=True)
            resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)
            for param in resnet101.parameters():  # Defrost model
                param.requires_grad = True
            print(resnet101)
            return resnet101.cuda()
    if model_name == "resnet152":
        if not pretrained:
            return ResNet152(num_classes=num_classes).cuda()
        else:
            resnet152 = models.resnet152(pretrained=True)
            resnet152.fc = nn.Linear(resnet152.fc.in_features, num_classes)
            for param in resnet152.parameters():  # Defrost model
                param.requires_grad = True
            print(resnet152)
            return resnet152.cuda()
    else:
        assert False, "Uknown model selected!"

def test():
    net = model_selector("resnet18", num_classes=9)
    y = net(torch.randn(1, 3, 32, 32).cuda())
    print(y.size())

#test()