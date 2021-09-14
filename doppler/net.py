#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision import models
#from torchsummary import summary

class DopplerBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(224*224*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.flatten(x) # flatten all dimensions except first?
        logits = self.linear_relu_stack(x) # shape is (1, 1)
        logits = logits.flatten(start_dim=0) # now shape is (1,)
        return logits

class Doppler(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()

        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)

        # Change the last FC layer to have a single output.
        num_features = self.model.fc.in_features # number of in_features in the last FC layer.
        self.model.fc = nn.Linear(num_features, 1) # Replace the last FC layer.

    def forward(self):
        return 

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
from torchvision.models.resnet import resnet18
#model = resnet18()
def myResNet18(freeze_layers=None, dropout_rate=0, pretrained=True, out_features=1):
    model = resnet18(pretrained=pretrained)
    fc_in_features = model.fc.in_features
    if dropout_rate:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_in_features, out_features),
        )
    else:
        model.fc = nn.Linear(fc_in_features, out_features)

    if freeze_layers:
        for layer in list(model.children())[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False # Freeze the layer.
    
    return model
    
if __name__ == '__main__':
#    model = Doppler()
#    print(model)
#    print(model.forward)

    ## Let's check to see if torch.cuda is available, else we continue to use the CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))
    
    #model = DopplerBasic()
    model = myResNet18(freeze_layers=5)
    model = model.to(device)
    #print(model)
    
    for layer in list(model.children()):
        print(layer)
        for param in layer.parameters():
            print(param.requires_grad)
