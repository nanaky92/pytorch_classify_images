from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

model_name_to_model = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg11_bn': models.vgg11_bn,
    'vgg13_bn': models.vgg13_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn
}


def get_model_arch_from_model_name(model_name):
    return model_name_to_model[model_name]


def get_pretrained_model(model_arch):
    model = model_arch(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    return model


class Model(nn.Module):
    def __init__(self, hidden_units_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(25088, hidden_units_size[0])
        self.fc2 = nn.Linear(hidden_units_size[0], hidden_units_size[1])
        self.fc3 = nn.Linear(hidden_units_size[1], hidden_units_size[2])
        self.fc4 = nn.Linear(hidden_units_size[2], 102)
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc4(x), dim=1)

def get_model(model_arch, hidden_units_size):
    # Get pretrained model
    model = get_pretrained_model(model_arch)
    # Create a classifier
    classifier = Model(hidden_units_size)
    # Overwrite default classifier with our own
    model.classifier = classifier
    
    return model
