'''
MIT License

Copyright (c) 2018 Udacity
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torch import nn, optim



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class FFClassifier(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.5):
        super(FFClassifier, self).__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        x = F.log_softmax(x, dim=1)
        return x


def build_model(fc_hidden, fc_out, arch='vgg16', learning_rate=0.001, drop_prob=0.5):


    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if arch.startswith('vgg'):
        fc_in = model.classifier[0].in_features
        model.classifier = FFClassifier(fc_in, fc_hidden, fc_out, drop_prob)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch.startswith('alexnet'):
        fc_in = model.classifier[1].in_features
        model.classifier = FFClassifier(fc_in, fc_hidden, fc_out, drop_prob)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch.startswith('res'):
        model.fc = FFClassifier(model.fc.in_features, fc_hidden, fc_out, drop_prob)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def save_model(image_datasets, model, arch, hidden_units, drop_prob, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'arch': arch,
                'hidden': hidden_units,
                'drop_prob':  drop_prob,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, 
                save_dir+'/'+arch+'_classifier.pt')


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']

    if checkpoint['arch'].startswith('vgg'):
        fc_in = model.classifier[0].in_features
        model.classifier = FFClassifier(fc_in, checkpoint['hidden'], len(model.class_to_idx), checkpoint['drop_prob'])
    elif checkpoint['arch'].startswith('alexnet'):
        fc_in = model.classifier[1].in_features
        model.classifier = FFClassifier(fc_in, checkpoint['hidden'], len(model.class_to_idx), checkpoint['drop_prob'])
    elif checkpoint['arch'].startswith('res'):
        model.fc = FFClassifier(model.fc.in_features, checkpoint['hidden'], len(model.class_to_idx), checkpoint['drop_prob'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model





