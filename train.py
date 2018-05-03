'''
MIT License

Copyright (c) 2018 Udacity
'''


import time

import torch
from torch.autograd import Variable

import argparse
import os

import numpy as np
from model import build_model, save_model
from utils import load_data


parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_directory', type=str, required=True,
                    help='training, validation and test directory in order separated by a comma')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
                   'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'alexnet', 'resnet101', 'resnet152', 'resnet18', 
                   'resnet34', 'resnet50'],
                    help='type of architecture used')
parser.add_argument('--hidden_units', type=int, default=512,
                    help='number of hidden units')
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs')
parser.add_argument('--print_every', type=int, default=20,
                    help='print frequency')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout_prob', type=float, default=0.5,
                    help='probability of dropping weights')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='run the network on the GPU')

args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    raise OSError('Directory'+str(args.save_dir)+'does not exist.')

def validation(model, val_data, criterion, cuda=False, is_test=False):
    val_start = time.time()
    running_val_loss = 0
    accuracy = 0
    for inputs, labels in val_data:
        inputs, labels = Variable(inputs), Variable(labels)
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        val_loss = criterion(outputs, labels)

        ps = torch.exp(outputs.data)
        
        _, predicted = ps.max(dim=1)
        
        equals = predicted == labels.data
        accuracy += float(torch.sum(equals))/len(equals)
        
        running_val_loss += val_loss.data[0]
    val_time = time.time() - val_start
    if not is_test:
    	print("Validation loss: {:.3f}".format(running_val_loss/len(val_data)),
          	"Validation Accuracy: {:.3f}".format(accuracy/len(val_data)),
          	"Validation time: {:.3f} s/batch".format(val_time/len(val_data)))
    else:
    	print("Test loss: {:.3f}".format(running_val_loss/len(val_data)),
          	"Test Accuracy: {:.3f}".format(accuracy/len(val_data)),
          	"Test time: {:.3f} s/batch".format(val_time/len(val_data)))


def train(dataloaders, model, criterion, optimizer, epochs=10, cuda=False):

    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.train()
    for e in range(epochs):
        print("Epoch  "+str(e+1)+'/'+str(epochs))
        counter = 0
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            counter += 1

            # Training pass
            inputs, labels = Variable(inputs), Variable(labels)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if counter % args.print_every == 0:
                print("Step: {}".format(counter))
                print("Training loss {:.3f}".format(running_loss/counter))
                model.eval()
                validation(model, dataloaders['valid'], criterion, cuda=cuda)
                model.train()
        else:
            # Validation pass
            train_end = time.time()
            model.eval()
            validation(model, dataloaders['valid'], criterion, cuda=cuda)

    if 'test' in dataloaders:
        if cuda:
            model.cuda()
        model.eval()
        validation(model, dataloaders['test'], criterion, cuda=cuda, is_test=True)


if __name__ == '__main__':
	train_dir, valid_dir, test_dir = str(args.data_directory).split(',')
	image_datasets, dataloaders = load_data(train_dir, valid_dir, test_dir, args.batch_size)
	model, criterion, optimizer = build_model(args.hidden_units, len(image_datasets['train'].classes), args.arch, args.learning_rate, args.dropout_prob)
	train(dataloaders, model, criterion, optimizer, args.num_epochs, args.gpu)
	save_model(image_datasets, model, args.arch, args.hidden_units, args.dropout_prob, args.save_dir)
