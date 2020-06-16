import torch
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import os
import sys

def _load_data(batch_size,num_workers) -> torch.utils.data.DataLoader:
    """ Returns training and test data """
    
    ## defining the folders
    train_data_raw = os.path.abspath('./data/train/')
    test_data_raw = os.path.abspath('./data/test/')
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    
    ## transformations
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor()])
    ## selecting the folders
    train_data = datasets.ImageFolder(train_data_raw, transform=data_transform)
    test_data = datasets.ImageFolder(test_data_raw, transform=data_transform)
    
    ## loading the data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               num_workers=num_workers, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                              num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def _build_pretrained_model(cuda=True) -> torchvision.models.vgg16:
    """ Returns a built model """

    model = models.vgg16(pretrained=True)
    
    for parameter in model.features.parameters():
        parameter.requires_grad = False
        
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=5, bias=True)
    
    if cuda:
        model.cuda()
    
    return model

def train_model(epochs:int) -> torchvision.models.vgg16:
    """ Train a model """
    
    train_loss = 0.0
    model = _build_pretrained_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    train,test = _load_data(32,0)
    
    for epoch in range(epochs):
        for data in train:
            
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            
            train_loss += loss.item()*inputs.size(0)
            
            optimizer.step()
            optimizer.zero_grad()

    train_loss = train_loss/len(train.dataset)
    
    return model, train, test, train_loss


def test_model(model:torchvision.models.vgg16, test:torch.utils.data.DataLoader):
    """ Eval a model using test data """
    
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    correct, total = 0,0
    predictions = []
    model.eval()
    
    for data in test:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs,labels)
        test_loss += loss.item()*inputs.size(0)

        predictions.append(predicted)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_loss = test_loss/len(test.dataset)
    acc = (100*correct/total)

    return round(acc,2), test_loss


def save_model(model:torchvision.models.vgg16, acc:int):
    torch.save(model.state_dict(),f'TrainModel/models/model{acc}')


def main(wanted_acc:float):

    acc = 0
    epochs = 1
    acc_wanted = wanted_acc*100

    while acc < acc_wanted:
        model, train, test, train_loss = train_model(epochs)
        acc, test_loss = test_model(model, test)
        epochs += 1

        if acc < acc_wanted:
            print(f'\nMODEL BELOW WANTED ACC\nAVG TRAIN LOSS : {train_loss}\nAVG TEST LOSS : {test_loss}\nACC : {acc}\nINCREASING EPOCHS TO {epochs}')

    save_model(model,acc)

    print(f'\nTraining finished!\nEPOCHS NEEDED {epochs-1}\nAVG TRAIN LOSS : {train_loss}\nAVG TEST LOSS : {test_loss}\nACC : {acc}\nMODEL SAVED: TrainModel/models/model{acc}\n')

if __name__ == "__main__":
    wanted_acc = sys.argv[1]
    main(float(wanted_acc))