import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR


def train_classifier(model, device, dataloaders, lr=0.001, epochs=30, model_checkpoint='trained_model.pt'):
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
    valid_loss_min = np.Inf

    for epoch in range(epochs):
        print("We are in epoch {}".format(epoch+1))
        scheduler.step()
        ######################    
        # train the model #
        ######################
        model.train()
        train_loss = 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*inputs.size(0)

        ######################    
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_accuracy = 0
            for inputs, labels in dataloaders["validation"]:
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                valid_loss += loss.item()*inputs.size(0)

                ps = torch.exp(logps)
                equality = (labels.data==ps.max(dim=1)[1])
                valid_accuracy += equality.type(torch.FloatTensor).mean()*inputs.size(0)
    
        # calculate average losses
        train_loss = train_loss/len(dataloaders["train"].dataset)
        valid_loss = valid_loss/len(dataloaders["validation"].dataset)
        valid_accuracy = valid_accuracy/len(dataloaders["validation"].dataset)
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValid Loss: {:.6f}\tValid Accuracy {:.6f}'.format(
            epoch+1, train_loss, valid_loss, valid_accuracy*100))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Valid loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), model_checkpoint)
            valid_loss_min = valid_loss

