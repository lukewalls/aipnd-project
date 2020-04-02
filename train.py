# Imports
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import argparse
import time
import json


# arch options: resnet101, densenet161, vgg16
def model_build(data_dir, arch, learning_rate, hidden_units, dropout, device):
    """
    Builds the selected deep learning model. Returns the model, optimizer, and criterion.
    """
    if arch == 'resnet101':
        print(f'Building model: {arch}')
        model = models.resnet101(pretrained=True)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif arch == 'densenet161':
        print(f'Building model: {arch}')
        model = models.densenet161(pretrained=True)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(model.classifier.in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch == 'vgg16':
        print(f'Building model: {arch}')
        model = models.vgg16(pretrained=True)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        print("Invalid model architecture. Available model options include: \n 1) resnet101 \n 2) densenet161 \n 3) vgg16")
        quit()

    model = model.to(device)

    return model, optimizer


def model_train(model, optimizer, criterion, trainloader, validloader, device, epochs):
    """
    Trains and validates the selected deep learning model.
    """
    since = time.time()

    train_losses, valid_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Move image and label tensors to device
            images, labels = images.to(device), labels.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Run training
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            # Place model into eval
            model.eval()

            # Turn off gradients
            with torch.no_grad():
                for images, labels in validloader:
                    # Move image and label tensors to device
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    valid_loss += loss.item()

                    # Calculate accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            model.train()

            print("Epoch: {}/{}   ".format(e+1, epochs),
                "Training Loss: {:.3f}   ".format(running_loss/len(trainloader)),
                "Valid Loss: {:.3f}   ".format(valid_loss/len(validloader)),
                "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
        
    time_elapsed = time.time() - since
    print("Training duration (seconds): {:.3f}".format(time_elapsed))

    return model


def main():
    # Initialize Parser
    parser = argparse.ArgumentParser(
        description="Udacity: Intro to Machine Learning Deep Learning Project. Student: Luke Walls")

    parser.add_argument('data_dir', action='store',
                        help="Location of the images you wish to use with the model.")
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='checkpoints', help="Location where you want to save checkpoints.")
    parser.add_argument('--arch', action='store', dest='arch', default='resnet101',
                        help="Model architecture. Option: resnet101, densenet161, vgg16")
    parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                        default=0.003, type=float, help="Optimizers learning rate.")
    parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                        default=852, type=int, help="The size of the hidden layer.")
    parser.add_argument('--epochs', action='store', dest='epochs',
                        default=7, type=int, help="Number of epochs or iterations.")
    parser.add_argument('--dropout', action='store', dest='dropout',
                        default=0.2, type=float, help="Dropout percent value for classifier.")
    parser.add_argument('--gpu', action='store_true', dest='gpu',
                        default=False, help="Use a GPU instead of a CPU.")
    results = parser.parse_args()

    # Assign arguments
    data_dir = results.data_dir
    save_dir = results.save_dir
    arch = results.arch
    learning_rate = results.learning_rate
    hidden_units = results.hidden_units
    epochs = results.epochs
    dropout = results.dropout
    gpu = results.gpu

    # Define sub-directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Open label map
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Define device: GPU or CPU
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    # Build model
    print('----- Building Model -----')
    model, optimizer = model_build(data_dir, arch, learning_rate, hidden_units, dropout, device)
    print(f'Selected model: {arch}')
    print('Classifier: ' + str(model.fc) if arch == 'resnet101' else str(model.classifier))
    print('Optimizer: {}'.format(optimizer))

    # Criterion
    criterion = nn.NLLLoss()
    print('Criterion: {}'.format(criterion))

    # Train model
    print('----- Training Model -----')
    model = model_train(model, optimizer, criterion, trainloader, validloader, device, epochs)

    # Test model
    print('----- Testing Model -----')
    since = time.time()

    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            # Move image and label tensors to device
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()

    print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

    time_elapsed = time.time() - since
    print("Testing duration (seconds): {:.3f}".format(time_elapsed))

    # Save Checkpoint
    # Attach class_to_idx as an attribute to our model
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                  'classifier': model.fc if arch == 'resnet101' else model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer,
                  'optimizer_state': optimizer.state_dict,
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('Checkpoint successfully saved to: ' + save_dir + '/checkpoint.pth')
    print('----- Training Completed -----')


if __name__ == '__main__':
    main()
