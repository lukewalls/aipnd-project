import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import argparse
import json


def load_checkpoint(img_path):
    """
    Loads a checkpoint of a previously trained neural network.
    """
    checkpoint = torch.load(img_path)
    arch = checkpoint['arch']
    
    if arch == 'resnet101':
        print(f'Building model: {arch}')
        model = models.resnet101(pretrained=True)
        model.fc = checkpoint['classifier']
    elif arch == 'densenet161':
        print(f'Building model: {arch}')
        model = models.densenet161(pretrained=True)
        model.classifier = checkpoint['classifier']
    elif arch == 'vgg16':
        print(f'Building model: {arch}')
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    else:
        print(f'Invalid model architecture of {arch} from checkpoint. Please use a checkpoint with a valid architecture. Options include: \n 1) resnet101 \n 2) densenet161 \n 3) vgg16')
        quit()

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.state_dict = checkpoint['optimizer_state']
    epochs = checkpoint['epochs']

    classifier = model.fc if arch == 'resnet101' else model.classifier
    print(f'Classifier: {classifier}')
    print(f'Optimizer: {optimizer}')
    print(f'Epochs: {epochs}')
    
    return model, optimizer, epochs


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Thumbnail or resize to 256 (thumbnail keeps aspect ratio)
    image.thumbnail((256, 256), Image.ANTIALIAS)
    
    # Crop center 224x224. Input is a 4-tuple: left, upper, right, lower
    (left, upper, right, lower) = (16, 16, 240, 240)
    image = image.crop((left, upper, right, lower))
    
    # Color channels
    # Normalize by dividing by max of 255
    image = np.array(image)/255
    
    # Normalize image. Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
    # image is already a np_array. Subtract mean, divide by std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image-mean)/std
    
    # First dimension: transpose
    # Color channel first
    image = np.transpose(image, (2, 0, 1))

    return image.astype(np.float32)


def predict(img_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    # Open in PIL and process image
    image = Image.open(img_path)
    image = process_image(image)
    
    # Move from np to tensor
    image = torch.from_numpy(image)
    
    # Convert to 1D vector
    image = image.unsqueeze(0)

    image = image.to(device)

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.cpu().topk(top_k, dim=1)
    
    # Invert original model.class_to_idx (swap key/value positions)
    class_idx = dict([[v,k] for k, v in model.class_to_idx.items()])
    
    # Retrieve indices to later convert from indices to class labels
    top_class = [class_idx[x] for x in top_class.numpy()[0]]
    top_p = top_p.detach().numpy()[0] #detach from gradient
    
    return top_p, top_class


def main():
    # Initialize Parser
    parser = argparse.ArgumentParser(
        description="Udacity: Intro to Machine Learning Deep Learning Project. Student: Luke Walls")

    parser.add_argument('img_path', action='store',
                        help="Location of the images you wish to use for your prediction.")
    parser.add_argument('checkpoint', action='store', default='flowers', help="Location of checkpoint you wish to use.")
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int,
                        help="Specify the number of top_k classes you would like returned.")
    parser.add_argument('--category_names', action='store', dest='category_names',
                        default='cat_to_name.json', help="Specify file for mapping categories to their real names.")
    parser.add_argument('--gpu', action='store_true', dest='gpu',
                        default=False, help="Use a GPU instead of a CPU.")
    results = parser.parse_args()
    
    # Assign arguments
    img_path = results.img_path
    checkpoint = results.checkpoint
    top_k = results.top_k
    category_names = results.category_names
    gpu = results.gpu

    # Device: GPU or CPU
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    # Open label map
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load checkpoint
    print('----- Load Checkpoint -----')
    model, optimizer, epochs = load_checkpoint(checkpoint)

    # Predict: flower name and probabilities
    print('----- Predict -----')
    probs, classes = predict(img_path, model, top_k, device)

    # map flower names
    class_names = [cat_to_name[x] for x in classes]

    for x in range(len(class_names)):
        print(f'Predicted Class {x+1}: {class_names[x]}  ||  ', f'Probability: {probs[x]:.5f}')

    print('----- Predict Complete -----')

if __name__ == '__main__':
    main()