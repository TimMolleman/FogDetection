import torch
from torch import nn
from torch.autograd import Variable
from helpers.model_plotters import show_cm

def test_model(model, dataloader, args):
    """
    Tests a specified model on all the manually labeled highway camera images. Shows confusion matrix and overall accuracy.
    
    :param model: trained model to evaluate
    :param dataloader: pytorch dataloader object for the train dataset
    :param args: parser arguments
    :return: confusion matrix
    """
    
    test_images, test_targets, idx, test_filepaths, meteo = next(iter(dataloader))
    
    # Loss criterion
    criterion = nn.CrossEntropyLoss(reduce=False)
    
    # Wrap tensors
    features = Variable(test_images)
    targets = Variable(test_targets)
    total = len(targets)
    
    if args.include_meteo:
        meteo = Variable(meteo.type(torch.FloatTensor))

    # Feed test features into model
    outputs = model(features)
    
    # Loss and optimization
    loss = criterion(outputs, targets)
    
    # Get test predictions and number of correct predictions
    _, predictions = torch.max(outputs.data, 1) 
    correct = torch.sum(predictions == targets.data)  
    corrects = predictions == targets.data
    test_accuracy = correct / total * 100

    # Print results and show the confusion matrix
    print('\nTest results:\nAccuracy of {}: {:.2f}%'.format(args.model_name, test_accuracy))
    cm = show_cm(list(targets.data), list(predictions))

    return cm

def load_model(filepath):
    '''
    Loads a trained model.
    
    :param filepath: path to the trained model
    :return: trained model
    '''
    model = torch.load(filepath, map_location=lambda storage, loc: storage)

    return model