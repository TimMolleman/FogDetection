import torch
from torch import nn
from torch.autograd import Variable
from helpers.model_plotters import show_cm

def test_model(model, dataloader, args):
    """
    Tests a specified model on all the manually labeled highway camera
    images. 
    
    :param model: Trained model to evaluate
    :param test_features: All test features as tensor
    :param test_targets: All test labels as tensor
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

    # Print results
    print('Accuracy of {}: {}'.format(args.model_name, test_accuracy))
    show_cm(list(targets.data), list(predictions))
    
