import train_utils
from torch import nn
from torch.autograd import Variable

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
    
    # plot_most_uncertain(outputs, corrects, predictions, loss, k=20)
    
#     for i, cor in enumerate(corrects):
#         if predictions[i] == 1 and cor == 0:
#             print(test_filepaths[i])
#             print(test_targets[i])
#             img = test_features[i]
#             plt.imshow(img)
#             plt.show()
             
    image_indices = list(range(0, total))
#     plot_images(loss, image_indices, test_filepaths, targets, predictions, phase='test', amount=15)
    
    test_accuracy = correct / total * 100
    
    print('Accuracy of model: {}'.format(test_accuracy)
    
    # show_cm(list(targets.data), list(predictions))

# print('Confusion matrix test set f1 macro:')
# test_model(model_f1, test_dataloader)
# print('Confusion matrix test set average accuracy:')
# test_model(model_avg, test_dataloader)