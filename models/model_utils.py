from torch import nn
import torch
import torchvision
import torch.nn as nn

def get_model(args):
    '''
    Retrieve model to train. 

    :param args: parser arguments
    :return: model that is specified in args.model_name
    '''

    if args.model_name == 'resnet18':

        model = torchvision.models.resnet18(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        for parameter in model.parameters():
            parameter.requires_grad = False

        # Replace the fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, args.num_classes)
        
        return model

    elif args.model_name == 'shallow_CNN':

        model = shallow_CNN()

        return model

    elif args.model_name == 'resnet+meteo_NN':
        
        resnet_model = torchvision.models.resnet18(pretrained=True)
        resnet_model.avgpool = nn.AdaptiveAvgPool2d(1)
        meteo_NN = meteo_NN(args.meteo_inputs, args.meteo_hidden_size, args.meteo_outputs)
        model = resnet18_meteo(resnet_model, meteo_NN, args.num_classes)
        
        return model

    else:
        raise ValueError("Define one of 'resnet18', 'simple_CNN' or 'resnet+meteo_NN'")

class meteo_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(meteo_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class resnet18_meteo(nn.Module):
    def __init__(self, resnet18, meteo_NN, num_classes):
        super(resnet18_meteo, self).__init__()
        
        # Respectively a torchvision resnet-18 and a 1-hidden layer NN
        self.resnet_CNN = resnet18
        self.meteo_net = meteo_NN
        
        # Sizes of the FC layers of both NN's
        self.len_fc_resnet = self.resnet_CNN.fc.in_features
        self.len_fc_meteo = self.meteo_net.fc2.out_features
        
        # Extract convolutional block out of predefined network
        self.modules=list(self.resnet_CNN.children())[:-1]
        self.resnet18_convblocks= nn.Sequential(*self.modules)

        self.fc = nn.Linear(self.len_fc_resnet + self.len_fc_meteo, num_classes)

    def forward(self, img_x, meteo_x):

        # Both should be flattened layers at end of networks
        img_x = self.resnet18_convblocks(img_x)
        meteo_x = self.meteo_net(meteo_x)

        # Flatten convolutional features
        img_x_flattened = img_x.view(img_x.size(0), -1)

        # Concat the outputs of CNN and meteo-NN in fully connected layer
        out = torch.cat([img_x_flattened, meteo_x], dim=1)

        out = self.fc(out)
        return out   

class shallow_CNN(nn.Module):
    def __init__(self):
        super(shallow_CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=3,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )     
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(576, 3)   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        output = self.out(x)
        return output