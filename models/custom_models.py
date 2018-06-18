from torch import nn

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

class simple_CNN(nn.Module):
    def __init__(self):
        super(simple_CNN, self).__init__()
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