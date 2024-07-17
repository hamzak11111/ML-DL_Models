import torch
import torch.nn as nn

##### Key Responsibilities of nn.Module
##### Parameter Management:

##### Attribute Registration: nn.Module automatically registers all the parameters (such as weights and biases) and submodules (layers, other modules) assigned as attributes. This allows easy access to all parameters and submodules when needed.
##### Parameter Storage: Parameters are stored as torch.nn.Parameter objects, which are essentially torch.Tensor objects with the additional property of being registered as a parameter of a module.
##### Forward and Backward Pass:
##### Forward Pass: The forward method of an nn.Module defines the computation performed at every call. It is overridden by subclasses to specify the forward computation.
##### Backward Pass: The gradients for parameters are automatically computed using backpropagation when loss gradients are propagated back through the network during training.
##### Parameter Initialization:
##### Parameters of submodules are automatically initialized according to best practices unless explicitly overridden by the user.
##### State Management:
##### State Dicts: nn.Module provides methods to save and load the state of the model, such as state_dict() and load_state_dict(). This is useful for saving models to disk and later restoring them.
##### Mode Switching: Methods like train() and eval() are used to switch the module between training and evaluation modes, affecting layers like dropout and batch normalization which behave differently during training and evaluation.
##### Device Management:
##### Device Handling: nn.Module supports moving the entire module to a specific device (CPU or GPU) using methods like to() and cuda(). This is crucial for utilizing hardware accelerators.


architecture_config = [
    # Tuple: Conv Layer(kernel_size,num_filters,stride,padding)
    (7, 64, 2, 3), 
    "M", # String: max pooling
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: [(kernel_size,num_filters,stide,padding),(kernel_size,num_filters,stide,padding),num_repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# new class named CNNBlock that inherits from nn.Module, 
# which is the base class for all neural network modules in PyTorch.
class CNNBlock(nn.Module):


    # **kwargs is used in function and method definitions to allow the function or method to accept an arbitrary number of keyword arguments. 
    # kwargs stands for "keyword arguments", and the ** syntax tells Python to gather any keyword arguments passed to the function 
    # and store them in a dictionary.
    def __init__(self,in_channels,out_channels,**kwargs):
        super(CNNBlock,self).__init__() # Calls the constructor of the parent class nn.Module to ensure that the object is properly initialized as an nn.Module.

        # Initializes a 2D convolutional layer with the specified number of input and output channels. 
        # The bias=False argument indicates that the convolutional layer will not use a bias term. 
        # **kwargs allows passing additional parameters like kernel size, stride, padding, etc
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) # normalizes the activations of the output from the convolutional layer.
        self.leakyrelu = nn.LeakyReLU(0.1) # Initializes a leaky ReLU activation function with a negative slope of 0.1

    def forward(self,x): 
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self,in_channels=3,**kwargs): # in_channels defined as 3 by default for RGB
        super(Yolov1,self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self,x):

        x = self.darknet(x)

        # torch.flatten(x,start_dim=1) means to flatten starting from 2nd dimension
        # eg torch.Size([2, 3, 4, 5]) -> torch.Size([2, 60])
        return self.fcs(torch.flatten(x,start_dim=1))
    
    def _create_conv_layers(self,architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers +=[ CNNBlock(in_channels,
                                    out_channels=x[1],
                                    kernel_size=x[0],
                                    stride=x[2],
                                    padding=x[3])
                        ]
                
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)] 

            elif type(x)==list:
                conv1 = x[0]
                conv2 = x[1]

                for _ in range(x[2]):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        # *layers syntax unpacks the list, passing each layer as a separate argument to the nn.Sequential constructor.
        return nn.Sequential(*layers)
    
    def _create_fcs(self,split_size,num_boxes,num_classes):
        # split size is S grid size
        # num_boxes is number of boxes we want that each cell should output
        # num_classes is the total number of classes we want to predict
        S,B,C = split_size, num_boxes, num_classes
        return nn.Sequential(
                nn.Flatten(), # flatten tensor starting from first non-batch dimension
                nn.Linear(1024*S*S,496), # in original paper it was 4096
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(496,S*S*(C+B*5)) # (S,S,30) as C+b*5 = 30
        )
    

def test(S=7,B=2,C=20):
    model = Yolov1(split_size=S,num_boxes=B,num_classes=C)
    # 2 batch size
    # 3 channels
    # 448,448 is the size that YOLOv1 takes
    x = torch.randn((2,3,448,448))
    print(model(x).shape)