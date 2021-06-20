"""
    Imports
"""
import torch as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

from tqdm import trange
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score



"""
    Base Model Architecture
"""
class Net( Module ):
    def __init__( self ):
        #init super class
        super(Net, self).__init__()
        
        #define conv layer arch
        self.conv1 = Conv2d(1, 32, 2)
        self.conv2 = Conv2d(32, 64, 2)
        self.conv3 = Conv2d(64, 128, 2)
        
        #1-step forwrd prop for conv output shape
        t = nn.randn( 28,28 ).view( -1, 1, 28, 28 )
        self._linear_shape = None
        self.conv_pipe( t )
                
        #fully-connected layers
        self.fc1 = Linear( self._linear_shape, 512 )
        self.fc2 = Linear( 512, 10 )
        
    def conv_pipe(self, x):
        #first conv layer with an activation function and pooling
        x = self.conv1( x )
        x = F.relu( x )
        x = F.max_pool2d( x, (2, 2) )
        
        #second conv layer with an activation function and pooling
        x = self.conv2( x )
        x = F.relu( x )
        x = F.max_pool2d( x, (2, 2) )
        
        #third conv layer with an activation function and pooling
        x = self.conv3( x )
        x = F.relu( x )
        x = F.max_pool2d( x, (2, 2) )
        
        #forward pass to get shape of the flattened conv layer,
        #only runs once when class instantiated
        if self._linear_shape is None:
            shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self._linear_shape = shape
            
        return x
    
    def forward( self, x ):
        #conv layers
        x = self.conv_pipe( x )
        
        #shape
        x = x.view( -1, self._linear_shape )
        
        #linear layers
        x = F.relu( self.fc1( x ) )
        x = self.fc2( x )
        
        #activated output classications
        #loss wasn't changing until I changed this.
        #return F.log_softmax( x, dim=1)
        return x #for the sake of generality the output should be logits.

"""
    Model training class
    Should make it easier to get model learned more quickly.
"""
class Train_Net_MNIST( object ):

    def __init__(self, ds_loader, batch_size=30, epochs=5, learning_rate=0.0001):

        self.loader = ds_loader
        self.loaded = False
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.LOSS_HISTORY = None

    def load_dataset(self):
        mnist_loader = self.loader()
        self.x_train, self.y_train, self.x_test, self.y_test = mnist_loader.load_mnist()

    def train_net_model(self, model):

        self.LOSS_HISTORY = []

        self.model = model

        if self.loaded == False:
            self.load_dataset()
            self.loaded = True

        optimizer = Adam( model.parameters(), lr=self.LEARNING_RATE )

        for epoch in range(self.EPOCHS): 
            for i in trange( 0, len(self.x_train), self.BATCH_SIZE):
                #X and ygt of batch
                bX = self.x_train[i:i+self.BATCH_SIZE].view(-1, 1, 28, 28)
                by = self.y_train[i:i+self.BATCH_SIZE]
                
                #train model, calculate loss
                model.zero_grad()
                outputs = model( bX.float() )
                outputs = F.log_softmax( outputs, dim=1)
                loss = F.nll_loss( outputs, by )
                
                #store loss
                self.LOSS_HISTORY.append( loss )
                
                #backward step
                loss.backward()
                optimizer.step()
                
                
            with nn.no_grad():
                output = model( self.x_test.view(-1, 1, 28, 28).float() )
            
            sf = F.softmax( output, dim=1 )
            prob = list( sf.numpy() )
            predictions = np.argmax(prob, axis=1)
            print( f"Epoch done: { epoch + 1 } / { self.EPOCHS }, accuracy: {accuracy_score( self.y_test, predictions )}", flush=True)

if __name__ == "__main__":
    import sys
    sys.path.append("/Users/jameskunstle/Documents/dev/Adversarial-ML-GK/loaders")

    from mnist_loader import MNIST_Dataset_Loader

    model = Net().float()
    training_object = Train_Net_MNIST(ds_loader = MNIST_Dataset_Loader)

    training_object.train_net_model( model )