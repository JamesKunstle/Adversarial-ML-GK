"""
    Imports
"""
import torch as nn 
import torch.nn.functional as F

from tqdm import trange

import sys
sys.path.append("/Users/jameskunstle/Documents/dev/Adversarial-ML-GK/loaders")
sys.path.append("/Users/jameskunstle/Documents/dev/Adversarial-ML-GK/models")
from mnist_loader import MNIST_Dataset_Loader
from base_model import Train_Net_MNIST, Net

import matplotlib.pyplot as plt

"""
    Attack Class
"""
class SimBA_Attack(object):

    def __init__(self, model=None ):
        self.model = model
    
    def get_probs(self, x):
        if self.model is None:
            print("Model not defined!")
            return

        with nn.no_grad():
            output = self.model( x.view(-1, 1, 28, 28).float() )

        sf = F.softmax( output, dim=1 )
        probs = list( sf.numpy() )
        
        return probs[0]
    
    def simba( self, x, y, epsilon=0.01, steps=100000 ):
        if self.model is None:
            print("Model not defined!")
            return
    
        #last classification distribution (softmax)
        #of input image.
        last_probs = self.get_probs( x )
        
        #number of dimensions of the input
        ndims = x.view( 1, -1 ).size(1)
        
        #random basis dimensions
        perm = nn.randperm( ndims )
        
        #collect history of successful perturbations
        pert_history = nn.zeros( ndims )
        
        for i in trange( steps ):
            
            #if the model misclassifies the input
            if( last_probs[y] != last_probs.max() ):
                break
            
            #make the perturbation for this step
            pert = nn.zeros( ndims )
            pert[ perm[ i % ndims ] ] = epsilon
            
            
            x_temp = (x - pert.view(x.size())).clamp(0, 1)
            left_prob = self.get_probs( x_temp )
            
            if( left_prob[y] <= last_probs[y] ):
                x = x_temp
                last_probs = left_prob
                pert_history -= pert
                
            else:
                x_temp = (x + pert.view(x.size())).clamp(0, 1)
                right_prob = self.get_probs( x_temp )
                
                if( right_prob[y] <= last_probs[y] ):
                    x = x_temp
                    last_probs = right_prob
                    pert_history += pert
        
        return x, pert_history, last_probs

if __name__ == "__main__":

    print("Loading Model")
    # Instantiate the model to be trained, convert internal datatypes to floats.
    model = Net().float()

    # Instantiate training object
    training_object = Train_Net_MNIST()

    print("Downloading data.")
    # Load the dataset from the loader.
    training_object.load_dataset( loader = MNIST_Dataset_Loader )

    print("Training model.")
    # Train the model
    training_object.train_net_model( model )

    # Run an attack on the model
    print("Starting SimBA attack")
    attack = SimBA_Attack(model=model)
    x_pert, history, x_pert_probs = attack.simba( training_object.x_test[0], training_object.y_test[0] )

    plt.imshow( x_pert.view(28,28) )
    plt.show()

    plt.imshow( history.view(28,28) )
    plt.show()

    print( x_pert_probs * 100 )
    