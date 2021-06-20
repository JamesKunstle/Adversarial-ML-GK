"""
    Imports
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import torch.nn as nn
import torch
import numpy as np

"""
    Loader Class
"""
class MNIST_Dataset_Loader( object ):

    def __init__(self, torch=True):

        # Shape of the data.
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        # Train / Test split percentage
        self.SPLIT = 0.3

        # Local references to loaded data.        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test =  None

    def _preprocess_mnist(self, x_train, y_train, x_test, y_test):
            """
                Formats X matrices to being ( num x channels, ROWS x COLS  ),
                scales the X matrices to [0, 1).
                converts all matrices to PyTorch Tensors.

                Returns the converted / formatted / scaled matrices.
            """
            X_TRAIN_SHAPE = x_train.shape[0]
            X_TEST_SHAPE = x_test.shape[0]

            x_train = np.array(x_train)
            x_train = x_train.reshape( X_TRAIN_SHAPE, 1, self.IMG_ROWS, self.IMG_COLS )
            x_train /= 255.0
            
            y_train = np.array(y_train)
            y_train = y_train.astype( int )
            
            x_test = np.array(x_test)
            x_test = x_test.reshape( X_TEST_SHAPE, 1, self.IMG_ROWS, self.IMG_COLS )
            x_test /= 255.0
            
            y_test = np.array(y_test)
            y_test = y_test.astype( int )
            
            """
            Convert to torch format for nn
            """
            x_train = torch.from_numpy( x_train )
            y_train = torch.from_numpy( y_train )
            
            x_test = torch.from_numpy( x_test )
            y_test = torch.from_numpy( y_test )


            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test

            return self.x_train, self.y_train, self.x_test, self.y_test

    def load_mnist(self, preprocess=True):

        # Download data
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

        # Split the data into training and testing populations
        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.SPLIT, 
                                                            random_state=42)

        # Specify whether the dataset ought to be returned raw or 
        # pre-processed for the repo's structure.
        if preprocess:
            x_train, y_train, x_test, y_test = self._preprocess_mnist(x_train, y_train, x_test, y_test)

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    # Instantiate the loader object
    loader = MNIST_Dataset_Loader()

    # Load the data, output local references.
    x_train, y_train, x_test, y_test = loader.load_mnist( )

    print( f"x_train shape: {x_train.shape}\n\t type: {type(x_train)}" )
    print( f"x_test  shape:  {x_test.shape}\n\t type: {type(x_test)}")
    print( f"y_train shape: {y_train.shape}\n\t type: {type(y_train)}")
    print( f"y_test  shape:  {y_test.shape}\n\t type: {type(y_test)}")
