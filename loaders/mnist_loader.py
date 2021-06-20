"""
    Imports
"""
import keras
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
        self.IMG_ROWS = 28
        self.IMG_COLS = 28
        self.NUM_CLASSES = 10
        self.BATCH_SIZE = 30
        self.SPLIT = 0.3
        self.torch = torch
        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test =  None

    def preprocess_mnist(self, x_train, y_train, x_test, y_test):
            """
                Formats X matrices to being ( num x ROWS x COLS x Channels )
                and converts Y vectors to categorical. 

                Returns the converted / formatted matrices and the shape
                of the X matrices.
            """
            X_TRAIN_SHAPE = x_train.shape[0]
            X_TEST_SHAPE = x_test.shape[0]

            x_train = np.array(x_train)
            x_train = x_train.reshape( X_TRAIN_SHAPE, 1, self.IMG_ROWS, self.IMG_COLS )
            
            y_train = np.array(y_train)
            y_train = y_train.astype( int )
            
            x_test = np.array(x_test)
            x_test = x_test.reshape( X_TEST_SHAPE, 1, self.IMG_ROWS, self.IMG_COLS )
            
            y_test = np.array(y_test)
            y_test = y_test.astype( int )
            
            """
            Convert to torch format for nn
            """
            x_train = torch.from_numpy( x_train )
            y_train = torch.from_numpy( y_train )
            
            x_test = torch.from_numpy( x_test )
            y_test = torch.from_numpy( y_test )

            self.x_train = x_train / 255.0
            self.y_train = y_train
            self.x_test = x_test / 255.0
            self.y_test =  y_test

            return self.x_train, self.y_train, self.x_test, self.y_test

    def load_mnist(self, preprocess = True):
        """
            Data Loading
        """
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.SPLIT, 
                                                            random_state=42)

        if preprocess:
            x_train, y_train, x_test, y_test = self.preprocess_mnist(x_train, y_train, x_test, y_test)

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    # Load and preprocess the dataset.
    loader = MNIST_Dataset_Loader()
    x_train, y_train, x_test, y_test = loader.load_mnist( preprocess=True )

    print( f"x_train shape: {x_train.shape}\n\t type: {type(x_train)}" )
    print( f"x_test  shape:  {x_test.shape}\n\t type: {type(x_test)}")
    print( f"y_train shape: {y_train.shape}\n\t type: {type(y_train)}")
    print( f"y_test  shape:  {y_test.shape}\n\t type: {type(y_test)}")
