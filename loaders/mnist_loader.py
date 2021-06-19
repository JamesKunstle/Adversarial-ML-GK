"""
    Imports
"""
import keras
from keras.datasets import mnist
import numpy as np

"""
    Loader Class
"""
class MNIST_Dataset_Loader( object ):

    def __init__(self):
        self.IMG_ROWS = 28
        self.IMG_COLS = 28
        self.NUM_CLASSES = 10
        
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
            x_train = x_train.reshape(x_train.shape[0], self.IMG_ROWS, self.IMG_COLS, 1)
            x_test = x_test.reshape(x_test.shape[0], self.IMG_ROWS, self.IMG_COLS, 1 )
                
            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255
            
            y_train = keras.utils.to_categorical(y_train, self.NUM_CLASSES)
            y_test = keras.utils.to_categorical(y_test, self.NUM_CLASSES)

            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test =  y_test

            return x_train, y_train, x_test, y_test

    def load_mnist(self, preprocess = True):
        """
            Data Loading
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

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