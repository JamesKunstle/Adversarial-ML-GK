import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
classes = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        #output = x
        output = F.softmax(x, dim=1)
        return output


model = SimpleNet().to( device )
model.load_state_dict( torch.load("models/mnist_fashion_SimpleNet.pt", map_location='cpu') )
### NOTE THAT map_location is set to 'cpu' when running locally! remove that when running on gpu/remotely.

print(model.eval())


"""
    Data loading, train and test set via the PyTorch dataloader.
"""

train_transform=transforms.Compose([
#         transforms.RandomCrop(28, padding=4),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

batch_size = 1

trainset = datasets.FashionMNIST('./data', train=True, download=True,
                   transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testset = datasets.FashionMNIST('./data', train=False,
                   transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


"""
    Function to return the current probabilities of the classes given the model.
    
    Input parameters:
        model : model that is performing the classification
        x : input data object being classified
        y : gt label of the input data
"""

def get_probs( model, x, y, device ):
    
    model = model.to( device )
    x = x.to( device )
    y = y.to( device )
    
    with torch.no_grad():
        output = model( x ) #model classifies the input
        
    probs = output #get the current probability of classification, model outputs log_softmax
    
    return probs, probs[0][y] #return the full distribution plus the probability of the gt class

"""
    Performs SimBA method for finding a minimum adversarial perturbation on some 
    input image x s.t. it is misclassified by some target model.
    
    Input parameters:
        model : model that is performing the classification
        x : input data object being classified
        y : gt label of the input data
        num_iters : maximum allowed iterations toward convergence
        epsilon : step size
        confidence : target gt classification probability
"""

def simba_single( model, x, y, device, iters = 10000, epsilon = 0.2 ):
    
    #number of dimensions of the input image
    n_dims = x.view( 1, -1 ).size(1)
    
    #random basis dimensions
    perm = torch.randperm( n_dims )
    
    #previous probability vector
    all_probs, last_prob = get_probs( model, x, y, device )
    
    #history of perturbation
    all_diffs = torch.zeros( n_dims )
    
    for i in trange( iters ):
        
        #if we have met our confidence requirement

        if( classes[y] != classes[all_probs.argmax(dim=1, keepdim=True) ] ):
            break
        else:
            
            # creation of adversarial step by step-size epsilon
            diff = torch.zeros( n_dims )
            diff[ perm[ i % n_dims ] ] = epsilon
            
            # subtract adversarial perturbation from the input
            x_temp = (x - diff.view(x.size())).clamp(-1, 1)
            
            # get the new probabilities after addition
            all_probs_temp, left_prob = get_probs( model, x_temp, y, device )
            
            if( left_prob < last_prob ):
                x = x_temp
                last_prob = left_prob
                all_probs = all_probs_temp
                all_diffs += diff 
            
            else:
                
                # subtract adversarial perturbation from the input
                x_temp = (x + diff.view(x.size())).clamp(-1, 1)

                # get the new probabilities after addition
                all_probs_temp, right_prob = get_probs( model, x_temp, y, device )
                
                if( right_prob < last_prob ):
                    x = x_temp
                    last_prob = right_prob
                    all_probs = all_probs_temp
                    all_diffs += diff
        
    # return the perturbed image, the perturbation, the final prob dist, and the iteration count    
    return x, all_diffs, all_probs, i


def run_simba( model, device, test_loader):
    
    first_ten_perts = []

    for idx, (img, lab) in enumerate(test_loader):
        
        pert_img, perts, probs, iters = simba_single( model, img, lab, device, iters = 100000, epsilon = 2.0 )
        
        if( iters < 100000 ):
            first_ten_perts.append( (img, lab, pert_img, perts, probs, iters ) )

        plt.imshow( img.view( 28, 28, 1) )
        plt.show()

        plt.imshow( pert_img.view( 28, 28, 1) )
        plt.show()

        plt.imshow( perts.view( 28, 28, 1) )
        plt.show()


    #     probs, class_prob = get_probs( model, img, lab, device )
        print( probs )
    #     print( class_prob )

    #     plt.imshow( img.view(28, 28, 1) )
    #     plt.show()

        if (idx >= 10):
            break
            
    return first_ten_perts
    
def build_conf_matrix(model, device, test_loader):
    matrix = [np.zeros(10) for i in range(10)]
    print(len(matrix), len(matrix[0]))
    for idx, (img, lab) in enumerate(test_loader):
        
        probs, gt_prob = get_probs(model, img, lab, device)
        



    #     probs, class_prob = get_probs( model, img, lab, device )
        max = torch.argmax(probs)

        matrix[lab][max] +=1 
    #     print( class_prob )

    #     plt.imshow( img.view(28, 28, 1) )
    #     plt.show()

            
    return matrix
    
conf_matrix = build_conf_matrix(model, device, test_loader)
cm = np.asarray(conf_matrix)
cm = cm.astype(int)

import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, testset.classes) ## this isnt working perfectly
## when not running in jupyter notebook. 


def build_pert_matrix( model, device, test_loader):
    
    matrix = [np.zeros(10) for i in range(10)]
    matrix = np.asarray(matrix)
    print(len(matrix), len(matrix[0]))
    
    perts_storage = []

    for idx, (img, lab) in enumerate(test_loader):
        
        
        pert_img, perts, probs, iters = simba_single( model, img, lab, device, iters = 5000, epsilon = 2.0 )
        
        if( iters < 5000 ):
            max = torch.argmax(probs)
            print(max.item())
            matrix[lab][max] +=1
            
        else:
            matrix[lab][lab] += 1
        


        if (idx >= 10000):
            break
            
    return matrix
    
        
pert_matrix = build_pert_matrix( model, device, test_loader )
pert_matrix = pert_matrix.astype(int)

plt.figure(figsize=(10,10))
plot_confusion_matrix(pert_matrix, testset.classes, title='Perturbation Matrix')


def build_heatmap_matrix( model, device, test_loader):
    
    matrix = [torch.zeros(784) for i in range(10)]
    
    print(len(matrix), len(matrix[0]))
    
    

    for idx, (img, lab) in enumerate(test_loader):
        
        
        pert_img, perts, probs, iters = simba_single( model, img, lab, device, iters = 5000, epsilon = 2.0 )
        
        if( iters < 5000 ):
            matrix[lab] += perts
            
        else:
            print('here')
            


        if (idx >= 1000):
            break
            
    return matrix


heatmap_matrix = build_heatmap_matrix( model, device, test_loader )

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

for i in range(10):
    if i ==6:
        continue
    plt.imshow( heatmap_matrix[i].view( 28, 28, 1),
               interpolation='nearest', cmap='coolwarm' )
    plt.colorbar()
    plt.xlabel(labels_map[i])
    plt.show()
