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


def test_gaussian_pert(model, device, test_loader, mean=0, var=0.01):
    sigma = var ** 0.5
    
    for idx, (img, lab) in enumerate(test_loader):
        
        probs, gt_prob = get_probs(model, img, lab, device)
        plt.imshow(img.view(28,28), cmap="gray")
        plt.show()
        max = torch.argmax(probs)
        if int(max) != int(lab):
            # model is wrong, skip this.
            print('skipping bc model is wrong')
            continue
        
        # generating noise. 
        
        row,col,ch= (1,28,28)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        noisy = noisy.float()
        
        n_probs, n_gt_prob = get_probs(model, noisy, lab, device)
        
        n_max = torch.argmax(n_probs)
        
        
        plt.imshow(noisy.view(28,28), cmap="gray")
        plt.show()
        
        
        if int(n_max) != int(lab):
            print('mislabeled.')
        
        
        if idx > 5:
            break
    return

test_gaussian_pert(model, device, test_loader, var=0.1)

def create_gaussian_pert_matrix(model, device, test_loader, mean=0, var=0.01):
    sigma = var ** 0.5
    matrix = [np.zeros(10) for i in range(10)]
    matrix = np.asarray(matrix)
    
    for idx, (img, lab) in enumerate( tqdm(test_loader)):
        probs, gt_prob = get_probs(model, img, lab, device)
        
        max = torch.argmax(probs)
        if int(max) != int(lab):
            # model is wrong, skip this.
            continue
        
        row,col,ch= (1,28,28)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        noisy = noisy.float()
        
        n_probs, n_gt_prob = get_probs(model, noisy, lab, device)
        
        n_max = torch.argmax(n_probs)
        
        matrix[lab][n_max] +=1 
        
    return matrix

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


matrix = create_gaussian_pert_matrix(model, device, test_loader, var=0.1)
pgm = matrix.astype(int)
plt.figure(figsize=(10,10))
plot_confusion_matrix(pgm, testset.classes, title='Gaussian Perturbations Matrix')


total_images = 0
perturbed = 0
for i in range(10):
    for j in range(10):
        total_images += matrix[i][j]
        if i != j:
            perturbed += matrix[i][j]

print(total_images, perturbed, perturbed/total_images)


## example case:
import random
random.seed(5)

def example_normal_test(model, device, test_loader):
    
    for idx, (img, lab) in enumerate(test_loader):
        
        probs, gt_prob = get_probs(model, img, lab, device)
        plt.imshow(img.view(28,28), cmap="gray")
        plt.show()
        max = torch.argmax(probs)
        if int(max) != int(lab):
            # model is wrong, skip this.
            print('skipping bc model is wrong')
            continue
        
        # generating noise. 
        pert = np.zeros((1,28,28))
        rows = np.floor(np.random.normal(14, 4, 20))
        cols = np.ceil(np.random.normal(14, 4, 20))
        rows = rows.astype(int)
        cols = cols.astype(int)
        for i in range(20):
            pert[0][rows[i]][cols[i]] += random.uniform(-1,1)
            
        noisy = img+pert
        noisy = noisy.float()
            
        n_probs, n_gt_prob = get_probs(model, noisy, lab, device)
        
        n_max = torch.argmax(n_probs)
        
        
        plt.imshow(noisy.view(28,28), cmap="gray")
        plt.show()
        
        
        if int(n_max) != int(lab):
            print('mislabeled.')
        
        
        if idx > 5:
            break
    return


example_normal_test(model, device, test_loader)

def create_normal_pert_matrix(model, device, test_loader, num_pix=20, delta=1, gauss=True):
    matrix = [np.zeros(10) for i in range(10)]
    matrix = np.asarray(matrix)
    
    for idx, (img, lab) in enumerate( tqdm(test_loader)):
        probs, gt_prob = get_probs(model, img, lab, device)
        
        max = torch.argmax(probs)
        if int(max) != int(lab):
            # model is wrong, skip this.
            continue
        
        pert = np.zeros((1,28,28))
        rows = np.floor(np.random.normal(14, 4, num_pix))
        cols = np.ceil(np.random.normal(14, 4, num_pix))
        rows = rows.astype(int)
        cols = cols.astype(int)
        for i in range(num_pix):
            if rows[i] < 0:
                rows[i] = 0
            elif rows[i] > 27:
                rows[i] = 27
            if cols[i] < 0:
                cols[i] = 0
            elif cols[i] > 27:
                cols[i] = 27
            if gauss == True:
                pert[0][rows[i]][cols[i]] += random.uniform(-1*delta,1*delta)
            else:
                if random.random() > 0.5:
                    pert[0][rows[i]][cols[i]] -= delta
                else:
                    pert[0][rows[i]][cols[i]] += delta
            
        noisy = img+pert
        noisy = noisy.float()
        
        n_probs, n_gt_prob = get_probs(model, noisy, lab, device)
        
        n_max = torch.argmax(n_probs)
        
        matrix[lab][n_max] +=1 
        
        if idx%500==0:
            plt.imshow(img.view(28,28), cmap="gray")
            plt.show()
            
            plt.imshow(noisy.view(28,28), cmap="gray")
            plt.show()
            
            if int(n_max)!=int(lab):
                print("Mislabeled noisy pert!")
            else:
                print("Did not mislabel noisy.")
        
    return matrix


norm_matrix = create_normal_pert_matrix(model, device, test_loader)
total_images = 0
perturbed = 0
for i in range(10):
    for j in range(10):
        total_images += norm_matrix[i][j]
        if i != j:
            perturbed += norm_matrix[i][j]

print(total_images, perturbed, perturbed/total_images)


norm_matrix_1 = create_normal_pert_matrix(model, device, test_loader, 50, 3)

total_images = 0
perturbed = 0
for i in range(10):
    for j in range(10):
        total_images += norm_matrix_1[i][j]
        if i != j:
            perturbed += norm_matrix_1[i][j]
            
print(total_images, perturbed, perturbed/total_images)


norm_matrix_2 = create_normal_pert_matrix(model, device, test_loader, 50, 3, False)

total_images = 0
perturbed = 0
for i in range(10):
    for j in range(10):
        total_images += norm_matrix_2[i][j]
        if i != j:
            perturbed += norm_matrix_2[i][j]
            
print(total_images, perturbed, perturbed/total_images)