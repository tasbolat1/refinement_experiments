import numpy as np
import matplotlib.pyplot as plt
from torch import distributions
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()

# save the model
def save_model_info(model, info, name='model'):
    torch.save(model.state_dict(), f'{name}_model.pt')
    pickle.dump(info, open(f'{name}_info.pkl','wb'))
    
def load_model_info(name, model):
    model.load_state_dict(torch.load(f'{name}_model.pt'))
    info = pickle.load(open(f'{name}_info.pkl', 'rb'))
    return model, info
    
    
    
def train_network(model, train_dataloader, train_dataset, test_dataloader, test_dataset, n_epochs = 100, print_freq = 10, optimizer=None):
    info = {
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[]
    }

    for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get statistics
            train_loss += loss.item()
            predicted = torch.sigmoid(outputs)
            train_accuracy += (torch.round(predicted) == labels).sum().item()

        # normalize
        train_loss /= len(train_dataset)
        train_accuracy /= len(train_dataset)

        info['train_acc'].append(train_accuracy)
        info['train_loss'].append(train_loss)
            
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # forward + backward + optimize
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                # get statistics
                test_loss += loss.item()
                predicted = torch.sigmoid(outputs)
                test_accuracy += (torch.round(predicted) == labels).sum().item()

        # normalize
        test_loss /= len(test_dataset)
        test_accuracy /= len(test_dataset)

        info['test_acc'].append(test_accuracy)
        info['test_loss'].append(test_loss)
        
        if epoch % print_freq == 0:

            print(f'Epoch: {epoch}')
            print(f'Train loss: {train_loss} | Test losses: {test_loss}')
            print(f'Train acc: {train_accuracy} | Test acc: {test_accuracy}')

    return model, info
    
    
# train classifier
class ClassifierNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=None):
        super(ClassifierNN, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.activation = activation
    def forward(self, x):
        hidden = self.fc1(x)
        relu = F.relu(hidden)
        if self.activation is None:
            output = self.fc2(relu)
        elif self.activation == 'tanh':
            output = F.tanh( self.fc2(relu) )
        elif self.activation == 'leaky_relu':
            output = F.leaky_relu( self.fc2(relu) )
        return output
    
        
        
# exact refinement
def refine_sample_exact(x, D, steps=10, f='KL', eta=0.001, noise_factor=0.0001, p=None, q=None):
    
    #f = 'KL'
    #eta = 0.001
    #noise_factor = 0.0001
    
    def _velocity(x):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
        
        if p is not None:
            # calculate d_score using analytical solution
            d_score = p.log_prob(x_t) - q.log_prob(x_t)
        else:
            d_score = D(x_t)

        if f == 'KL':
            s = torch.ones_like(d_score.detach())

        elif f == 'logD':
            s = 1 / (1 + d_score.detach().exp())

        elif f == 'JS':
            s = 1 / (1 + 1 / d_score.detach().exp())

        else:
            raise ValueError()

        s.expand_as(x_t)
        d_score.backward(torch.ones_like(d_score).to(x_t.device))
        grad = x_t.grad
        #print(d_score, grad)
        return s.data * grad.data
    
    all_x = [x]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        v = _velocity(x)
        all_v.append(v.detach())
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach())
    return all_x, all_v

def refine_sample(x, D, steps=10, f='KL', eta=0.001, noise_factor=0.0001):
    
    #f = 'KL'
    #eta = 0.001
    #noise_factor = 0.0001
    
    def _velocity(x):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
        d_score = D(x_t)
        
        # calculate d_score using analytical solution

        if f == 'KL':
            s = torch.ones_like(d_score.detach())

        elif f == 'logD':
            s = 1 / (1 + d_score.detach().exp())

        elif f == 'JS':
            s = 1 / (1 + 1 / d_score.detach().exp())

        else:
            raise ValueError()

        s.expand_as(x_t)
        d_score.backward(torch.ones_like(d_score).to(x_t.device))
        grad = x_t.grad
        return s.data * grad.data
    
    all_x = [x]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        v = _velocity(x)
        all_v.append(v.detach())
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach())
    return all_x, all_v

def draw_decision_boundaries(ax, model, x_lim=[-10,10],
                                        y_lim=[-10,10],
                                        step_size=0.2,
                                        threshold=0.5,
                                        cmap=None,
                                        alpha=0.3):
    if cmap is None:
        cmap = plt.cm.Paired
    
    xx, yy = np.meshgrid(np.arange(x_lim[0], x_lim[1], step_size),
                     np.arange(y_lim[0], y_lim[1], step_size))
    
    data = np.c_[xx.ravel(), yy.ravel()]
    dataloader = DataLoader(torch.from_numpy(data).float(), batch_size=256, shuffle=False)
    
    model.eval()
    zz = []
    for s in dataloader:
        output = torch.sigmoid(model(s))
        zz.append(output)
        
    zz=torch.cat(zz)
    zz[zz < threshold ] = 0
    zz[zz >= threshold ] = 1.0
    Z=zz.detach().cpu().numpy().reshape(xx.shape)
    return ax.contourf(xx, yy, Z, cmap=cmap, alpha=alpha)
