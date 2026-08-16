lam=1e-4     # Regularisation parameter
threshold=[0,0.1,1,10,20]  # list of thresholding parameter $\mu$ in paper 

import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch_geometric
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch_geometric.nn import GCNConv
from torch_geometric.nn import SGConv
from sklearn.metrics import f1_score
import numpy as np

def train_and_test(model,data,lmbda):       # training and testing GCN models
    data.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        reg = sum(p.pow(2.0).sum()
                  for p in model.parameters())
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])+lmbda*reg  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        mac_f1=f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro');
        return test_acc,mac_f1


    for epoch in range(1, 251):
        loss = train()

    test_acc,mac_f1 = test()
    return test_acc,mac_f1

def train_and_test_val(model,data,lmbda):
    data.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        reg = sum(p.pow(2.0).sum()
                  for p in model.parameters())
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])+lmbda*reg  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def val():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val1=((True^data.train_mask)*(True^data.test_mask))
        val_correct = pred[val1] == data.y[val1]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val1.sum())  # Derive ratio of correct predictions.
        return val_acc
    
    def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      mac_f1=f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro');
      return test_acc,mac_f1

    for epoch in range(1, 251):
        loss = train()

    val_acc = val()
    test_acc,mac_f1=test()

    #print(f'Test Accuracy: {test_acc:.4f}')
    return val_acc,test_acc,mac_f1


class SGCN(torch.nn.Module):
    def __init__(self,hidden_channels=16,seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(dataset.num_features,hidden_channels)
        self.conv2=GCNConv(hidden_channels,dataset.num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


def create_noisy(data,noise_level):   # add SLN noise to data
    data1=data.clone()

    rand_num=torch.rand(len(data.y))
    for k in range(no_of_classes):
        for j in range(no_of_classes-1):
            pos_to_flip=(((rand_num<(j+1)*noise_level/(no_of_classes-1)).to(device))*(((j)*noise_level/(no_of_classes-1)<=rand_num).to(device))*data.train_mask)*(data.y==k)
            data1.y[pos_to_flip]=(data1.y[pos_to_flip]+(j+1))%no_of_classes
    return(data1)

dataset = Planetoid(root='data/py311/Planetoid', name='Cora', split="random", num_train_per_class=172,num_val=50,num_test=1000,transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
datap=data.clone()
no_of_classes=len(datap.y.unique());
datap.to(device)


def fltn(gr):  
  temp=torch.tensor([])
  for j in range(len(gr)):
    temp=torch.cat((temp.to(device),torch.flatten(gr[j])))
  return temp

def calc_grad1(model,data,lamdaa,reg):
    params = list(model.parameters())
    model.eval()
    out1=model(data.x,data.edge_index)
    loss=torch.nn.CrossEntropyLoss()
    l1=loss(out1,data.y)+lamdaa*reg
    grad_z=torch.autograd.grad(l1,params,retain_graph=True)
    return(grad_z)


def calc_grad(model,data,pos,lamdaa,reg):
    params = list(model.parameters())
    model.eval()
    out1=model(data.x,data.edge_index)
    loss=torch.nn.CrossEntropyLoss()
    l1=loss(out1[pos].view(1,7),data.y[pos].view(1))+lamdaa*reg
    grad_z=torch.autograd.grad(l1,params,retain_graph=True)
    return(grad_z)

def mod_graph(g_nx,i):
    g_nx_new=g_nx.copy()
    a=g_nx_new[i].copy()
    for k in a:
        g_nx_new.remove_edge(i,k)
    pyg_graph=torch_geometric.utils.convert.from_networkx(g_nx_new)
    pyg_graph.x=data_bn.x; pyg_graph.y=data_bn.y
    del g_nx_new
    return pyg_graph

def main_func(data_bn,g_nx,sed):
    cleaned_acc=[]
    cleaned_mac=[]
    best_acc=0
    lambd=lam
    model_sgcn=SGCN(hidden_channels=16,seed=sed)  #1. Trains model_sgcn on noisy data to obtain weights required for influence calculation     
    train_and_test(model_sgcn,data_bn,lam)
    torch.cuda.empty_cache()
    model=model_sgcn
    data=data_bn

    #2. Calculate hessian, hessian inverse and desired gradients
    reg = sum(p.pow(2.0).sum() for p in model.parameters())
    loss=torch.nn.CrossEntropyLoss()
    out=model(data.x,data.edge_index)
    l=loss(out[data.train_mask], data.y[data.train_mask])+lambd*reg
    grads = torch.autograd.grad(l, model.parameters(), create_graph=True,retain_graph=True,allow_unused=True)
    grad_flatten=fltn(grads)
    # Initialize the Hessian tensor
    hessian_tensor = torch.zeros(len(grad_flatten),len(grad_flatten))

    # Compute the Hessian by iterating over the model parameters and gradients
    for i in range(len(grad_flatten)):
        grad_grad=torch.autograd.grad(grad_flatten[i],model.parameters(),retain_graph=True)
        hessian_tensor[i] = fltn(grad_grad)
    hessian_tensor=hessian_tensor.to(device)
    hess_inv=torch.linalg.inv(hessian_tensor.to(device))
    # 3. Calculate influence function and predict noisy points ( for different values of threshold )
    infl=torch.zeros(len(data.train_mask),len(grad_flatten)).to(device)
    for i,b in enumerate(data.train_mask):
        if(b):
            grad_temp=fltn(calc_grad(model_sgcn,data_bn,i,lambd,reg))
            grad_temp=grad_temp/torch.norm(grad_temp)
            infl[i]=-1*torch.matmul(hess_inv.to(device),grad_temp)
            del grad_temp


    del hess_inv
    torch.cuda.empty_cache()
    grad_test= torch.zeros(len(data.val_mask),len(grad_flatten)).to(device)
    for i,b in enumerate(datap.val_mask):
        if(b):
            grad_test[i]=fltn(calc_grad(model_sgcn,data_bn,i,lambd,reg))
    infl_final_val=torch.matmul(grad_test,torch.transpose(infl,0,1))
    infl_sum=infl_final_val.sum(dim=0)
    best_val=0
    best_test=0
    best_mac=0
    for th in threshold:
        test_1=(infl_sum>th)
        data1=data_bn.clone()
        
        model.eval()
        out=model(data.x,data.edge_index)
        pred = out.argmax(dim=1)
        assign_max=(pred!=data_bn.y)*data_bn.train_mask*test_1
        data1.y[assign_max]=pred[assign_max]
        assign_smax=(data_bn==pred)*data_bn.train_mask*test_1
        preds = torch.topk(out,2,dim=1).indices[:,1]
        data1.y[assign_smax]=preds[assign_smax]
        # 4. Relabel noisy points and train a new model on denoised data aset
        model_sgcn_n=SGCN(hidden_channels=16,seed=sed)
        val_acc,test_acc,mac_f1=train_and_test_val(model_sgcn_n,data1,lam)
        torch.cuda.empty_cache()
        if val_acc>best_val:
            best_test=test_acc
            best_mac=mac_f1
            best_val=val_acc
        del val_acc,test_acc,model_sgcn_n
        torch.cuda.empty_cache()
    del model_sgcn, hessian_tensor,out,l,grads, model,data
    torch.cuda.empty_cache()
    return best_test,best_mac


it=1;   # Increase the number of iteration to perform experiment multile times
nl=0.2  # Noise level 0.2 means 20% symmetric label noise
c_acc=np.zeros(it);
c_mac=np.zeros(it);
for i in tqdm(range(it)):
    torch.manual_seed(i+2)
    dataset = Planetoid(root='data/py311/Planetoid', name='Cora', split="random", num_train_per_class=172,num_val=50,num_test=1000,transform=NormalizeFeatures())
    datap = (dataset[0]).to(device)
    data_bn=create_noisy(datap,nl)     # binary noisy data set
    g_nx=torch_geometric.utils.convert.to_networkx(data_bn,to_undirected='True',remove_self_loops='True')
    c_temp_acc,c_temp_mac=main_func(data_bn,g_nx,i+1);
    c_acc[i]=np.array(c_temp_acc)
    c_mac[i]=np.array(c_temp_mac)
print(" denoised accuracy =",np.mean(c_acc),"+-", np.std(c_acc))
print(" cleaned mac =",np.mean(c_mac),"+-", np.std(c_mac))


