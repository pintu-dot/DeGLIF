import os
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from tqdm.auto import tqdm

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import torch_geometric

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------------
LAMBDA = 1e-4  # Regularisation parameter
THRESHOLDS = [0, 0.1, 1, 10, 20]  # List of thresholding parameter mu in paper


# -------------------------------------------------------------------------
# 1. Models & Training
# -------------------------------------------------------------------------
class SGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


def train_eval_model(model, data, lmbda, with_val=False, epochs=250, lr=0.005):
    """
    Trains the GCN model and optionally evaluates on a validation set.
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def train_step():
        model.train()
        optimizer.zero_grad()
        reg = sum(p.pow(2.0).sum() for p in model.parameters())
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask]) + lmbda * reg
        loss.backward()
        optimizer.step()
        return loss.item()

    # Train loop
    for epoch in range(1, epochs + 1):
        train_step()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Test Evaluation
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        mac_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro')

        if not with_val:
            return test_acc, mac_f1

        # Validation Evaluation
        # Original logic: val1 = ((True^data.train_mask)*(True^data.test_mask))
        val_mask = (~data.train_mask) & (~data.test_mask)
        val_correct = pred[val_mask] == data.y[val_mask]
        val_acc = int(val_correct.sum()) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0

    return val_acc, test_acc, mac_f1


# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------
def create_noisy(data, noise_level, num_classes):
    """Adds symmetric label noise (SLN) to the data."""
    data1 = data.clone()
    rand_num = torch.rand(len(data.y))

    for k in range(num_classes):
        for j in range(num_classes - 1):
            mask1 = (rand_num < (j + 1) * noise_level / (num_classes - 1)).to(device)
            mask2 = ((j) * noise_level / (num_classes - 1) <= rand_num).to(device)
            pos_to_flip = mask1 * mask2 * data.train_mask * (data.y == k)
            data1.y[pos_to_flip] = (data1.y[pos_to_flip] + (j + 1)) % num_classes
    return data1


def fltn(grad_tuple):
    """Efficiently flattens a tuple of tensors."""
    return torch.cat([g.reshape(-1) for g in grad_tuple])


def calc_loss(model, data, lmbda, mask=None):
    """Calculates the loss for a given mask, or full dataset if mask is None."""
    out = model(data.x, data.edge_index)
    reg = sum(p.pow(2.0).sum() for p in model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if mask is not None:
        loss = criterion(out[mask], data.y[mask]) + lmbda * reg
    else:
        loss = criterion(out, data.y) + lmbda * reg
    return loss


def calc_node_grad(model, data, node_idx, lmbda):
    """Calculates gradient of the loss with respect to a single node."""
    params = list(model.parameters())
    out = model(data.x, data.edge_index)
    reg = sum(p.pow(2.0).sum() for p in model.parameters())

    criterion = torch.nn.CrossEntropyLoss()
    # CrossEntropyLoss requires 2D input and 1D target
    loss = criterion(out[node_idx].view(1, -1), data.y[node_idx].view(1)) + lmbda * reg
    grad = torch.autograd.grad(loss, params, create_graph=False)
    return fltn(grad)


def calc_full_grad(model, data, lmbda):
    """Calculates gradient of the loss with respect to the full graph."""
    params = list(model.parameters())
    loss = calc_loss(model, data, lmbda, mask=None)
    grad = torch.autograd.grad(loss, params, create_graph=False)
    return fltn(grad)


def mod_graph(g_nx, node_idx, orig_data):
    """Removes edges connected to node_idx and returns the modified PyG graph."""
    g_nx_new = g_nx.copy()
    neighbors = list(g_nx_new[node_idx])
    for k in neighbors:
        g_nx_new.remove_edge(node_idx, k)

    pyg_graph = torch_geometric.utils.convert.from_networkx(g_nx_new)
    pyg_graph.x = orig_data.x
    pyg_graph.y = orig_data.y
    return pyg_graph


# -------------------------------------------------------------------------
# 3. Main Logic (Memory Optimized)
# -------------------------------------------------------------------------
def main_func(data_bn, orig_data, g_nx, sed, num_features, num_classes):
    """
    Refactored main function.
    Calculates exact Hessian without explicitly forming the inverse.
    """
    lambd = LAMBDA
    
    # 1. Trains model on noisy data to obtain base weights
    model_sgcn = SGCN(num_features, num_classes, hidden_channels=16, seed=sed)
    train_eval_model(model_sgcn, data_bn, lambd, with_val=False)
    
    # Eager clear
    torch.cuda.empty_cache()

    # 2. Calculate Exact Hessian efficiently
    # We evaluate the loss and gradients, keeping the graph alive only as long as needed.
    loss = calc_loss(model_sgcn, data_bn, lambd, mask=data_bn.train_mask)
    grads = torch.autograd.grad(loss, model_sgcn.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
    grad_flatten = fltn(grads)
    
    num_params = len(grad_flatten)
    hessian_tensor = torch.zeros((num_params, num_params), device=device)

    # Compute row by row. Notice we release retain_graph on the final iteration
    for i in range(num_params):
        grad_grad = torch.autograd.grad(
            grad_flatten[i], 
            model_sgcn.parameters(), 
            retain_graph=(i < num_params - 1) # Release graph on last iteration!
        )
        hessian_tensor[i] = fltn(grad_grad).detach()
        
        # Clear intermediates to keep memory stable
        del grad_grad

    # CRITICAL MEMORY STEP: Delete original graph tensors
    del loss, grads, grad_flatten
    torch.cuda.empty_cache()

    # 3. Calculate influence function (Batched Solve approach)
    infl = torch.zeros((len(data_bn.train_mask), num_params), device=device)
    
    train_indices = data_bn.train_mask.nonzero(as_tuple=True)[0]
    num_train = len(train_indices)
    
    # Collect all gradients into a single matrix of shape (num_params, num_train)
    # This matrix is very small (e.g., P x 140) compared to a P x P inverse.
    grad_temps = torch.zeros((num_params, num_train), device=device)
    
    for idx, i in enumerate(train_indices):
        grad_temp = calc_node_grad(model_sgcn, data_bn, i, lambd)
        grad_temps[:, idx] = -(grad_temp / torch.norm(grad_temp))
        
        # Clear node grad intermediate
        del grad_temp
        torch.cuda.empty_cache()

    # Solve all equations simultaneously: H * X = grad_temps
    # This is blazingly fast (factorizes H exactly once internally using BLAS level 3)
    # and entirely avoids creating an explicit PxP inverse matrix!
    X = torch.linalg.solve(hessian_tensor, grad_temps)
    
    # Map the solved batched influences back to the infl tensor
    for idx, i in enumerate(train_indices):
        infl[i] = X[:, idx]
        
    # CRITICAL MEMORY STEP: Eagerly delete large tensors
    del hessian_tensor, grad_temps, X
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    grad_test = torch.zeros((len(data_bn.val_mask), num_params), device=device)
    for i, b in enumerate(orig_data.val_mask):
        if b:
            grad_test[i] = calc_node_grad(model_sgcn, data_bn, i, lambd)
            
    # infl_final_val = grad_test * infl^T
    infl_final_val = torch.matmul(grad_test, infl.T)
    infl_sum = infl_final_val.sum(dim=0)
    
    del grad_test, infl, infl_final_val
    torch.cuda.empty_cache()

    best_val = 0
    best_test = 0
    best_mac = 0
    
    # 5. Relabel noisy points and evaluate thresholds
    model_sgcn.eval()
    out = model_sgcn(data_bn.x, data_bn.edge_index)
    pred = out.argmax(dim=1)
    preds_top2 = torch.topk(out, 2, dim=1).indices[:, 1]
    
    for th in THRESHOLDS:
        test_1 = (infl_sum > th)
        data1 = data_bn.clone()
        
        assign_max = (pred != data_bn.y) * data_bn.train_mask * test_1
        data1.y[assign_max] = pred[assign_max]
        
        assign_smax = (data_bn.y == pred) * data_bn.train_mask * test_1
        data1.y[assign_smax] = preds_top2[assign_smax]
        
        # Train new model on denoised dataset
        model_sgcn_n = SGCN(num_features, num_classes, hidden_channels=16, seed=sed)
        val_acc, test_acc, mac_f1 = train_eval_model(model_sgcn_n, data1, lambd, with_val=True)
        
        if val_acc > best_val:
            best_test = test_acc
            best_mac = mac_f1
            best_val = val_acc
            
        del val_acc, test_acc, model_sgcn_n, data1
        torch.cuda.empty_cache()

    del model_sgcn, out, pred, preds_top2, infl_sum
    torch.cuda.empty_cache()
    
    return best_test, best_mac

# -------------------------------------------------------------------------
# Execution Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ITERATIONS = 1
    NOISE_LEVEL = 0.2
    
    c_acc = np.zeros(ITERATIONS)
    c_mac = np.zeros(ITERATIONS)
    
    for i in tqdm(range(ITERATIONS)):
        torch.manual_seed(i + 2)
        dataset = Planetoid(
            root='data/py311/Planetoid', 
            name='Cora', 
            split="random", 
            num_train_per_class=172, 
            num_val=50, 
            num_test=1000, 
            transform=NormalizeFeatures()
        )
        
        orig_data = dataset[0].to(device)
        num_features = dataset.num_features
        num_classes = len(orig_data.y.unique())
        
        # Binary noisy data set
        data_bn = create_noisy(orig_data, NOISE_LEVEL, num_classes)
        
        # Convert to undirected networkx graph
        g_nx = torch_geometric.utils.convert.to_networkx(
            data_bn, 
            to_undirected=True, 
            remove_self_loops=True
        )
        
        c_temp_acc, c_temp_mac = main_func(
            data_bn, orig_data, g_nx, 
            sed=(i + 1), 
            num_features=num_features, 
            num_classes=num_classes
        )
        
        c_acc[i] = c_temp_acc
        c_mac[i] = c_temp_mac
        
        print(f"Iteration {i+1}: Test Acc: {c_temp_acc:.4f}, Macro F1: {c_temp_mac:.4f}")

    print("--------------------------------------------------")
    print(f" Denoised Accuracy = {np.mean(c_acc):.4f} +- {np.std(c_acc):.4f}")
    print(f" Cleaned Macro F1  = {np.mean(c_mac):.4f} +- {np.std(c_mac):.4f}")
