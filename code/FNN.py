"""
Feed-Forward Neural Network example. FashionMNIST and classification.
"""

import os
import torch
import torchvision 
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotting_utils as plu
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import ToTensor
matplotlib.use('macOSX')


##


# Paths
path_main = '/Users/IEO5505/Desktop/AI and DL/learn/my_DL_playground'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results')


##


# Check type of available devices
torch.accelerator.is_available() 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 


##


# Dataloaders, iterators and batches

# Load FashionMNIST
training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Load data
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Viz batch of 64 images
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# mapping = {
#     k.item():v for k,v in zip(training_data.targets.unique(),training_data.classes)
# }
# mapping
# 
# fig, axs = plt.subplots(2,5,figsize=(10,4))
# for i,x in enumerate(train_features):
#     ax = axs.flatten()[i]
#     ax.imshow(x.squeeze().numpy())
#     plu.format_ax(ax, title=mapping[train_labels[i].item()])
#     if i>8:
#         break
# 
# fig.tight_layout()
# plt.show()


## 


# Data batches visualization 
# for i,x in enumerate(train_features):
#     print(i,x.size())


##


# FFNN Model building
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# # Instantiate model class
# 
# # Obtain probs via model() call: execute forward method by default
# model = NeuralNetwork().to(device)     
# print(model)
# 
# X = torch.rand(1, 28, 28, device=device)    # 1,28,28 size
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
# pred_probab.size()                          # 1,10 size    
# 
# # Inspect model structure 
# print(f"Model structure: {model}\n\n")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


##

# # Obtain probs by step-by-step execution, change input image dim 0 1-->3
# input_image = torch.rand(3,28,28)           # 3,28,28 dim
# print(input_image.size())
# 
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
# 
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())
# 
# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")
# hidden1.size()                      # 3,10 dim
# 
# seq_modules = nn.Sequential(
#     flatten,
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20, 10)
# )
# input_image = torch.rand(3,28,28)   # 3,28,28 size
# logits = seq_modules(input_image)
# logits.size()
# 
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(logits)
# pred_probab.size()                  # 3,10 size


##


# Disable gradient tracking
# torch.set_grad_enabled(False) or with torch.no_grad():

# Set grad to zero
# params.grad.zero_()  


##


# Model training

# Hyper-parameters
learning_rate = 1e-3
batch_size = 64
epochs = 10


##


# Train and test loops
def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


##


# Instantiate model class
model = NeuralNetwork().to(device)     
model
# Set up optimizer, loss and n epochs
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
epochs = 10

# Train and evaluate the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)


##


# Final, test set losses
test_losses = []
for X, y in test_dataloader:
    pred = model(X)
    test_losses.append(loss_fn(pred, y).item())

fig, ax = plt.subplots(figsize=(3.5, 3.5))
sns.kdeplot(test_losses, fill=True, alpha=0.5)
plu.format_ax(ax=ax, title='Test losses')
fig.tight_layout()
plt.show()

# Mean test set loss
f'Mean test set loss: {np.mean(test_losses):.2f}'  # Mean test loss across all batches

# Save and re-load model
torch.save(model.state_dict(), os.path.join(path_results, 'model.pth'))
model_reloaded = NeuralNetwork().to(device)
model_reloaded.load_state_dict(torch.load(os.path.join(path_results, 'model.pth'))) 


##


# # Manual cross-entropy calculation
# manual_pred = pred.detach()         # (n obs x n classes)
# manual_log_softmax = manual_pred - torch.log(torch.exp(manual_pred).sum(dim=1, keepdim=True))
# manual_loss = -manual_log_softmax.gather(1, y.unsqueeze(1)).squeeze().mean()
# 
# # Assess differences
# print(f"Manual calculation (exact): {manual_loss}")
# print(f"PyTorch CrossEntropyLoss: {loss}")
# print(f"Difference: {abs(manual_loss - loss)}")


##


# # NBB: Demonstration. Standard Cross-Entropy vs PyTorch Implementation
# print("\n" + "="*60)
# print("BRIDGING THE GAP: Standard Definition vs PyTorch Implementation")
# print("="*60)
# 
# # Take just the first sample for clarity
# sample_logits = pred[0]  # Shape: [num_classes]
# sample_target = y[0]     # Shape: [] (scalar class index)
# 
# print(f"Sample logits: {sample_logits}")
# print(f"Sample target (class index): {sample_target}")
# 
# # Method 1: Standard cross-entropy definition with one-hot
# print("\n--- Method 1: Standard Definition (with one-hot) ---")
# # Convert class index to one-hot
# one_hot = torch.zeros_like(sample_logits)
# one_hot[sample_target] = 1.0
# print(f"One-hot encoding: {one_hot}")
# 
# # Compute softmax probabilities
# softmax_probs = torch.exp(sample_logits) / torch.exp(sample_logits).sum()
# print(f"Softmax probabilities: {softmax_probs}")
# 
# # Standard cross-entropy: -Σ(p_true * log(p_pred))
# standard_ce = -(one_hot * torch.log(softmax_probs)).sum()
# print(f"Standard CE: -Σ(p_true * log(p_pred)) = {standard_ce}")
# 
# # Method 2: Simplified (only true class contributes)
# print("\n--- Method 2: Simplified (true class only) ---")
# true_class_prob = softmax_probs[sample_target]
# simplified_ce = -torch.log(true_class_prob)
# print(f"True class probability: {true_class_prob}")
# print(f"Simplified CE: -log(p_true_class) = {simplified_ce}")
# 
# # Method 3: PyTorch's approach (log-softmax + selection)
# print("\n--- Method 3: PyTorch's Implementation ---")
# log_softmax = sample_logits - torch.log(torch.exp(sample_logits).sum())
# pytorch_ce = -log_softmax[sample_target]
# print(f"Log-softmax: {log_softmax}")
# print(f"PyTorch CE: -log_softmax[true_class] = {pytorch_ce}")
# 
# print(f"\n--- Verification ---")
# print(f"Method 1 (standard): {standard_ce:.6f}")
# print(f"Method 2 (simplified): {simplified_ce:.6f}")
# print(f"Method 3 (PyTorch): {pytorch_ce:.6f}")
# print(f"All methods equal? {torch.allclose(standard_ce, simplified_ce) and torch.allclose(simplified_ce, pytorch_ce)}")


##


