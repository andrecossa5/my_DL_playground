"""
Data, Datasets, and DataLoaders.
"""

import os
import torch
import torchvision 
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotting_utils as plu
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer
matplotlib.use('macOSX')


##


# Paths
path_main = '/Users/IEO5505/Desktop/AI and DL/learn/my_DL_playground'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'data')


##


# Check type of available devices
torch.accelerator.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device=device)


##


# Datasets

# IMAGES

# From torchvision

# Fashion MNIST: download and visualize data
training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
# ...

# 

labels_map = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
    9: "Ankle Boot"
}

#

# Show some random example of images
plu.set_rcParams({'dpi':30})

fig = plt.figure(figsize=(3, 3))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    idx = np.random.randint(0,len(training_data),size=1)[0]
    img, label = training_data[idx]
    ax = fig.add_subplot(rows, cols, i)
    ax.set_title(labels_map[label])
    ax.axis("off")
    ax.imshow(img.squeeze(), cmap="grey")

fig.tight_layout()
plt.show()


##


# Load single image and play
# img = Image.open(os.path.join(path_data, 'images', 'image_files', image1.jpeg'))
# 'get_child_images', 'get_format_mimetype', 
# 'getbands', 'getbbox', 'getchannel', 'getcolors', 
# 'getdata', 'getexif', 'getextrema', 'getim', 
# 'getpalette', 'getpixel', 'getprojection', 'getxmp

# img.mode
# img.size
# img.getdata()
# img.getcolors()

# Covert, transform
# t = ToTensor()
# T = t(img)


##


# Custom Dataset class for images
class MyImageDataset(Dataset):
    """
    Custom pytorch Dataset for loading image data from a folder 
    and (optional) labels stored in a CSV file (matched by image name to
    actual images).
    """
    def __init__(
        self, 
        path_data: str, 
        file_format: str = '.jpeg', 
        path_labels: Optional[str] = None,
        transform: Optional[Callable] = None
        ):
        self.path_data = path_data
        self.transform = transform

        image_files = [
            f for f in os.listdir(path_data) if f.endswith(file_format)
        ]
        self.image_ids = [os.path.splitext(f)[0] for f in image_files]
        self.image_paths = [
            os.path.join(path_data, f) for f in image_files
        ]
        if path_labels is not None and os.path.exists(path_labels):
            df = pd.read_csv(path_labels, index_col=0)
            if df.index.isin(self.image_ids).all():
                self.labels = df.loc[self.image_ids]['label'].to_list()
            else:
                raise ValueError('Not all examples are labelled in the label file!')
        else:
            self.labels = None

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx] if self.labels is not None else -1
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


##


# Example
path_data = '/Users/IEO5505/Desktop/AI and DL/learn/my_DL_playground/data/images/'
path_images = os.path.join(path_data, 'image_files')
path_labels = os.path.join(path_data, 'image_labels.csv')
dataset = MyImageDataset(
    path_images, 
    path_labels=path_labels, 
    transform=None #ToTensor() for conversion
) 
dataset.image_paths
dataset.labels
dataset[0]

##

# Viz MyImageDataset
n = len(dataset)
plu.set_rcParams({"dpi":50})

fig, axs = plt.subplots(1,n,figsize=(6,2))
for i, ax in enumerate(axs):
    img, label = dataset[i]
    image_id = dataset.image_ids[i]
    ax.imshow(img)
    ax.set(title=f'{image_id}: {label}')

fig.tight_layout()
plt.show()


##


# TEXT, NLP

# Public, HugginFace
dataset = load_dataset("imdb")
dataset['unsupervised'][5]['text']

# Mine
# path_data = '/Users/IEO5505/Desktop/AI and DL/projects/vMTB/data/report.json'
# dataset = load_dataset('json', data_files=path_data)
# dataset['train'][0]


# Tokenize one example
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Example
tokens = tokenizer.tokenize("Hello world!")
ids = tokenizer.convert_tokens_to_ids(tokens)
enc = tokenizer("Hello world!", return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))

# Whole dataset
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset['train'][0].keys()
tokenized_dataset['train'][0]['input_ids']

# To torch
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
tokenized_dataset['train'][0]


##


# TABULAR, standard

# Iris example
df = sns.load_dataset("iris")
mapping = { x:i for i,x in enumerate(df['species'].unique()) }
df['species'] = df['species'].map(mapping).astype('float')

##

class MyDataset(torch.utils.data.Dataset):
    """
    Small dataset from numeric DataFrame.
    """
    def __init__(self, df, labels='species'):
        self.data = torch.tensor(df.drop(columns=[labels]).values) 
        self.labels = torch.tensor(df[labels].values) 
    
    def __getitem__(self, idx):
        return self.data[idx,:], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

dataset = MyDataset(df)
dataset[0]


##


# Dataloaders, iterators and batches

# FashionMNIST
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

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Viz 1 batch of 64 images
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
mapping = {
    k.item():v for k,v in zip(training_data.targets.unique(),training_data.classes)
}
mapping

fig, axs = plt.subplots(2,5,figsize=(10,4))
for i,x in enumerate(train_features):
    ax = axs.flatten()[i]
    ax.imshow(x.squeeze().numpy())
    plu.format_ax(ax, title=mapping[train_labels[i].item()])
    if i>8:
        break

fig.tight_layout()
plt.show()


##
