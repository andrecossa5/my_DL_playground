"""
Wine dataset. Classification: NN vs xgboost.
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
import plotting_utils as plu
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader


##


# Set results path
path_results = '/Users/IEO5505/Desktop/AI and DL/learn/my_DL_playground/results/wine'


##


# ======================================================= # Utils

def train_test_split(X, y, train_size=.8, random_state=1234):
    """
    Split data into train and test sets.
    """
    ss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    for train_idx, test_idx in ss.split(X, y):
        X_train = X[train_idx,:]; y_train = y[train_idx]
        X_test = X[test_idx,:]; y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


##


def xgboost_classification(X_train, y_train, X_test, y_test, n_iter=5):
    """
    XGBoost-based classification.
    """
    T = plu.Timer()
    T.start()

    params = {
        "xgboost__n_estimators": [100, 200, 300, 500],
        "xgboost__max_depth": [4, 6, 8, 10],
        "xgboost__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "xgboost__num_leaves": [31, 50, 100, 200],
        "xgboost__min_data_in_leaf": [10, 20, 50],
        "xgboost__feature_fraction": [0.8, 0.9, 1.0],
        "xgboost__bagging_fraction": [0.8, 0.9, 1.0]
    }
    pipe = Pipeline(
        steps=[ 
            ('pp', StandardScaler()), 
            ('xgboost', LGBMClassifier(objective='multiclass', n_jobs=-1))
        ]   
    )

    # Tune hyper-parameters
    xgboost = RandomizedSearchCV(
        pipe, 
        param_distributions=params, 
        n_iter=n_iter,
        refit=True,
        n_jobs=1,
        scoring='f1_macro',
        cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
        verbose=True
    )
    xgboost.fit(X_train, y_train)

    # Evaluation
    xgboost_classes = xgboost.predict(X_test) 
    report = pd.DataFrame.from_dict(
        classification_report(y_test, xgboost_classes, output_dict=True)
    )
    confusion = pd.DataFrame(confusion_matrix(y_test, xgboost_classes))

    # Save model and stats
    with open(os.path.join(path_results, 'xgboost.pickle'), 'wb') as f:
        pickle.dump(xgboost, f)
    report.to_csv(os.path.join(path_results, 'report_xgboost.csv'))
    confusion.to_csv(os.path.join(path_results, 'confusion_xgboost.csv'))

    print(confusion)
    print(report)
    print(f'xgboost finished in {T.stop()}')


##


class MyDataset(Dataset):
    """
    Dataset subclass for wine data.
    """
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)  # Changed to torch.long for classification
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return len(self.labels)


##

    
class NN(nn.Module):
    """
    FFNN.
    """
    def __init__(self, input_size=11, hidden_size=35, num_classes=6, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First layer: 11 → 64
            nn.ReLU(),                           # Activation function
            nn.Dropout(dropout),                 # Regularization: drop 20% of neurons
            nn.Linear(hidden_size, hidden_size), # Second layer: 64 → 64  
            nn.ReLU(),                           # Activation function
            nn.Dropout(dropout),                 # More regularization
            nn.Linear(hidden_size, num_classes)  # Output layer: 64 → 6
        )
    
    def forward(self, x):
        return self.network(x)


##


def nn_classification(X_train, y_train, X_test, y_test, n_epochs=100, hidden_size=128, dropout=0.3, batch_size=32):
    """
    NN-based classification with proper validation, class weighting, and early stopping.
    """
    
    T = plu.Timer()
    T.start()

    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Handle class imbalance with class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    # print(f"Class weights: {dict(zip(np.unique(y_train), class_weights))}")
    
    # 2. Create train/validation split
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(val_split.split(X_train_scaled, y_train))
    X_train_final = X_train_scaled[train_idx]
    y_train_final = y_train[train_idx]
    X_val = X_train_scaled[val_idx]
    y_val = y_train[val_idx]
    # print(f"Train size: {len(X_train_final)}, Validation size: {len(X_val)}")
    
    # 3. Create datasets and loaders
    train_dataset = MyDataset(X_train_final, y_train_final)
    val_dataset = MyDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model with configurable architecture
    nn_model = NN(hidden_size=hidden_size, dropout=dropout)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    # print(f"Model architecture: {sum(p.numel() for p in nn_model.parameters())} parameters")
    
    # 5. Training with validation and early stopping
    best_val_acc = 0
    patience_counter = 0
    patience = 15
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(n_epochs):

        # Training phase
        nn_model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = nn_model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        nn_model.eval()
        epoch_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = nn_model(X_batch)
                loss = loss_fn(outputs, y_batch)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model state
            best_model_state = nn_model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.4f}')
            break
    
    # Load best model
    nn_model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_dataset = MyDataset(X_test_scaled, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    nn_model.eval()  
    with torch.no_grad(): 
        X_, _ = next(iter(test_dataloader))
        nn_pred_logits = nn_model(X_)
        nn_classes = nn_pred_logits.argmax(1).detach().numpy()
    
    # Save model and performance
    report = pd.DataFrame.from_dict(
        classification_report(y_test, nn_classes, output_dict=True)
    )
    confusion = pd.DataFrame(confusion_matrix(y_test, nn_classes))
    
    # Save results
    report.to_csv(os.path.join(path_results, 'report_nn.csv'))
    confusion.to_csv(os.path.join(path_results, 'confusion_nn.csv'))

    print(confusion)
    print(report)
    print(f'NN finished in {T.stop()}')
    

##


# Get data
df = fetch_openml("wine-quality-red", version=1, as_frame=True).frame
df.columns
df.head()
df.isna().sum()
df.describe()

# To numeric class
df['class'] = df['class'].astype(int)
mapping = { x:i for i,x in enumerate(sorted(df['class'].unique())) }
df['class'] = df['class'].map(mapping)

# Get X and y
y = df['class'].values
X = df.drop(columns=['class']).values

# Split into train and test
X_train, y_train, X_test, y_test = train_test_split(X, y)

# XGBoost classification
xgboost_classification(X_train, y_train, X_test, y_test, n_iter=25)

# Neural Network classification
nn_classification(X_train, y_train, X_test, y_test, n_epochs=20)


##