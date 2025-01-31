import torch
import torch as nn
import numpy as np
import pandas as pd
from torch import nn
import copy
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


class EarlyStopping:
    def __init__(self, patience=400, min_delta=1e-3, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


# GPU
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_data():
    df = pd.read_csv("winequality_data.csv")
    df = df.drop(
        columns=["Id", "fixed acidity", "chlorides", "free sulfur dioxide"], axis=1
    )
    df = df[df["total sulfur dioxide"] < 200]

    # Binning the 'quality' column into 'low', 'average', 'high'
    bins = [2, 4, 6, 9]
    labels = ["low", "average", "high"]
    df["quality"] = pd.cut(df["quality"], bins=bins, labels=labels, include_lowest=True)
    # Encoding 'quality' into integers (0 for 'low', 1 for 'average', 2 for 'high')
    label_mapping = {"low": 0, "average": 1, "high": 2}
    df["quality"] = df["quality"].map(label_mapping)

    X = df.drop(columns=["quality"]).values
    y = df["quality"].values
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # SMOTE
    smote = SMOTE(sampling_strategy={0: 150, 2: 150}, random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to Torch tensors
    x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.long)
    x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test, device=device, dtype=torch.long)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data()

# Create dataloaders
BATCH_SIZE = 32

dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# Create model using nn.Sequential
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 3),
    nn.LogSoftmax(dim=1),
)

model = torch.compile(model, backend="aot_eager").to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
es = EarlyStopping()

train_accuracies = []
val_accuracies = []
epoch = 0
done = False
while epoch < 1000 and not done:  # 200
    epoch += 1
    steps = list(enumerate(dataloader_train))
    pbar = tqdm.tqdm(steps)
    model.train()
    correct_train = 0
    total_train = 0

    for i, (x_batch, y_batch) in pbar:
        y_batch_pred = model(x_batch.to(device))
        loss = loss_fn(y_batch_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(y_batch_pred, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

        loss, current = loss.item(), (i + 1) * len(x_batch)
        if i == len(steps) - 1:
            model.eval()
            val_pred = model(x_test)
            _, predicted_val = torch.max(val_pred, 1)
            correct_val = (predicted_val == y_test).sum().item()
            total_val = y_test.size(0)
            val_accuracy = correct_val / total_val

            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            vloss = loss_fn(val_pred, y_test)
            if es(model, vloss):
                done = True
            pbar.set_description(
                f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, {es.status}"
            )
        else:
            pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")

# Final evaluation
pred = model(x_test)
_, predict_classes = torch.max(pred, 1)
correct = accuracy_score(y_test.cpu(), predict_classes.cpu())
print(f"Test Accuracy: {correct * 100:.2f}%")
print(classification_report(y_test.cpu(), predict_classes.cpu()))

# Plot learning curve
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
