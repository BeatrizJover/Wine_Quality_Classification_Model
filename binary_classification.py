import torch
import torch as nn
import numpy as np
import pandas as pd
from torch import nn
import copy
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class EarlyStopping:
    def __init__(self, patience=160, min_delta=1e-3, restore_best_weights=True):
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

# Random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_data():
    df = pd.read_csv("winequality_data.csv")
    df = df.drop(columns=[
        "Id","density",'fixed acidity','chlorides', "free sulfur dioxide"], axis=1)
    df = df[df['total sulfur dioxide'] < 200]
    
    # Binning the 'quality' column into binary classification: 'low', 'high'
    bins = [2, 5, 9]  
    group_names = ['low', 'high']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
    # Encoding (0 for 'low', 1 for 'high')
    le = LabelEncoder()
    df['quality'] = le.fit_transform(df['quality'])
    
    X = df.drop(columns=["quality"]).values
    y = df["quality"].values  
    
    # Split into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to Torch tensors
    x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.float32)  
    x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test, device=device, dtype=torch.float32)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data()

# Create dataloaders
BATCH_SIZE = 32
dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 50),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(25, 15),
    nn.ReLU(),
    nn.Linear(15, 1),  
)

model = torch.compile(model, backend="aot_eager").to(device)

# BCEWithLogitsLoss better for binary classification
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
es = EarlyStopping()

# Training
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
epoch = 0
done = False
while epoch < 1000 and not done:
    epoch += 1
    steps = list(enumerate(dataloader_train))
    pbar = tqdm.tqdm(steps)
    model.train()
    epoch_train_loss = 0  
    correct_train = 0
    total_train = 0

    for i, (x_batch, y_batch) in pbar:
        y_batch_pred = model(x_batch.to(device)).squeeze(1)
        loss = loss_fn(y_batch_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_train_loss += loss.item() * len(x_batch) 

        # Calculate train accuracy
        predicted_labels = (torch.sigmoid(y_batch_pred) > 0.5).float()
        correct_train += (predicted_labels == y_batch.to(device)).sum().item()
        total_train += len(y_batch)

        if i == len(steps) - 1:
            model.eval()
            pred = model(x_test).squeeze(1)
            vloss = loss_fn(pred, y_test)

            # Calculate validation accuracy
            predicted_labels_val = (torch.sigmoid(pred) > 0.5).float()
            correct_val = (predicted_labels_val == y_test).sum().item()
            total_val = len(y_test)
            val_accuracy = correct_val / total_val

            if es(model, vloss):
                done = True
            pbar.set_description(
                f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, {es.status}"
            )
        else:
            pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
    
    # Store the epoch's training loss and validation loss
    epoch_train_loss /= len(dataloader_train.dataset)  
    val_losses.append(vloss.item())  
    train_losses.append(epoch_train_loss)  
    
    # Calculate training accuracy
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# Evaluation
model.eval()
pred = model(x_test).squeeze(1)
y_pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy() 
y_test_np = y_test.cpu().numpy()

accuracy = accuracy_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)
balanced_acc = balanced_accuracy_score(y_test_np, y_pred_np)
roc_auc = roc_auc_score(y_test_np, torch.sigmoid(pred).cpu().detach().numpy())

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"F1-score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test_np, y_pred_np)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase 0 (<=5)", "Clase 1 (>=6)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_test_np, y_pred_np, target_names=["Clase 0 (<=5)", "Clase 1 (>=6)"])
print("Classification Report:\n", report)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch+1), train_losses, label='Training Loss')
plt.plot(range(1, epoch+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# Plot learning curve for accuracy
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()




