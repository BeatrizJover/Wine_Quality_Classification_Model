import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
import random

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic behavior in CUDA (if using GPU)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
data = pd.read_csv("dataset/winequality_data.csv")
data = data.drop(
    columns=["Id", "fixed acidity", "chlorides", "free sulfur dioxide"], axis=1
)
data = data[data["total sulfur dioxide"] < 200]

# Binning the 'quality' column into 'low', 'average', 'high'
bins = [2, 4, 6, 9]
labels = ["low", "average", "high"]
data["quality"] = pd.cut(data["quality"], bins=bins, labels=labels, include_lowest=True)
# Encoding 'quality' into integers (0 for 'low', 1 for 'average', 2 for 'high')
label_mapping = {"low": 0, "average": 1, "high": 2}
data["quality"] = data["quality"].map(label_mapping)

# Separate features and target
X = data.drop("quality", axis=1).values
y = data["quality"].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the baseline neural network
class WineQualityModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WineQualityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


# Initialize the model
input_dim = X_train.shape[1]
output_dim = len(set(y))
model = WineQualityModel(input_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epoch_train_accuracies = []
epoch_test_accuracies = []
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

    train_accuracy = correct_train / total_train
    epoch_train_accuracies.append(train_accuracy)

    # Calculate test accuracy
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total_test += y_batch.size(0)
            correct_test += (predicted == y_batch).sum().item()

    test_accuracy = correct_test / total_test
    epoch_test_accuracies.append(test_accuracy)

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
        f"Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%"
    )

# Plot the accuracy vs epochs for both training and test
plt.plot(
    range(1, epochs + 1), epoch_train_accuracies, label="Train Accuracy", color="blue"
)
plt.plot(
    range(1, epochs + 1), epoch_test_accuracies, label="Test Accuracy", color="red"
)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy (Train vs Test)")
plt.legend()
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
plt.show()

# Evaluate the model
model.eval()
y_true = []
y_pred = []
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

# Convert to numpy arrays for metric calculations
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)

# Calculate metrics
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
outputs = model(X_test_tensor)
vloss = criterion(outputs, y_test_tensor)
print(f"Loss = {vloss.item()}")
print(classification_report(y_true_np, y_pred_np))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
