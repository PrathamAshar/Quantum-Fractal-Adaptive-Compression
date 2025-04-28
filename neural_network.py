import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def build_and_train_neural_network():
    # Generate synthetic training data
    x_train = np.array([[np.random.randint(1, 100), np.random.random()] for _ in range(100)])
    y_train = np.array([np.random.randint(0, 2) for _ in range(100)])
    
    # Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    # Split the data into training and validation sets
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_split, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
    x_val_tensor = torch.tensor(x_val_split, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_split, dtype=torch.float32).unsqueeze(1)
    
    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    
    return model, scaler

def predict_compressibility(model, scaler, data):
    # Prepare test data
    x_test = np.array([[len(data), np.random.random()]])
    x_test_scaled = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor)
    return predictions[0].item()