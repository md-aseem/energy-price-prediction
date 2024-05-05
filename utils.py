import numpy as np
import torch

def remove_outliers_2(X_train, y_train, lower_bound, upper_bound):
    out_of_range_indices = np.where((y_train < lower_bound) | (y_train > upper_bound))[0]
    X_train_clean = np.delete(X_train, out_of_range_indices, axis=0)
    y_train_clean = np.delete(y_train, out_of_range_indices, axis=0)
    return X_train_clean, y_train_clean

def classify_hour(hour):
    if 8 <= hour < 24:
        return 'on_peak'
    else:
        return 'off_peak'
    
def evaluate_model_with_dataloader(model, criterion, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    total_absolute_error = 0.0
    total_relative_error = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets.unsqueeze(1))
            total_loss += loss.item() * inputs.size(0)

            absolute_errors = torch.abs(predictions - targets)
            safe_targets = torch.clamp(targets, min=1e-4)
            relative_errors = absolute_errors / safe_targets

            total_absolute_error += absolute_errors.sum().item()
            total_relative_error += relative_errors.sum().item()

    num_samples = len(dataloader.dataset)
    mean_squared_error = total_loss / num_samples
    root_mean_squared_error = np.sqrt(mean_squared_error)
    mean_absolute_percentage_error = (total_relative_error / num_samples) * 100

    return {
        "RMSE": root_mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "Total Loss": total_loss,
        "MAE": total_absolute_error / num_samples
    }

def evaluate_model(model, criterion, X_test, y_test):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    total_absolute_error = 0.0
    total_relative_error = 0.0

    # Assuming that X_test and y_test are already tensors
    with torch.no_grad():
        predictions = model(X_test)
        loss = criterion(predictions, y_test.unsqueeze(1))
        total_loss += loss.item() * X_test.size(0)

        absolute_errors = torch.abs(predictions - y_test)
        safe_targets = torch.clamp(y_test, min=1e-4)
        relative_errors = absolute_errors / safe_targets

        total_absolute_error += absolute_errors.sum().item()
        total_relative_error += relative_errors.sum().item()

    num_samples = X_test.shape[0]
    mean_squared_error = total_loss / num_samples
    root_mean_squared_error = np.sqrt(mean_squared_error)
    mean_absolute_percentage_error = (total_relative_error / num_samples) * 100

    return {
        "RMSE": root_mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "MAE": total_absolute_error / num_samples
    }
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def train_one_epoch(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def validate_one_epoch(model, val_loader, device, criterion):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            running_val_loss += loss.item() * inputs.size(0)
    return running_val_loss / len(val_loader.dataset)

def train_model(model, train_loader, val_loader, device, optimizer, criterion, num_epochs, model_save_path):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        epoch_train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = validate_one_epoch(model, val_loader, device, criterion)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model, model_save_path)

    plot_losses(train_losses, val_losses)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/model_training_loss.png")
    plt.show()
