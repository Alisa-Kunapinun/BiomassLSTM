#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import numpy as np
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
DEBUG = False
#%%
class PhysicsLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PhysicsLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs_many_to_one, y_true, outputs_many_to_many, y_ideal):
        if DEBUG:
            print(f'outputs_many_to_one: {outputs_many_to_one.shape}, y_true: {y_true.shape}')
            print(f'outputs_many_to_many: {outputs_many_to_many.shape}, y_ideal: {y_ideal.shape}')
        
        # Adjust shapes to match
        if outputs_many_to_many.shape[-1] == 1:
            outputs_many_to_many = outputs_many_to_many.squeeze(-1)

        # MSE loss for true values (many-to-one)
        loss_true = self.mse_loss(outputs_many_to_one, y_true.unsqueeze(-1))
        # MSE loss for ideal values (many-to-many)
        if alpha <= 0.0:
            y_ideal[:,-1] = y_true.unsqueeze(-1)[:,-1]
        mask = (y_ideal != 0)
        y_ideal_filtered = y_ideal[mask]
        outputs_many_to_many_filtered = outputs_many_to_many[mask]
        loss_ideal = self.mse_loss(outputs_many_to_many_filtered, y_ideal_filtered)

        # Combined loss
        loss = self.alpha * loss_true + (1 - self.alpha) * loss_ideal
        return loss, loss_true, loss_ideal
    
# %%
def read_files_recursively(folder):
    all_files = []
    print(folder)
    def read_files(folder):
        print(folder)
        for entry in os.scandir(folder):
            if entry.is_file():
                all_files.append(entry.path)
            elif entry.is_dir():
                read_files(entry.path)

    read_files(folder)
    return all_files
# %%
import numpy as np
# Dummy function for loading data from a CSV file
def load_data(file_path, sensor_val = 'sensor_val', ideal_biomass_kg = 'ideal_biomass_kg'):
    # print(file_path)
    data = pd.read_csv(file_path)

    data['datetime'] = pd.to_datetime(data['datetime'])
    start_time = data['datetime'].min()
    data['timestamp'] = (data['datetime'] - start_time).dt.total_seconds() / 60.0

    inputs = data[['timestamp', sensor_val,'Air_Temp','Solar_radiation','humidity','Precipitation Intensity mm/h',
                   'Total Precipitation Millimeters','takeout_kg']].values
    biomass_kg = data['biomass_kg'].values
    # print('here')
    ideal_biomass_kg = data[ideal_biomass_kg].values
    # print('here')
    # count = data[data[ideal_biomass_kg] != 0.0].shape[0]
    
    return inputs, biomass_kg, ideal_biomass_kg

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]
        y = data[i+seq_length, -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class SeaweedDataset(Dataset):
    def __init__(self, file_paths, sensor_val = 'sensor_val', ideal_biomass_kg = 'ideal_biomass_kg'):
        self.file_paths = file_paths
        self.sensor_val = sensor_val
        self.ideal_biomass_kg = ideal_biomass_kg
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        inputs, biomass_kg, ideal_biomass_kg = load_data(file_path, self.sensor_val, self.ideal_biomass_kg)
        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        biomass_kg = torch.tensor(biomass_kg, dtype=torch.float32)
        ideal_biomass_kg = torch.tensor(ideal_biomass_kg, dtype=torch.float32)
        y_true = biomass_kg[-1]  # Assume last value is the true value for prediction
        y_ideal = ideal_biomass_kg  # Assume the whole sequence is the ideal output
        
        return inputs, y_true, y_ideal
# %%
def collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    inputs, y_true, y_ideal = zip(*batch)
    
    # Pad sequences
    lengths = [len(seq) for seq in inputs]
    max_len = max(lengths)
    
    padded_inputs = torch.zeros(len(inputs), max_len, inputs[0].size(1))
    for i, seq in enumerate(inputs):
        end = lengths[i]
        padded_inputs[i, :end, :] = seq
    
    y_true = torch.stack(y_true)
    padded_y_ideal = torch.zeros(len(inputs), max_len)
    for i, seq in enumerate(y_ideal):
        end = lengths[i]
        padded_y_ideal[i, :end] = seq
    
    # print(f"Lengths: {lengths}")
    # print(f"Padded inputs shape: {padded_inputs.shape}")
    # print(f"Padded y_ideal shape: {padded_y_ideal.shape}")
    
    return padded_inputs, y_true, padded_y_ideal, lengths
# %%
main_folder = 'generate_reduce_sensors'
file_paths = read_files_recursively(main_folder)
# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, alpha=0.5, device="cpu", epochs_start=1):
    model.train()
    best_val_loss = float('inf')
    log_file = 'log_lstm/training_log.txt'
    os.makedirs('log_lstm', exist_ok=True)
    
    if epochs_start == 1:
        with open(log_file, 'w') as f:
            f.write('Epoch,Train_Loss,Val_Loss,Val_Loss_True,Val_Loss_Ideal\n')

    for epoch in range(epochs_start-1, num_epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, y_true, y_ideal, lengths) in enumerate(train_loader):
            if i % 1 == 0:
                print(f'\rEpoch {epoch} Iteration {i+1}', end='', flush=True)
            inputs = inputs.to(device)
            y_true = y_true.to(device)
            y_ideal = y_ideal.to(device)
            
            outputs_many_to_many, outputs_many_to_one = model(inputs)
            if DEBUG:
                print(f'inputs: {inputs.shape}')
                print(f'outputs_many_to_many: {outputs_many_to_many.shape}')
                print(f'outputs_many_to_one: {outputs_many_to_one.shape}')
                print(f'y_true: {y_true.shape}')
                print(f'y_ideal: {y_ideal.shape}')
            
            loss, loss_true, loss_ideal = criterion(outputs_many_to_one, y_true, outputs_many_to_many, y_ideal)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_loss_true = 0
        val_loss_ideal = 0
        with torch.no_grad():
            for inputs, y_true, y_ideal, lengths in val_loader:
                inputs = inputs.to(device)
                y_true = y_true.to(device)
                y_ideal = y_ideal.to(device)

                outputs_many_to_many, outputs_many_to_one = model(inputs)
                if DEBUG:
                    print(f'Validation - inputs: {inputs.shape}')
                    print(f'Validation - outputs_many_to_many: {outputs_many_to_many.shape}')
                    print(f'Validation - outputs_many_to_one: {outputs_many_to_one.shape}')
                    print(f'Validation - y_true: {y_true.shape}')
                    print(f'Validation - y_ideal: {y_ideal.shape}')

                loss, loss_true, loss_ideal = criterion(outputs_many_to_one, y_true, outputs_many_to_many, y_ideal)
                val_loss += loss.item()
                val_loss_true += loss_true.item()
                val_loss_ideal += loss_ideal.item()

        val_loss /= len(val_loader)
        val_loss_true /= len(val_loader)
        val_loss_ideal /= len(val_loader)

        # Update the scheduler
        scheduler.step(val_loss)

        # Get the last learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Learning rate after epoch {epoch+1}: {current_lr}')

        # Logging
        with open(log_file, 'a') as f:
            f.write(f'{epoch+1},{epoch_loss:.4f},{val_loss:.4f},{val_loss_true:.4f},{val_loss_ideal:.4f}\n')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'log_lstm/seaweed_growth_lstm_{epoch+1}.pth')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'log_lstm/seaweed_growth_lstm_best.pth')

        print(f'\rEpoch [{epoch+1}/{num_epochs}], Learning rate: {current_lr}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Loss True: {val_loss_true:.4f}, Val Loss Ideal: {val_loss_ideal:.4f}')

# %%
class SeaweedGrowthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SeaweedGrowthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out_many_to_many = self.fc(out)
        out_many_to_one = self.fc(out[:, -1, :])
        return out_many_to_many, out_many_to_one
# %%
from torch.utils.data import Dataset, DataLoader, random_split

# normal training dataset
alpha = 0.8
dataset = SeaweedDataset(file_paths)
#####################################################
# Split the dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True, collate_fn=collate_fn)
# %%
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Hyperparameters
input_size = 8
hidden_size = 64
output_size = 1
num_layers = 5
num_epochs = 100
learning_rate = 0.001
epoch_start = 1
#%%
# Model, Loss, Optimizer
# model = SeaweedGrowthLSTM(input_size, hidden_size, output_size, num_layers).to(device)
model = SeaweedGrowthLSTM(input_size, hidden_size, output_size, num_layers)  # Ensure these parameters match your saved model
# model.load_state_dict(torch.load('log_lstm/seaweed_growth_lstm_100.pth'))
model = model.to(device)
criterion = PhysicsLoss(alpha)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
# %%
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, alpha, device, epoch_start)

# Save the model
torch.save(model.state_dict(), 'seaweed_growth_lstm.pth')
# %%
import matplotlib.pyplot as plt

def plot_sensor_vs_outputs(sensor_data, y_true, y_ideal, y_pred, y_pred_last, title="Sensor Data vs Outputs", is_save = False, file_name = ""):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Sensor Data', color='tab:blue')
    ax1.plot(sensor_data, color='tab:blue', label='Sensor Data')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Output', color='tab:orange')
    ax2.set_ylim(ymax=y_true[-1] * 1.2)
    ax2.plot(y_true, color='tab:orange', label='True Output')
    ax2.plot(len(y_true)-1, y_true[-1], 'o', color='tab:orange', markersize=10, label='True Output (Last Point)')
    ax2.plot(len(y_true)-1, y_ideal[-1], '*', color='tab:green', markersize=10, label='Ideal Output (Last Point)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add legends
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(title)

    if is_save:
        plt.savefig(file_name)
    plt.show()

def plot_sensor_vs_predictions(sensor_data, y_pred, y_pred_last, y_true_last, title="Sensor Data vs Predictions", is_save = False, file_name = ""):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Sensor Data', color='tab:blue')
    ax1.plot(sensor_data, color='tab:blue', label='Sensor Data')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Predictions', color='tab:orange')
    _max = y_pred.max()
    if _max < y_true_last:
        _max = y_true_last
    ax2.set_ylim(ymax=_max * 1.2)
    ax2.plot(y_pred, color='tab:orange', label='Predicted Output')
    ax2.plot(len(y_pred)-1, y_pred_last, '*', color='tab:red', markersize=10, label='Predicted Output (Last Point)')
    ax2.plot(len(y_pred)-1, y_true_last, 'o', color='tab:blue', markersize=10, label='True Output (Last Point)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add legends
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(title)

    if is_save:
        plt.savefig(file_name)
    plt.show()

def plot_input_vs_predictions(input_data, y_pred, y_pred_last, y_true_last, is_save = False, file_name = ""):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    inputs = ['timestamp', 'sensor_val','Air_Temp','Solar_radiation','humidity','Precipitation Intensity mm/h',
                   'Total Precipitation Millimeters','takeout_kg']

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Sensor Data', color='tab:blue')
    ax1.plot(input_data[:, 1], color='tab:blue', label='Sensor Data')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Sensor Data")

    if is_save:
        plt.savefig(file_name + "_sensorval.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Air Temperature', color='tab:blue')
    ax1.plot(input_data[:, 2], color='tab:blue', label='Air Temperature')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Air Temperature")

    if is_save:
        plt.savefig(file_name + "_Air_Temp.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Solar radiation', color='tab:blue')
    ax1.plot(input_data[:, 3], color='tab:blue', label='Solar radiation')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Solar radiation")

    if is_save:
        plt.savefig(file_name + "_Solar_radiation.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Humidity', color='tab:blue')
    ax1.plot(input_data[:, 4], color='tab:blue', label='Humidity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Humidity")

    if is_save:
        plt.savefig(file_name + "_humidity.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Precipitation Intensity', color='tab:blue')
    ax1.plot(input_data[:, 5], color='tab:blue', label='Precipitation Intensity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Precipitation Intensity mm/h")

    if is_save:
        plt.savefig(file_name + "_Precipitation Intensity.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot sensor data
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Total Precipitation', color='tab:blue')
    ax1.plot(input_data[:, 6], color='tab:blue', label='Sensor Data')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Total Precipitation Millimeters")

    if is_save:
        plt.savefig(file_name + "_Total Precipitation.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set_ylabel('Predictions', color='tab:orange')
    _max = y_pred.max()
    if _max < y_true_last:
        _max = y_true_last
    ax1.set_ylim(ymax=_max * 1.2)
    ax1.plot(y_pred, color='tab:orange', label='Predicted Output')
    ax1.plot(len(y_pred)-1, y_pred_last, '*', color='tab:red', markersize=10, label='Predicted Output (Last Point)')
    ax1.plot(len(y_pred)-1, y_true_last, 'o', color='tab:blue', markersize=10, label='True Output (Last Point)')
    ax1.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Predictions vs True Output")

    if is_save:
        plt.savefig(file_name)
    plt.show()

def plot_ideal_vs_predictions(y_ideal, y_true_last, y_pred, y_pred_last, title="Ideal vs Predictions", is_save = False, file_name = ""):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    m1 = y_ideal.max()
    m2 = y_pred.max()

    _max = 0
    if m1 > m2:
        _max = m1
    else:
        _max = m2

    if _max < y_true_last:
        _max = y_true_last

    # Plot ideal output
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Ideal Output', color='tab:blue')
    ax1.set_ylim(ymax=_max * 1.2)
    ax1.plot(y_ideal, color='tab:blue', label='Ideal Output')
    ax1.plot(len(y_ideal)-1, y_true_last, 'o', color='tab:blue', markersize=10, label='True Output (Last Point)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot predicted output
    ax2 = ax1.twinx()
    ax2.set_ylabel('Predictions', color='tab:orange')
    ax2.set_ylim(ymax=_max * 1.2)
    ax2.plot(y_pred, color='tab:orange', label='Predicted Output')
    ax2.plot(len(y_pred)-1, y_pred_last, '*', color='tab:red', markersize=10, label='Predicted Output (Last Point)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add legends
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(title)

    if is_save:
        plt.savefig(file_name)
    plt.show()

#%%
# Example usage
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

# Load the model state dictionary
model = SeaweedGrowthLSTM(input_size, hidden_size, output_size, num_layers)  # Ensure these parameters match your saved model
model.load_state_dict(torch.load('log_lstm_00/seaweed_growth_lstm_best.pth'))
model.to(device)

# Example usage
# Load a dataset from the validation set and use the model to make predictions
sample_file = 'generate_reduce_sensors/tank_4/sensor_data_00011.csv'  # Replace with your validation file path
inputs, y_true, y_ideal = load_data(sample_file)

# Convert inputs to tensor
inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    outputs_many_to_many, outputs_many_to_one = model(inputs_tensor)
    y_pred_tensor = outputs_many_to_many
    print(f"y_pred_tensor shape: {y_pred_tensor.shape}")  # Debug print
    y_pred = y_pred_tensor.squeeze().cpu().numpy()
    print(f"y_pred shape: {y_pred.shape}")  # Debug print
    y_pred_last = y_pred[-1] if y_pred.ndim > 0 else y_pred
    print(f"y_pred data: {y_pred_last}")  # Debug print

#%%
# Plotting
plot_sensor_vs_outputs(inputs[:, 1], y_true, y_ideal, y_pred, y_pred_last, title="Sensor Data vs True Output vs Ideal Output")
plot_sensor_vs_predictions(inputs[:, 1], y_pred, y_pred_last, y_true[-1], title="Sensor Data vs Predicted Output")
plot_ideal_vs_predictions(y_ideal, y_true[-1], y_pred, y_pred_last, title="Ideal Output vs Predicted Output")
# plot_input_vs_predictions(inputs, y_pred, y_pred_last, y_true[-1])

#%%
def predicteach(model, inputs_tensor, y_true, y_ideal):
    model.eval()
    with torch.no_grad():
        outputs_many_to_many, outputs_many_to_one = model(inputs_tensor)
        y_pred_tensor = outputs_many_to_many
        print(f"y_pred_tensor shape: {y_pred_tensor.shape}")  # Debug print
        y_pred = y_pred_tensor.squeeze().cpu().numpy()
        print(f"y_pred shape: {y_pred.shape}")  # Debug print
        y_pred_last = y_pred[-1] if y_pred.ndim > 0 else y_pred
        print(f"y_pred last: {y_pred_last}")  # Debug print
        y_true_last = y_true[-1]
        print(f"y_true last: {y_true_last}")  # Debug print
        y_ideal_last = y_ideal[-1]
        print(f"y_ideal last: {y_ideal_last}")  # Debug print

    return y_pred, y_pred_last
# %%
for i in range(0,9):
    sample_file = f'real_sensor_ideal/data{i + 1}.csv'  # Replace with your validation file path
    # inputs, y_true, y_ideal = load_data(sample_file, sensor_val='sensor_val')
    inputs, y_true, y_ideal = load_data(sample_file, sensor_val='sensor_avg')

    # Convert inputs to tensor
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)
    y_pred, y_pred_last = predicteach(model, inputs_tensor, y_true, y_ideal)

    y_pred_last = y_pred[-1] if y_pred.ndim > 0 else y_pred
    y_true_last = y_true[-1]
    y_ideal_last = y_ideal[-1]

    Graph_result_lstm = 'Graph_result_lstm_avg_095'

    output_file = f'{Graph_result_lstm}/result_log.txt'
    with open(output_file, 'a') as f:
        f.write(f'{sample_file}\n')
        f.write(f"y_pred last: {y_pred_last}\n")
        f.write(f"y_true last: {y_true_last}\n")
        f.write(f"y_ideal last: {y_ideal_last}\n")

    plot_sensor_vs_outputs(inputs[:, 1], y_true, y_ideal, y_pred, y_pred_last, title="Sensor Data vs True Output vs Ideal Output", is_save=True, file_name=f'{Graph_result_lstm}/data{i + 1}_ideal.png')
    plot_sensor_vs_predictions(inputs[:, 1], y_pred, y_pred_last, y_true[-1], title="Sensor Data vs Predicted Output", is_save=True, file_name=f'{Graph_result_lstm}/data{i + 1}_predict.png')
    plot_ideal_vs_predictions(y_ideal, y_true[-1], y_pred, y_pred_last, title="Ideal Output vs Predicted Output", is_save=True, file_name=f'{Graph_result_lstm}/data{i + 1}_compare.png')
    plot_input_vs_predictions(inputs, y_pred, y_pred_last, y_true_last, is_save = True, file_name = f"{Graph_result_lstm}/data{i + 1}")
# %%
