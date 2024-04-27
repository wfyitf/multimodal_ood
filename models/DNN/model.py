import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiLabelNN(nn.Module):
    def __init__(self, input_size=512, output_size=11):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, 64) 
        self.fc4 = nn.Linear(64, output_size)    
        self.relu = nn.ReLU()           
        self.sigmoid = nn.Sigmoid()    

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  
        return x

    def evaluate(model, data_loader, loss_function):
        model.eval()  
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        with torch.no_grad():  
            for inputs, labels in data_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                predictions = outputs > 0.5 
                total_accuracy += (predictions == labels.byte()).all(dim=1).float().mean().item()
                total_samples += 1

        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples

        return average_loss, average_accuracy
    


