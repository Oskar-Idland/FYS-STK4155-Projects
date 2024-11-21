import torch
import torch.nn as nn
import torchvision.models as models 
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time
import copy
import torch.nn.functional as F
from sklearn.metrics import classification_report

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable convolutions
        self.conv2_dw = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv2_pw = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.conv3_pw = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4_dw = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.conv4_pw = nn.Conv2d(128, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5_dw = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv5_pw = nn.Conv2d(128, 256, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 3)  # 3 classes
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable conv blocks
        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        x = F.relu(self.bn4(self.conv4_pw(self.conv4_dw(x))))
        x = F.relu(self.bn5(self.conv5_pw(self.conv5_dw(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(model, lossFunction, optimizer, scheduler, dataloaders, dataset_sizes, class_names, device, num_epochs=20, patience=2, min_epochs=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    consecutive_epochs_without_improvement = 0

    # Dictionary to store metrics for each epoch
    epoch_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = lossFunction(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Store metrics
            if phase == 'train':
                epoch_metrics['train_loss'].append(epoch_loss)
                epoch_metrics['train_acc'].append(epoch_acc.item())
            else:
                epoch_metrics['val_loss'].append(epoch_loss)
                epoch_metrics['val_acc'].append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'validation':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    consecutive_epochs_without_improvement = 0
                else:
                    consecutive_epochs_without_improvement += 1

        if epoch >= min_epochs and consecutive_epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    
    # Validation metrics
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    target_names = [str(class_names[i]) for i in range(len(class_names))]
    print(classification_report(y_true, y_pred, target_names=target_names))

    return model, epoch_metrics

