import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import json

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_dataset(directory):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                # Assuming label is the parent folder name
                label = os.path.basename(os.path.dirname(root))
                image_paths.append(image_path)
                labels.append(label)
    return image_paths, labels

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# input train dataset
directory = 'augmented_images/augmented_data'
image_paths, labels = load_dataset(directory)

label_set = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(label_set)}

train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, train_size=0.8, random_state=42)

train_dataset = CustomDataset(train_image_paths, train_labels, label_to_idx, transform)
test_dataset = CustomDataset(test_image_paths, test_labels, label_to_idx, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 64 * 64, len(label_set))  # Adjust the size according to your image size

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
training_metrics = {'epoch': [], 'accuracy': [], 'train_size': len(train_loader.dataset)}

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Evaluate the model at the end of each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    training_metrics['epoch'].append(epoch + 1)
    training_metrics['accuracy'].append(accuracy)
    print(f'Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')

# Save the model and training metrics
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/saved_model.pth')
os.makedirs('metrics', exist_ok=True)
with open('metrics/training_metrics.json', 'w') as f:
    json.dump(training_metrics, f)