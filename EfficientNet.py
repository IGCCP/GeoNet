import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import torch.nn as nn

import torch.optim as optim
from efficientnet_pytorch import EfficientNet
import cv2
from google.colab import drive
import matplotlib.pyplot as plt

# 自定义数据集类
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

# 加载数据集函数
def load_dataset(directory):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                label = filename.split('.')[0]  # 使用文件名（不含扩展名）作为标签
                image_paths.append(image_path)
                labels.append(label)
    return image_paths, labels

# 指定数据存放的目录
directory = '/model/max48'
image_paths, labels = load_dataset(directory)

# 创建标签索引并保存
label_set = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(label_set)}

# 保存标签索引
with open('label_to_idx.pkl', 'wb') as f:
    pickle.dump(label_to_idx, f)

# 分割数据集
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, train_size=0.8, random_state=42)

# 打印所有图片路径和它们的标签
# for path, label in zip(image_paths, labels):
#     print("Path:", path, "Label:", label)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use of custom dataset classes and data loaders
train_dataset = CustomDataset(train_image_paths, train_labels, label_to_idx, transform)
test_dataset = CustomDataset(test_image_paths, test_labels, label_to_idx, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# Modify the final fully-connected layer to accommodate the new classification task
num_ftrs = model._fc.in_features
num_classes = len(label_set)
model._fc = nn.Linear(num_ftrs, num_classes)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss functions and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training models
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# save model
torch.save(model.state_dict(), 'model.pth')

# test model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')

# load model
model.load_state_dict(torch.load('model.pth'))

# label
def draw_labels_on_image(image_path, model, label_to_idx, transform):
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    image = Image.open(image_path).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape
    stride = 256
    labels_added = set()
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            box = image.crop((x, y, x + stride, y + stride))
            if transform:
                box = transform(box).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(box)
                _, predicted = torch.max(outputs.data, 1)
                label = idx_to_label[predicted.item()]
                if label not in labels_added:
                    labels_added.add(label)
                    cv2.putText(image_cv, label, (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

    output_path = 'labeled_large_image.jpg'
    cv2.imwrite(output_path, image_cv)
    print(f'Saved labeled image to {output_path}')

    # 使用matplotlib显示图像
    plt.figure(figsize=(20, 20))
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Image to be processed
large_image_path = '/content/drive/My Drive/Agency_Creek_DWM-182_m_0.png'
draw_labels_on_image(large_image_path, model, label_to_idx, transform)