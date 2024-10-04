import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet101, ResNet101_Weights
from tqdm import tqdm

num_classes = 5 # [Asian, Black, Indian, Others, White]
model = resnet101(weights=ResNet101_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

num_epochs = int(input("Set the number of epochs: "))
save_path = input("Enter save directory for your model: ")

# 과적합 방지를 위해 기존 레이어 freeze
for param in model.parameters():
    param.requires_grad = False

# 일부 레이어 unfreeze
blocks = [model.layer2, model.layer3, model.layer4]
for block in blocks:
    for param in block.parameters():
        param.requires_grad = True

model.fc.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else torch.device("cpu"))
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 학습 단계
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 검증 단계
    model.eval()
    val_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)
    val_acc = corrects.double() / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
