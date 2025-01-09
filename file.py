import os
import shutil
import random
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader



#Setting up directories here
leaf3_dir = '/Users/danielpalin/Documents/3leaf2'
leaf4_dir = '/Users/danielpalin/Documents/4leaf2'

base_dir = '/Users/danielpalin/Documents/clover_dataset'
os.makedirs(base_dir, exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

classes = ['3_leaf', '4_leaf']

#Split data into train and validate
def split_data(src_dir, dest_train_dir, dest_val_dir, split_ratio = 0.7):
    images = os.listdir(src_dir)
    random.shuffle(images)
    train_count = round(len(images) * split_ratio)

    os.makedirs(dest_train_dir, exist_ok = True)
    os.makedirs(dest_val_dir, exist_ok = True)

    for i, img in enumerate(images):
        if i < train_count:
            shutil.copy(os.path.join(src_dir, img), dest_train_dir)
        else:
            shutil.copy(os.path.join(src_dir, img), dest_val_dir)

#Splitting 3 leaf images
split_data(
    leaf3_dir, 
    os.path.join(train_dir, '3_leaf'),
    os.path.join(val_dir, '3_leaf')
)
#Splitting 4 leaf iamages
split_data(
    leaf4_dir,
    os.path.join(train_dir, '4_leaf'),
    os.path.join(val_dir, '4_leaf')
)

transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

train_dataset = datasets.ImageFolder('/Users/danielpalin/Documents/clover_dataset/train', transform=transforms)
val_dataset = datasets.ImageFolder('/Users/danielpalin/Documents/clover_dataset/val', transform=transforms)


#Augmenting the image set
from PIL import Image
from torchvision import transforms

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.5),
    transforms.Resize((224,224))
])

augmented_dir = os.path.join(base_dir, 'augmented')
os.makedirs(augmented_dir, exist_ok = True)

for c in classes:
    os.makedirs(os.path.join(augmented_dir, c), exist_ok = True)

aug_number = 10
valid_extensions = {".jpg", ".jpeg", ".png"}  # Define valid image extensions

for c in classes:
    class_dir = os.path.join(train_dir, c)
    for image_name in os.listdir(class_dir):
        if os.path.splitext(image_name)[1].lower() not in valid_extensions:
            continue  # Skip non-image files

        img_path = os.path.join(class_dir, image_name)
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # Ensure the image is in RGB mode

            # Save the original image in the augmented directory
            original_name = f"{os.path.splitext(image_name)[0]}_original.jpg"
            img.save(os.path.join(augmented_dir, c, original_name))

            # Generate augmented images
            for n in range(1, aug_number + 1):  # Augmented images
                aug_img = augment_transforms(img)  # Apply transformations
                aug_img = aug_img.convert("RGB")  # Ensure augmented images are in RGB mode
                new_name = f"{os.path.splitext(image_name)[0]}_aug{n}.jpg"
                aug_img.save(os.path.join(augmented_dir, c, new_name))



aug_dataset = datasets.ImageFolder(augmented_dir, transform=transforms.ToTensor())

train_loader = DataLoader(aug_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

#Loading the model into ResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

model = models.resnet18(pretrained=True)

num_features = model.fc.in_features  
model.fc = nn.Linear(num_features, 2) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Try to run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        #Put inputs through a model, give outputs
        outputs = model(inputs) #Put the value through the model
        loss = criterion(outputs, labels) #Compute how accurate it was

        #optimization
        optimizer.zero_grad() # Reset gradient from last time
        loss.backward() #Find gradient of loss function
        optimizer.step() #Update parameters
        
        #Metrics (keeping track)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)  # Count total predictions
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

#Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #Put inputs through a model, give outputs
            outputs = model(inputs) #Put the value through the model
            loss = criterion(outputs, labels) #Compute how accurate it was

            #Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)  # Count total predictions
    
    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


#Training the data

num_epochs = 10
best_validation_accuracy = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    print(f"Training loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    #Validation
    validation_loss, validation_accuracy = validate(model, val_loader, criterion, device)
    print(f"Validation loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), "best_resnet_model.pth")
        print("Saved best model!")

print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")

