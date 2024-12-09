import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MihiranNet(nn.Module):
        def __init__(self, num_classes=17):
            super(MihiranNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                #nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            
            self.fc = nn.Sequential(
                nn.Linear(36864, 4096),
                nn.ReLU())
            self.fc1 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(4096, num_classes)
                #nn.Softmax()
                )
           

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = torch.flatten(out, 1)
            #print("After layer2 shape:", out.shape)
            out = self.fc(out)
            out = self.fc1(out)
            #out = self.fc2(out)
            return out

# Load the trained model
model = MihiranNet(num_classes=17)
model.load_state_dict(torch.load('mihirannet.pth'))
model.eval()


# Define the test data transformations
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 32

data_dir = "/workspace/pattern/main/data/Jute_Pest_Dataset"
# Load the test dataset
test_dir = os.path.join(data_dir, 'test')
test_dataset = datasets.ImageFolder(test_dir, test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Calculate and print the accuracy
accuracy = calculate_accuracy(test_loader, model)
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# Function for inference
def inference(image_path, model):
    image = Image.open(image_path)
    image = test_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Test the model on a sample image         
import matplotlib.pyplot as plt

# Function to show images with true and predicted labels
def show_images_with_labels(loader, model, num_images=5):
    model.eval()
    images_shown = 0
    fig = plt.figure(figsize=(15, 15))
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                ax = fig.add_subplot(num_images // 5 + 1, 5, images_shown + 1, xticks=[], yticks=[])
                img = images[i].permute(1, 2, 0).numpy()
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f'True: {labels[i].item()} Pred: {predicted[i].item()}')
                images_shown += 1
            
            if images_shown >= num_images:
                break

    plt.show()

# Show some images with their true and predicted labels
show_images_with_labels(test_loader, model, num_images=40)


# Function to calculate class-wise accuracy
def calculate_class_wise_accuracy(loader, model, num_classes):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of class {i}: N/A (no samples)')

# Calculate and print the class-wise accuracy
calculate_class_wise_accuracy(test_loader, model, num_classes=17)
