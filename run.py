
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
            out = out.reshape(out.size(0), -1)
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
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Example inference
image_path = 'path_to_single_test_image'
predicted_class = inference(image_path, model)
print(f'Predicted class for the image: {predicted_class}')           