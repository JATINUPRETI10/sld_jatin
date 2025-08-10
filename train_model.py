import os #Used to work with file paths (check if model file exists, etc.).
import torch #main PyTorch library for dl
import torch.nn as nn #Neural network module from PyTorch
import torch.optim as optim #optimizer like adam
from torchvision import datasets, transforms #datasets and transforms for image processing
from torch.utils.data import DataLoader #DataLoader to load images in batches
from PIL import Image #open iamges for prediction


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")


DATA_DIR = "asl_alphabet_train"   # Folder containing asl dataset
BATCH_SIZE = 32 # Batch size for training
EPOCHS = 10
MODEL_PATH = "model.pth" # Path to save the trained model
CLASSES_FILE = "classes.txt"#A text file to store all ASL class names for later.




transform = transforms.Compose([
    transforms.Resize((64, 64)), #bring evry image to 64x64
    transforms.ToTensor(), # Converts images to PyTorch tensors ([0,1] range).
    transforms.Normalize([0.5], [0.5]) # Normalize images to [-1, 1] range.
])


train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform) #image folder automatically labels each folder as seperate class
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #dataloader split data in batches and shuffles them for each epoch 



classes = train_dataset.classes 
num_classes = len(classes)
with open(CLASSES_FILE, "w") as f:
    for c in classes:
        f.write(f"{c}\n")
print(f" Found {num_classes} classes: {classes}")


# cnn architecture 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),# Detects image patterns (edges, curves, shapes).
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),# Activation function to add non-linearity.
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)# Reduces image size while keeping important features.
        )
        self.fc = nn.Sequential(
            nn.Flatten(),#converts 3D features into 1D for fully connected layers.
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(),#Fully connected layers for classification.
            nn.Dropout(0.5),#Randomly drops 50% of neurons to prevent overfitting.
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = SimpleCNN(num_classes).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f" Loaded existing model from {MODEL_PATH}, skipping training.")
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(" Training started...\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f" Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")


def predict_image(image_path):
    """Predict ASL letter from an image."""
    model.eval() # Sets model to evaluation mode (no dropout, etc.).
    image = Image.open(image_path).convert("RGB") # Open image and convert to RGB
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension(mdoel expects batches)
    # gets index of highest probability class
    # convert index to actual letter




    with torch.no_grad():
        output = model(image)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = classes[predicted_idx]

    return predicted_label



# prediction
test_image_path = "C:/MINOR/asl_alphabet_test/P_Test.jpg"  # Change as needed
if os.path.exists(test_image_path):
    prediction = predict_image(test_image_path)
    print(f"🔍 Predicted ASL Letter: {prediction}")
else:
    print(f" Test image not found at {test_image_path}")
