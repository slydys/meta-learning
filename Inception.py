import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader, Dataset


device = torch.device("mps")


BASELINE_MODELS = {
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'AB': AdaBoostClassifier(),
    'GBM': GradientBoostingClassifier(),
    'NB': GaussianNB(),
    'LR': LogisticRegression(),
    'SVM': SVC(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP': MLPClassifier(),
    'GNB': GaussianNB(),
    'XGB': XGBClassifier(),
    'LGBM': LGBMClassifier(),
    'CB': CatBoostClassifier(),
    'ET': ExtraTreesClassifier(),
    'RR': RidgeClassifier(),
}


class InceptionV3TransferLearning(nn.Module):
    def __init__(self, num_classes=17):
        super(InceptionV3TransferLearning, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Sequential(
            nn.Linear(self.inception.fc.in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x, _ = self.inception(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=(299, 299)):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label


def extract_samples_and_labels_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    unique_colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)
    label_map = {tuple(color): i for i, color in enumerate(unique_colors)}
    samples, labels = [], []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = tuple(image[i, j])
            samples.append(image[i, j])
            labels.append(label_map[pixel])
    return np.array(samples), np.array(labels)


def extract_inception_features(model, image_paths, batch_size=32):
    model.eval()
    model_features = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                image = transforms.Resize((299, 299))(image)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
                images.append(image)
            images = torch.stack(images).to(device)
            features = model(images)
            model_features.append(features.cpu().numpy())
    return np.vstack(model_features)


def recommend_algorithm(features, labels):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    best_model, best_accuracy, best_algorithm = None, 0, None
    for algorithm_name, model in BASELINE_MODELS.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f'{algorithm_name} accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_algorithm = algorithm_name
    return best_algorithm, best_model


def train_model(model, train_loader, criterion, optimizer, num_epochs=80):
    model.train()
    for epoch in range(num_epochs):
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    print("Training complete")


def main(dataset_folder):
    image_paths, labels = [], []
    for dataset_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset_name)
        if os.path.isdir(dataset_path):
            for image_name in os.listdir(dataset_path):
                image_path = os.path.join(dataset_path, image_name)
                if image_name.endswith('.png') or image_name.endswith('.jpg'):
                    samples, sample_labels = extract_samples_and_labels_from_image(image_path)
                    image_paths.extend([image_path] * len(sample_labels))
                    labels.extend(sample_labels)


    dataset = ImageDataset(image_paths=image_paths, labels=labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


    num_classes = len(set(labels))
    model = InceptionV3TransferLearning(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

    train_model(model, train_loader, criterion, optimizer, num_epochs=80)

    features = extract_inception_features(model, image_paths)
    best_algorithm, best_model = recommend_algorithm(features, labels)
    print(f"Best algorithm: {best_algorithm}")


dataset_folder = 't-SNE_pictures'
main(dataset_folder)
