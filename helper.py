import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)


def load_image(train_path: str, test_path: str):
    """Load and resize the training and test image datasets."""
    image_transforms = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ]
    )

    return (
        datasets.ImageFolder(train_path, transform=image_transforms),
        datasets.ImageFolder(test_path, transform=image_transforms),
    )


def sampling(train_path: str, test_path: str):
    """Create randomized samplers for the training and test datasets."""
    train_set, test_set = load_image(
        train_path=train_path,
        test_path=test_path,
    )

    train_indices = np.random.permutation(len(train_set))
    test_indices = np.random.permutation(len(test_set))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    return train_set, test_set, train_sampler, test_sampler


def image_processing(
    train_path: str,
    test_path: str,
    batch_train: int,
    batch_test: int,
):
    """Prepare shuffled data loaders for training and evaluation."""
    train_set, test_set, train_sampler, test_sampler = sampling(
        train_path=train_path,
        test_path=test_path,
    )

    return (
        torch.utils.data.DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_train
        ),
        torch.utils.data.DataLoader(
            test_set, sampler=test_sampler, batch_size=batch_test
        ),
    )


def cnn_model(model):
    """Attach a binary classifier head and prepare the model for training."""
    classifier = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 2),
        torch.nn.LogSoftmax(dim=1),
    )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc")

    model.fc = classifier
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
    error = torch.nn.CrossEntropyLoss()

    model.to(device)

    return model, error, optimizer


def model_training(
    data_train,
    num_epochs: int,
    model,
    loss_func,
    optimizer,
):
    """Train the model and return it."""
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()

        for images, labels in data_train:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {train_loss / len(data_train.dataset):.4f}"
        )

    return model



def model_testing(data_test, model):
    """Evaluate the model and return its confusion matrix and report."""
    actual, predicted = [], []
    model.eval()

    with torch.no_grad():
        for images, labels in data_test:
            images, labels = images.to(device), labels.to(device)
            actual.extend(labels.tolist())
            outputs = model(images)
            predicted.extend(outputs.argmax(dim=1).tolist())

    return (
        confusion_matrix(actual, predicted),
        classification_report(actual, predicted),
    )


def adversarial_training(
    data_train,
    num_epochs: int,
    model,
    loss_func,
    optimizer,
    adv_method,
):
    """Train the model on adversarially perturbed images."""
    atk = adv_method(model)

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()

        for images, labels in data_train:
            images, labels = images.to(device), labels.to(device)
            images = atk(images, labels)
            outputs = model(images)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {train_loss / len(data_train.dataset):.4f}"
        )

    return model
