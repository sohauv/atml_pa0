import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.models import resnet152, ResNet152_Weights


# -------------------------
# Configuration
# -------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
RESULTS_DIR = Path("task1_resnet/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Reproducibility
# -------------------------
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -------------------------
# Pretrained ResNet-152
# -------------------------
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

# Replace ImageNet classifier with CIFAR-10 classifier
model.fc = nn.Linear(2048, 10)

# Freeze entire network
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only classification head
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)


# -------------------------
# Verify trainable parameters
# -------------------------
trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

total_params = sum(
    param.numel()
    for param in model.parameters()
)

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")


# -------------------------
# CIFAR-10 preprocessing
# -------------------------
transform = weights.transforms()


# -------------------------
# Load CIFAR-10
# -------------------------
full_train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)


# -------------------------
# Train / validation split
# -------------------------
split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000],
    generator=split_generator,
)


# -------------------------
# DataLoaders
# -------------------------
train_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    generator=train_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# -------------------------
# Loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# -------------------------
# Training function
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# -------------------------
# Evaluation function
# -------------------------
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# -------------------------
# Training
# -------------------------
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


# -------------------------
# Final test evaluation
# -------------------------
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device,
)

print(
    f"Test Loss: {test_loss:.4f} | "
    f"Test Acc: {test_acc:.4f}"
)


# -------------------------
# Save model
# -------------------------
model_path = RESULTS_DIR / "resnet152_cifar10_baseline.pth"

torch.save(
    model.state_dict(),
    model_path,
)

print(f"Model saved to: {model_path}")


# -------------------------
# Save training history
# -------------------------
history_path = RESULTS_DIR / "baseline_history.csv"

with open(history_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ])

    for epoch in range(EPOCHS):
        writer.writerow([
            epoch + 1,
            history["train_loss"][epoch],
            history["train_acc"][epoch],
            history["val_loss"][epoch],
            history["val_acc"][epoch],
        ])

print(f"History saved to: {history_path}")


# -------------------------
# Save summary
# -------------------------
summary_path = RESULTS_DIR / "baseline_summary.csv"

with open(summary_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "final_train_loss",
        "final_train_acc",
        "final_val_loss",
        "final_val_acc",
        "test_loss",
        "test_acc",
    ])

    writer.writerow([
        history["train_loss"][-1],
        history["train_acc"][-1],
        history["val_loss"][-1],
        history["val_acc"][-1],
        test_loss,
        test_acc,
    ])

print(f"Summary saved to: {summary_path}")


# -------------------------
# Plot loss
# -------------------------
epochs_range = range(1, EPOCHS + 1)

plt.figure()

plt.plot(
    epochs_range,
    history["train_loss"],
    marker="o",
    label="Train Loss",
)

plt.plot(
    epochs_range,
    history["val_loss"],
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline ResNet-152 Loss")
plt.legend()
plt.grid()

loss_plot_path = RESULTS_DIR / "baseline_loss.png"

plt.savefig(
    loss_plot_path,
    bbox_inches="tight",
)

plt.close()

print(f"Loss plot saved to: {loss_plot_path}")


# -------------------------
# Plot accuracy
# -------------------------
plt.figure()

plt.plot(
    epochs_range,
    history["train_acc"],
    marker="o",
    label="Train Accuracy",
)

plt.plot(
    epochs_range,
    history["val_acc"],
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline ResNet-152 Accuracy")
plt.legend()
plt.grid()

accuracy_plot_path = RESULTS_DIR / "baseline_accuracy.png"

plt.savefig(
    accuracy_plot_path,
    bbox_inches="tight",
)

plt.close()

print(f"Accuracy plot saved to: {accuracy_plot_path}")