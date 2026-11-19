"""Fashion-MNIST MLP: data preparation, training, validation, and test evaluation.

Verified environment: Python 3.9.6, macOS arm64, CPU, seed 42.
Packages: torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, matplotlib 3.9.4,
          seaborn 0.13.2, scikit-learn 1.6.1.

Run from the project folder:
    python3 -m venv .venv                  # Only if .venv does not exist.
    .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 numpy==2.0.2 matplotlib==3.9.4 seaborn==0.13.2 scikit-learn==1.6.1
    .venv/bin/python train_model.py        # Train, then evaluate best checkpoint.
    .venv/bin/python train_model.py --eval-only


"""


# 1. Import libraries
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import matplotlib

# Save plots to files without requiring a graphical window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 2. Class names and configuration
# The position of each name matches its numeric label (0 through 9).
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
MEAN = 0.2860
STD = 0.3530
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SAMPLE_GRID_PATH = PROJECT_DIR / "fashion_mnist_samples.png"


def build_model(device):
    return nn.Sequential(
        # Keep the batch dimension; flatten each [1, 28, 28] image to 784 values.
        nn.Flatten(start_dim=1),
        nn.Linear(784, 256),  # First hidden layer.
        nn.ReLU(),           # Replace negative values with zero.
        nn.Linear(256, 128),  # Second hidden layer.
        nn.ReLU(),
        nn.Linear(128, 10),   # Raw class scores (logits), without softmax.
    ).to(device)


# Final evaluation helper 
def evaluate_checkpoint(model, test_loader, device):
    checkpoint_path = PROJECT_DIR / "model_proj1.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}. Train first.")

    # Always replace in-memory weights with the selected validation checkpoint.
    # The existing baseline checkpoint was selected at epoch 8. Running
    # may select a different epoch; we never assume the final epoch is best.
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True,
    ))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    count = 0
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            count += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())

    # Precision: how often predictions of a class are correct.
    # Recall: how many actual examples of that class are found.
    # F1 balances precision and recall; support is the true example count.
    # Macro averages give each class equal weight; weighted averages use support.
    report = classification_report(
        true_labels, predicted_labels, labels=list(range(10)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        true_labels, predicted_labels, labels=list(range(10)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / count
    metrics = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "test_examples": count, "test_loss": loss_sum / count,
        "test_accuracy": accuracy, "classification_report": report,
    }
    (PROJECT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )
    with (PROJECT_DIR / "test_classification_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "precision", "recall", "f1-score", "support",
        ])
        writer.writeheader()
        for name in CLASS_NAMES + ["macro avg", "weighted avg"]:
            writer.writerow({"class": name, **report[name]})
    summary = (
        f"Reloaded checkpoint: {checkpoint_path.name}\n"
        f"Test examples: {count:,}\n"
        f"Test loss: {metrics['test_loss']:.6f}\n"
        f"Test accuracy: {accuracy:.2%}\n\n{report_text}"
    )
    (PROJECT_DIR / "test_results.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Number of examples"}, ax=ax)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title="Fashion-MNIST test confusion matrix — selected checkpoint")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.savefig(PROJECT_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved test metrics, classification report, text results, and confusion matrix.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate model_proj1.pth without training or replacing training outputs.")
    args = parser.parse_args()
    
    # 3. Reproducibility and device
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    # Deliberately use CPU even when MPS is available. DataLoaders yield CPU
    # tensors; this device will also be used in the later model stage.
    device = torch.device("cpu")
    print(f"Random seed: {SEED}")
    print(f"Device: {device}")

    
    # 4. Transform and load the official datasets
    # ToTensor converts grayscale pixels from [0, 255] to floats in [0, 1]
    # with shape [1, 28, 28]. Normalize then applies (pixel - mean) / std.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    # Return before creating training data, sample plots, optimizers, or logs.
    if args.eval_only:
        if not (PROJECT_DIR / "model_proj1.pth").is_file():
            parser.error("--eval-only requires an existing model_proj1.pth")
        test_dataset = datasets.FashionMNIST(
            root=DATA_DIR, train=False, download=True, transform=transform,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        evaluate_checkpoint(build_model(device), test_loader, device)
        return
    original_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform,
    )

    # 5. Seeded training/validation split
    # Only the original training set is split. The official test set stays
    # separate, so it can be reserved for final evaluation later.
    train_dataset, validation_dataset = random_split(
        original_train_dataset,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    # 6. DataLoaders
    # A separate generator makes training shuffling repeatable across runs.
    # drop_last=False (the default) keeps the smaller final batch.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # 7. Check dataset sizes and one training batch
    print(f"Original training examples: {len(original_train_dataset):,}")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")
    print(f"Test examples: {len(test_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Batches per split: train={len(train_loader)}, "
        f"validation={len(validation_loader)}, test={len(test_loader)}"
    )
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"Batch device: {images.device}")

    # 8. Save a labeled sample grid (one training example per class)
    # Select from the training subset only, without advancing its DataLoader.
    samples = {}
    for image, label in train_dataset:
        if label not in samples:
            samples[label] = image
        if len(samples) == len(CLASS_NAMES):
            break

    fig, axes = plt.subplots(2, 5, figsize=(12, 6.5))
    for label, ax in enumerate(axes.flat):
        # Undo normalization for display; the dataset remains normalized.
        display_image = (samples[label] * STD + MEAN).clamp(0, 1)
        ax.imshow(display_image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}: {CLASS_NAMES[label]}")
        ax.axis("off")
    fig.suptitle("Fashion-MNIST: training samples", fontsize=16)
    # Leave enough space between rows for the second row's class labels.
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.86,
                        wspace=0.15, hspace=0.35)
    fig.savefig(SAMPLE_GRID_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved sample grid: {SAMPLE_GRID_PATH}")

    # 9. Define the MLP
    model = build_model(device)
    print("\nModel (our design choice):")
    print(model)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")

    # 10. Verify one forward pass without training
    # Reuse the normalized batch from section 7. No gradients are recorded,
    # and no weights are updated. These untrained scores are not predictions
    # whose accuracy we should interpret yet.
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    expected_shape = (images.shape[0], len(CLASS_NAMES))
    assert tuple(outputs.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(outputs.shape)}"
    )
    assert torch.isfinite(outputs).all().item(), "Output contains NaN or infinity"
    print(f"Model output shape: {tuple(outputs.shape)}")
    print(f"Output device: {outputs.device}")
    print("Forward-pass check passed; starting training next.")

    # 11. Configure baseline training and CSV logs
    # CrossEntropyLoss accepts raw logits and integer labels directly.
    # Adam adjusts weights using their gradients. This baseline has no dropout
    # and no weight decay.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0,
    )
    batch_log_path = PROJECT_DIR / "batch_losses.csv"
    history_path = PROJECT_DIR / "training_history.csv"
    checkpoint_path = PROJECT_DIR / "model_proj1.pth"
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    global_step = 0
    print(f"\nTraining: {EPOCHS} epochs, Adam lr={LEARNING_RATE}, CPU")

    # Opening with 'w' starts fresh logs each run. Accuracy in the CSV is a
    # fraction from 0 to 1; console output displays it as a percentage.
    with batch_log_path.open("w", newline="") as batch_file, \
            history_path.open("w", newline="") as history_file:
        batch_writer = csv.DictWriter(batch_file, fieldnames=[
            "epoch", "batch_number", "global_step", "batch_size", "loss",
        ])
        history_writer = csv.DictWriter(history_file, fieldnames=[
            "epoch", "train_loss", "train_accuracy", "validation_loss",
            "validation_accuracy",
        ])
        batch_writer.writeheader()
        history_writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            # 12. Train for one epoch (one pass through the training subset)
            
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            for batch_number, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()       # Clear gradients from the last batch.
                outputs = model(images)     # Calculate raw class scores.
                loss = criterion(outputs, labels)
                loss.backward()             # Calculate how weights affect loss.
                optimizer.step()            # Adjust weights to reduce loss.

                count = labels.size(0)
                loss_value = loss.item()
                # The loss is a batch mean. Multiply by its actual size so the
                # smaller final batch receives the correct weight in the epoch.
                train_loss_sum += loss_value * count
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_count += count
                global_step += 1
                batch_writer.writerow({
                    "epoch": epoch, "batch_number": batch_number,
                    "global_step": global_step, "batch_size": count,
                    "loss": loss_value,
                })

            # 13. Validate without gradients or weight updates
            
            model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0
            validation_count = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    count = labels.size(0)
                    validation_loss_sum += loss.item() * count
                    validation_correct += (
                        outputs.argmax(dim=1) == labels
                    ).sum().item()
                    validation_count += count

            # Training metrics summarize batches while weights were changing;
            # validation metrics use the fixed weights at the end of the epoch.
            row = {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "train_accuracy": train_correct / train_count,
                "validation_loss": validation_loss_sum / validation_count,
                "validation_accuracy": validation_correct / validation_count,
            }
            history.append(row)
            history_writer.writerow(row)
            batch_file.flush()
            history_file.flush()

            
            # 14. Keep the weights with the lowest validation loss
            
            improved = row["validation_loss"] < best_validation_loss
            if improved:
                best_validation_loss = row["validation_loss"]
                best_epoch = epoch
                # Save only the state_dict, using the requested filename.
                torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Epoch {epoch:02d}/{EPOCHS}: "
                f"train loss={row['train_loss']:.4f}, "
                f"accuracy={row['train_accuracy']:.2%} | "
                f"validation loss={row['validation_loss']:.4f}, "
                f"accuracy={row['validation_accuracy']:.2%}"
                + (" [saved best weights]" if improved else "")
            )

    # 15. Save learning curves and report the selected epoch
    for metric, ylabel, filename in [
        ("loss", "Cross-entropy loss", "training_validation_loss.png"),
        ("accuracy", "Accuracy (%)", "training_validation_accuracy.png"),
    ]:
        scale = 100 if metric == "accuracy" else 1
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for split, label in [("train", "Training"), ("validation", "Validation")]:
            ax.plot(
                [row["epoch"] for row in history],
                [row[f"{split}_{metric}"] * scale for row in history],
                marker="o", label=label,
            )
        ax.axvline(best_epoch, color="gray", linestyle="--",
                   label=f"Selected epoch: {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=ylabel,
               title=f"Fashion-MNIST baseline: {metric}")
        ax.set_xticks(range(1, EPOCHS + 1))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(PROJECT_DIR / filename, dpi=150)
        plt.close(fig)

    print(f"\nSelected epoch: {best_epoch} (validation loss={best_validation_loss:.6f})")
    print(f"Saved weights: {checkpoint_path}; exists={checkpoint_path.is_file()}")
    print(f"Recorded {global_step:,} batch losses and {len(history)} epoch summaries.")
    print("Saved training/validation loss and accuracy plots.")

    
    # 16. Evaluate the selected checkpoint, not the final in-memory weights
    evaluate_checkpoint(model, test_loader, device)


if __name__ == "__main__":
    main()

# References and supporting documentation

# [1] H. Xiao, K. Rasul, and R. Vollgraf,
#     "Fashion-MNIST: A novel image dataset for benchmarking machine learning
#     algorithms," arXiv, 2017.
#     https://arxiv.org/abs/1708.07747
#
# [2] PyTorch Contributors, "FashionMNIST,"
#     Torchvision 0.23 documentation, n.d.
#     https://docs.pytorch.org/vision/0.23/generated/torchvision.datasets.FashionMNIST.html
#
# [3] PyTorch Contributors, "ToTensor,"
#     Torchvision 0.23 documentation, n.d.
#     https://docs.pytorch.org/vision/0.23/generated/torchvision.transforms.ToTensor.html
#
# [4] PyTorch Contributors, "Normalize,"
#     Torchvision 0.23 documentation, n.d.
#     https://docs.pytorch.org/vision/0.23/generated/torchvision.transforms.Normalize.html
#
# [5] PyTorch Contributors, "torch.utils.data,"
#     PyTorch 2.8 documentation, n.d.
#     Covers DataLoader and random_split.
#     https://docs.pytorch.org/docs/2.8/data.html
#
# [6] PyTorch Contributors, "Build the neural network,"
#     PyTorch tutorials, n.d.
#     Covers neural network construction using layers and activation functions.
#     https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#
# [7] PyTorch Contributors, "Optimizing model parameters,"
#     PyTorch tutorials, n.d.
#     Covers training, backpropagation, parameter updates, and evaluation.
#     https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
#
# [8] PyTorch Contributors, "CrossEntropyLoss,"
#     PyTorch 2.8 documentation, n.d.
#     https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html
#
# [9] PyTorch Contributors, "Adam,"
#     PyTorch 2.8 documentation, n.d.
#     https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
#
# [10] PyTorch Contributors, "Save and load the model,"
#      PyTorch tutorials, n.d.
#      Covers saving and loading model weights using state_dict.
#      https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
#
# [11] PyTorch Contributors, "Reproducibility,"
#      PyTorch 2.8 documentation, n.d.
#      Covers random seeds and deterministic execution.
#      https://docs.pytorch.org/docs/2.8/notes/randomness.html
#
# [12] Scikit-learn Developers, "classification_report,"
#      Scikit-learn 1.6 documentation, n.d.
#      Covers precision, recall, F1-score, and support.
#      https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.classification_report.html
#
# [13] Scikit-learn Developers, "confusion_matrix,"
#      Scikit-learn 1.6 documentation, n.d.
#      https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.confusion_matrix.html
#
# [14] Matplotlib Development Team, "Pyplot tutorial,"
#      Matplotlib documentation, n.d.
#      Supports plotting training and validation loss and accuracy.
#      https://matplotlib.org/stable/tutorials/pyplot.html
#
# [15] Seaborn Developers, "seaborn.heatmap,"
#      Seaborn documentation, n.d.
#      Supports visualizing the confusion matrix.
#      https://seaborn.pydata.org/generated/seaborn.heatmap.html
#
# [16] Python Software Foundation, "csv—CSV file reading and writing,"
#      Python 3.9 documentation, n.d.
#      Supports exporting batch losses and training history.
#      https://docs.python.org/3.9/library/csv.html
