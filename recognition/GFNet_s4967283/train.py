"""
train.py

Trains a model on the ADNI AD vs NC task, validates it after every epoch,
saves the best weights, evaluates on the test set, and writes out everything
the report needs.

Usage examples:

    # Short smoke test to check the loop works at all
    python train.py --model baseline --epochs 3

    # Full baseline run
    python train.py --model baseline --epochs 30

    # Point at a local subset instead of the Rangpur path
    python train.py --model baseline --epochs 2 --data-root /path/to/adni_subset/AD_NC

What it writes into the output directory:

    best_<model>.pt            the weights from the best validation epoch
    history_<model>.json       per-epoch losses and metrics
    curves_<model>.png         training/validation loss and accuracy plots
    test_predictions_<model>.npz   per-image probabilities, labels, patient IDs
    summary_<model>.json       final test metrics plus peak VRAM, timing, params

That .npz file matters more than it looks. The whole dilemma investigation -
calibration, expected calibration error, the reject-option threshold - needs
the per-image probabilities, not just the final accuracy. Saving them here
means I can do all of that analysis later without retraining anything.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib

# Pick a non-interactive plotting backend before importing pyplot. On a compute
# node there is no display, and without this matplotlib tries to find one and
# crashes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataloaders, DEFAULT_ADNI_ROOT, LABEL_TO_CLASS, IMAGE_SIZE
from modules import build_model, count_trainable_parameters

def set_all_seeds(seed):
    """
    Fix every random source I can so a rerun gives the same result.

    The assignment asks about reproducibility, and without this the weight
    initialisation, the augmentation choices and the batch order all change
    between runs. Then I could not tell whether a difference between the
    baseline and GFNet was real or just luck.

    Note the honest limitation: cudnn.benchmark picks the fastest convolution
    algorithm for the hardware, and that choice is not guaranteed identical
    across runs. I leave benchmark on because the speed is worth more than
    bit-exact reproducibility here, so results are repeatable to within a small
    margin rather than exactly. Setting deterministic=True instead would fix
    that but slows training noticeably.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pick_device():
    """Use the GPU if there is one, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: %s" % torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, running on CPU. This will be slow.")
    return device

def run_one_epoch(model, loader, loss_function, optimiser, device, is_training):
    """
    Run a single pass over one dataloader.

    Used for both training and evaluation. When is_training is False it skips
    the backward pass and wraps everything in no_grad, which saves memory and
    time.

    Returns (average_loss, accuracy).
    """
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # torch.enable_grad / no_grad as a context, chosen by the flag.
    gradient_context = torch.enable_grad() if is_training else torch.no_grad()

    with gradient_context:
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimiser.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_function(logits, labels)

            if is_training:
                loss.backward()
                optimiser.step()

            # Multiply by batch size so the average at the end is weighted
            # correctly even if the last batch is smaller.
            running_loss += loss.item() * labels.size(0)

            predicted_classes = logits.argmax(dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy


def evaluate_and_collect_probabilities(model, loader, device):
    """
    Run the model over a dataloader and keep every prediction.

    Returns three arrays: predicted probabilities for each class, the true
    labels, and the patient ID each image came from.

    The patient IDs are the reason this function exists separately from
    run_one_epoch. For the failure analysis I need to know which patient a
    wrong prediction belongs to - several mistakes from one patient is a very
    different story from mistakes scattered across many patients.
    """
    model.eval()

    all_probabilities = []
    all_labels = []
    all_patient_ids = []

    with torch.no_grad():
        for images, labels, patient_ids in loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            # Softmax turns the raw logits into probabilities that sum to 1.
            probabilities = torch.softmax(logits, dim=1)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())
            all_patient_ids.extend(list(patient_ids))

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_labels),
        np.array(all_patient_ids),
    )


def compute_classification_metrics(probabilities, true_labels):
    """
    Work out accuracy and the per-class precision, recall and F1 by hand.

    I compute these directly rather than calling sklearn so I can be sure I
    know what each number means - the assignment asks me to defend every metric
    in the report, and "sklearn said so" is not a defence.

    AD is the positive class (label 1), so recall on AD is the fraction of
    actual Alzheimer's cases the model caught. That is the number that matters
    clinically: a missed AD case is the expensive mistake.
    """
    predicted_labels = probabilities.argmax(axis=1)

    accuracy = float((predicted_labels == true_labels).mean())

    metrics = {"accuracy": accuracy, "per_class": {}}

    for class_index in (0, 1):
        class_name = LABEL_TO_CLASS[class_index]

        true_positives = int(((predicted_labels == class_index) & (true_labels == class_index)).sum())
        false_positives = int(((predicted_labels == class_index) & (true_labels != class_index)).sum())
        false_negatives = int(((predicted_labels != class_index) & (true_labels == class_index)).sum())

        # Guard the divisions - if the model never predicts a class, precision
        # would be 0/0.
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["per_class"][class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    # Macro F1 is the unweighted mean of the two per-class F1 scores. Unweighted
    # means it treats both classes as equally important regardless of how many
    # examples each has, which is what I want for a screening task.
    metrics["macro_f1"] = float(
        np.mean([metrics["per_class"][c]["f1"] for c in metrics["per_class"]])
    )

    return metrics


def measure_inference_latency(model, loader, device, num_batches=20):
    """
    Time how long inference takes, in images per second.

    This is one of the three numbers the resource profiling table needs. I warm
    up first and synchronise the GPU around the timed section, because CUDA
    calls are asynchronous - without synchronising, the timer would stop before
    the GPU had finished and report a figure that is far too good.
    """
    model.eval()

    batches = []
    for i, (images, _, _) in enumerate(loader):
        if i >= num_batches:
            break
        batches.append(images.to(device))

    if not batches:
        return None

    with torch.no_grad():
        # Warm-up pass. The first few calls include one-off setup costs.
        for images in batches[:3]:
            model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        total_images = 0
        for images in batches:
            model(images)
            total_images += images.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

    return {
        "images_per_second": total_images / elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / total_images,
    }


def plot_training_curves(history, model_name, output_directory):
    """
    Plot loss and accuracy over the epochs, training against validation.

    The assignment specifically asks for losses and metrics plotted during
    training. These curves are also the fastest way to see overfitting: if
    training loss keeps falling while validation loss turns upward, the model
    has started memorising rather than learning.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    loss_axis.plot(epochs, history["val_loss"], label="validation", marker="s", markersize=3)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("cross-entropy loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, history["train_accuracy"], label="train", marker="o", markersize=3)
    accuracy_axis.plot(epochs, history["val_accuracy"], label="validation", marker="s", markersize=3)
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)

    figure.suptitle("%s on ADNI (AD vs NC)" % model_name)
    figure.tight_layout()

    output_path = os.path.join(output_directory, "curves_%s.png" % model_name)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print("Saved training curves to %s" % output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model on ADNI AD vs NC.")

    parser.add_argument("--model", default="baseline", choices=["baseline", "gfnet"],
                        help="Which model to train.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="How many epochs to train for.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Images per batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Starting learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularisation strength.")
    parser.add_argument("--data-root", default=DEFAULT_ADNI_ROOT,
                        help="Path to the AD_NC folder.")
    parser.add_argument("--output-dir", default="outputs",
                        help="Where to write weights, plots and metrics.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader worker processes. Rangpur allows 8 CPUs max.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, also used for the train/val split.")
    parser.add_argument("--augmentation", default="normal", choices=["normal", "strong"],
                        help="Training augmentation strength.")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override the model's default dropout rate.")
    parser.add_argument("--slice-context", type=int, default=1, choices=[1, 3],
                        help="1 feeds one slice as one channel. 3 feeds the "
                             "slice plus its two neighbours as three channels, "
                             "which gives the model depth context a single "
                             "slice cannot carry.")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="GFNet patch size. 16 gives a 14x14 token grid "
                             "and 112 frequency components. 8 gives 28x28 and "
                             "420, at about four times the compute. Ignored by "
                             "the baseline.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    set_all_seeds(arguments.seed)
    device = pick_device()

    os.makedirs(arguments.output_dir, exist_ok=True)

    # ---- Data ----
    print()
    print("Loading data from %s" % arguments.data_root)
    train_loader, validation_loader, test_loader = get_dataloaders(
        adni_root=arguments.data_root,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        augmentation_strength=arguments.augmentation,
        slice_context=arguments.slice_context,
    )
    print("  train batches: %d" % len(train_loader))
    print("  validation batches: %d" % len(validation_loader))
    print("  test batches: %d" % len(test_loader))
    
    # ---- Model ----
    # in_channels must match what the loader produces, or the first layer
    # will not accept the batch.
    model = build_model(
        arguments.model,
        image_size=IMAGE_SIZE,
        in_channels=arguments.slice_context,
        dropout_rate=arguments.dropout,
        patch_size=arguments.patch_size,
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    print()
    print("Model: %s with %d trainable parameters" % (arguments.model, parameter_count))

    # CrossEntropyLoss expects raw logits and applies the softmax itself, which
    # is why the model does not apply one.
    loss_function = nn.CrossEntropyLoss()

    # Adam with weight decay. Adam adapts the step size per parameter, which
    # means it usually trains sensibly without much learning rate tuning -
    # worth a lot when I have one week.
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    # Cosine annealing winds the learning rate down smoothly to near zero by
    # the final epoch. Big steps early to make progress, small steps late to
    # settle into a minimum instead of bouncing around it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=arguments.epochs
    )

    # Reset the VRAM counter so the peak figure covers only this run.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- Training ----
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "epoch_seconds": [], "learning_rate": [],
    }

    best_validation_accuracy = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(arguments.output_dir, "best_%s.pt" % arguments.model)

    print()
    print("Training for %d epochs" % arguments.epochs)
    print("-" * 72)

    training_start_time = time.time()

    for epoch in range(1, arguments.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, loss_function, optimiser, device, is_training=True
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model, validation_loader, loss_function, optimiser, device, is_training=False
        )

        scheduler.step()
        epoch_seconds = time.time() - epoch_start_time

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        history["epoch_seconds"].append(epoch_seconds)
        history["learning_rate"].append(optimiser.param_groups[0]["lr"])

        # Keep the weights from the best validation epoch, not the last one.
        # Validation accuracy usually peaks before training ends, and the final
        # epoch is often slightly overfitted. Checkpointing on validation also
        # means the test set stays genuinely untouched until the very end.
        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        print("epoch %3d/%d  train loss %.4f acc %.4f  |  val loss %.4f acc %.4f  |  %.1fs%s"
              % (epoch, arguments.epochs, train_loss, train_accuracy,
                 validation_loss, validation_accuracy, epoch_seconds,
                 "  <- best" if is_best else ""))

    total_training_seconds = time.time() - training_start_time

    print("-" * 72)
    print("Training finished in %.1f minutes" % (total_training_seconds / 60.0))
    print("Best validation accuracy %.4f at epoch %d" % (best_validation_accuracy, best_epoch))

    # ---- Test evaluation ----
    # Reload the best checkpoint so the test numbers come from the best model
    # rather than whatever state the last epoch left behind.
    print()
    print("Loading best checkpoint and evaluating on the test set")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_probabilities, test_labels, test_patient_ids = evaluate_and_collect_probabilities(
        model, test_loader, device
    )
    test_metrics = compute_classification_metrics(test_probabilities, test_labels)

    print()
    print("TEST RESULTS")
    print("  accuracy: %.4f" % test_metrics["accuracy"])
    print("  macro F1: %.4f" % test_metrics["macro_f1"])
    for class_name, class_metrics in test_metrics["per_class"].items():
        print("  %s: precision %.4f  recall %.4f  f1 %.4f"
              % (class_name, class_metrics["precision"],
                 class_metrics["recall"], class_metrics["f1"]))

    # ---- Resource profiling ----
    latency = measure_inference_latency(model, test_loader, device)

    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print()
    print("RESOURCE PROFILE")
    print("  trainable parameters: %d" % parameter_count)
    if peak_vram_gb is not None:
        print("  peak GPU memory: %.2f GB" % peak_vram_gb)
    print("  total training time: %.1f minutes" % (total_training_seconds / 60.0))
    print("  mean seconds per epoch: %.1f" % float(np.mean(history["epoch_seconds"])))
    if latency:
        print("  inference: %.0f images/second (%.3f ms per image)"
              % (latency["images_per_second"], latency["milliseconds_per_image"]))

    # ---- Save everything ----
    history_path = os.path.join(arguments.output_dir, "history_%s.json" % arguments.model)
    with open(history_path, "w") as handle:
        json.dump(history, handle, indent=2)

    # Per-image probabilities, for the calibration and failure analysis later.
    predictions_path = os.path.join(
        arguments.output_dir, "test_predictions_%s.npz" % arguments.model
    )
    np.savez_compressed(
        predictions_path,
        probabilities=test_probabilities,
        labels=test_labels,
        patient_ids=test_patient_ids,
    )

    summary = {
        "model": arguments.model,
        "arguments": vars(arguments),
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "test_metrics": test_metrics,
        "resources": {
            "peak_vram_gb": peak_vram_gb,
            "total_training_seconds": total_training_seconds,
            "mean_epoch_seconds": float(np.mean(history["epoch_seconds"])),
            "inference": latency,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    summary_path = os.path.join(arguments.output_dir, "summary_%s.json" % arguments.model)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_training_curves(history, arguments.model, arguments.output_dir)

    print()
    print("Wrote:")
    print("  %s" % checkpoint_path)
    print("  %s" % history_path)
    print("  %s" % predictions_path)
    print("  %s" % summary_path)


if __name__ == "__main__":
    main()
