import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_log = []  # Global log

def plot_metrics(log_df, model_name, fold):
    plot_dir = f'Results/final_summary/{model_name}/plots/fold_{fold}'
    os.makedirs(plot_dir, exist_ok=True)

    for metric in ['Accuracy', 'Balanced_Accuracy']:
        plt.figure()
        plt.plot(log_df['Epoch'], log_df[f'Train_{metric}'], label=f'Train {metric}')
        plt.plot(log_df['Epoch'], log_df[f'Val_{metric}'], label=f'Val {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{model_name} - Fold {fold} - {metric}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{plot_dir}/{metric}.png')
        plt.close()

    for metric in ['Loss']:
        plt.figure()
        plt.plot(log_df['Epoch'], log_df[f'Train_{metric}'], label=f'Train {metric}')
        plt.plot(log_df['Epoch'], log_df[f'Val_{metric}'], label=f'Val {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{model_name} - Fold {fold} - {metric}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{plot_dir}/{metric}.png')
        plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, fold, split):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'{model_name} - Fold {fold} - {split} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_dir = f'Results/final_summary/{model_name}/confusion_matrices/fold_{fold}'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/{split}_conf_matrix.png')
    plt.close()

def train(model, train_loader, val_loader, config, model_name, fold):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    num_epochs = config['num_epochs']

    best_val_balanced_accuracy = -1  # To track the best model based on validation balanced accuracy
    best_model_wts = None  # To store the best model weights

    patience = 10  # Early stopping patience
    counter = 0  # Counter to track no improvement in validation loss

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass without autocast
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_accuracy = (torch.tensor(train_preds) == torch.tensor(train_labels)).float().mean().item()
        train_balanced_accuracy = balanced_accuracy_score(train_labels, train_preds)

        # VALIDATION
        model.eval()
        val_preds = []
        val_labels = []
        val_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = (torch.tensor(val_preds) == torch.tensor(val_labels)).float().mean().item()
        val_balanced_accuracy = balanced_accuracy_score(val_labels, val_preds)

        print(f"[Epoch {epoch+1}] Train Loss: {sum(train_losses)/len(train_losses):.4f} Train Acc: {train_accuracy:.4f} Train Bal Acc: {train_balanced_accuracy:.4f}")
        print(f"[Epoch {epoch+1}] Val Loss: {sum(val_losses)/len(val_losses):.4f} Val Acc: {val_accuracy:.4f} Val Bal Acc: {val_balanced_accuracy:.4f}")
        print("---------------------------------------------------")

        results_log.append({
            'Model': model_name,
            'Fold': fold,
            'Epoch': epoch + 1,
            'Train_Loss': sum(train_losses)/len(train_losses),
            'Train_Accuracy': train_accuracy,
            'Train_Balanced_Accuracy': train_balanced_accuracy,
            'Val_Loss': sum(val_losses)/len(val_losses),
            'Val_Accuracy': val_accuracy,
            'Val_Balanced_Accuracy': val_balanced_accuracy
        })

        # Update best model based on validation balanced accuracy
        if val_balanced_accuracy > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_balanced_accuracy
            best_model_wts = model.state_dict()
            print(f"New best model found (Val Balanced Acc: {best_val_balanced_accuracy:.4f})")

        # Early stopping check (based on validation loss improvement)
        avg_val_loss = sum(val_losses) / len(val_losses)

        if epoch == 0:
            best_val_loss = avg_val_loss
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0  # Reset the counter if validation loss improves
                print(f"Validation loss improved to {best_val_loss:.4f}")
            else:
                counter += 1
                print(f"No improvement in val loss. Counter: {counter}")

        # Early stopping if no improvement in `patience` epochs
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Plot confusion matrices for train and validation
    plot_confusion_matrix(train_labels, train_preds, model_name, fold, 'train')
    plot_confusion_matrix(val_labels, val_preds, model_name, fold, 'val')

    # Plot metrics
    log_df = pd.DataFrame([r for r in results_log if r['Model'] == model_name and r['Fold'] == fold])
    plot_metrics(log_df, model_name, fold)

    # Saving the best model checkpoint
    checkpoint_dir = f"Results/final_summary/{model_name}/checkpoints/fold_{fold}"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

    save_path = os.path.join(checkpoint_dir, f"model_fold_{fold}_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model checkpoint at {save_path}")
 