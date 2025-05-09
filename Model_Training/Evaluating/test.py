import sys
import os
# Add the base directory to the system path to locate Model_Training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from Model_Training.Data.datasets.isic2020_dataset import ISIC2020Dataset
from Model_Training.Models.model import get_model
from Model_Training.Data.transforms.isic2020_transforms import get_augmentation
import json

# CUDA settings
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = '/work/c-2iia/hn977782/torch_cache'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = False

# Load the test samples from the provided JSON file
def load_test_data(test_json_path):
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    return test_data['testing']  # Assumed that test data is under 'testing' key

# Define the test function
def test(model, test_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    precision = precision_score(targets, preds, average='weighted')

    print(f"Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")
    return accuracy, f1, recall, precision, preds, targets

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, fold, split, fold_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'{model_name} - Fold {fold} - {split} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plot_dir = f'{fold_dir}/final_summary/{model_name}/confusion_matrices/fold_{fold}'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/fold_{fold}_{split}_conf_matrix.png')  # This 'split' will indicate 'test'
    plt.close()

# Define function to test across folds and models
def test_models(model_list, folds_dir, image_dir, config, test_json_path):
    results_log = []

    # Load test data from the JSON file
    test_data = load_test_data(test_json_path)
    print(f"Loaded {len(test_data)} test samples from {test_json_path}")

    for model_name in model_list:
        print(f"Testing Model: {model_name}")

        model_dir = os.path.join(folds_dir, 'final_summary', model_name)  # Path to the model folder
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')  # Path to the checkpoints folder

        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory for model {model_name} does not exist.")
            continue

        # Prepare the test dataset
        test_transforms = get_augmentation(source='torchvision', level=5, is_train=False)
        test_dataset = ISIC2020Dataset(
            json_path=test_json_path,
            split='testing',  # Use the provided test split
            images_dir=image_dir,
            transform=test_transforms,
            label_map={'benign': 0, 'malignant': 1}
        )

        # DataLoader for test data
        batch_size = 8
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

        # Loop over each fold and test the model checkpoints
        for fold_idx in range(1, 6):  # Iterate through the 5 folds
            print(f"Testing Fold {fold_idx} for model {model_name}...")

            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}', f'model_fold_{fold_idx}_best.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint for fold {fold_idx} does not exist at {checkpoint_path}")
                continue

            # Load the model and the checkpoint
            model = get_model(model_name=model_name, num_classes=config['num_classes'])

            # Handle multi-GPU if available
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)  # Wrap the model in DataParallel

            model.load_state_dict(torch.load(checkpoint_path))
            model = model.to(config['device'])

            # Test the model
            accuracy, f1, recall, precision, preds, targets = test(model, test_loader, config['device'])

            # Plot confusion matrix for the test set
            plot_confusion_matrix(targets, preds, model_name, fold_idx, 'test', folds_dir)

            # Log results
            results_log.append({
                'Model': model_name,
                'Fold': fold_idx,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Recall': recall,
                'Precision': precision
            })

            print(f"Fold {fold_idx} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Check if results_log has been populated
    if len(results_log) == 0:
        print("Warning: No results logged. Check the model testing process.")
    else:
        # Save results to CSV
        results_df = pd.DataFrame(results_log)
        print(f"Saving results to {folds_dir}/model_testing_results.csv")
        results_df.to_csv(f'{folds_dir}/model_testing_results.csv', index=False)

    print("Testing results saved to CSV")

# Define main function to start testing process
def main():
    folds_dir = '/work/c-2iia/hn977782/Thesis/Code/Results'  # the model directory
    image_dir = '/work/c-2iia/hn977782/Thesis/Code/Datasets/ISIC2020/Images'
    test_json_path = '/work/c-2iia/hn977782/Thesis/Code/Datasets/ISIC2020/Data_splitting/Random_patient_equal_split/train_test_split.json'  # Path to the provided test JSON file

    model_list = ["densenet121", "densenet1", "densenet169","densenet201","googlenet","mobilenet_v2"]
    
    config = {
        'image_size': 224,
        'batch_size': 8,
        'num_epochs': 60,
        'learning_rate': 0.00001,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'scheduler': 'StepLR',
        'step_size': 10,
        'gamma': 0.5,
        'num_classes': 2,
        'device': 'cuda'  # or 'cpu'
    }

    # Call the test models function
    test_models(model_list, folds_dir, image_dir, config, test_json_path)

if __name__ == "__main__":
    main()