import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

def load_data(csv_file):
    """
    Load the cleaned dataset.
    Args:
        csv_file (str): Path to the cleaned dataset CSV file.
    Returns:
        df (pd.DataFrame): Dataframe containing image names and labels.
    """
    df = pd.read_csv(csv_file)
    print(f"Dataset loaded: {csv_file}")
    return df

def split_train_test(df, test_size=0.2, patient_level_split=False, stratified=True, class_equalization=False):
    """
    Split the data into train and test sets based on the configuration.
    Args:
        df (pd.DataFrame): Dataframe containing image names and labels.
        test_size (float): Percentage of data to be used for testing.
        patient_level_split (bool): Whether to ensure no patient data leakage.
        stratified (bool): Whether to use stratified splitting to balance classes.
        class_equalization (bool): Whether to balance class distribution across train/test splits.
    Returns:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
    """
    if patient_level_split:
        # Ensure no patient data leakage between train and test sets
        patients = df['patient_id'].unique()
        train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=42)

        # Filter data based on patients for train/test split
        train_df = df[df['patient_id'].isin(train_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]
    else:
        # Standard random split, stratified by benign/malignant
        if stratified:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['benign_malignant'])
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # If class equalization is required, balance the classes
    if class_equalization:
        train_df = balance_classes(train_df)
        test_df = balance_classes(test_df)

    return train_df, test_df

def balance_classes(df):
    """
    Balance the classes in the dataframe by oversampling the minority class.
    Args:
        df (pd.DataFrame): Dataframe containing image names and labels.
    Returns:
        balanced_df (pd.DataFrame): Dataframe with balanced classes.
    """
    # Find the minimum class size
    min_class_size = df['benign_malignant'].value_counts().min()

    # Sample the dataframe to balance the classes
    df_benign = df[df['benign_malignant'] == 'benign'].sample(min_class_size, random_state=42)
    df_malignant = df[df['benign_malignant'] == 'malignant'].sample(min_class_size, random_state=42)

    # Combine the two balanced dataframes
    balanced_df = pd.concat([df_benign, df_malignant])

    return balanced_df

def save_split_to_json(train_df, test_df, split_folder_name):
    """
    Save the split data into a JSON file for train and test sets.
    Args:
        train_df (pd.DataFrame): Training set dataframe.
        test_df (pd.DataFrame): Testing set dataframe.
        split_folder_name (str): Path to the folder where JSON files will be saved.
    """
    # Convert the DataFrame to JSON format
    train_images = [{"image_name": row['image_name'], "label": row['benign_malignant']} for _, row in train_df.iterrows()]
    test_images = [{"image_name": row['image_name'], "label": row['benign_malignant']} for _, row in test_df.iterrows()]

    split_data = {
        "training": train_images,
        "testing": test_images,
        "statistics": {
            "total_images": len(train_df) + len(test_df),
            "training_images": len(train_df),
            "testing_images": len(test_df),
            "training_benign": len(train_df[train_df['benign_malignant'] == 'benign']),
            "training_malignant": len(train_df[train_df['benign_malignant'] == 'malignant']),
            "testing_benign": len(test_df[test_df['benign_malignant'] == 'benign']),
            "testing_malignant": len(test_df[test_df['benign_malignant'] == 'malignant']),
        }
    }

    # Save the split data to JSON
    os.makedirs(split_folder_name, exist_ok=True)
    with open(os.path.join(split_folder_name, 'train_test_split.json'), 'w') as f:
        json.dump(split_data, f, indent=4)
    print(f"Train/test split saved to {split_folder_name}/train_test_split.json")

def create_cross_validation_folds(train_df, num_folds=5, stratified=True, plot_folder=None):
    """
    Split the training set into k-folds for cross-validation and generate plots for each fold.
    Args:
        train_df (pd.DataFrame): Training dataset.
        num_folds (int): Number of folds for cross-validation.
        stratified (bool): Whether to use stratified splitting to balance classes.
        plot_folder (str): Path to the folder where class distribution plots will be saved.
    Returns:
        folds (list): List of train/validation splits for each fold.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = []

    X = train_df['image_name'].values
    y = train_df['benign_malignant'].values

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        fold_data = {
            "fold": fold + 1,
            "train": train_df.iloc[train_index],
            "validation": train_df.iloc[val_index]
        }
        folds.append(fold_data)

        # Plot the class distribution for each fold
        if plot_folder:
            fold_plot_folder = os.path.join(plot_folder, f"fold_{fold + 1}")
            os.makedirs(fold_plot_folder, exist_ok=True)

            # Generate and save the class distribution plots for training and validation sets
            plot_class_distribution(fold_data["train"], fold_data["validation"],fold_plot_folder,
    train_title=f"Fold {fold + 1} - Training Set Class Distribution",
    test_title=f"Fold {fold + 1} - Validation Set Class Distribution"
)
    
    return folds

def save_folds_to_json(folds, split_folder_name):
    """
    Save cross-validation folds into JSON files.
    Args:
        folds (list): List of train/validation splits for each fold.
        split_folder_name (str): Path to the folder where fold JSON files will be saved.
    """

    for fold_data in folds:
        fold_num = fold_data["fold"]
        fold_df_train = fold_data["train"]
        fold_df_val = fold_data["validation"]

        fold_images_train = [{"image_name": row['image_name'], "label": row['benign_malignant']} for _, row in fold_df_train.iterrows()]
        fold_images_val = [{"image_name": row['image_name'], "label": row['benign_malignant']} for _, row in fold_df_val.iterrows()]

        fold_json = {
            "training": fold_images_train,
            "validation": fold_images_val,
            "statistics": {
                "training_images": len(fold_df_train),
                "validation_images": len(fold_df_val),
                "training_benign": len(fold_df_train[fold_df_train['benign_malignant'] == 'benign']),
                "training_malignant": len(fold_df_train[fold_df_train['benign_malignant'] == 'malignant']),
                "validation_benign": len(fold_df_val[fold_df_val['benign_malignant'] == 'benign']),
                "validation_malignant": len(fold_df_val[fold_df_val['benign_malignant'] == 'malignant']),
            }
        }

        # Save fold data to JSON file
        fold_file_path = os.path.join(split_folder_name, f"fold_{fold_num}",f"fold_{fold_num}_split.json")
        with open(fold_file_path, 'w') as f:
            json.dump(fold_json, f, indent=4)
        print(f"Fold {fold_num} saved to {fold_file_path}")


import matplotlib.pyplot as plt
import os

def plot_class_distribution(train_df, test_df, split_folder_name, train_title="Training Set Class Distribution", test_title="Testing Set Class Distribution"):
    """
    Generates a class distribution plot for training and testing (or validation) sets.
    Also includes a plot showing the number of samples in training and testing/validation sets.
    Args:
        train_df (pd.DataFrame): Training set dataframe.
        test_df (pd.DataFrame): Testing or validation set dataframe.
        split_folder_name (str): Folder name to save the plot.
        train_title (str): Title for the training set plot.
        test_title (str): Title for the testing/validation set plot.
    """
    # 1. Class distribution in train set
    train_class_counts = train_df['benign_malignant'].value_counts()
    plt.figure(figsize=(6, 6))
    train_class_counts.plot(kind='bar', color=['lightgreen', 'orange'])
    plt.title(train_title)
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    for i, value in enumerate(train_class_counts):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(split_folder_name, f"{train_title}.png"))
    plt.close()

    # 2. Class distribution in test/validation set
    test_class_counts = test_df['benign_malignant'].value_counts()
    plt.figure(figsize=(6, 6))
    test_class_counts.plot(kind='bar', color=['lightgreen', 'orange'])
    plt.title(test_title)
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    for i, value in enumerate(test_class_counts):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(split_folder_name, f"{test_title}.png"))
    plt.close()

    # 3. Sample distribution (Training vs. Testing/Validation)
    plt.figure(figsize=(6, 6))
    sample_counts = [len(train_df), len(test_df)]
    sample_labels = ['Training Set', 'Test/Validation Set']
    plt.bar(sample_labels, sample_counts, color=['lightblue', 'salmon'])
    plt.title('Sample Distribution (Training vs Testing/Validation)')
    plt.ylabel('Number of Samples')

    # Annotate the bars with the count of samples
    for i, value in enumerate(sample_counts):
        plt.text(i, value + 10, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(split_folder_name, 'sample_distribution.png'))
    plt.close()

    # 4. Number of patients in the training and testing sets
    train_patient_count = len(train_df['patient_id'].unique())
    test_patient_count = len(test_df['patient_id'].unique())
    
    plt.figure(figsize=(6, 6))
    patient_counts = [train_patient_count, test_patient_count]
    patient_labels = ['Training Set', 'Test/Validation Set']
    plt.bar(patient_labels, patient_counts, color=['lightblue', 'salmon'])
    plt.title('Number of Patients (Training vs Testing/Validation)')
    plt.ylabel('Number of Patients')

    # Annotate the bars with the number of patients
    for i, value in enumerate(patient_counts):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(split_folder_name, 'patient_distribution.png'))
    plt.close()

    # 5. Number of patients with benign and malignant conditions (Training vs Testing)

    # Get the unique patients for each condition
    train_benign_patients = len(train_df[train_df['benign_malignant'] == 'benign'].groupby('patient_id').size())
    train_malignant_patients = len(train_df[train_df['benign_malignant'] == 'malignant'].groupby('patient_id').size())
    test_benign_patients = len(test_df[test_df['benign_malignant'] == 'benign'].groupby('patient_id').size())
    test_malignant_patients = len(test_df[test_df['benign_malignant'] == 'malignant'].groupby('patient_id').size())

    # Plot the number of patients with benign and malignant conditions
    plt.figure(figsize=(6, 6))
    benign_malignant_counts_train = [train_benign_patients, train_malignant_patients]
    benign_malignant_counts_test = [test_benign_patients, test_malignant_patients]

    bar_width = 0.35  # Set the bar width for grouped bars
    index = range(2)  # Two categories: benign, malignant

    # Plot the bars for training and testing sets
    plt.bar(index, benign_malignant_counts_train, bar_width, color='lightgreen', label='Training Set')
    plt.bar([i + bar_width for i in index], benign_malignant_counts_test, bar_width, color='orange', label='Test/Validation Set')

    plt.title('Number of Patients with Benign and Malignant Conditions')
    plt.xticks([i + bar_width / 2 for i in index], ['Benign', 'Malignant'])
    plt.ylabel('Number of Patients')
    plt.legend()

    # Annotate the bars with the number of patients
    for i, value in enumerate(benign_malignant_counts_train):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=12)
    for i, value in enumerate(benign_malignant_counts_test):
        plt.text(i + bar_width, value + 1, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(split_folder_name, 'benign_malignant_patient_distribution.png'))
    plt.close()


    print(f"Class distribution, sample distribution, patient distribution, and benign/malignant patient distribution plots saved to {split_folder_name}")
