import os
import json
import pandas as pd
from utils import plot_class_distribution, load_data, split_train_test, save_split_to_json, create_cross_validation_folds, save_folds_to_json
import matplotlib.pyplot as plt
import config

def main():
    # Use paths from config.py
    cleaned_dataset_path = config.CSV_FILE_PATH_CLEANED_DATASET
    config_path = config.SPLITTING_STRATEGY_PATH_JSON_FILE
    
    # Load the dataset
    df = load_data(cleaned_dataset_path)
    
    # Load the splitting configuration
    with open(config_path, 'r') as f:
        splits_config = json.load(f)
    
    # Process each splitting strategy in the JSON configuration
    for split_config in splits_config:
        print(f"Processing split strategy: {split_config['Description']}")
        
        # Extract configuration details
        split_folder_name = split_config["Split_Folder_Name"]
        patient_level_split = split_config["Patient_level_split"]
        split_type = split_config["Split_Type"]
        stratified = split_config["Stratified"]
        class_equalization = split_config["Class_Equalization"]
        cross_validation = split_config["Cross_Validation_folders"]
        validation_percentage = split_config["Validation_Percentage"]
        plots_folder = split_config["Plots_Folder"]
        
        # Create the folder structure for this split
        os.makedirs(split_folder_name, exist_ok=True)
        
        # Perform train/test split
        print(f"Performing train/test split for {split_folder_name}...")
        train_df, test_df = split_train_test(df, test_size=(100 - split_config['training_Percentage']) / 100,
                                             patient_level_split=patient_level_split,
                                             stratified=stratified,
                                             class_equalization=class_equalization)
        
        # Save train/test split to JSON
        save_split_to_json(train_df, test_df, split_folder_name)
        
        # Generate visualizations if required
        if split_config["Visualization"]:
            print(f"Generating visualizations for {split_folder_name}...")
            # You can generate class distribution plots here if needed
            # Example:
            plot_class_distribution(train_df, test_df, split_folder_name)
        
        # Perform cross-validation splits if required
        if cross_validation:
            print(f"Performing cross-validation splits for {split_folder_name}...")
            folds = create_cross_validation_folds(train_df, num_folds=5, stratified=stratified,plot_folder=split_folder_name)
            save_folds_to_json(folds, split_folder_name)
        
        print(f"Split strategy completed for {split_folder_name}\n")


if __name__ == '__main__':
    main()
