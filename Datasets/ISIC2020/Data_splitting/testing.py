import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/work/c-2iia/hn977782/Thesis/Code/Datasets/ISIC2020/cleaned_dataset.csv')

patients = df['patient_id'].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)

# Filter data based on patients for train/test split
train_df = df[df['patient_id'].isin(train_patients)]
test_df = df[df['patient_id'].isin(test_patients)]

patients_train = train_df['patient_id'].unique()
print("Total Number of patients in the training", len(patients_train))

patients_test = test_df['patient_id'].unique()
print("Total number of patients in the testing set", len(patients_test))

# Get the unique patients with benign or malignant conditions
benign_patients_test = test_df[test_df['benign_malignant'] == 'benign']['patient_id'].unique()
malignant_patients_test = test_df[test_df['benign_malignant'] == 'malignant']['patient_id'].unique()

# Find the intersection of benign and malignant patients
overlap_patients_test = set(benign_patients_test).intersection(set(malignant_patients_test))

# Total unique patients should be the size of the union of benign and malignant patients
unique_patients_test = set(benign_patients_test).union(set(malignant_patients_test))

# Number of overlapping patients (those who have both benign and malignant labels)
overlap_count = len(overlap_patients_test)

# Correct the count: Total unique patients = len(benign) + len(malignant) - len(overlap)
corrected_total = len(benign_patients_test) + len(malignant_patients_test) - overlap_count

print("Number of overlapping patients (both benign and malignant):", overlap_count)
print("Corrected total number of unique patients:", corrected_total)

print("Number of patients with benign:", len(benign_patients_test))
print("Number of patients with malignant:", len(malignant_patients_test))
print("Total unique patients in the testing set:", len(unique_patients_test))
