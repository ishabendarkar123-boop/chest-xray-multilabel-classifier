import pandas as pd
import os
import glob

# Load metadata
df = pd.read_csv('data/raw/Data_Entry_2017.csv')
print("CSV rows:", len(df))

# Find all image files across all 12 folders (adjust pattern based on actual structure)
image_paths = glob.glob('data/raw/images_*/images/*.png')
print("Image files found:", len(image_paths))

# Check for duplicate labels
print("Duplicate Image Index entries:", df['Image Index'].duplicated().sum())
print("Unique images referenced in CSV:", df['Image Index'].nunique())

# Age anomaly check
print("\nPatient Age stats:")
print(df['Patient Age'].describe())
print("Suspicious ages (>100):", (df['Patient Age'] > 100).sum())

# View Position distribution
print("\nView Position distribution:")
print(df['View Position'].value_counts())

# Curation: flag and null anomalous ages rather than dropping rows
df['Age_Flag'] = df['Patient Age'] > 100
flagged_count = df['Age_Flag'].sum()
df.loc[df['Age_Flag'], 'Patient Age'] = None

print(f"\nFlagged and nulled {flagged_count} anomalous age records")
print(f"Retained all {len(df)} rows (image + disease labels still valid)")

# Save curated metadata
df.to_csv('data/processed/metadata_curated.csv', index=False)
print("Saved curated metadata to data/processed/metadata_curated.csv")
