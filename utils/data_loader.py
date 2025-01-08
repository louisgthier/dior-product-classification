import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_folder, seed=42, test_size=0.2, val_size=0.1):
    """
    Load images from the DAM folder based on filenames listed in a CSV file, 
    associate each image with its label, and split the data into train, val, and test DataFrames.

    Parameters:
        data_folder (str): Path to the data folder containing 'DAM' and 'product_list.csv'.
        seed (int): Random seed for reproducibility.
        test_size (float): Proportion of data to be used for testing.
        val_size (float): Proportion of remaining data to be used for validation after splitting off the test set.

    Returns:
        pd.DataFrame: train_df, val_df, test_df
    """
    # Paths to CSV and image folder
    csv_path = os.path.join(data_folder, 'product_list.csv')
    dam_folder = os.path.join(data_folder, 'DAM')

    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Create a list of image paths and labels
    data = []
    for _, row in df.iterrows():
        mmc = row['MMC']
        label = row['Product_BusinessUnitDesc']
        image_path = os.path.join(dam_folder, f"{mmc}.jpeg")
        
        # Check if image exists before adding it to the list
        if os.path.exists(image_path):
            data.append({"image_path": image_path, "label": label})

    # Convert to DataFrame
    data_df = pd.DataFrame(data)

    # Ensure reproducibility
    train_val_df, test_df = train_test_split(
        data_df, 
        test_size=test_size, 
        random_state=seed, 
        stratify=data_df['label']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=seed, 
        stratify=train_val_df['label']
    )

    # Return the splits
    return train_df, val_df, test_df

# Example usage
if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split_data("data")
    print("Train DataFrame:")
    print(train_df.head())
    print("\nValidation DataFrame:")
    print(val_df.head())
    print("\nTest DataFrame:")
    print(test_df.head())