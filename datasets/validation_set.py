import json
import random
import os
from typing import List, Tuple

def generate_datasets_with_merge(
    directory: str,
    target_name: str,
    x: float,
    col1: str,
    col2: str
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generates non-member, train, and member sets from a dataset in a JSON file.
    In the non-member and member sets, merges two specified columns into a new column 'text' and discards all others.
    
    Parameters:
    - directory (str): Path to the directory containing the dataset and where outputs will be saved.
    - target_name (str): Base name for the generated dataset files.
    - x (float): Percentage (0 < x < 1) of the dataset to sample for non-member and member sets.
    - col1 (str): Name of the first column to merge.
    - col2 (str): Name of the second column to merge.
    
    Returns:
    - Tuple[List[dict], List[dict], List[dict]]: Non-member set, train set, member set.
    """
    # Construct the input dataset path
    dataset_path = os.path.join(directory, f"{target_name}.json")
    
    # Load the dataset
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
    
    # Shuffle the dataset to ensure randomness
    random.shuffle(dataset)
    
    # Calculate sample size
    sample_size = int(x * len(dataset))
    
    # Split into non-member set and rest of the dataset (train candidate set)
    non_member_set = dataset[:sample_size]
    train_candidate_set = dataset[sample_size:]
    
    # Shuffle train candidate set and select member set from it
    random.shuffle(train_candidate_set)
    member_set = train_candidate_set[:sample_size]
    
    # Remaining becomes the final train set
    train_set = train_candidate_set
    
    # Function to merge columns and discard the rest
    def merge_columns(data, col1, col2):
        return [{"text": f"{item[col1]} {item[col2]}"} for item in data if col1 in item and col2 in item]
    
    # Process non-member and member sets
    non_member_set = merge_columns(non_member_set, col1, col2)
    member_set = merge_columns(member_set, col1, col2)
    
    # Automatically generate output file paths
    non_member_file = os.path.join(directory, f"{target_name}_non_member.json")
    train_file = os.path.join(directory, f"{target_name}_train.json")
    member_file = os.path.join(directory, f"{target_name}_member.json")
    
    # Save sets to JSON files
    with open(non_member_file, 'w') as file:
        json.dump(non_member_set, file, indent=4)
    
    with open(train_file, 'w') as file:
        json.dump(train_set, file, indent=4)
    
    with open(member_file, 'w') as file:
        json.dump(member_set, file, indent=4)
    
    print(f"Generated datasets:")
    print(f"Non-member set saved to: {non_member_file}")
    print(f"Train set saved to: {train_file}")
    print(f"Member set saved to: {member_file}")
    
    return non_member_set, train_set, member_set


# Usage example:
directory = "/nfs-share/pa511/llm_memorisation/datasets/medical_dataset"
target_name = "deduplicated_medical_meadow_flashcards"
col1 = "input"
col2 = "output"
x = 0.1  # 20% sampling
generate_datasets_with_merge(directory, target_name, x, col1, col2)
