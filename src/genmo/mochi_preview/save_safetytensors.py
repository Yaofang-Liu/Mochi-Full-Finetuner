import argparse
import os
import torch
from safetensors.torch import save_file

def extract_sub_model_weights(checkpoint, prefix):
    """
    Extract sub-model weights by removing the specified prefix from each key.
    
    Args:
        checkpoint (dict): The state dictionary of the model.
        prefix (str): The prefix to remove from each key.
    
    Returns:
        dict: A dictionary containing the sub-model weights with updated keys.
    """
    sub_model_weights = {}
    for key, value in checkpoint.items():
        if key.startswith(prefix):  # Match the sub-model prefix
            new_key = key[len(prefix):]  # Remove the prefix
            sub_model_weights[new_key] = value
    return sub_model_weights

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to safetensors format.")
    parser.add_argument(
        "--path",
        type=str,
        default="./checkpoints/consolidated.ckpt",
        help="Path to the PyTorch checkpoint file (.ckpt)"
    )
    args = parser.parse_args()


    checkpoint_path = args.path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"The checkpoint file '{checkpoint_path}' does not exist.")

    # Load the consolidated checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    else:
        checkpoint = checkpoint  # If the checkpoint is already a state_dict

    # Define sub-model prefixes
    sub_model_prefixes = {
        "dit": "dit.",
        # "text_encoder": "text_encoder.",
        # "encoder": "encoder."
    }

    base_dir, filename = os.path.split(checkpoint_path)
    basename, _ = os.path.splitext(filename)

    # Iterate over sub-model prefixes and save safetensor checkpoints
    for sub_model_name, prefix in sub_model_prefixes.items():
        # Extract weights for the sub-model
        sub_model_weights = extract_sub_model_weights(checkpoint, prefix)
        
        # Define the save path using the base path
        save_path = os.path.join(base_dir, f"{basename}_{sub_model_name}.safetensors")
        save_file(sub_model_weights, save_path)
        print(f"Saved {sub_model_name} checkpoint to {save_path}")

if __name__ == "__main__":
    main()
