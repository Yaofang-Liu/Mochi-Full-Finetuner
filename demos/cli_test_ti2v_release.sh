#!/bin/bash
# Model directory
MODEL_DIR="<path_to_Pusa-V0.5_dir>"

# Checkpoint paths
CHECKPOINTS=(
    "<path_to_pusa_dit_checkpoint>"
)

# I2V - default conditioned on the second frame
image_dir="<path_to_input_image_dir>"
# prompt_dir=None
prompt="The camera remains still, the boy waves the baseball bat and knocks the baseball away."
cond_position=1
num_steps=30

# image_dir="<path_to_input_image_dir>"
# prompt_dir="<path_to_prompt_dir>"
# prompt=None
# cond_position=1
# num_steps=30

# Generate with 1 GPU, generated video will be splited into 4 parts for decoding
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python3 ./demos/cli_test_ti2v_release.py --model_dir "$MODEL_DIR" --dit_path "$CHECKPOINT" --image_dir "$image_dir" --prompt "$prompt" --prompt_dir "$prompt_dir" --cond_position "$cond_position" --num_steps "$num_steps"
done


# Generate with 4 GPUs
# for CHECKPOINT in "${CHECKPOINTS[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./demos/cli_test_ti2v_release.py --model_dir "$MODEL_DIR" --dit_path "$CHECKPOINT" --image_dir "$image_dir" --prompt "$prompt" --prompt_dir "$prompt_dir" --cond_position "$cond_position" --num_steps "$num_steps"
# done