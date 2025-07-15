# Full Fine-tuner for Mochi and Pusa 0.5

[![Code](https://img.shields.io/badge/Code-Pusa%20Repo-32CD32?logo=github)](https://github.com/Yaofang-Liu/Pusa-VidGen) [![ModelHub](https://img.shields.io/badge/⚡-Model%20Hub-FFD700?logo=huggingface)](https://huggingface.co/RaphaelLiu/Pusa-V0.5) [![DataRepo](https://img.shields.io/badge/📁-Dataset%20Repo-6495ED?logo=huggingface)](https://huggingface.co/datasets/RaphaelLiu/PusaV0.5_Training) 
[![Paper](https://img.shields.io/badge/📜-FVDM%20Paper-B31B1B?logo=arxiv)](https://arxiv.org/abs/2410.03160) [![Twitter](https://img.shields.io/badge/🐦-Twitter-1DA1F2?logo=twitter)](https://x.com/stephenajason)
[![Xiaohongshu](https://img.shields.io/badge/📕-Xiaohongshu-FF2442)](https://www.xiaohongshu.com/user/profile/5c6f928f0000000010015ca1?xsec_token=YBEf_x-s5bOBQIMJuNQvJ6H23Anwey1nnDgC9wiLyDHPU=&xsec_source=app_share&xhsshare=CopyLink&appuid=5c6f928f0000000010015ca1&apptime=1752622393&share_id=60f9a8041f974cb7ac5e3f0f161bf748)

## Overview

This repository provides tools for full fine-tuning or partial fine-tuning (e.g., specific DiT blocks) of the Mochi and Pusa 0.5 video generation models. It supports training on both single-node and multi-node configurations with our provided dataset or your custom data.

## System Requirements

- **GPU**: 8x H800 or above for full fine-tuning
- **Video Length Support**: Up to 163 frames (~5.4 seconds at 30 FPS)
  - Choose frame counts in increments of 6: 25, 31, 37, ... 163.
  - Single node (8 GPUs): 163 frames with batch_size_per_worker=1 uses ~59GB VRAM per GPU
  - Two nodes (16 GPUs): supports up to 163 frames with batch_size_per_worker=2 (total batch size 32), [Pusa-V0.5](https://huggingface.co/RaphaelLiu/Pusa-V0.5) model was trained for 500 steps in ~7 hours
  - Multi-node training above two nodes is also supported


## Installation

Set up the environment and install dependencies:

```bash
git clone https://github.com/Yaofang-Liu/Mochi-Full-Finetuner.git
cd Mochi-Full-Finetuner

pip install uv
uv venv pusa
source pusa/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
uv pip install torchmetrics ipdb opencv-python pyarrow "ray[train]" lightning 
uv pip install flash-attn --no-build-isolation
uv pip install decord PyAV
pip install ray[client]
```

Download Mochi 1 weights:

```bash
huggingface-cli download genmo/mochi-1-preview --repo-type model --local-dir <path_to_model_directory>
```

## Dataset Preparation

### Option 1: Use our pre-processed dataset

Download our training dataset (52695 pre-encoded latent samples from [VIDGEN-1M](https://huggingface.co/datasets/Fudan-FUXI/VIDGEN-1M), Pusa V0.5 only used 16000 samples):

```bash
huggingface-cli download RaphaelLiu/PusaV0.5_Training --repo-type dataset --local-dir <path_to_dataset_directory>
```

Alternatively, you can use your own dataset following the instructions as Mochi Lora Training [here](https://github.com/genmoai/mochi/tree/main/demos/fine_tuner). Note that your final dataset structure should be arranged like this:

```
path/to/datasets/
  videos/
    xxxx.latent.pt
    xxxx.latent.pt
    ...
  captions/
    xxxx.embed.pt
    xxxx.embed.pt
    ...
``` 

## Fine-tuning

### Single Node (8 GPUs)

```bash
python -u /path/to/src/genmo/mochi_preview/train_xxxx.py \
  --world_size=8 \
  --model_dir="/path/to/model/directory" \
  --data_path="/path/to/datasets/videos"
```
Note: 
-`/path/to/src/genmo/mochi_preview/train_xxxx.py` can be `train_mochi.py` if you want to train original mochi model or `train_pusa.py` if you want to train Pusa model.
- please provide only the path to the videos directory for `--data_path` argument, the captions directory will be automatically derived by replacing base directory name"videos" with "captions". 
- `os.path.join(args.model_dir, 'checkpoints')` will be used as the checkpoint directory.

### Multi-Node Configuration

Edit the SLURM configuration in `src/genmo/mochi_preview/train_multi_nodes.sh`:

1. Update SLURM parameters:
   - `--partition`: Your cluster's partition
   - `--nodes`: Number of nodes
   - `--nodelist`: Node names (optional)
   - `--cpus-per-task`: CPUs per node
   - `--mem`: Memory per node
   - `--gres`: GPU resources per node

2. Update paths:
   - Project directory
   - Model directory
   - Data directory
   - Training script path (train_mochi.py or train_pusa.py)

3. Adjust training parameters:
   - `--num_frames`: Frame count
   - `--frame_interval`: Frame interval
   - `--width` and `--height`: Frame dimensions

### Launch Multi-Node Training

```bash
sbatch ./src/genmo/mochi_preview/train_multi_nodes.sh
```

## Monitoring

Training logs are saved to:
- `logs/mochi_[job_id].out`: Standard output
- `logs/mochi_[job_id].err`: Standard error

## Using Fine-tuned Models

You can find the training checkpoints in `os.path.join(args.model_dir, 'checkpoints')` directory. Since we use fsdp to train the model (splited the model into multiple parts), we need to convert your saved checkpoints to safety tensors:
```bash
bash ./src/genmo/mochi_preview/convert_checkpoint.sh /path/to/your/xxxxx.ckpt
```
Please give the path to the local checkpoint file.

After the conversion, you will have a finetuned dit safetytensor file and you can use the file to replace the original dit safetytensor file and use in the same way as the original Mochi or Pusa model. For example with Pusa, you can use the following command to generate video:
```bash
bash ./demos/cli_test_ti2v_release.sh
```
and give the finetuned dit safetytensor file as the checkpoint path in `cli_test_ti2v_release.sh`.
```bash
CHECKPOINTS=(
    "<path_to_finetuned_dit_safetytensor_file>"
)
```

## LoRA Training

For LoRA fine-tuning, refer to:
- [Mochi LoRA Training](https://github.com/genmoai/mochi)
- [Diffusers Python Library](https://github.com/huggingface/diffusers)

## Limitations

Currently, we do not support training with context parallelism/sequence parallelism/tensor parallelism, which can make the training process much more memory efficient. 
Contributions to implement these memory-efficient training methods are welcome!


## Citation

If you use this work in your project, please cite:

```bibtex
@misc{Liu2025pusa,
  title={Pusa: Thousands Timesteps Video Diffusion Model},
  author={Yaofang Liu and Rui Liu},
  year={2025},
  url={https://github.com/Yaofang-Liu/Pusa-VidGen},
}
```

```bibtex
@article{liu2024redefining,
  title={Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach},
  author={Liu, Yaofang and Ren, Yumeng and Cun, Xiaodong and Artola, Aitor and Liu, Yang and Zeng, Tieyong and Chan, Raymond H and Morel, Jean-michel},
  journal={arXiv preprint arXiv:2410.03160},
  year={2024}
}
```
