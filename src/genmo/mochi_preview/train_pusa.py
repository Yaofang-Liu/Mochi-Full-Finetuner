import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torchmetrics import MeanMetric
from ray.train.lightning import RayFSDPStrategy, RayLightningEnvironment
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from ray.train import ScalingConfig
from lightning.pytorch.callbacks import ModelCheckpoint

from einops import repeat
import argparse
import functools

import ipdb
import json
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block
from lightning.pytorch import LightningModule, Trainer, seed_everything

import genmo.mochi_preview.dit_pusa.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.models import (
    Decoder,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
)
from genmo.mochi_preview.vae.models import (Encoder, encode_latents)

from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents
from genmo.lib.utils import Timer, save_video
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from genmo.mochi_preview.vae.vae_stats import vae_latents_to_dit_latents

from datetime import datetime


from genmo.mochi_preview.vae.latent_dist import LatentDistribution
from genmo.mochi_preview.datasets.embedding_datasets import VideoEmbeddingDataset, get_video_embedding_dataloader

def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256

def get_conditioning(tokenizer, encoder, device, batch_inputs, *, prompt: list[str], negative_prompt: list[str]): 
    cond_input = get_conditioning_for_prompts(tokenizer, encoder, device, prompt)  
    null_input = get_conditioning_for_prompts(tokenizer, encoder, device, negative_prompt)
    return dict(cond=cond_input, null=null_input)

def get_conditioning_for_prompts(tokenizer, encoder, device, prompts: List[str]):
    # assert len(prompts) in [1, 2]  # [neg] or [pos] or [pos, neg]
    B = len(prompts)
    t5_toks = tokenizer(prompts,padding="max_length",truncation=True,max_length=MAX_T5_TOKEN_LENGTH,return_tensors="pt",return_attention_mask=True,)
    caption_input_ids_t5 = t5_toks["input_ids"]
    caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
    del t5_toks

    assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
    assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

    # Special-case empty negative prompt by zero-ing it
    if prompts[-1] == "":
        caption_input_ids_t5[-1] = 0
        caption_attention_mask_t5[-1] = False

    caption_input_ids_t5 = caption_input_ids_t5.to(device, non_blocking=True)
    caption_attention_mask_t5 = caption_attention_mask_t5.to(device, non_blocking=True)

    y_mask = [caption_attention_mask_t5]
    y_feat = [encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state.detach()]
    # Sometimes returns a tensor, othertimes a tuple, not sure why
    # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
    assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
    del caption_input_ids_t5, caption_attention_mask_t5

    return dict(y_mask=y_mask, y_feat=y_feat)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def t5_tokenizer():
    return T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)

from genmo.mochi_preview.vae.models import Encoder, add_fourier_features
# Define the LightningModule
class MochiModel(LightningModule):
    def __init__(
        self,
        lr: float,
        betas: tuple,
        weight_decay: float,
        min_lr: float,
        training_steps: int,
        checkpointing_period: int,
        model_dir_path,
        attention_mode = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['text_encoder', 'encoder'])

        MOCHI_DIR = model_dir_path
        self.tokenizer = t5_tokenizer()
        dit_path=f"{MOCHI_DIR}/dit.safetensors"
        from genmo.mochi_preview.dit_pusa.joint_model.asymm_models_joint import (
            AsymmDiTJoint,
        )
        attention_mode = "flash"
        self.dit = AsymmDiTJoint(
            depth=48,
            # depth=8,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
            attention_mode=attention_mode,
        )
        model_state_dict = load_file(dit_path)
        self.dit.load_state_dict(model_state_dict, strict=True)
        self.dit.train()
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.training_steps = training_steps
        self.checkpointing_period = checkpointing_period
        self.loss_metric = None
        self.automatic_optimization = True 
        self.stored_loss = None  # Store loss for manual backward

    def training_step(self, batch, batch_idx):
        # Get optimizer
        opt = self.optimizers()
        opt.zero_grad()
        
        # Use data directly from the VideoEmbeddingDataset
        # instead of manual loading
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            z_0 = batch["z_0"].to(self.device)
            y_feat = batch["y_feat"].to(self.device)
            y_mask = batch["y_mask"].to(self.device)
            
            conditioning = dict(cond=dict(y_mask=[y_mask], y_feat=[y_feat]))
            
            # Apply caption dropout with probability 0.1
            caption_dropout_prob = 0.1
            if random.random() < caption_dropout_prob:
                conditioning["cond"]["y_feat"][0].zero_()
                conditioning["cond"]["y_mask"][0].zero_()
        
        batch_size = z_0.shape[0]
        cfg_scale = 4.5
        SPATIAL_DOWNSAMPLE = 8
        TEMPORAL_DOWNSAMPLE = 6
        latent_t = z_0.shape[2]
        latent_w, latent_h = z_0.shape[3] , z_0.shape[4] 

        num_total_steps=1000
        threshold_noise=0.025
        sigma_schedule = linear_quadratic_schedule(num_total_steps, threshold_noise)
        sigma_schedule = torch.tensor(sigma_schedule, device=z_0.device)

        B = batch_size 

        random_steps = torch.randint(0, num_total_steps - 1, (B, latent_t), device=z_0.device)
        sigma = sigma_schedule[random_steps]
        next_sigma = sigma_schedule[random_steps + 1]
        dsigma = sigma - next_sigma

        num_latents = latent_t * latent_h * latent_w
        
        epsilon = torch.randn_like(z_0)
        sigma_reshaped = sigma.view(B, 1, latent_t, 1, 1)
        z_t = (1.0 - sigma_reshaped) * z_0 + sigma_reshaped * epsilon
        
        # Forward pass with gradient computation
        with torch.set_grad_enabled(True):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cond_indices = compute_packed_indices(z_0.device, conditioning["cond"]["y_mask"][0], num_latents)
                
                # Add packed indices to the conditioning dictionaries
                conditioning["cond"]["packed_indices"] = cond_indices
                pred = self.dit(z_t, sigma, **conditioning["cond"])
                with torch.no_grad():
                    target = z_0 - epsilon
                    if "cond" not in conditioning:  
                        target = target[:B]

                target_dit_space = vae_latents_to_dit_latents(target.float()) # Important to convert to dit space

                loss = F.mse_loss(pred.float(), target_dit_space, reduction='mean')
        
            # Store loss for backward in on_train_batch_end
            self.stored_loss = loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.global_rank == 0:
            log_dict = {
                "learning_rate": opt.param_groups[0]['lr'],
                "global_step": self.global_step,
                "batch_idx": batch_idx,
                "train_loss": loss,
            }
            self.logger.experiment.log(log_dict)

        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.trainer.model)
        trainable_params = [p for p in self.trainer.model.dit.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        return self.optimizer

    def on_train_epoch_end(self):
        pass

import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
from lightning.pytorch  import LightningDataModule


class MochiDataModule(LightningDataModule):
    def __init__(self, configs = None, data_path=None, batch_size=1, dataset_length=1000, num_workers=2, num_frames=16, frame_interval=1, width=848, height=480):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset_length = dataset_length
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.frame_interval = frame_interval
        self.configs = configs

    def setup(self, stage=None):
        # Create dataset using VideoEmbeddingDataset instead of manual loading
        if not hasattr(self, 'dataset'):
            from genmo.mochi_preview.datasets.embedding_datasets import VideoEmbeddingDataset
            
            self.dataset = VideoEmbeddingDataset(
                data_dir=self.data_path,
                device="cpu",  # Load to CPU first, will move to device in training step
                use_bfloat16=False,  # Convert to bfloat16 during training
            )

    def train_dataloader(self):
        # Get the current worker's rank and world size from Ray context
        from ray import train
        context = train.get_context()
        rank = context.get_world_rank()
        
        # Create sampler for distributed training
        sampler = DistributedSampler(
            self.dataset,
            rank=rank,
            shuffle=True
        )
        
        from genmo.mochi_preview.datasets.embedding_datasets import VideoEmbeddingDataset
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

# Training function
def train_mochi(config):
    import ray
    from ray.train.torch import TorchTrainer
    
    # Initialize Ray with more explicit resource configuration
    if config['address'] is not None:
        ray.init(
            address=os.environ["ip_head"],  # Force new cluster creation
            logging_level="INFO",  # Set logging level to INFO
            log_to_driver=True,    # Ensure logs are sent to the driver
        )
    else:
        ray.init(
            address="local",  # Force new cluster creation
            include_dashboard=False,
            num_cpus=config['world_size']*10,  # 10 CPUs per worker
            num_gpus=config['world_size'],     # 1 GPU per worker
            object_store_memory=1000 * 1024 * 1024 * 1024, # 1000G
        )

    seed_everything(42, workers=True)

    from datetime import datetime

    config['checkpoint_dir'] = config['checkpoint_dir'] + '/' + 'mochi_train_'+ datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Define the training function for each worker
    def train_loop_per_worker(config):
        from lightning.pytorch import LightningModule, Trainer, seed_everything
        import functools
        import lightning.pytorch as pl 
        from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

        from ray.train.lightning import RayFSDPStrategy, RayDDPStrategy
        from genmo.mochi_preview.dit_pusa.joint_model.asymm_models_joint import AsymmetricJointBlock
        from genmo.mochi_preview.vae.models import ResBlock, CausalUpsampleBlock, DownsampleBlock
        from transformers.models.t5.modeling_t5 import T5Block
        from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
        from ray.train import get_context
   
        context = get_context()
        rank = context.get_world_rank()
        world_size = context.get_world_size()

        # Initialize distributed context first
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size
            )
        
        # Safely set device by getting local rank
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
        device_id = local_rank % device_count
        torch.cuda.set_device(device_id)
        
        # Initialize context parallel after dist init
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), rank)

        callbacks = []
      
        # Rest of the FSDP strategy configuration remains the same
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                AsymmetricJointBlock, 
            }
        )

        fsdp_strategy = RayFSDPStrategy(
            state_dict_type="sharded", # 1. Select the FSDP strategy and set the sharded/distributed checkpoint format
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            auto_wrap_policy=auto_wrap_policy,
            limit_all_gathers=True,
            activation_checkpointing_policy=auto_wrap_policy,   
            mixed_precision=MixedPrecision(
                # Use bfloat16 for parameters and reduction
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ), 
        )

        # Initialize wandb only for rank 0 worker
        if rank == 0:
            wandb_logger = WandbLogger(
                project="mochi-training",  # Change this to your project name
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,  # This will log your config parameters
                log_model=False,  # Enable model checkpointing in wandb
            )
        else:
            wandb_logger = None

        checkpoint_callback = ModelCheckpoint(
            dirpath=config['checkpoint_dir'],
            filename='mochi-{step:07d}-{train_loss:.2f}',
            save_top_k=-1, # if save_top_k == k, the best k models according to the quantity monitored will be saved. If save_top_k == 0, no models are saved. If save_top_k == -1, all models are saved.
            monitor='train_loss',
            # mode='min',
            every_n_train_steps=config['checkpointing_period'],
            save_weights_only=True,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)
        # Add LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        dm = MochiDataModule(
            configs=config,
            data_path=config['data_path'],
            batch_size=config['batch_size_per_worker'],
            dataset_length=config['dataset_length'],
            num_workers=config['num_workers'],
            num_frames=config['num_frames'],  # Pass num_frames from config
            frame_interval=config['frame_interval'],  # Pass frame_interval from config
            height=config['height'],
            width=config['width'],
        )
        dm.setup()
        
        # Initialize the model
        model = MochiModel(
            lr=config['lr'],
            betas=config['betas'],
            weight_decay=config['weight_decay'],
            min_lr=config['min_lr'],
            training_steps=config['training_steps'],
            checkpointing_period=config['checkpointing_period'],
            model_dir_path=config['model_dir'],
        )

        # Initialize the Trainer with the callbacks
        pl_trainer = Trainer(
            max_epochs=config['max_epochs'],
            accelerator="gpu",
            devices=config['devices'],
            precision="bf16",
            strategy=fsdp_strategy,
            plugins=[RayLightningEnvironment()],
            callbacks=callbacks,  # Use the conditionally added callbacks
            enable_checkpointing=True,
            max_steps=config['training_steps'],
            logger=wandb_logger
        )

        # Prepare trainer with Ray
        pl_trainer = prepare_trainer(pl_trainer)

        # Train the model
        if config['resume_from_checkpoint']:
            pl_trainer.fit(model, dm.train_dataloader(), ckpt_path=config['resume_from_checkpoint'])
        else:
            pl_trainer.fit(model, dm.train_dataloader())

        # Finish wandb run
        if rank == 0:
            wandb.finish()

    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    # Configure the scaling
    scaling_config = ScalingConfig(
        num_workers=config['world_size'],
        use_gpu=True,
        resources_per_worker={
            "CPU": 6,  # Explicitly request 6 CPUs per worker
            "GPU": 1   # Explicitly request 1 GPU per worker
        },
        placement_strategy="PACK", # Try t  o pack workers on same node
    )

    # Create the trainer with the updated scaling config
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="mochi_training",
            log_to_file=True,  # Save logs to file
            storage_path=config['checkpoint_dir'],  # Where to save logs
        )
    )
    
    # Run the training with a timeout
    try:
        result = trainer.fit()
        print(f"Training completed: {result}")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        ray.shutdown()  # Ensure Ray is shut down properly


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_addr", type=str, default=None,
                      help="Ray head node address")
    parser.add_argument("--world_size", type=int, required=True,
                      help="Number of nodes")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Path to model directory")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to data directory")
    parser.add_argument("--num_frames", type=int, default=163,
                      help="Number of frames for video processing")
    parser.add_argument("--width", type=int, default=848, help="Width of the video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the video.")
    parser.add_argument("--frame_interval", type=int, default=1,
                      help="Frame interval for video processing")
    parser.add_argument("--lr", type=float, default=1e-5,
                      help="Learning rate for training")
    parser.add_argument("--training_steps", type=int, default=50000,
                      help="Training steps for training")
    parser.add_argument("--checkpointing_period", type=int, default=100,
                      help="Checkpointing period for training")
    parser.add_argument("--batch_size_per_worker", type=int, default=1,
                      help="Batch size per worker")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    from lightning.pytorch.callbacks import ModelCheckpoint
    import os   
    config = {
        'lr': args.lr,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
        'min_lr': None,
        'training_steps': args.training_steps,
        'checkpointing_period': args.checkpointing_period,
        'data_path': args.data_path,
        'model_dir': args.model_dir,
        'num_workers': 4,
        'world_size': args.world_size,
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'max_epochs': 1000,
        'devices': 1,  # Number of GPUs per process
        'cpu_offload': True,  # Whether to offload model to CPU
        'dataset_length': 100,  # No meaning, just for testing
        'batch_size_per_worker': args.batch_size_per_worker,  # Batch size per worker
        'address': args.head_addr,
        'num_frames': args.num_frames,
        'width': args.width,
        'height': args.height,
        'frame_interval': args.frame_interval,
        'checkpoint_dir': os.path.join(args.model_dir, 'checkpoints'),
        'resume_from_checkpoint': None,  # Set to checkpoint path if resuming training
        'wandb': {
            'project': 'mochi-training',
            'entity': None,  # Change this to your wandb username or team name
            'tags': ['mochi', 'training'],
        },
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    train_mochi(config)
