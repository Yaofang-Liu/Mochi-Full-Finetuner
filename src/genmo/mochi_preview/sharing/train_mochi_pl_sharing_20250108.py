import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torchmetrics import MeanMetric
from ray.train.lightning import RayFSDPStrategy, RayLightningEnvironment
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy
)
from ray.train import ScalingConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from einops import repeat
import argparse
import functools
import ipdb
import json
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast
import numpy as np
import ray
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block
from lightning.pytorch import LightningModule, Trainer, seed_everything
import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer, save_video
from genmo.mochi_preview.vae.models import (
    Decoder,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
    Encoder
)
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents


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
        model_dir_path: str,
        attention_mode: str = None,
    ):
        """
        Initialize the MochiModel LightningModule.

        Args:
            lr (float): Learning rate.
            betas (tuple): Betas for the Adam optimizer.
            weight_decay (float): Weight decay for the optimizer.
            min_lr (float): Minimum learning rate for scheduler.
            training_steps (int): Total number of training steps.
            checkpointing_period (int): Steps between checkpoints.
            model_dir_path (str): Path to the model directory.
            attention_mode (str, optional): Attention mode to use. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters() # Note, It will enable Lightning to store all the provided arguments under the self.hparams attribute. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.

        # Setup model directory
        MOCHI_DIR = model_dir_path

        # Initialize tokenizer
        self.tokenizer = t5_tokenizer()

        # Define paths to model components
        dit_path = f"{MOCHI_DIR}/dit.safetensors"
        decoder_path = f"{MOCHI_DIR}/decoder.safetensors"
        encoder_path = f"{MOCHI_DIR}/encoder.safetensors"

        # Set attention mode
        attention_mode = "flash"

        # Initialize the DiT model
        self.dit = AsymmDiTJoint(
            depth=48,
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
        # Load pre-trained state
        model_state_dict = load_file(dit_path)
        self.dit.load_state_dict(model_state_dict, strict=True)
        self.dit.train()

        # Initialize text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(T5_MODEL)
        self.text_encoder.eval()

        # Note:Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Configuration for VAE encoder
        config = dict(
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate",
        )

        # Create VAE encoder
        self.encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            **config,
        )
        self.encoder = self.encoder.to(memory_format=torch.channels_last_3d)

        # Load pre-trained encoder state
        state_dict = load_file(encoder_path)
        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.eval()

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Initialize hyperparameters
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.training_steps = training_steps
        self.checkpointing_period = checkpointing_period
        self.loss_metric = None
        self.automatic_optimization = True  # Note:Enable automatic optimization, recommended for FSDP

    def forward(self, z, sigma, cfg_scale, num_latents, conditioning):
        """
        Forward pass of the model.

        Args:
            z (Tensor): Input tensor.
            sigma (Tensor): Noise scale.
            cfg_scale (float): Configuration scale.
            num_latents (int): Number of latent variables.
            conditioning (dict): Conditioning information.

        Returns:
            Tensor: Output tensor after processing.
        """
        cond_batched = cond_text = cond_null = None
        if "cond" in conditioning:
            cond_text = conditioning["cond"]
            cond_null = conditioning["null"]
            cond_text["packed_indices"] = compute_packed_indices(z.device, cond_text["y_mask"][0], num_latents)
            cond_null["packed_indices"] = compute_packed_indices(z.device, cond_null["y_mask"][0], num_latents)
        else:
            cond_batched = conditioning["batched"]
            cond_batched["packed_indices"] = compute_packed_indices(z.device, cond_batched["y_mask"][0], num_latents)
            z = repeat(z, "b ... -> (repeat b) ...", repeat=2)
        if cond_batched:
            # with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            # nonlocal cond_text, cond_null
            # with torch.autocast("cuda", dtype=torch.bfloat16):
            out_cond = self.dit(z, sigma, **cond_text)
            out_uncond = self.dit(z, sigma, **cond_null)
        # ipdb.set_trace()
        assert out_cond.shape == out_uncond.shape
        out_uncond = out_uncond.to(z)
        out_cond = out_cond.to(z)
        return out_uncond + cfg_scale * (out_cond - out_uncond)


    def training_step(self, batch, batch_idx):
        """
        Training step executed on each batch.

        Args:
            batch (dict): Batch data containing 'video' and 'caption'.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        # Extract input data
        z = batch["video"]
        prompt = batch["caption"]

        cfg_scale = 4.5
        SPATIAL_DOWNSAMPLE = 8
        TEMPORAL_DOWNSAMPLE = 6
        IN_CHANNELS = 12

        # Calculate latent dimensions
        latent_t = ((z.shape[2] - 1) // TEMPORAL_DOWNSAMPLE) + 1
        latent_w, latent_h = z.shape[3] // SPATIAL_DOWNSAMPLE, z.shape[4] // SPATIAL_DOWNSAMPLE

        batch_size = z.shape[0]

        num_total_steps = 1000
        threshold_noise = 0.025

        # Generate noise schedule
        sigma_schedule = linear_quadratic_schedule(num_total_steps, threshold_noise)
        sigma_schedule = torch.tensor(sigma_schedule, device=z.device)

        # Randomly select steps for each sample in the batch
        random_steps = torch.randint(0, num_total_steps, (batch_size,), device=z.device)
        sigma = sigma_schedule[random_steps]  # Shape: (batch_size,)

        with torch.no_grad():
            # Encode input with Fourier features
            z_with_features = add_fourier_features(z)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ldist = self.encoder(z_with_features)
                z = ldist.sample()  # Sample from the latent distribution

            # Get conditioning information
            conditioning = get_conditioning(
                self.tokenizer,
                self.text_encoder,
                z.device,
                True,  # batch_cfg
                prompt=prompt,
                negative_prompt=prompt,
            )

        B = batch_size if "cond" in conditioning else batch_size * 2

        # Expand sigma to match latent dimensions
        sigma = sigma.unsqueeze(1).expand(B, latent_t)

        # Calculate sigma difference for scheduler
        next_sigma = sigma_schedule[random_steps + 1]
        dsigma = sigma - next_sigma.unsqueeze(1).expand(B, latent_t)

        num_latents = latent_t * latent_h * latent_w

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = self.forward(z, sigma, cfg_scale, num_latents, conditioning)
            target = z
            z = z if "cond" in conditioning else z[:B]
            loss = F.mse_loss(pred, target)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            optimizer: Configured optimizer.
        """
        self.optimizer = AdamW(
            self.trainer.model.dit.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return self.optimizer
        
        # Note: Below one is also okay
        # optimizer = torch.optim.AdamW(self..dit.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.training_steps, eta_min=self.min_lr)
        # return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        """Hook for actions at the end of each training epoch."""
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Hook for actions at the end of each training batch."""
        pass


# Define the DataModule for handling data loading
class MochiDataModule(LightningDataModule):
    def __init__(
        self,
        configs: dict = None,
        data_path: str = None,
        batch_size: int = 1,
        dataset_length: int = 1000,
        num_workers: int = 2,
        num_frames: int = 16,
        frame_interval: int = 1,
    ):
        """
        Initialize the MochiDataModule.

        Args:
            configs (dict, optional): Configuration settings. Defaults to None.
            data_path (str, optional): Path to the data directory. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.
            dataset_length (int, optional): Length of the dataset. Defaults to 1000.
            num_workers (int, optional): Number of data loader workers. Defaults to 2.
            num_frames (int, optional): Number of frames per video sample. Defaults to 16.
            frame_interval (int, optional): Frame interval for video sampling. Defaults to 1.
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset_length = dataset_length
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.configs = configs

    def setup(self, stage=None):
        """
        Setup the dataset for training.

        Args:
            stage (str, optional): Stage identifier. Defaults to None.
        """
        if not hasattr(self, "dataset"):
            temporal_sample = TemporalRandomCrop(self.num_frames * self.frame_interval)
            transform_VIDGEN = transforms.Compose([
                ToTensorVideo(),  # Convert video to Tensor (TCHW)
                RandomHorizontalFlipVideo(),
                UCFCenterCropVideo(256),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

            self.dataset = VIDGEN(
                configs=self.configs,
                transform=transform_VIDGEN,
                temporal_sample=temporal_sample,
            )

    def train_dataloader(self):
        """
        Create the DataLoader for training.

        Returns:
            DataLoader: Configured DataLoader instance.
        """
        # Get Ray's distributed context
        context = train.get_context()
        rank = context.get_world_rank()

        # Create a sampler for distributed training
        sampler = DistributedSampler(
            self.dataset,
            rank=rank,
            shuffle=True,
        )

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Define the training function
def train_mochi(config: dict):
    """
    Train the Mochi model using Ray and PyTorch Lightning.

    Args:
        config (dict): Configuration settings for training.
    """
    # Initialize Ray with specified resources
    ray.init(
        address="local",  # Note:Start a new Ray cluster locally
        include_dashboard=False,
        num_cpus=config['world_size'] * 4,  # Note:4 CPUs per worker
        num_gpus=config['world_size'],      # Note:1 GPU per worker
        object_store_memory=1000 * 1024 * 1024 * 1024,  # Note:1 TB
    )
    seed_everything(42, workers=True)  # Set random seed for reproducibility

    # Create checkpoint directory with timestamp
    config['checkpoint_dir'] = os.path.join(
        config['checkpoint_dir'],
        'mochi_train_' + datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_loop_per_worker(config: dict):
        """
        Note: Training loop executed by each Ray worker.

        Args:
            config (dict): Configuration settings for training.
        """
        context = get_context()
        rank = context.get_world_rank()

        callbacks = []

        # Note: Define the model sharding policy for FSDP
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # transformer_layer_cls = {AsymmetricJointBlock, T5Block, ResBlock} # TODO this setting works, fsdp all correspoding blocks successfully 
            transformer_layer_cls = {AsymmetricJointBlock, T5Block} # TODO this setting works, fsdp all correspoding blocks successfully 
            # transformer_layer_cls = {AsymmetricJointBlock} # TODO this setting works, fsdp all correspoding blocks successfully 
            # transformer_layer_cls = {T5Block}
        ) 
        # auto_wrap_policy = functools.partial(
        #     size_based_auto_wrap_policy, min_num_params=100000000
        # )

        fsdp_strategy = RayFSDPStrategy(
            state_dict_type="sharded",
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            auto_wrap_policy=auto_wrap_policy,
            limit_all_gathers=True,
            activation_checkpointing_policy=auto_wrap_policy,
            mixed_precision=MixedPrecision( # Note,bfloat16 to save memory
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )

        # Add ModelCheckpoint and LearningRateMonitor callbacks only for rank 0
        if rank == 0:
            checkpoint_callback = ModelCheckpoint(
                dirpath=config['checkpoint_dir'],
                filename='mochi-{step:07d}-{train_loss:.2f}',
                save_top_k=1,
                monitor='train_loss',
                mode='min',
                every_n_train_steps=config['checkpointing_period'],
                save_last=True,
            )
            callbacks.append(checkpoint_callback)

            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

        # Initialize the DataModule
        dm = MochiDataModule(
            configs=config,
            data_path=config['data_path'],
            batch_size=config['batch_size_per_worker'],
            dataset_length=config['dataset_length'],
            num_workers=config['num_workers'],
            num_frames=config.get('num_frames', 16),
            frame_interval=config.get('frame_interval', 1),
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

        # Initialize the Trainer with the defined strategy and callbacks
        pl_trainer = Trainer(
            max_epochs=config['max_epochs'],
            accelerator="gpu",
            devices=config['devices'],
            precision="bf16",
            strategy=fsdp_strategy,
            plugins=[RayLightningEnvironment()],
            callbacks=callbacks,
            enable_checkpointing=True,
            max_steps=50,
        )

        # Prepare the trainer with Ray
        pl_trainer = prepare_trainer(pl_trainer)

        # Start training, optionally resuming from a checkpoint
        if config['resume_from_checkpoint']:
            pl_trainer.fit(
                model,
                dm.train_dataloader(),
                ckpt_path=config['resume_from_checkpoint'],
            )
        else:
            pl_trainer.fit(model, dm.train_dataloader())

    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    # Note for ray RunConfig checkpointing, use automatic optimization is recommended
    # run_config=RunConfig( 
    #     storage_path=config['checkpoint_dir'],
    #     name="finetune_mochi1",
    #     sync_config=ray.train.SyncConfig(sync_artifacts=True), #If RunConfig(SyncConfig(sync_artifacts=True)), then all artifacts saved in this directory will be persisted to storage. The frequency of artifact syncing can be configured via SyncConfig. Note that this behavior is off by default. https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html#overview-of-ray-train-outputs
    # )

    # Configure the scaling
    scaling_config = ScalingConfig(
        num_workers=config['world_size'],
        use_gpu=True,
        resources_per_worker={
            "CPU": 4,  # Explicitly request 4 CPUs per worker
            "GPU": 1   # Explicitly request 1 GPU per worker
        },
        placement_strategy="PACK",  # Pack workers on the same node
    )

    # Initialize the TorchTrainer with the training loop and scaling config
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        # run_config=run_config,
    )

    # Execute the training process with error handling
    try:
        result = trainer.fit()
        print(f"Training completed: {result}")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        ray.shutdown()  # Ensure Ray shuts down properly


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Mochi Model")
    parser.add_argument("--head_addr", type=str, default=None,
                      help="Ray head node address")
    parser.add_argument("--world_size", type=int, required=True,
                      help="Number of nodes")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Path to model directory")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to data directory")
    parser.add_argument("--num_frames", type=int, default=16,
                      help="Number of frames for video processing")
    parser.add_argument("--frame_interval", type=int, default=1,
                      help="Frame interval for video processing")
    return parser.parse_args()


# Entry point for script execution
if __name__ == "__main__":-
    args = parse_args()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # Example import

    # Define training configuration
    config = {
        'lr': 1e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
        'min_lr': 1e-6,
        'training_steps': 1000,
        'checkpointing_period': 20,
        'data_path': args.data_path,
        'model_dir': args.model_dir,
        'num_workers': 4,
        'world_size': args.world_size,
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'max_epochs': 10,
        'devices': 1,  # Number of GPUs per process
        'cpu_offload': True,  # Whether to offload model to CPU
        'dataset_length': 100,  # Length of the synthetic dataset
        'batch_size_per_worker': 2,  # Batch size per worker
        'address': args.head_addr,
        'num_frames': args.num_frames,
        'frame_interval': args.frame_interval,
        'checkpoint_dir': os.path.join(args.model_dir, 'checkpoints'),
        'resume_from_checkpoint': None,  # Set to checkpoint path if resuming training
    }

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Start the training process
    train_mochi(config)
