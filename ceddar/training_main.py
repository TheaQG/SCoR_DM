# sbgm/training_main.py
import os
import torch
import logging

import numpy as np
import matplotlib.pyplot as plt

from sbgm.training_utils import get_model_string, get_model, get_optimizer, get_dataloader, get_scheduler
from sbgm.plotting_utils import plot_sample
from sbgm.training import TrainingPipeline_general
from sbgm.score_unet import marginal_prob_std_fn, diffusion_coeff_fn

# Set up logging
logger = logging.getLogger(__name__)

def train_main(cfg):
    """
    Main function to run the training process.
    
    Args:
        cfg (dict): Configuration dictionary containing all necessary parameters.
    """

    logger.info("\n\n=== Starting SBGM_SD Training Pipeline ===")
    logger.info(f"          Experiment name: {cfg['experiment']['name']}")

    # Set path to figures, samples, losses
    save_str = get_model_string(cfg)
    path_samples = os.path.join(cfg['paths']['path_save'], 'samples', save_str)
    path_figures = os.path.join(path_samples, 'Figures')

    # Make sure figures directory exists
    os.makedirs(path_figures, exist_ok=True)

    # Read device str from cfg
    device_str = cfg['training']['device']

    # Set device
    if device_str == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"          ▸ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info(f"          ▸ CUDA is not available, using CPU instead.")
    else:
        device = torch.device('cpu')
        logger.info(f"          ▸ Using CPU for training.")

    # Load data
    train_dataloader, val_dataloader, gen_dataloader = get_dataloader(cfg)

    # # ------------------------------------------------------------------------
    # # Quick data-loader throughput check: ~100 batches warm-up + timed 
    # # ------------------------------------------------------------------------
    # from time import perf_counter
    # start = perf_counter()
    # for i, _ in enumerate(train_dataloader):
    #     if i == 100:
    #         avg = (perf_counter() - start) / 100
    #         logger.info(f"          ▸ Dataloader average fetch time ~{avg:.3f} s / batch\n\n")



    # Examine sample from train dataloader (sample is full batch)
    sample = train_dataloader.dataset[0]
    for key, value in sample.items():
        try:
            # Log the shape of the tensor
            logger.info(f'          {key}: {value.shape}')
            # Log the device of the tensor
            logger.info(f'              {key} device: {value.device}')
        except AttributeError:
            logger.info(f'          {key}: {value}')
        if key == 'classifier':
            logger.info(f'          ▸ Classifier: {value}')


    if cfg['visualization']['plot_initial_sample']:
        fig, _ = plot_sample(sample, cfg)
        if cfg['visualization']['show_figs']:
            plt.show()
        else:
            plt.close(fig)
        # Save the figure
        SAVE_NAME = 'Initial_sample_plot.png'
        save_path = os.path.join(path_figures, SAVE_NAME)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        logger.info(f"\n\n          ▸ Saved initial sample plot to {save_path}")
    
    
    #Setup checkpoint path
    checkpoint_dir = os.path.join(cfg['paths']['path_save'], cfg['paths']['checkpoint_dir'])

    checkpoint_name = save_str + '.pth.tar'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Define the seed for reproducibility, and set seed for torch, numpy and random
    torch.manual_seed(cfg['training']['seed'])
    torch.cuda.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True
    
    # Get the model
    model, checkpoint_path, checkpoint_name = get_model(cfg)
    model = model.to(device)

    # Get the optimizer
    optimizer = get_optimizer(cfg, model)

    # Get the learning rate scheduler (if applicable)
    lr_scheduler_type = cfg['training'].get('lr_scheduler', None)
    
    if lr_scheduler_type is not None:
        logger.info(f"          ▸ Using learning rate scheduler: {lr_scheduler_type}")
        scheduler = get_scheduler(cfg, optimizer)
    else:
        scheduler = None
        logger.info(f"          ▸ No learning rate scheduler specified, using default learning rate.")

    # Define the training pipeline
    pipeline = TrainingPipeline_general(model=model,
                                        marginal_prob_std_fn=marginal_prob_std_fn,
                                        diffusion_coeff_fn=diffusion_coeff_fn,
                                        optimizer=optimizer,
                                        device=device,
                                        lr_scheduler=scheduler,
                                        cfg=cfg
                                        )

    
    # Load checkpoint if it exists
    if cfg['training']['load_checkpoint'] and os.path.exists(checkpoint_path):
        logger.info(f"          ▸ Loading pretrained weights from checkpoint {checkpoint_path}")

        pipeline.load_checkpoint(checkpoint_path, load_ema=cfg['training']['load_ema'],)
    else:
        logger.info(f"          ▸ No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    
    # If training on cuda, print device name and empty cache
    if cfg['training']['device'] == 'cuda' and torch.cuda.is_available():
        logger.info(f"\n\n          ▸ Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"          ▸ Model is using {torch.cuda.memory_allocated() / 1e9:.2f} GB of GPU memory.")
        logger.info(f"          ▸ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        logger.info(f"\n          ▸ Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"          ▸ Number of trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad and p.requires_grad):,}")
        logger.info(f"          ▸ Number of non-trainable parameters in model: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
        torch.cuda.empty_cache()
    else:
        logger.info("\n\n          ▸ Using CPU for training.")
        logger.info(f"          ▸ Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"          ▸ Number of trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad and p.requires_grad):,}")
        logger.info(f"          ▸ Number of non-trainable parameters in model: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    # Perform training
    logger.info(f"\n\n          === STARTING TRAINING MAIN LOOP ===\n")
    pipeline.train(train_dataloader,
                   val_dataloader,
                   gen_dataloader,
                   cfg,
                   epochs=cfg['training']['epochs'],
                   verbose=cfg['training']['verbose'],
                   use_mixed_precision= cfg['training']['use_mixed_precision'],
    )
    logger.info("\n\n       === TRAINING COMPLETE ===\n\n")


















