#!/usr/bin/env python3
"""Training script for conditional text generation."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from transformers import GPT2Tokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from conditional_text_generation.models import GPT2ConditionalGenerator
from conditional_text_generation.data import ConditionalTextDataModule
from conditional_text_generation.utils import set_seed, get_device, load_config


class ConditionalTextLightningModule(pl.LightningModule):
    """PyTorch Lightning module for conditional text generation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Lightning module.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self.model = GPT2ConditionalGenerator(
            model_name=config["model"]["name"],
            device=self.device,
        )
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Model outputs.
        """
        return self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch data.
            batch_idx: Batch index.
            
        Returns:
            Training loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch data.
            batch_idx: Batch index.
            
        Returns:
            Validation loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer.
        
        Returns:
            Optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        
        return optimizer


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train conditional text generation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config["data"]["data_path"] = args.data_path
    if args.max_epochs:
        config["training"]["num_epochs"] = args.max_epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
        
    # Set device
    device = get_device()
    config["model"]["device"] = str(device)
    
    # Initialize data module
    data_module = ConditionalTextDataModule(
        data_path=config["data"]["data_path"],
        dataset_name=config["data"]["dataset_name"],
        tokenizer=None,  # Will be set after model initialization
        max_length=config["data"]["max_length"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        seed=config["data"]["seed"],
    )
    
    # Initialize tokenizer for data module
    tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_module.tokenizer = tokenizer
    
    # Prepare data
    data_module.prepare_data()
    
    # Initialize Lightning module
    lightning_module = ConditionalTextLightningModule(config)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["mode"],
        save_top_k=config["logging"]["save_top_k"],
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["mode"],
        patience=3,
        verbose=True,
    )
    
    # Initialize logger
    if config["logging"]["experiment_name"]:
        logger = TensorBoardLogger(
            save_dir=config["logging"]["log_dir"],
            name=config["logging"]["experiment_name"],
        )
    else:
        logger = True
        
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        gpus=args.gpus if torch.cuda.is_available() else 0,
        precision=config["training"]["precision"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        val_check_interval=config["logging"]["val_check_interval"],
        deterministic=True,
    )
    
    # Train model
    trainer.fit(lightning_module, data_module)
    
    # Test model
    trainer.test(lightning_module, data_module)
    
    print(f"Training completed! Best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
