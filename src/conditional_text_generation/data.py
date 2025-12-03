"""Data handling for conditional text generation."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset, load_dataset
from transformers import PreTrainedTokenizer
from .utils import set_seed


class TextDataset(Dataset):
    """Dataset for conditional text generation."""
    
    def __init__(
        self,
        texts: List[str],
        conditions: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize text dataset.
        
        Args:
            texts: List of text samples.
            conditions: List of conditions for each text.
            tokenizer: Tokenizer for text processing.
            max_length: Maximum sequence length.
            **kwargs: Additional arguments.
        """
        self.texts = texts
        self.conditions = conditions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if len(texts) != len(conditions):
            raise ValueError("Number of texts and conditions must match.")
            
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index.
        
        Args:
            idx: Item index.
            
        Returns:
            Dictionary containing tokenized text and condition.
        """
        text = self.texts[idx]
        condition = self.conditions[idx]
        
        # Format conditional text
        conditional_text = f"{condition}: {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            conditional_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "text": text,
            "condition": condition,
        }


class ConditionalTextDataModule:
    """Data module for conditional text generation."""
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize data module.
        
        Args:
            data_path: Path to data file.
            dataset_name: HuggingFace dataset name.
            tokenizer: Tokenizer for text processing.
            max_length: Maximum sequence length.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes.
            train_split: Training set split ratio.
            val_split: Validation set split ratio.
            test_split: Test set split ratio.
            seed: Random seed.
            **kwargs: Additional arguments.
        """
        self.data_path = Path(data_path) if data_path else None
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        self.train_dataset: Optional[TextDataset] = None
        self.val_dataset: Optional[TextDataset] = None
        self.test_dataset: Optional[TextDataset] = None
        
        if data_path or dataset_name:
            self.prepare_data()
            
    def prepare_data(self) -> None:
        """Prepare datasets."""
        if self.dataset_name:
            self._load_huggingface_dataset()
        elif self.data_path:
            self._load_local_data()
        else:
            self._create_toy_dataset()
            
    def _load_huggingface_dataset(self) -> None:
        """Load dataset from HuggingFace."""
        dataset = load_dataset(self.dataset_name)
        
        # Extract texts and conditions
        texts = []
        conditions = []
        
        for split_name, split_data in dataset.items():
            for item in split_data:
                # This is a generic approach - specific datasets may need custom handling
                if "text" in item:
                    texts.append(item["text"])
                    conditions.append(item.get("condition", "general"))
                elif "content" in item:
                    texts.append(item["content"])
                    conditions.append(item.get("condition", "general"))
                else:
                    # Use the first text-like field
                    text_fields = [k for k, v in item.items() if isinstance(v, str) and len(v) > 10]
                    if text_fields:
                        texts.append(item[text_fields[0]])
                        conditions.append(item.get("condition", "general"))
                        
        self._create_datasets(texts, conditions)
        
    def _load_local_data(self) -> None:
        """Load data from local file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        if self.data_path.suffix == ".json":
            with open(self.data_path, "r") as f:
                data = json.load(f)
        elif self.data_path.suffix == ".jsonl":
            data = []
            with open(self.data_path, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path)
            data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
        # Extract texts and conditions
        texts = []
        conditions = []
        
        for item in data:
            if isinstance(item, dict):
                text = item.get("text", item.get("content", ""))
                condition = item.get("condition", "general")
            else:
                text = str(item)
                condition = "general"
                
            if text.strip():
                texts.append(text.strip())
                conditions.append(condition)
                
        self._create_datasets(texts, conditions)
        
    def _create_toy_dataset(self) -> None:
        """Create a toy dataset for demonstration."""
        toy_data = [
            ("I love this product because it works perfectly.", "positive review"),
            ("This is terrible and doesn't work at all.", "negative review"),
            ("The weather is sunny and beautiful today.", "weather description"),
            ("Once upon a time, there was a brave knight.", "story beginning"),
            ("The recipe calls for flour, eggs, and sugar.", "cooking instructions"),
            ("The movie was exciting and full of action.", "movie review"),
            ("I think this policy will help many people.", "political opinion"),
            ("The sunset over the ocean was breathtaking.", "nature description"),
            ("She walked into the room with confidence.", "character description"),
            ("The experiment showed interesting results.", "scientific observation"),
        ]
        
        texts = [item[0] for item in toy_data]
        conditions = [item[1] for item in toy_data]
        
        # Duplicate data to make it larger
        texts = texts * 10
        conditions = conditions * 10
        
        self._create_datasets(texts, conditions)
        
    def _create_datasets(self, texts: List[str], conditions: List[str]) -> None:
        """Create train/val/test datasets from texts and conditions."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
            
        # Set seed for reproducible splits
        set_seed(self.seed)
        
        # Shuffle data
        combined = list(zip(texts, conditions))
        random.shuffle(combined)
        texts, conditions = zip(*combined)
        
        # Calculate split indices
        total_size = len(texts)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        
        # Split data
        train_texts = texts[:train_size]
        train_conditions = conditions[:train_size]
        
        val_texts = texts[train_size:train_size + val_size]
        val_conditions = conditions[train_size:train_size + val_size]
        
        test_texts = texts[train_size + val_size:]
        test_conditions = conditions[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = TextDataset(
            train_texts, train_conditions, self.tokenizer, self.max_length
        )
        self.val_dataset = TextDataset(
            val_texts, val_conditions, self.tokenizer, self.max_length
        )
        self.test_dataset = TextDataset(
            test_texts, test_conditions, self.tokenizer, self.max_length
        )
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train/val/test sets.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            raise ValueError("Datasets must be prepared first.")
            
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader
