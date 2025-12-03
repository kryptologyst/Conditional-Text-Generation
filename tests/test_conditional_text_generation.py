"""Unit tests for conditional text generation."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from conditional_text_generation.models import GPT2ConditionalGenerator, AutoConditionalGenerator
from conditional_text_generation.data import TextDataset, ConditionalTextDataModule
from conditional_text_generation.evaluation import EvaluationMetrics, TextEvaluator
from conditional_text_generation.utils import set_seed, get_device, format_text_for_condition


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        assert torch.initial_seed() == 42
        
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
    def test_format_text_for_condition(self):
        """Test text formatting."""
        result = format_text_for_condition("positive review", "I love this")
        assert result == "positive review: I love this"
        
        result = format_text_for_condition("story", "Once upon a time", " - ")
        assert result == "story - Once upon a time"


class TestModels:
    """Test model classes."""
    
    def test_gpt2_conditional_generator_init(self):
        """Test GPT-2 generator initialization."""
        generator = GPT2ConditionalGenerator(model_name="gpt2")
        assert generator.model_name == "gpt2"
        assert generator.model is not None
        assert generator.tokenizer is not None
        
    def test_gpt2_generate(self):
        """Test GPT-2 text generation."""
        generator = GPT2ConditionalGenerator(model_name="gpt2")
        
        # Test generation
        texts = generator.generate(
            prompt="I love this product because",
            condition="positive review",
            max_length=50,
            temperature=0.7,
            num_return_sequences=1,
        )
        
        assert isinstance(texts, list)
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        assert len(texts[0]) > 0


class TestData:
    """Test data handling."""
    
    def test_text_dataset(self):
        """Test TextDataset class."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        texts = ["I love this product", "This is terrible"]
        conditions = ["positive", "negative"]
        
        dataset = TextDataset(texts, conditions, tokenizer, max_length=128)
        
        assert len(dataset) == 2
        
        # Test getting an item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "text" in item
        assert "condition" in item
        
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        
    def test_conditional_text_data_module(self):
        """Test ConditionalTextDataModule."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        data_module = ConditionalTextDataModule(
            tokenizer=tokenizer,
            max_length=128,
            batch_size=2,
            seed=42,
        )
        
        # Test toy dataset creation
        data_module._create_toy_dataset()
        
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None
        
        # Test dataloaders
        train_loader, val_loader, test_loader = data_module.get_dataloaders()
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_evaluation_metrics_init(self):
        """Test EvaluationMetrics initialization."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        metrics = EvaluationMetrics(tokenizer)
        
        assert metrics.tokenizer is not None
        assert metrics.rouge_scorer is not None
        
    def test_bleu_score(self):
        """Test BLEU score calculation."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        metrics = EvaluationMetrics(tokenizer)
        
        predictions = ["I love this product", "This is great"]
        references = ["I really love this product", "This is excellent"]
        
        scores = metrics.bleu_score(predictions, references)
        
        assert "bleu_mean" in scores
        assert "bleu_std" in scores
        assert isinstance(scores["bleu_mean"], float)
        
    def test_rouge_score(self):
        """Test ROUGE score calculation."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        metrics = EvaluationMetrics(tokenizer)
        
        predictions = ["I love this product", "This is great"]
        references = ["I really love this product", "This is excellent"]
        
        scores = metrics.rouge_score(predictions, references)
        
        assert "rouge1_mean" in scores
        assert "rouge2_mean" in scores
        assert "rougeL_mean" in scores
        
    def test_distinct_n(self):
        """Test distinct-n calculation."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        metrics = EvaluationMetrics(tokenizer)
        
        texts = ["I love this product", "I hate this product", "This is great"]
        
        distinct_2 = metrics.distinct_n(texts, n=2)
        distinct_3 = metrics.distinct_n(texts, n=3)
        
        assert isinstance(distinct_2, float)
        assert isinstance(distinct_3, float)
        assert 0 <= distinct_2 <= 1
        assert 0 <= distinct_3 <= 1
        
    def test_length_statistics(self):
        """Test length statistics calculation."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        metrics = EvaluationMetrics(tokenizer)
        
        texts = ["Short text", "This is a longer text with more words", "Medium length text here"]
        
        stats = metrics.length_statistics(texts)
        
        assert "avg_length" in stats
        assert "std_length" in stats
        assert "min_length" in stats
        assert "max_length" in stats
        
        assert stats["min_length"] == 2  # "Short text"
        assert stats["max_length"] == 8  # "This is a longer text with more words"
        
    def test_text_evaluator(self):
        """Test TextEvaluator."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        evaluator = TextEvaluator(tokenizer)
        
        predictions = ["I love this product", "This is great"]
        references = ["I really love this product", "This is excellent"]
        conditions = ["positive", "positive"]
        
        metrics = evaluator.evaluate(predictions, references, conditions)
        
        assert "avg_length" in metrics
        assert "distinct_2" in metrics
        assert "distinct_3" in metrics
        assert "bleu_mean" in metrics
        assert "rouge1_mean" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
