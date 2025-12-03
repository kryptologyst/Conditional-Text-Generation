"""Conditional text generation models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from .utils import get_device


class ConditionalTextGenerator(nn.Module):
    """Base class for conditional text generation models."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the conditional text generator.
        
        Args:
            model_name: Name of the pre-trained model to use.
            device: Device to run the model on.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device or get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        raise NotImplementedError
        
    def generate(
        self,
        prompt: str,
        condition: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        **kwargs: Any,
    ) -> List[str]:
        """Generate conditional text.
        
        Args:
            prompt: Input prompt text.
            condition: Conditioning text.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate.
            do_sample: Whether to use sampling.
            repetition_penalty: Penalty for repetition.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated text sequences.
        """
        raise NotImplementedError


class GPT2ConditionalGenerator(ConditionalTextGenerator):
    """GPT-2 based conditional text generator."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GPT-2 conditional generator.
        
        Args:
            model_name: GPT-2 model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl).
            device: Device to run the model on.
            **kwargs: Additional arguments.
        """
        super().__init__(model_name, device, **kwargs)
        self.load_model()
        
    def load_model(self) -> None:
        """Load GPT-2 model and tokenizer."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
    def generate(
        self,
        prompt: str,
        condition: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        **kwargs: Any,
    ) -> List[str]:
        """Generate conditional text using GPT-2.
        
        Args:
            prompt: Input prompt text.
            condition: Conditioning text.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate.
            do_sample: Whether to use sampling.
            repetition_penalty: Penalty for repetition.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated text sequences.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
            
        # Format conditional prompt
        conditional_prompt = f"{condition}: {prompt}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(conditional_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt from the generated text
            text = text[len(conditional_prompt):].strip()
            generated_texts.append(text)
            
        return generated_texts


class AutoConditionalGenerator(ConditionalTextGenerator):
    """AutoModel based conditional text generator for various model types."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize AutoModel conditional generator.
        
        Args:
            model_name: HuggingFace model name.
            device: Device to run the model on.
            trust_remote_code: Whether to trust remote code.
            **kwargs: Additional arguments.
        """
        super().__init__(model_name, device, **kwargs)
        self.trust_remote_code = trust_remote_code
        self.load_model()
        
    def load_model(self) -> None:
        """Load AutoModel and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
    def generate(
        self,
        prompt: str,
        condition: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        **kwargs: Any,
    ) -> List[str]:
        """Generate conditional text using AutoModel.
        
        Args:
            prompt: Input prompt text.
            condition: Conditioning text.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate.
            do_sample: Whether to use sampling.
            repetition_penalty: Penalty for repetition.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated text sequences.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
            
        # Format conditional prompt
        conditional_prompt = f"{condition}: {prompt}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(conditional_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt from the generated text
            text = text[len(conditional_prompt):].strip()
            generated_texts.append(text)
            
        return generated_texts
