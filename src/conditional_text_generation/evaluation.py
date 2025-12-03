"""Evaluation metrics for conditional text generation."""

import re
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from transformers import PreTrainedTokenizer

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class EvaluationMetrics:
    """Collection of evaluation metrics for text generation."""
    
    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        """Initialize evaluation metrics.
        
        Args:
            tokenizer: Tokenizer for text processing.
        """
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def perplexity(
        self,
        model: torch.nn.Module,
        texts: List[str],
        conditions: List[str],
        device: torch.device,
    ) -> float:
        """Calculate perplexity of generated texts.
        
        Args:
            model: Language model.
            texts: List of reference texts.
            conditions: List of conditions.
            device: Device to run computation on.
            
        Returns:
            Average perplexity score.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for perplexity calculation.")
            
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text, condition in zip(texts, conditions):
                conditional_text = f"{condition}: {text}"
                inputs = self.tokenizer.encode(conditional_text, return_tensors="pt")
                inputs = inputs.to(device)
                
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                
                total_loss += loss.item() * inputs.size(1)
                total_tokens += inputs.size(1)
                
        return np.exp(total_loss / total_tokens)
        
    def bleu_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            Dictionary of BLEU scores.
        """
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=self.smoothing_function,
            )
            bleu_scores.append(score)
            
        return {
            "bleu_mean": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
        }
        
    def rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            Dictionary of ROUGE scores.
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
                
        return {
            "rouge1_mean": np.mean(rouge_scores["rouge1"]),
            "rouge1_std": np.std(rouge_scores["rouge1"]),
            "rouge2_mean": np.mean(rouge_scores["rouge2"]),
            "rouge2_std": np.std(rouge_scores["rouge2"]),
            "rougeL_mean": np.mean(rouge_scores["rougeL"]),
            "rougeL_std": np.std(rouge_scores["rougeL"]),
        }
        
    def bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts.
            
        Returns:
            Dictionary of BERTScore metrics.
        """
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item(),
        }
        
    def distinct_n(self, texts: List[str], n: int = 2) -> float:
        """Calculate distinct-n metric for diversity.
        
        Args:
            texts: List of generated texts.
            n: N-gram size.
            
        Returns:
            Distinct-n score.
        """
        ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngrams.add(ngram)
                total_ngrams += 1
                
        return len(ngrams) / total_ngrams if total_ngrams > 0 else 0.0
        
    def self_bleu(self, texts: List[str], n: int = 4) -> float:
        """Calculate self-BLEU for diversity evaluation.
        
        Args:
            texts: List of generated texts.
            n: N-gram size.
            
        Returns:
            Self-BLEU score.
        """
        if len(texts) < 2:
            return 0.0
            
        bleu_scores = []
        
        for i, text in enumerate(texts):
            references = [t.split() for j, t in enumerate(texts) if j != i]
            if not references:
                continue
                
            score = sentence_bleu(
                references,
                text.split(),
                smoothing_function=self.smoothing_function,
            )
            bleu_scores.append(score)
            
        return np.mean(bleu_scores)
        
    def length_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length statistics.
        
        Args:
            texts: List of texts.
            
        Returns:
            Dictionary of length statistics.
        """
        lengths = [len(text.split()) for text in texts]
        
        return {
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
        }
        
    def repetition_ratio(self, texts: List[str]) -> float:
        """Calculate repetition ratio.
        
        Args:
            texts: List of texts.
            
        Returns:
            Average repetition ratio.
        """
        ratios = []
        
        for text in texts:
            tokens = text.split()
            if len(tokens) < 2:
                ratios.append(0.0)
                continue
                
            unique_tokens = set(tokens)
            repetition_ratio = 1.0 - len(unique_tokens) / len(tokens)
            ratios.append(repetition_ratio)
            
        return np.mean(ratios)


class TextEvaluator:
    """Comprehensive text evaluation system."""
    
    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        """Initialize text evaluator.
        
        Args:
            tokenizer: Tokenizer for text processing.
        """
        self.metrics = EvaluationMetrics(tokenizer)
        
    def evaluate(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of generated texts.
        
        Args:
            predictions: List of predicted texts.
            references: List of reference texts (optional).
            conditions: List of conditions (optional).
            model: Language model for perplexity calculation (optional).
            device: Device for model computation (optional).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        results = {}
        
        # Length statistics
        results.update(self.metrics.length_statistics(predictions))
        
        # Diversity metrics
        results["distinct_2"] = self.metrics.distinct_n(predictions, n=2)
        results["distinct_3"] = self.metrics.distinct_n(predictions, n=3)
        results["self_bleu"] = self.metrics.self_bleu(predictions)
        results["repetition_ratio"] = self.metrics.repetition_ratio(predictions)
        
        # Reference-based metrics
        if references is not None:
            results.update(self.metrics.bleu_score(predictions, references))
            results.update(self.metrics.rouge_score(predictions, references))
            results.update(self.metrics.bert_score(predictions, references))
            
        # Perplexity calculation
        if model is not None and device is not None and conditions is not None:
            try:
                perplexity = self.metrics.perplexity(model, predictions, conditions, device)
                results["perplexity"] = perplexity
            except Exception as e:
                print(f"Warning: Could not calculate perplexity: {e}")
                
        return results
        
    def evaluate_conditioning(
        self,
        predictions: List[str],
        conditions: List[str],
        condition_classifier: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate how well generated texts match their conditions.
        
        Args:
            predictions: List of predicted texts.
            conditions: List of conditions.
            condition_classifier: Optional classifier for condition evaluation.
            
        Returns:
            Dictionary of conditioning evaluation metrics.
        """
        results = {}
        
        # Simple keyword-based conditioning evaluation
        condition_keywords = {
            "positive": ["good", "great", "excellent", "amazing", "wonderful", "love", "like"],
            "negative": ["bad", "terrible", "awful", "hate", "dislike", "horrible", "worst"],
            "neutral": ["okay", "fine", "average", "normal", "standard"],
        }
        
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred, condition in zip(predictions, conditions):
            pred_lower = pred.lower()
            
            # Simple keyword matching
            if condition.lower() in condition_keywords:
                keywords = condition_keywords[condition.lower()]
                if any(keyword in pred_lower for keyword in keywords):
                    correct_predictions += 1
                    
        results["conditioning_accuracy"] = correct_predictions / total_predictions
        
        return results
