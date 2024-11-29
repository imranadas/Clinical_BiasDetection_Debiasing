import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class BiasAnalysisResult:
    is_biased: bool
    confidence: float
    important_words: List[Tuple[str, float]]
    bias_probability: float
    prediction: str
    biased_phrases: List[str]

class BiasDetector:
    def __init__(self, model_path: str = "models/bias_bert_model_synthetic"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self._load_model()
        
    def _load_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                output_attentions=True  # Request attention outputs
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_word_importance_from_embeddings(self, text: str, hidden_states) -> List[Tuple[str, float]]:
        """Calculate word importance using hidden states"""
        tokens = self.tokenizer.tokenize(text)
        
        # Get embeddings from last hidden state
        last_hidden = hidden_states[-1][0]  # Take first item from batch
        
        # Calculate importance scores using L2 norm of hidden states
        importance_scores = torch.norm(last_hidden, dim=1).cpu().numpy()
        
        # Pair tokens with their importance scores
        word_importance = []
        for token, score in zip(tokens, importance_scores):
            if not token.startswith('##'):  # Skip subword tokens
                word_importance.append((token, float(score)))
                
        # Sort by importance
        word_importance.sort(key=lambda x: x[1], reverse=True)
        return word_importance

    def find_biased_phrases(self, text: str, word_importances: List[Tuple[str, float]], 
                          importance_threshold: float = 0.1) -> List[str]:
        """Extract potentially biased phrases using important words as anchors"""
        words = text.split()
        phrases = []
        window_size = 3

        # Find phrases around important words
        for important_word, importance in word_importances:
            if importance < importance_threshold:
                continue
                
            for i, word in enumerate(words):
                if important_word in word.lower():
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    phrase = ' '.join(words[start:end])
                    phrases.append(phrase)

        return list(set(phrases))  # Remove duplicates
    
    def analyze_text(self, text: str, threshold: float = 0.5) -> BiasAnalysisResult:
        """Perform detailed bias analysis on text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = probabilities.argmax(dim=1)
            confidence = probabilities[0][prediction[0]].item()
            bias_probability = probabilities[0][1].item()
            
            # Get word importance from hidden states
            hidden_states = outputs.hidden_states
            word_importance = self.get_word_importance_from_embeddings(text, hidden_states)

        is_biased = bias_probability >= threshold
        
        # Find potentially biased phrases
        biased_phrases = self.find_biased_phrases(text, word_importance)

        return BiasAnalysisResult(
            is_biased=is_biased,
            confidence=confidence,
            important_words=word_importance[:10],
            bias_probability=bias_probability,
            prediction='Biased' if is_biased else 'Unbiased',
            biased_phrases=biased_phrases
        )

    def __call__(self, text: str) -> BiasAnalysisResult:
        """Convenience method to allow direct calling of instance"""
        return self.analyze_text(text)

if __name__ == "__main__":
    detector = BiasDetector()
    text = "The south Indians believe in alternate medicine than surgery."
    result = detector.analyze_text(text)
    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2f}")
    print("\nBiased phrases:")
    for phrase in result.biased_phrases:
        print(f"  {phrase}")