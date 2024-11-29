import re
import logging
from typing import List, Optional, Dict
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict

class LLMTextDebiaser:
    """Uses T5 to debias clinical text by generating unbiased alternatives"""
    
    def __init__(self, model_name: str = "google/flan-t5-large"):
        """Initialize the debiaser with T5 model"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.logger.info(f"Initialized T5 model {model_name} successfully")
        except Exception as e:
            self.logger.error(f"Error loading T5 model: {e}")
            raise

        # Categories of bias to check for
        self.bias_categories = {
            'gender': ['he', 'she', 'man', 'woman', 'male', 'female', 'gender'],
            'racial': ['race', 'ethnic', 'cultural', 'minority', 'black', 'white', 'asian'],
            'age': ['young', 'old', 'elderly', 'age', 'generation'],
            'socioeconomic': ['poor', 'wealthy', 'income', 'class'],
            'stigmatizing': ['difficult', 'challenging', 'non-compliant', 'refusing', 'denies'],
            'judgment': ['demanding', 'aggressive', 'angry', 'uncooperative']
        }

    def analyze_text_for_bias(self, text: str) -> Dict[str, List[str]]:
        """Analyze text for different types of potential bias"""
        found_bias = defaultdict(list)
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for category-specific bias terms
        for category, terms in self.bias_categories.items():
            for term in terms:
                if term in words:
                    context = self._get_context(text_lower, term)
                    if context:
                        found_bias[category].append(context)
        
        return dict(found_bias)

    def _get_context(self, text: str, term: str, window: int = 5) -> str:
        """Extract context around a biased term"""
        words = text.split()
        for i, word in enumerate(words):
            if term in word:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                return " ".join(words[start:end])
        return ""

    def create_debiasing_prompt(self, text: str, biased_phrases: Optional[List[str]] = None) -> str:
        """Create a prompt instructing T5 how to debias the text"""
        # Analyze text for specific types of bias
        bias_analysis = self.analyze_text_for_bias(text)
        
        prompt = """Rewrite the following medical text to remove bias while preserving clinical information.
Guidelines:
- Maintain medical facts and observations
- Use neutral language
- Remove stereotypes
- Keep clinical meaning
- Use person-first language
- Stay professional

Text: {text}"""

        # Add specific bias concerns if found
        if bias_analysis:
            prompt += "\nAddress these biases:\n"
            for category, instances in bias_analysis.items():
                if instances:
                    prompt += f"{category}: {', '.join(instances)}\n"

        # Add specific phrases to address
        if biased_phrases:
            prompt += f"\nRevise these phrases: {', '.join(biased_phrases)}\n"

        return prompt.format(text=text)

    def generate_debiased_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate debiased text using T5"""
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_length, 
                              truncation=True).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True
            )
        
        # Decode and return the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def validate_output(self, original: str, debiased: str) -> bool:
        """Validate that the debiased output maintains key characteristics"""
        # Check length similarity (within 20%)
        orig_len = len(original.split())
        new_len = len(debiased.split())
        if not (0.8 <= new_len/orig_len <= 1.2):
            return False
            
        # Check for key content words preserved
        key_words = set(w.lower() for w in original.split() if len(w) > 4)
        debiased_words = set(w.lower() for w in debiased.split())
        content_preservation = len(key_words & debiased_words) / len(key_words)
        
        return content_preservation >= 0.7

    def debias_text(self, text: str, biased_phrases: Optional[List[str]] = None) -> str:
        """Generate unbiased version of potentially biased text using T5"""
        try:
            # Create appropriate prompt
            prompt = self.create_debiasing_prompt(text, biased_phrases)

            # Generate debiased text
            debiased_text = self.generate_debiased_text(prompt)
            
            # Validate output
            if not self.validate_output(text, debiased_text):
                self.logger.warning("First attempt failed validation, retrying...")
                return self._retry_debiasing(text, biased_phrases)
            
            return debiased_text

        except Exception as e:
            self.logger.error(f"Error generating debiased text: {e}")
            return self._get_fallback_response(text)

    def _retry_debiasing(self, text: str, biased_phrases: Optional[List[str]] = None) -> str:
        """Retry debiasing with modified parameters"""
        try:
            prompt = self.create_debiasing_prompt(text, biased_phrases)
            prompt += "\nImportant: Maintain exact clinical meaning while removing bias."
            
            debiased_text = self.generate_debiased_text(
                prompt,
                max_length=512,  # Increased max length
            )
            
            return debiased_text if debiased_text else self._get_fallback_response(text)
            
        except Exception as e:
            self.logger.error(f"Error in retry debiasing: {e}")
            return self._get_fallback_response(text)

    def _get_fallback_response(self, text: str) -> str:
        """Provide a fallback response that maintains clinical meaning"""
        # Simple fallback that attempts to neutralize obvious biased terms
        for category, terms in self.bias_categories.items():
            for term in terms:
                if category == 'stigmatizing':
                    text = text.replace('difficult patient', 'patient requiring additional support')
                    text = text.replace('non-compliant', 'has not followed treatment plan')
                    text = text.replace('refusing', 'declining')
                    text = text.replace('denies', 'reports no')
                elif category == 'judgment':
                    text = text.replace('demanding', 'expressing needs')
                    text = text.replace('aggressive', 'agitated')
                    text = text.replace('angry', 'distressed')
                    text = text.replace('uncooperative', 'not in agreement with plan')
        return text

    def __call__(self, text: str, biased_phrases: Optional[List[str]] = None) -> str:
        """Convenience method to allow direct calling of instance"""
        return self.debias_text(text, biased_phrases)