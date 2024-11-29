import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from bias_model_BERT import BiasDetector
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm

class BertValidator:
    def __init__(self, model_path: str = "models/bias_bert_model_synthetic"):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.detector = BiasDetector(model_path)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        # Remove quotes if present
        text = text.strip('"')
        # Remove newlines and extra spaces
        text = text.replace('\n', ' ').strip()
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text

    def load_validation_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess validation data"""
        try:
            # Read CSV file, specify index_col=0 to handle the unnamed index column
            df = pd.read_csv(filepath, index_col=0)
            
            # Ensure we have the required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns")
            
            # Clean text data
            df['text'] = df['text'].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['text'].str.len() > 0].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading validation data: {e}")
            raise

    def predict_batch(self, texts: list) -> list:
        """Get predictions for a batch of texts"""
        predictions = []
        for text in tqdm(texts, desc="Processing texts"):
            try:
                result = self.detector.analyze_text(text)
                predictions.append(1 if result.is_biased else 0)
            except Exception as e:
                self.logger.error(f"Error processing text: {e}")
                predictions.append(0)  # Default to unbiased in case of error
        return predictions

    def plot_confusion_matrix(self, y_true: list, y_pred: list, save_path: str):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def validate(self, validation_file: str, output_dir: str = "validation_results"):
        """Run validation and generate metrics"""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load validation data
        self.logger.info(f"Loading validation data from {validation_file}")
        val_data = self.load_validation_data(validation_file)
        
        # Labels are already binary (0 or 1) based on the sample, no need for conversion
        
        # Get predictions
        self.logger.info("Generating predictions")
        y_pred = self.predict_batch(val_data['text'].tolist())
        y_true = val_data['label'].tolist()

        # Calculate metrics
        self.logger.info("Calculating metrics")
        metrics = {
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }

        # Generate detailed classification report
        class_report = classification_report(y_true, y_pred, 
                                          target_names=['Unbiased', 'Biased'])

        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, 
            y_pred, 
            str(output_dir / 'confusion_matrix.png')
        )

        # Save results
        results = {
            'metrics': metrics,
            'classification_report': class_report
        }

        # Save metrics to file
        with open(output_dir / 'validation_results.txt', 'w') as f:
            f.write("Validation Results\n")
            f.write("=================\n\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(class_report)

        # Save all predictions and their corresponding texts
        predictions_df = pd.DataFrame({
            'text': val_data['text'],
            'true_label': y_true,
            'predicted_label': y_pred
        })
        predictions_df.to_csv(output_dir / 'predictions.csv')

        self.logger.info("Validation complete. Results saved to validation_results/")
        return results

if __name__ == "__main__":
    validator = BertValidator()
    results = validator.validate(
        validation_file="data/bias-valid.csv",
        output_dir="validation_results"
    )
    
    # Print summary results
    print("\nValidation Results Summary:")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print("\nDetailed Classification Report:")
    print(results['classification_report'])