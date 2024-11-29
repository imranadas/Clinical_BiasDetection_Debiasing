import re
import gc
import pickle
import logging
import pandas as pd
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from datetime import datetime
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class BiasDataProcessor:
    def __init__(self, data_dir: str, cache_dir: str):
        """
        Initialize the bias data processor for local datasets
        
        Args:
            data_dir: Directory containing raw data files
            cache_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to both file and console
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / f'data_processing_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_local_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all local datasets
        """
        self.logger.info("Loading local datasets...")
        datasets = {}
        
        # 1. Medical Bias Tagged Data
        try:
            self.logger.info("Loading medical bias tagged data...")
            medical_bias = pd.read_csv(self.data_dir / 'medical_bias_tagged_data.csv')
            datasets['medical_bias'] = medical_bias
        except Exception as e:
            self.logger.error(f"Error loading medical bias tagged data: {e}")
            
        # 2. News Bias Data (train and test)
        try:
            self.logger.info("Loading news bias data...")
            news_train = pd.read_csv(self.data_dir / 'data_train.csv')
            news_test = pd.read_csv(self.data_dir / 'data_test.csv')
            datasets['news_train'] = news_train
            datasets['news_test'] = news_test
        except Exception as e:
            self.logger.error(f"Error loading news bias data: {e}")
            
        # 3. Medical Chatbot Data
        try:
            self.logger.info("Loading medical chatbot data...")
            chatbot = pd.read_csv(self.data_dir / 'ai_medical_chatbot.csv')
            datasets['chatbot'] = chatbot
        except Exception as e:
            self.logger.error(f"Error loading medical chatbot data: {e}")
            
        return datasets

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove numeric values
        text = re.sub(r'\d+', ' NUM ', text)
        
        return text.strip()

    def create_binary_label(self, row: pd.Series) -> int:
        """
        Create binary bias label from different dataset formats
        """
        if 'is_biased' in row:
            return 1 if row['is_biased'] else 0
        elif 'label' in row:
            return 1 if row['label'] in ['Highly Biased', 'Biased'] else 0
        elif 'biased_words' in row and isinstance(row['biased_words'], str):
            # Check if biased_words is not empty (handling string representation of lists)
            return 1 if row['biased_words'] not in ('[]', '', '[]', 'nan') else 0
        return 0

    def process_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process and combine all datasets
        """
        processed_data = []
        
        for name, df in datasets.items():
            self.logger.info(f"Processing {name}...")
            
            # Determine text column
            text_col = None
            if 'TEXT' in df.columns:
                text_col = 'TEXT'
            elif 'text' in df.columns:
                text_col = 'text'
            elif 'Description' in df.columns:
                text_col = 'Description'
            
            if text_col is None:
                self.logger.warning(f"No text column found in {name}, skipping...")
                continue
                
            # Create processed dataset
            processed_texts = []
            labels = []
            
            for _, row in tqdm(df.iterrows(), total=len(df)):
                processed_text = self.preprocess_text(row[text_col])
                if len(processed_text.split()) > 3:  # Skip very short texts
                    processed_texts.append(processed_text)
                    labels.append(self.create_binary_label(row))
            
            processed_data.append(pd.DataFrame({
                'text': processed_texts,
                'is_biased': labels,
                'source': name
            }))
        
        return pd.concat(processed_data, ignore_index=True)

    def create_bert_inputs(self, text: str, max_length: int = 512) -> Dict:
        """
        Create BERT input features
        """
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def process_and_save_data(self):
        """
        Main function to process and save the data
        """
        # Load datasets
        datasets = self.load_local_datasets()
        
        # Process all datasets
        combined_data = self.process_datasets(datasets)
        
        # Create splits
        train_data, temp = train_test_split(combined_data, test_size=0.3, random_state=42, stratify=combined_data['is_biased'])
        val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['is_biased'])
        
        # Save splits
        train_data.to_pickle(self.cache_dir / "train.pkl")
        val_data.to_pickle(self.cache_dir / "val.pkl")
        test_data.to_pickle(self.cache_dir / "test.pkl")
        
        # Save dataset statistics
        stats = {
            'total_samples': len(combined_data),
            'biased_samples': combined_data['is_biased'].sum(),
            'non_biased_samples': len(combined_data) - combined_data['is_biased'].sum(),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'source_distribution': combined_data['source'].value_counts().to_dict()
        }
        
        with open(self.cache_dir / 'dataset_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        
        self.logger.info("\nDataset statistics:")
        self.logger.info(f"Total samples: {stats['total_samples']}")
        self.logger.info(f"Biased samples: {stats['biased_samples']}")
        self.logger.info(f"Non-biased samples: {stats['non_biased_samples']}")
        self.logger.info(f"Train samples: {stats['train_samples']}")
        self.logger.info(f"Validation samples: {stats['val_samples']}")
        self.logger.info(f"Test samples: {stats['test_samples']}")
        
        # Clear memory
        gc.collect()

if __name__ == "__main__":
    # Initialize processor
    processor = BiasDataProcessor(
        data_dir="data",  # Directory containing the CSV files
        cache_dir="processed_data"  # Directory to save processed data
    )
    
    # Process and save data
    processor.process_and_save_data()