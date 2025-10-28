"""
Advanced data processing module for pseudo-civility detection.
Handles multiple datasets, preprocessing, and augmentation.
"""
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Import with fallbacks
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available. Chinese text processing will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class DataExample:
    """Enhanced data container with metadata."""
    text: str
    label: int
    source: str = ""
    original_label: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


class TextPreprocessor:
    """Advanced text preprocessing with support for multiple languages."""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 512,
                 remove_duplicates: bool = True,
                 normalize_text: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_extra_whitespace: bool = True):
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.normalize_text = normalize_text
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Strip and validate length
        text = text.strip()
        
        if len(text) < self.min_length or len(text) > self.max_length:
            return ""
        
        return text
    
    def preprocess_chinese(self, text: str) -> str:
        """Preprocess Chinese text with jieba if available."""
        if not JIEBA_AVAILABLE:
            return text
        
        # Basic Chinese text cleaning
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s.,!?;:]', ' ', text)
        return text.strip()
    
    def preprocess_english(self, text: str) -> str:
        """Preprocess English text."""
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', ' ', text)
        return text.strip()
    
    def preprocess(self, text: str, language: str = "auto") -> str:
        """Main preprocessing pipeline."""
        # Basic cleaning
        text = self.clean_text(text)
        if not text:
            return ""
        
        # Language-specific preprocessing
        if language == "chinese" or (language == "auto" and self._is_chinese(text)):
            text = self.preprocess_chinese(text)
        elif language == "english" or (language == "auto" and self._is_english(text)):
            text = self.preprocess_english(text)
        
        return text
    
    def _is_chinese(self, text: str) -> bool:
        """Detect if text is primarily Chinese."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars > len(text) * 0.3
    
    def _is_english(self, text: str) -> bool:
        """Detect if text is primarily English."""
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        return english_chars > len(text) * 0.5


class DatasetLoader:
    """Enhanced dataset loader with support for multiple sources."""
    
    def __init__(self, archive_root: str = "Archive"):
        self.archive_root = Path(archive_root)
        self.preprocessor = TextPreprocessor()
    
    def load_chnsenticorp(self, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load ChnSentiCorp dataset."""
        data_path = self.archive_root / "chinesenlpdataset" / "ChnSentiCorp" / "ChnSentiCorp"
        examples = []
        
        for split in ["train", "dev", "test"]:
            file_path = data_path / f"{split}.tsv"
            if not file_path.exists():
                continue
            
            logger.info(f"Loading ChnSentiCorp {split} from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                f.readline()  # Skip header
                
                for line in f:
                    if max_samples and len(examples) >= max_samples:
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    label_idx = 0
                    text_start = 1
                    first = parts[0].strip()
                    second = parts[1].strip() if len(parts) > 1 else ""
                    # Some splits contain an ID column before the label
                    if len(parts) >= 3:
                        if not first.isdigit() or int(first) not in (0, 1):
                            if second.isdigit() and int(second) in (0, 1):
                                label_idx = 1
                                text_start = 2
                    original_label = parts[label_idx].strip()
                    try:
                        label = int(original_label)
                    except ValueError:
                        continue
                    if label not in (0, 1):
                        # Treat non-binary labels as negative/positive using sign heuristic
                        label = 1 if label > 0 else 0
                    text = '\t'.join(parts[text_start:]).strip()
                    text = self.preprocessor.preprocess(text, language="chinese")
                    
                    if text:
                        examples.append(DataExample(
                            text=text,
                            label=label,
                            source="chnsenticorp",
                            original_label=original_label
                        ))
        
        logger.info(f"Loaded {len(examples)} examples from ChnSentiCorp")
        return examples
    
    def load_wikipedia_politeness(self, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load Wikipedia politeness dataset."""
        try:
            from convokit import Corpus, download
        except ImportError:
            logger.error("convokit not available. Install with: pip install convokit")
            return []
        
        examples = []
        
        # Try to load from local CSV first
        csv_path = self.archive_root / "datasets" / "wikipedia_politeness"
        if csv_path.exists():
            examples = self._load_from_csv(csv_path, "wikipedia_politeness", max_samples)
        else:
            # Download and process from convokit
            logger.info("Downloading Wikipedia politeness corpus...")
            corpus_path = download("wikipedia-politeness-corpus")
            corpus = Corpus(filename=corpus_path)
            
            for utt in corpus.iter_utterances():
                if max_samples and len(examples) >= max_samples:
                    break
                
                text = getattr(utt, 'text', None)
                meta = getattr(utt, 'meta', {}) or {}
                
                if not text:
                    continue
                
                # Extract label from metadata
                label = self._extract_politeness_label(meta)
                if label is None:
                    continue
                
                text = self.preprocessor.preprocess(text, language="english")
                if text:
                    examples.append(DataExample(
                        text=text,
                        label=label,
                        source="wikipedia_politeness",
                        original_label=str(label),
                        metadata=meta
                    ))
        
        logger.info(f"Loaded {len(examples)} examples from Wikipedia politeness")
        return examples
    
    def load_go_emotions(self, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load GoEmotions dataset and map to binary sentiment."""
        examples = []
        
        # Try local CSV first
        csv_path = self.archive_root / "datasets" / "go_emotions"
        if csv_path.exists():
            examples = self._load_from_csv(csv_path, "go_emotions", max_samples)
        else:
            # Load from HuggingFace
            logger.info("Loading GoEmotions from HuggingFace...")
            ds = load_dataset("SetFit/go_emotions")
            
            # Emotion mappings
            positive_emotions = {1, 2, 4, 6, 7, 9, 13, 16, 17, 22, 23}
            negative_emotions = {0, 3, 5, 8, 10, 11, 12, 14, 15, 18, 19, 20, 21, 24, 25, 26}
            
            for split_name in ["train", "validation", "test"]:
                if split_name not in ds:
                    continue
                
                split = ds[split_name]
                # Use simple iteration to avoid type issues
                for i, record in enumerate(split):
                    if max_samples and len(examples) >= max_samples:
                        break
                    
                    text = str(record["text"]).strip()
                    labels = record["labels"]
                    
                    if not labels:
                        continue
                    
                    # Map emotions to binary sentiment
                    pos_count = sum(1 for l in labels if l in positive_emotions)
                    neg_count = sum(1 for l in labels if l in negative_emotions)
                    
                    if pos_count > neg_count:
                        label = 1  # Positive
                    elif neg_count > pos_count:
                        label = 0  # Negative
                    else:
                        continue  # Ambiguous or neutral
                    
                    text = self.preprocessor.preprocess(text, language="english")
                    if text:
                        examples.append(DataExample(
                            text=text,
                            label=label,
                            source="go_emotions",
                            original_label=str(labels),
                            metadata={"split": split_name}
                        ))
        
        logger.info(f"Loaded {len(examples)} examples from GoEmotions")
        return examples
    
    def load_civil_comments(self, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load Civil Comments dataset."""
        examples = []
        
        # Try local CSV first
        csv_path = self.archive_root / "datasets" / "civil_comments"
        if csv_path.exists():
            examples = self._load_from_csv(csv_path, "civil_comments", max_samples)
        else:
            # Load from HuggingFace
            logger.info("Loading Civil Comments from HuggingFace...")
            ds = load_dataset("google/civil_comments")
            
            for split_name in ["train", "validation", "test"]:
                if split_name not in ds:
                    continue
                
                split = ds[split_name]
                for record in split:
                    if max_samples and len(examples) >= max_samples:
                        break
                    
                    text = record["text"].strip()
                    toxicity = float(record["toxicity"])
                    label = 1 if toxicity >= 0.5 else 0
                    
                    text = self.preprocessor.preprocess(text, language="english")
                    if text:
                        examples.append(DataExample(
                            text=text,
                            label=label,
                            source="civil_comments",
                            original_label=str(toxicity),
                            metadata={"toxicity_score": toxicity, "split": split_name}
                        ))
        
        logger.info(f"Loaded {len(examples)} examples from Civil Comments")
        return examples
    
    def load_toxigen(self, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load ToxiGen dataset."""
        examples = []
        
        # Try local CSV first
        csv_path = self.archive_root / "datasets" / "toxigen"
        if csv_path.exists():
            examples = self._load_from_csv(csv_path, "toxigen", max_samples)
        else:
            # Load from HuggingFace
            logger.info("Loading ToxiGen from HuggingFace...")
            ds = load_dataset("toxigen/toxigen-data", "annotations")
            
            split = ds["train"]
            for record in split:
                if max_samples and len(examples) >= max_samples:
                    break
                
                text = record["Input.text"]
                label = int(record["Input.binary_prompt_label"] == 1)
                
                text = self.preprocessor.preprocess(text, language="english")
                if text:
                    examples.append(DataExample(
                        text=text,
                        label=label,
                        source="toxigen",
                        original_label=record["Input.binary_prompt_label"],
                        metadata={"split": "train"}
                    ))
        
        logger.info(f"Loaded {len(examples)} examples from ToxiGen")
        return examples
    
    def _load_from_csv(self, csv_path: Path, dataset_name: str, max_samples: Optional[int] = None) -> List[DataExample]:
        """Load dataset from local CSV files."""
        examples = []
        
        for split in ["train", "dev", "test"]:
            file_path = csv_path / f"{split}.csv"
            if not file_path.exists():
                continue
            
            logger.info(f"Loading {dataset_name} {split} from {file_path}")
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                if max_samples and len(examples) >= max_samples:
                    break
                
                text = str(row.get('text', '')).strip()
                label = int(row.get('label', 0))
                
                text = self.preprocessor.preprocess(text)
                if text:
                    examples.append(DataExample(
                        text=text,
                        label=label,
                        source=dataset_name,
                        original_label=str(label),
                        metadata={"split": split}
                    ))
        
        return examples
    
    def _extract_politeness_label(self, meta: Dict) -> Optional[int]:
        """Extract politeness label from metadata."""
        if not isinstance(meta, dict):
            return None
        
        # Try different label formats
        if 'Binary' in meta:
            val = meta['Binary']
            if isinstance(val, dict):
                if 'polite' in val:
                    return 1 if bool(val['polite']) else 0
                elif 'impolite' in val:
                    return 0 if bool(val['impolite']) else 1
            elif isinstance(val, (str, int, bool)):
                s = str(val).lower()
                if s in ('polite', '1', 'true'):
                    return 1
                elif s in ('impolite', '0', 'false'):
                    return 0
        
        if 'politeness_label' in meta:
            s = str(meta['politeness_label']).lower()
            return 1 if 'polite' in s else 0 if 'impolite' in s else None
        
        if 'is_polite' in meta:
            return 1 if bool(meta['is_polite']) else 0
        
        return None


class DataBalancer:
    """Advanced data balancing and augmentation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def balance_dataset(self, examples: List[DataExample], strategy: str = "undersample") -> List[DataExample]:
        """Balance dataset using different strategies."""
        if not examples:
            return examples
        
        # Group by label
        label_groups = {}
        for ex in examples:
            if ex.label not in label_groups:
                label_groups[ex.label] = []
            label_groups[ex.label].append(ex)
        
        if len(label_groups) < 2:
            return examples  # Already balanced or single class
        
        # Find minority and majority classes
        class_counts = {label: len(group) for label, group in label_groups.items()}
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        logger.info(f"Class distribution before balancing: {class_counts}")
        
        if strategy == "undersample":
            # Undersample majority classes
            balanced_examples = []
            for label, group in label_groups.items():
                if len(group) > min_count:
                    sampled = resample(group, n_samples=min_count, random_state=self.random_state)
                    balanced_examples.extend(sampled)
                else:
                    balanced_examples.extend(group)
        
        elif strategy == "oversample":
            # Oversample minority classes
            balanced_examples = []
            for label, group in label_groups.items():
                if len(group) < max_count:
                    # Calculate how many times to repeat
                    repeat_factor = max_count // len(group)
                    remainder = max_count % len(group)
                    
                    # Repeat the group
                    oversampled = group * repeat_factor
                    if remainder > 0:
                        oversampled.extend(resample(group, n_samples=remainder, random_state=self.random_state))
                    
                    balanced_examples.extend(oversampled)
                else:
                    balanced_examples.extend(group)
        
        elif strategy == "hybrid":
            # Hybrid: undersample majority to 2x minority, then oversample minority
            target_count = min_count * 2
            balanced_examples = []
            
            for label, group in label_groups.items():
                if len(group) > target_count:
                    sampled = resample(group, n_samples=target_count, random_state=self.random_state)
                    balanced_examples.extend(sampled)
                else:
                    balanced_examples.extend(group)
            
            # Now balance to the new max
            new_label_groups = {}
            for ex in balanced_examples:
                if ex.label not in new_label_groups:
                    new_label_groups[ex.label] = []
                new_label_groups[ex.label].append(ex)
            
            new_max_count = max(len(group) for group in new_label_groups.values())
            final_examples = []
            
            for label, group in new_label_groups.items():
                if len(group) < new_max_count:
                    repeat_factor = new_max_count // len(group)
                    remainder = new_max_count % len(group)
                    
                    oversampled = group * repeat_factor
                    if remainder > 0:
                        oversampled.extend(resample(group, n_samples=remainder, random_state=self.random_state))
                    
                    final_examples.extend(oversampled)
                else:
                    final_examples.extend(group)
            
            balanced_examples = final_examples
        
        else:
            logger.warning(f"Unknown balancing strategy: {strategy}")
            return examples
        
        # Shuffle the balanced dataset
        random.shuffle(balanced_examples)
        
        new_class_counts = {}
        for ex in balanced_examples:
            new_class_counts[ex.label] = new_class_counts.get(ex.label, 0) + 1
        
        logger.info(f"Class distribution after balancing: {new_class_counts}")
        
        return balanced_examples
    
    def remove_duplicates(self, examples: List[DataExample]) -> List[DataExample]:
        """Remove duplicate examples based on text content."""
        seen_texts = set()
        unique_examples = []
        
        for ex in examples:
            if ex.text not in seen_texts:
                seen_texts.add(ex.text)
                unique_examples.append(ex)
        
        logger.info(f"Removed {len(examples) - len(unique_examples)} duplicate examples")
        return unique_examples


class DataSplitter:
    """Advanced data splitting with stratification and metadata preservation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _can_stratify(labels: List[int], split_size: Union[float, int], total_size: int) -> bool:
        """Check if stratified splitting is feasible."""
        if not labels or total_size == 0:
            return False
        label_counts = Counter(labels)
        if len(label_counts) < 2:
            return False
        if min(label_counts.values()) < 2:
            return False
        if isinstance(split_size, float):
            target = int(round(split_size * total_size))
        else:
            target = int(split_size)
        target = max(target, 1)
        if target < len(label_counts):
            return False
        return True
    
    def split_data(self, 
                   examples: List[DataExample], 
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   stratify: bool = True) -> Tuple[List[DataExample], List[DataExample], List[DataExample]]:
        """Split data into train, validation, and test sets."""
        if not examples:
            return [], [], []
        
        texts = [ex.text for ex in examples]
        labels = [ex.label for ex in examples]
        total_samples = len(labels)
        
        # First split: train+val vs test
        stratify_labels = labels if self._can_stratify(labels, test_size, total_samples) and stratify else None
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        # Second split: train vs val
        if val_size > 0:
            # Adjust val_size relative to trainval size
            adjusted_val_size = val_size / (1 - test_size)
            
            stratify_trainval = y_trainval if self._can_stratify(y_trainval, adjusted_val_size, len(y_trainval)) and stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=adjusted_val_size,
                random_state=self.random_state,
                stratify=stratify_trainval
            )
        else:
            X_train, X_val, y_train, y_val = X_trainval, [], y_trainval, []
        
        # Reconstruct DataExample objects
        def reconstruct_examples(texts_list, labels_list):
            examples_dict = {(ex.text, ex.label): ex for ex in examples}
            reconstructed = []
            for text, label in zip(texts_list, labels_list):
                key = (text, label)
                if key in examples_dict:
                    reconstructed.append(examples_dict[key])
                else:
                    # Fallback: create new example
                    reconstructed.append(DataExample(text=text, label=label))
            return reconstructed
        
        train_examples = reconstruct_examples(X_train, y_train)
        val_examples = reconstruct_examples(X_val, y_val) if X_val else []
        test_examples = reconstruct_examples(X_test, y_test)
        
        logger.info(f"Data split - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        return train_examples, val_examples, test_examples


class DataProcessor:
    """Main data processing orchestrator."""
    
    def __init__(self, config):
        self.config = config
        self.loader = DatasetLoader(config.data.archive_root)
        self.balancer = DataBalancer(config.data.random_state)
        self.splitter = DataSplitter(config.data.random_state)
        
        # Update preprocessor with config
        self.loader.preprocessor.min_length = config.data.min_text_length
        self.loader.preprocessor.max_length = config.data.max_text_length
        self.loader.preprocessor.remove_duplicates = config.data.remove_duplicates
    
    def load_dataset(self, dataset_name: str) -> Dict[str, List[DataExample]]:
        """Load, preprocess, balance, and split a single dataset."""
        loader_map = {
            "chnsenticorp": self.loader.load_chnsenticorp,
            "wikipedia_politeness": self.loader.load_wikipedia_politeness,
            "go_emotions": self.loader.load_go_emotions,
            "civil_comments": self.loader.load_civil_comments,
            "toxigen": self.loader.load_toxigen,
        }
        if dataset_name not in loader_map:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return {}
        try:
            examples = loader_map[dataset_name](self.config.data.max_samples)
        except Exception as exc:
            logger.error(f"Failed to load {dataset_name}: {exc}")
            return {}
        if not examples:
            logger.warning(f"No samples found for dataset {dataset_name}")
            return {}

        logger.info(f"Processing dataset {dataset_name} with {len(examples)} raw samples")

        processed_examples = examples
        if self.config.data.remove_duplicates:
            processed_examples = self.balancer.remove_duplicates(processed_examples)
        if self.config.data.balance_classes:
            processed_examples = self.balancer.balance_dataset(processed_examples, strategy="hybrid")

        train_examples, val_examples, test_examples = self.splitter.split_data(
            processed_examples,
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size
        )

        logger.info(
            "Dataset %s splits - Train: %d, Val: %d, Test: %d",
            dataset_name,
            len(train_examples),
            len(val_examples),
            len(test_examples)
        )

        return {
            "train": train_examples,
            "val": val_examples,
            "test": test_examples,
        }

    def load_datasets_individually(self) -> Dict[str, Dict[str, List[DataExample]]]:
        """Load all configured datasets individually and return their splits."""
        dataset_splits: Dict[str, Dict[str, List[DataExample]]] = {}
        for dataset_name in self.config.data.datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            splits = self.load_dataset(dataset_name)
            if splits:
                dataset_splits[dataset_name] = splits
        if not dataset_splits:
            raise ValueError("No examples loaded from any dataset")
        return dataset_splits

    def load_and_process_datasets(self) -> Tuple[List[DataExample], List[DataExample], List[DataExample]]:
        """Load and process all configured datasets."""
        dataset_splits = self.load_datasets_individually()
        train_examples: List[DataExample] = []
        val_examples: List[DataExample] = []
        test_examples: List[DataExample] = []

        for splits in dataset_splits.values():
            train_examples.extend(splits.get("train", []))
            val_examples.extend(splits.get("val", []))
            test_examples.extend(splits.get("test", []))

        return train_examples, val_examples, test_examples
    
    def get_data_statistics(self, train_examples: List[DataExample], 
                           val_examples: List[DataExample], 
                           test_examples: List[DataExample]) -> Dict:
        """Generate comprehensive data statistics."""
        def analyze_split(examples: List[DataExample], split_name: str) -> Dict:
            if not examples:
                return {"name": split_name, "count": 0, "class_distribution": {}}
            
            class_counts = {}
            source_counts = {}
            text_lengths = []
            
            for ex in examples:
                # Class distribution
                class_counts[ex.label] = class_counts.get(ex.label, 0) + 1
                
                # Source distribution
                source_counts[ex.source] = source_counts.get(ex.source, 0) + 1
                
                # Text length statistics
                text_lengths.append(len(ex.text))
            
            return {
                "name": split_name,
                "count": len(examples),
                "class_distribution": class_counts,
                "source_distribution": source_counts,
                "avg_text_length": np.mean(text_lengths) if text_lengths else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "std_text_length": np.std(text_lengths) if text_lengths else 0
            }
        
        stats = {
            "train": analyze_split(train_examples, "train"),
            "val": analyze_split(val_examples, "validation"),
            "test": analyze_split(test_examples, "test"),
            "total": {
                "count": len(train_examples) + len(val_examples) + len(test_examples)
            }
        }
        
        return stats

    def get_dataset_statistics(self, dataset_splits: Dict[str, Dict[str, List[DataExample]]]) -> Dict[str, Dict]:
        """Generate statistics per dataset."""
        detailed_stats: Dict[str, Dict] = {}
        for name, splits in dataset_splits.items():
            detailed_stats[name] = self.get_data_statistics(
                splits.get("train", []),
                splits.get("val", []),
                splits.get("test", [])
            )
        return detailed_stats
