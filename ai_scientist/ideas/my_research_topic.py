import warnings
from datetime import datetime
import numpy as np
import time  # For measuring execution time

warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

from datasets import load_dataset
from torchvision.models import resnet50
from huggingface_hub import login
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

## CONSTANTS AND CONFIGURATION

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 4
NUM_EPOCHS = 30
STEPS_TO_LOG = 25
NUM_TEST_BATCHES = 20
WARMUP_EPOCHS = 2
EXPERIMENT_NAME = "vlm_ood_detection"

# VLM models to evaluate
VLM_MODELS = {
    "CLIP_VIT_B32": "openai/clip-vit-base-patch32",
    "CLIP_VIT_L14": "openai/clip-vit-large-patch14",
    "DINOV2_VIT_B": "facebook/dinov2-base",
    "DINOV2_VIT_L": "facebook/dinov2-large",
    "RAD_DINO": "microsoft/rad-dino"  # Medical domain specialized model
}

# Create timestamp for experiment
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"{EXPERIMENT_NAME}_{timestamp}"
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(experiment_dir, "embeddings"), exist_ok=True)
os.makedirs(os.path.join(experiment_dir, "figures"), exist_ok=True)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## DATASET AND DATA LOADING

class AircraftDataset(Dataset):
    """Dataset for aircraft images with hierarchical labels"""
    def __init__(self, data, transform=None, known_classes=None):
        """
        Args:
            data: List of dicts with image path, manufacturer, model, variant
            transform: Transform to apply to images
            known_classes: Set of known class IDs (variants) for OOD setup
        """
        self.data = data
        self.transform = transform
        self.known_classes = known_classes
        
        # Filter for known classes if specified
        if known_classes is not None:
            self.data = [d for d in data if d['variant_id'] in known_classes]
            
        # Create label mappings
        self.manufacturer_to_idx = self._create_mapping([d['manufacturer'] for d in self.data])
        self.model_to_idx = self._create_mapping([d['model'] for d in self.data])
        self.variant_to_idx = self._create_mapping([d['variant'] for d in self.data])

    def _create_mapping(self, labels):
        """Create mapping from label to index"""
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Get hierarchical labels
        manufacturer_idx = self.manufacturer_to_idx[item['manufacturer']]
        model_idx = self.model_to_idx[item['model']]
        variant_idx = self.variant_to_idx[item['variant']]
        
        return {
            'image': img,
            'manufacturer_idx': manufacturer_idx,
            'model_idx': model_idx, 
            'variant_idx': variant_idx,
            'manufacturer': item['manufacturer'],
            'model': item['model'],
            'variant': item['variant'],
            'variant_id': item['variant_id']
        }

# Define image transformations
transform_train = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_aircraft_dataset(data_path):
    """
    Load aircraft dataset from path
    Returns train/val/test splits and metadata
    """
    # In a real implementation, this would load actual data
    # Here we mock the data structure
    print(f"Loading aircraft dataset from: {data_path}")
    
    # Mock data structure (in real implementation, load from files)
    manufacturers = ['Boeing', 'Airbus', 'Embraer', 'Bombardier', 'Cessna', 'Gulfstream']
    
    # Create hierarchical structure: manufacturer -> models -> variants
    aircraft_data = []
    variant_id = 0
    
    for manuf in manufacturers:
        # Each manufacturer has 3-5 models
        num_models = np.random.randint(3, 6)
        for model_idx in range(num_models):
            model_name = f"{manuf} {model_idx+1}00"
            
            # Each model has 2-4 variants
            num_variants = np.random.randint(2, 5)
            for variant_idx in range(num_variants):
                variant_name = f"{model_name}-{variant_idx+1}00"
                
                # Each variant has 20-50 images
                num_images = np.random.randint(20, 51)
                for img_idx in range(num_images):
                    # In a real implementation, this would be actual image paths
                    image_path = f"{data_path}/{manuf}/{model_name}/{variant_name}/{img_idx}.jpg"
                    
                    aircraft_data.append({
                        'image_path': image_path,
                        'manufacturer': manuf,
                        'model': model_name,
                        'variant': variant_name,
                        'variant_id': variant_id
                    })
                variant_id += 1
    
    # Split into train/val/test (70%/15%/15%)
    np.random.shuffle(aircraft_data)
    train_size = int(0.7 * len(aircraft_data))
    val_size = int(0.15 * len(aircraft_data))
    
    train_data = aircraft_data[:train_size]
    val_data = aircraft_data[train_size:train_size+val_size]
    test_data = aircraft_data[train_size+val_size:]
    
    # Extract all unique variant IDs
    all_variant_ids = set([d['variant_id'] for d in aircraft_data])
    
    # Extract metadata for reference
    metadata = {
        'num_manufacturers': len(manufacturers),
        'num_models': variant_id,
        'num_images': len(aircraft_data),
        'all_variant_ids': all_variant_ids
    }
    
    print(f"Dataset loaded: {metadata['num_images']} images across {metadata['num_manufacturers']} manufacturers")
    return train_data, val_data, test_data, metadata

## VLM EMBEDDING EXTRACTION

def initialize_vlm(model_name):
    """Initialize a vision-language model for embedding extraction"""
    print(f"Initializing model: {model_name}")
    
    feature_extractor = pipeline(
        task="image-feature-extraction",
        model=model_name,
        device=device if str(device) != "cpu" else -1,  # -1 for CPU
    )
    
    return feature_extractor

def create_text_prompts(labels, prompt_type='basic'):
    """
    Create text prompts for different labels
    prompt_type: 'basic' or 'enhanced'
    """
    prompts = {}
    
    if prompt_type == 'basic':
        for label in labels:
            prompts[label] = f"A photo of a {label}"
    
    elif prompt_type == 'enhanced':
        # In real implementation, this would include detailed descriptions
        # For this example, we use a simplified enhanced prompt
        for label in labels:
            if "Boeing" in label:
                prompts[label] = f"A photo of a {label}, a commercial passenger aircraft with distinctive wing design"
            elif "Airbus" in label:
                prompts[label] = f"A photo of a {label}, a modern commercial aircraft with characteristic cockpit windows"
            elif "Embraer" in label:
                prompts[label] = f"A photo of a {label}, a regional jet aircraft with T-tail configuration"
            else:
                prompts[label] = f"A photo of a {label}, an aircraft used for transportation"
    
    return prompts

def extract_embeddings(model, dataloader, level='variant', prompt_type=None):
    """
    Extract embeddings using the specified VLM
    
    Args:
        model: VLM model
        dataloader: DataLoader with images
        level: 'manufacturer', 'model', or 'variant'
        prompt_type: None for direct embedding, 'basic' or 'enhanced' for text-guided embedding
    
    Returns:
        embeddings: numpy array of embeddings
        labels: corresponding labels
        ids: variant IDs
    """
    embeddings = []
    labels = []
    ids = []
    
    # Create text prompts if needed
    if prompt_type:
        unique_labels = set()
        for batch in dataloader:
            unique_labels.update(batch[level])
        text_prompts = create_text_prompts(unique_labels, prompt_type)
    
    print(f"Extracting {level}-level embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image']
            batch_labels = batch[level]
            batch_ids = batch['variant_id'].numpy()
            
            # Process images in batch
            for i in range(len(images)):
                img = T.ToPILImage()(images[i])
                
                # Extract embeddings based on prompt type
                if prompt_type:
                    label = batch_labels[i]
                    prompt = text_prompts[label]
                    embedding = model(img, prompt)[0]
                else:
                    embedding = model(img)[0]
                
                # Store results
                embeddings.append(embedding)
                labels.append(batch_labels[i])
                ids.append(batch_ids[i])
    
    return np.array(embeddings), labels, np.array(ids)

## OOD DETECTION METHODS

def knn_ood_detection(train_embeddings, train_labels, test_embeddings, k=5):
    """
    OOD detection using k-nearest neighbors distance
    Returns anomaly scores where higher = more likely to be OOD
    """
    # Fit KNN model on training embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings)
    
    # Calculate distances to k nearest neighbors
    distances, _ = knn.kneighbors(test_embeddings)
    
    # Use mean distance as anomaly score
    ood_scores = distances.mean(axis=1)
    
    return ood_scores

def mahalanobis_ood_detection(train_embeddings, train_labels, test_embeddings):
    """
    OOD detection using Mahalanobis distance
    Returns anomaly scores where higher = more likely to be OOD
    """
    # Calculate class-wise mean and shared covariance
    unique_labels = np.unique(train_labels)
    class_means = {}
    
    all_centered = []
    for label in unique_labels:
        class_indices = np.where(np.array(train_labels) == label)[0]
        class_embeddings = train_embeddings[class_indices]
        class_means[label] = np.mean(class_embeddings, axis=0)
        
        # Center the embeddings for covariance calculation
        centered = class_embeddings - class_means[label]
        all_centered.append(centered)
    
    # Calculate shared covariance
    all_centered = np.vstack(all_centered)
    cov = np.cov(all_centered, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    
    # Calculate Mahalanobis distance for each test sample
    ood_scores = []
    for embedding in test_embeddings:
        # Find minimum distance to any class centroid
        min_dist = float('inf')
        for label in unique_labels:
            diff = embedding - class_means[label]
            dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
            min_dist = min(min_dist, dist)
        ood_scores.append(min_dist)
    
    return np.array(ood_scores)

def evaluate_ood_detection(ood_scores, is_ood):
    """
    Evaluate OOD detection performance
    
    Args:
        ood_scores: OOD anomaly scores
        is_ood: Binary labels (1 for OOD, 0 for in-distribution)
    
    Returns:
        metrics: Dict with AUROC, FPR@95, AUPR
    """
    # Calculate AUROC
    auroc = roc_auc_score(is_ood, ood_scores)
    
    # Calculate FPR@95% TPR
    fpr, tpr, thresholds = roc_curve(is_ood, ood_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = fpr[idx]
    
    # Calculate AUPR
    precision, recall, _ = precision_recall_curve(is_ood, ood_scores)
    aupr = auc(recall, precision)
    
    return {
        'auroc': auroc, 
        'fpr@95': fpr_at_95, 
        'aupr': aupr
    }

def roc_curve(y_true, y_score):
    """Calculate ROC curve points"""
    # Sort scores and corresponding labels
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Count positive labels
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    
    # Accumulate true positives and false positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    # Calculate TPR and FPR
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    thresholds = np.concatenate([[np.inf], y_score])
    
    return fpr, tpr, thresholds

## VISUALIZATION FUNCTIONS

def plot_embeddings_2d(embeddings, labels, title, save_path):
    """
    Plot 2D visualization of embeddings using TSNE
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        title: Plot title
        save_path: Path to save figure
    """
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Convert labels to colors
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    label_ids = [label_to_id[label] for label in labels]
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=label_ids, cmap='tab20', alpha=0.7, s=30)
    
    # Add legend and labels
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ood_histogram(ood_scores, is_ood, title, save_path):
    """
    Plot histogram of OOD scores for in-distribution and OOD samples
    
    Args:
        ood_scores: OOD anomaly scores
        is_ood: Binary labels (1 for OOD, 0 for in-distribution)
        title: Plot title
        save_path: Path to save figure
    """
    in_dist_scores = ood_scores[is_ood == 0]
    ood_sample_scores = ood_scores[is_ood == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(in_dist_scores, bins=50, alpha=0.7, label='In-Distribution', density=True)
    plt.hist(ood_sample_scores, bins=50, alpha=0.7, label='OOD', density=True)
    plt.xlabel('OOD Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(results, metric='auroc', title=None, save_path=None):
    """
    Create bar chart comparing model performance
    
    Args:
        results: Dict of performance results
        metric: Metric to plot ('auroc', 'fpr@95', or 'aupr')
        title: Plot title
        save_path: Path to save figure
    """
    # Extract data for plotting
    models = []
    values = []
    categories = []
    
    for model, model_results in results.items():
        for granularity, metrics in model_results.items():
            models.append(model)
            categories.append(granularity)
            values.append(metrics[metric])
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Model': models,
        'Granularity': categories, 
        metric.upper(): values
    })
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y=metric.upper(), hue='Granularity', data=df)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f"{metric.upper()} Comparison Across Models and Granularity Levels", fontsize=16)
        
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric.upper(), fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Granularity')
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

## DATA COLLATION

def custom_collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    manufacturer_idx = torch.tensor([item['manufacturer_idx'] for item in batch])
    model_idx = torch.tensor([item['model_idx'] for item in batch])
    variant_idx = torch.tensor([item['variant_idx'] for item in batch])
    variant_id = torch.tensor([item['variant_id'] for item in batch])
    
    manufacturers = [item['manufacturer'] for item in batch]
    models = [item['model'] for item in batch]
    variants = [item['variant'] for item in batch]
    
    return {
        'image': images,
        'manufacturer_idx': manufacturer_idx,
        'model_idx': model_idx,
        'variant_idx': variant_idx,
        'variant_id': variant_id,
        'manufacturer': manufacturers,
        'model': models,
        'variant': variants
    }

## HYBRID OOD DETECTION APPROACH

class HybridOODDetector:
    """
    Hybrid approach for OOD detection combining embedding similarity
    with classifier confidence
    """
    def __init__(self, models, classifier, alpha=0.6):
        """
        Args:
            models: Dict of VLM models
            classifier: Trained classifier for confidence scores
            alpha: Weight for embedding score vs confidence score
        """
        self.models = models
        self.classifier = classifier
        self.alpha = alpha
        
        # Storage for class statistics
        self.class_means = {}
        self.inv_cov = None
        
    def fit(self, train_embeddings, train_labels):
        """
        Calculate class statistics from training data
        
        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
        """
        # Calculate class-wise means
        unique_labels = np.unique(train_labels)
        all_centered = []
        
        for label in unique_labels:
            class_indices = np.where(np.array(train_labels) == label)[0]
            class_embeddings = train_embeddings[class_indices]
            self.class_means[label] = np.mean(class_embeddings, axis=0)
            
            # Center embeddings for covariance calculation
            centered = class_embeddings - self.class_means[label]
            all_centered.append(centered)
        
        # Calculate shared covariance
        all_centered = np.vstack(all_centered)
        cov = np.cov(all_centered, rowvar=False)
        self.inv_cov = np.linalg.pinv(cov)
        
        return self
    
    def calculate_embedding_score(self, embedding):
        """
        Calculate embedding-based anomaly score using Mahalanobis distance
        
        Args:
            embedding: Sample embedding
            
        Returns:
            score: Anomaly score
        """
        # Find minimum distance to any class centroid
        min_dist = float('inf')
        for label, centroid in self.class_means.items():
            diff = embedding - centroid
            dist = np.sqrt(np.dot(np.dot(diff, self.inv_cov), diff))
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def predict(self, image):
        """
        Predict OOD score for an image
        
        Args:
            image: Input image
            
        Returns:
            ood_score: Anomaly score (higher = more likely OOD)
        """
        # 1. Extract embeddings from multiple VLMs
        embeddings = []
        for model_name, model in self.models.items():
            embedding = model(image)[0]
            embeddings.append(embedding)
        
        combined_embedding = np.concatenate(embeddings)
        
        # 2. Calculate embedding similarity score
        emb_score = self.calculate_embedding_score(combined_embedding)
        
        # 3. Get classifier confidence
        with torch.no_grad():
            outputs = self.classifier(image)
            confidence = torch.max(torch.softmax(outputs, dim=1)).item()
        
        # 4. Combine scores (higher = more likely OOD)
        ood_score = self.alpha * emb_score + (1 - self.alpha) * (1 - confidence)
        
        return ood_score

## MAIN EXPERIMENT CODE

def create_ood_scenarios(train_data, val_data, test_data, metadata):
    """
    Create different OOD detection scenarios
    
    Args:
        train_data, val_data, test_data: Dataset splits
        metadata: Dataset metadata
    
    Returns:
        scenarios: Dict of OOD scenarios
    """
    all_variant_ids = metadata['all_variant_ids']
    
    # For experiment, we'll use 80% of variant IDs as known classes
    num_known = int(0.8 * len(all_variant_ids))
    all_variant_ids = list(all_variant_ids)
    np.random.shuffle(all_variant_ids)
    
    known_variant_ids = set(all_variant_ids[:num_known])
    unknown_variant_ids = set(all_variant_ids[num_known:])
    
    print(f"Created OOD scenario with {len(known_variant_ids)} known variants and "
          f"{len(unknown_variant_ids)} unknown variants")
    
    # Create datasets for known classes
    known_train_dataset = AircraftDataset(train_data, transform=transform_train, 
                                         known_classes=known_variant_ids)
    known_val_dataset = AircraftDataset(val_data, transform=transform_test,
                                       known_classes=known_variant_ids)
    
    # Create combined test dataset (including both known and unknown classes)
    test_dataset = AircraftDataset(test_data, transform=transform_test)
    
    # Create test labels for OOD detection (1 = OOD, 0 = in-distribution)
    test_ood_labels = [1 if variant_id in unknown_variant_ids else 0 
                       for variant_id in [d['variant_id'] for d in test_data]]
    
    scenarios = {
        'known_train_dataset': known_train_dataset,
        'known_val_dataset': known_val_dataset,
        'test_dataset': test_dataset,
        'test_ood_labels': test_ood_labels,
        'known_variant_ids': known_variant_ids,
        'unknown_variant_ids': unknown_variant_ids
    }
    
    return scenarios

def run_vlm_experiment(scenario):
    """
    Run VLM embedding experiment for OOD detection
    
    Args:
        scenario: OOD scenario dict
    
    Returns:
        results: Experiment results
    """
    # Create data loaders
    train_loader = DataLoader(
        scenario['known_train_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        scenario['known_val_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        scenario['test_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Initialize results dictionary
    results = {
        model_name: {
            'manufacturer': {},
            'model': {},
            'variant': {}
        }
        for model_name in VLM_MODELS
    }
    
    # Run experiment for each VLM model
    for model_name, model_path in VLM_MODELS.items():
        print(f"\n--- Running experiment with {model_name} ---")
        
        # Initialize VLM
        try:
            vlm = initialize_vlm(model_path)
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
            continue
        
        # Extract embeddings at different granularity levels
        for level in ['manufacturer', 'model', 'variant']:
            print(f"\nProcessing {level}-level embeddings")
            
            # Standard embeddings (no prompt)
            train_embeddings, train_labels, _ = extract_embeddings(
                vlm, train_loader, level=level, prompt_type=None
            )
            
            test_embeddings, test_labels, test_ids = extract_embeddings(
                vlm, test_loader, level=level, prompt_type=None
            )
            
            # Enhanced prompt embeddings
            train_embeddings_enhanced, train_labels_enhanced, _ = extract_embeddings(
                vlm, train_loader, level=level, prompt_type='enhanced'
            )
            
            test_embeddings_enhanced, test_labels_enhanced, test_ids_enhanced = extract_embeddings(
                vlm, test_loader, level=level, prompt_type='enhanced'
            )
            
            # Save embeddings
            np.save(os.path.join(experiment_dir, 'embeddings', 
                                f"{model_name}_{level}_train.npy"), train_embeddings)
            np.save(os.path.join(experiment_dir, 'embeddings',
                                f"{model_name}_{level}_test.npy"), test_embeddings)
            
            # Create OOD labels
            is_ood = np.array([1 if id in scenario['unknown_variant_ids'] else 0 
                              for id in test_ids])
            
            # KNN-based OOD detection
            knn_scores = knn_ood_detection(train_embeddings, train_labels, test_embeddings)
            knn_metrics = evaluate_ood_detection(knn_scores, is_ood)
            
            # Mahalanobis-based OOD detection
            mahalanobis_scores = mahalanobis_ood_detection(train_embeddings, train_labels, test_embeddings)
            mahalanobis_metrics = evaluate_ood_detection(mahalanobis_scores, is_ood)
            
            # Enhanced prompt results
            knn_scores_enhanced = knn_ood_detection(train_embeddings_enhanced, train_labels_enhanced, test_embeddings_enhanced)
            knn_metrics_enhanced = evaluate_ood_detection(knn_scores_enhanced, is_ood)
            
            mahalanobis_scores_enhanced = mahalanobis_ood_detection(train_embeddings_enhanced, train_labels_enhanced, test_embeddings_enhanced)
            mahalanobis_metrics_enhanced = evaluate_ood_detection(mahalanobis_scores_enhanced, is_ood)
            
            # Store results
            results[model_name][level] = {
                'knn': knn_metrics,
                'mahalanobis': mahalanobis_metrics,
                'knn_enhanced': knn_metrics_enhanced,
                'mahalanobis_enhanced': mahalanobis_metrics_enhanced
            }
            
            # Visualize embeddings
            plot_embeddings_2d(
                test_embeddings, 
                test_labels,
                f"{model_name} {level}-level Embeddings",
                os.path.join(experiment_dir, 'figures', f"{model_name}_{level}_embeddings.png")
            )
            
            # Visualize OOD scores
            plot_ood_histogram(
                mahalanobis_scores,
                is_ood,
                f"{model_name} {level}-level Mahalanobis OOD Scores",
                os.path.join(experiment_dir, 'figures', f"{model_name}_{level}_mahalanobis_hist.png")
            )
            
            # Log results
            print(f"\n{level.capitalize()}-level OOD detection results for {model_name}:")
            print(f"  KNN Standard - AUROC: {knn_metrics['auroc']:.4f}, FPR@95: {knn_metrics['fpr@95']:.4f}")
            print(f"  Mahalanobis Standard - AUROC: {mahalanobis_metrics['auroc']:.4f}, FPR@95: {mahalanobis_metrics['fpr@95']:.4f}")
            print(f"  KNN Enhanced - AUROC: {knn_metrics_enhanced['auroc']:.4f}, FPR@95: {knn_metrics_enhanced['fpr@95']:.4f}")
            print(f"  Mahalanobis Enhanced - AUROC: {mahalanobis_metrics_enhanced['auroc']:.4f}, FPR@95: {mahalanobis_metrics_enhanced['fpr@95']:.4f}")
    
    return results

## ICIC RATIO CALCULATION

def calculate_icic_ratio(embeddings, labels):
    """
    Calculate Intra-Class over Inter-Class (ICIC) ratio
    
    Args:
        embeddings: Sample embeddings
        labels: Class labels
        
    Returns:
        icic_stats: Dict with mean, std, and max ICIC ratios
    """
    unique_labels = np.unique(labels)
    label_indices = {label: np.where(np.array(labels) == label)[0] for label in unique_labels}
    
    # Calculate centroids for each class
    centroids = {label: np.mean(embeddings[indices], axis=0) for label, indices in label_indices.items()}
    
    # Calculate ICIC ratios for each sample
    icic_ratios = []
    
    for sample_idx, (embedding, label) in enumerate(zip(embeddings, labels)):
        # Skip if there's only one sample in the class
        if len(label_indices[label]) <= 1:
            continue
        
        # Calculate mean intra-class distance (excluding self)
        intra_indices = [idx for idx in label_indices[label] if idx != sample_idx]
        if len(intra_indices) == 0:
            continue
            
        intra_class_distances = [cosine(embedding, embeddings[idx]) for idx in intra_indices]
        mean_intra_distance = np.mean(intra_class_distances)
        
        # Calculate mean distance to nearest different class
        nearest_inter_distance = float('inf')
        for other_label in unique_labels:
            if other_label == label:
                continue
                
            inter_indices = label_indices[other_label]
            if len(inter_indices) == 0:
                continue
                
            inter_class_distances = [cosine(embedding, embeddings[idx]) for idx in inter_indices]
            min_inter_distance = np.min(inter_class_distances)
            nearest_inter_distance = min(nearest_inter_distance, min_inter_distance)
        
        # Calculate ICIC ratio
        if nearest_inter_distance > 0:
            icic_ratio = mean_intra_distance / nearest_inter_distance
            icic_ratios.append(icic_ratio)
    
    # Calculate statistics
    icic_stats = {
        'mean': np.mean(icic_ratios),
        'std': np.std(icic_ratios),
        'max': np.max(icic_ratios)
    }
    
    return icic_stats

## CLASSIFICATION MODEL

class AircraftClassifier(nn.Module):
    """ResNet-based classifier for aircraft classification"""
    def __init__(self, num_classes, pretrained=True):
        super(AircraftClassifier, self).__init__()
        # Load pretrained ResNet50
        self.model = resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                    device, num_epochs=NUM_EPOCHS, patience=5):
    """
    Train a classifier model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs
        patience: Early stopping patience
    
    Returns:
        trained_model: Trained model
        metrics: Training metrics
    """
    model = model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            images = batch['image'].to(device)
            labels = batch['variant_idx'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                images = batch['image'].to(device)
                labels = batch['variant_idx'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), 
                      os.path.join(experiment_dir, 'models', 'best_classifier.pt'))
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'models', 'best_classifier.pt')))
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    return model, metrics

def evaluate_classifier(model, test_loader, device):
    """
    Evaluate classifier on test data
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        accuracy: Test accuracy
        confidences: Softmax confidences
    """
    model.eval()
    correct = 0
    total = 0
    all_confidences = []
    all_variant_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['variant_idx'].to(device)
            variant_ids = batch['variant_id'].numpy()
            
            outputs = model(images)
            softmax_probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(softmax_probs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_confidences.extend(confidences.cpu().numpy())
            all_variant_ids.extend(variant_ids)
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, np.array(all_confidences), np.array(all_variant_ids)

## HYBRID APPROACH IMPLEMENTATION

def implement_hybrid_approach(vlm_results, scenario):
    """
    Implement and evaluate the hybrid approach
    
    Args:
        vlm_results: Results from VLM experiments
        scenario: OOD scenario
    
    Returns:
        hybrid_results: Results from hybrid approach
    """
    print("\n--- Implementing Hybrid Approach ---")
    
    # 1. Train classifier on known classes
    train_loader = DataLoader(
        scenario['known_train_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        scenario['known_val_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        scenario['test_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Count number of unique variant classes in train dataset
    num_classes = len(scenario['known_train_dataset'].variant_to_idx)
    print(f"Training classifier with {num_classes} known variant classes")
    
    # Initialize model
    classifier = AircraftClassifier(num_classes=num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Train classifier
    classifier, classifier_metrics = train_classifier(
        classifier, train_loader, val_loader, criterion, optimizer, scheduler, device
    )
    
    # Evaluate classifier
    test_acc, confidences, variant_ids = evaluate_classifier(classifier, test_loader, device)
    
    # 2. Initialize and evaluate VLM models for hybrid approach
    # Select best performing VLMs from previous experiments
    best_vlms = {}
    for model_name in VLM_MODELS:
        if model_name in vlm_results:
            variant_auroc = vlm_results[model_name]['variant']['mahalanobis_enhanced']['auroc']
            if variant_auroc > 0.8:  # Threshold for selecting good models
                best_vlms[model_name] = VLM_MODELS[model_name]
    
    if not best_vlms:
        print("No VLMs performed well enough for hybrid approach. Using top 2 models.")
        sorted_models = sorted(
            [(m, vlm_results.get(m, {}).get('variant', {}).get('mahalanobis_enhanced', {}).get('auroc', 0)) 
             for m in VLM_MODELS],
            key=lambda x: x[1], reverse=True
        )
        best_vlms = {model: VLM_MODELS[model] for model, _ in sorted_models[:2]}
    
    print(f"Selected {len(best_vlms)} VLMs for hybrid approach: {list(best_vlms.keys())}")
    
    # Initialize VLM models
    vlm_models = {}
    for model_name, model_path in best_vlms.items():
        try:
            vlm_models[model_name] = initialize_vlm(model_path)
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
    
    # 3. Extract embeddings for hybrid approach
    print("Extracting embeddings for hybrid approach...")
    
    # Extract train embeddings for each selected VLM
    train_embeddings_by_model = {}
    train_labels = []
    
    for model_name, model in vlm_models.items():
        embeddings, labels, _ = extract_embeddings(
            model, train_loader, level='variant', prompt_type='enhanced'
        )
        train_embeddings_by_model[model_name] = embeddings
        if not train_labels:
            train_labels = labels
    
    # Combine embeddings from multiple models
    combined_train_embeddings = np.hstack([train_embeddings_by_model[model_name] 
                                         for model_name in vlm_models])
    
    # Extract test embeddings
    test_embeddings_by_model = {}
    test_labels = []
    test_ids = []
    
    for model_name, model in vlm_models.items():
        embeddings, labels, ids = extract_embeddings(
            model, test_loader, level='variant', prompt_type='enhanced'
        )
        test_embeddings_by_model[model_name] = embeddings
        if not test_labels:
            test_labels = labels
            test_ids = ids
    
    combined_test_embeddings = np.hstack([test_embeddings_by_model[model_name] 
                                        for model_name in vlm_models])
    
    # 4. Create hybrid OOD detector
    hybrid_detector = HybridOODDetector(vlm_models=vlm_models, classifier=classifier)
    hybrid_detector.fit(combined_train_embeddings, train_labels)
    
    # 5. Calculate hybrid OOD scores
    # For simplicity, we'll use a weighted combination of embedding score and classifier confidence
    embedding_scores = mahalanobis_ood_detection(combined_train_embeddings, train_labels, combined_test_embeddings)
    
    # Create OOD labels
    is_ood = np.array([1 if id in scenario['unknown_variant_ids'] else 0 for id in test_ids])
    
    # Calculate hybrid scores for different alpha values
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hybrid_results = {}
    
    for alpha in alpha_values:
        hybrid_scores = alpha * embedding_scores + (1 - alpha) * (1 - confidences)
        hybrid_metrics = evaluate_ood_detection(hybrid_scores, is_ood)
        hybrid_results[alpha] = hybrid_metrics
        
        print(f"Hybrid approach (alpha={alpha}) - "
              f"AUROC: {hybrid_metrics['auroc']:.4f}, "
              f"FPR@95: {hybrid_metrics['fpr@95']:.4f}, "
              f"AUPR: {hybrid_metrics['aupr']:.4f}")
        
        # Plot histogram for selected alpha values
        if alpha in [0.0, 0.6, 1.0]:
            plot_ood_histogram(
                hybrid_scores,
                is_ood,
                f"Hybrid OOD Scores (alpha={alpha})",
                os.path.join(experiment_dir, 'figures', f"hybrid_alpha{alpha}_hist.png")
            )
    
    # Find best alpha
    best_alpha = max(alpha_values, key=lambda a: hybrid_results[a]['auroc'])
    print(f"\nBest hybrid approach uses alpha={best_alpha} with "
          f"AUROC: {hybrid_results[best_alpha]['auroc']:.4f}")
    
    # Compare with best individual methods
    best_embedding_auroc = max([
        vlm_results[model_name]['variant']['mahalanobis_enhanced']['auroc']
        for model_name in vlm_results
    ])
    
    print("\nPerformance comparison:")
    print(f"  Best VLM embedding only (alpha=1.0): AUROC = {hybrid_results[1.0]['auroc']:.4f}")
    print(f"  Classifier confidence only (alpha=0.0): AUROC = {hybrid_results[0.0]['auroc']:.4f}")
    print(f"  Hybrid approach (alpha={best_alpha}): AUROC = {hybrid_results[best_alpha]['auroc']:.4f}")
    
    return hybrid_results

## MAIN FUNCTION

def main():
    """Main execution function"""
    print("Starting VLM Embeddings for OOD Detection experiment")
    start_time = time.time()
    
    # 1. Load dataset
    data_path = "aircraft_dataset"  # This is a mock path
    train_data, val_data, test_data, metadata = load_aircraft_dataset(data_path)
    
    # 2. Create OOD scenarios
    scenario = create_ood_scenarios(train_data, val_data, test_data, metadata)
    
    # 3. Run VLM embedding experiments
    vlm_results = run_vlm_experiment(scenario)
    
    # 4. Calculate ICIC ratios for each model at variant level
    print("\n--- Calculating ICIC Ratios ---")
    icic_results = {}
    
    for model_name in VLM_MODELS:
        if model_name in vlm_results:
            # Load test embeddings
            test_embeddings = np.load(os.path.join(experiment_dir, 'embeddings',
                                                 f"{model_name}_variant_test.npy"))
            
            # Create dataloader to get labels
            test_loader = DataLoader(
                scenario['test_dataset'],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=custom_collate_fn
            )
            
            # Extract labels
            labels = []
            for batch in test_loader:
                labels.extend(batch['variant'])
            
            # Only use in-distribution samples for ICIC calculation
            test_ids = []
            for batch in test_loader:
                test_ids.extend(batch['variant_id'].numpy())
            
            is_id = np.array([1 if id in scenario['known_variant_ids'] else 0 for id in test_ids])
            id_indices = np.where(is_id == 1)[0]
            
            id_embeddings = test_embeddings[id_indices]
            id_labels = [labels[i] for i in id_indices]
            
            # Calculate ICIC ratio
            icic_stats = calculate_icic_ratio(id_embeddings, id_labels)
            icic_results[model_name] = icic_stats
            
            print(f"{model_name} ICIC Ratio - "
                  f"Mean: {icic_stats['mean']:.4f}, "
                  f"Std: {icic_stats['std']:.4f}, "
                  f"Max: {icic_stats['max']:.4f}")
    
    # 5. Implement hybrid approach
    hybrid_results = implement_hybrid_approach(vlm_results, scenario)
    
    # 6. Save results
    results = {
        'vlm_results': vlm_results,
        'icic_results': icic_results,
        'hybrid_results': hybrid_results
    }
    
    np.save(os.path.join(experiment_dir, 'results', 'all_results.npy'), results)
    
    # 7. Create comparative visualizations
    # Plot comparative bar charts for different metrics
    for metric in ['auroc', 'fpr@95', 'aupr']:
        plot_performance_comparison(
            {model: vlm_results[model] for model in vlm_results},
            metric=metric,
            title=f"{metric.upper()} Comparison Across Models",
            save_path=os.path.join(experiment_dir, 'figures', f"comparison_{metric}.png")
        )
    
    # 8. Report total execution time
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    
    print(f"\nExperiment completed in {hours:.0f}h {minutes:.0f}m {seconds:.2f}s")
    print(f"Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()