import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


class TrainingError(Exception):
    """Raised when model training fails"""
    pass


class AccessPattern(Enum):
    """Types of access patterns we can predict"""
    FREQUENT = "frequent"
    BURST = "burst" 
    DECLINING = "declining"
    RANDOM = "random"


@dataclass
class EmbeddingFeatures:
    """Features for a single embedding entry"""
    vector_id: str
    access_count_1h: int
    access_count_24h: int
    access_count_7d: int
    time_since_creation: float  # hours
    time_since_last_access: float  # hours
    avg_access_interval: float  # hours
    access_variance: float
    vector_dimension: int
    memory_size_kb: float
    cache_hit_ratio: float
    similar_vectors_accessed: int  # count of similar vectors accessed recently


@dataclass
class PredictionResult:
    """Result of access pattern prediction"""
    vector_id: str
    predicted_pattern: AccessPattern
    access_probability_1h: float
    access_probability_24h: float
    eviction_priority: float  # 0.0 = keep, 1.0 = evict first
    confidence: float
    features_used: Dict[str, float]


class EmbeddingDataset(Dataset):
    """PyTorch dataset for embedding access patterns"""
    
    def __init__(self, features: List[EmbeddingFeatures], labels: Optional[List[AccessPattern]] = None):
        self.features = features
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature = self.features[idx]
        
        # Convert features to tensor
        feature_vector = torch.tensor([
            float(feature.access_count_1h),
            float(feature.access_count_24h),
            float(feature.access_count_7d),
            feature.time_since_creation,
            feature.time_since_last_access,
            feature.avg_access_interval,
            feature.access_variance,
            float(feature.vector_dimension),
            feature.memory_size_kb,
            feature.cache_hit_ratio,
            float(feature.similar_vectors_accessed)
        ], dtype=torch.float32)
        
        if self.labels is not None:
            label = torch.tensor(list(AccessPattern).index(self.labels[idx]), dtype=torch.long)
            return feature_vector, label
        
        return feature_vector, None


class AccessPatternNet(nn.Module):
    """Neural network for predicting embedding access patterns"""
    
    def __init__(self, input_dim: int = 11, hidden_dims: List[int] = [64, 32], num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
        # Additional heads for regression tasks
        self.prob_1h_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.prob_24h_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.eviction_priority_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get features from second-to-last layer
        features = x
        for layer in self.network[:-1]:
            features = layer(features)
        
        # Classification output
        pattern_logits = self.network[-1](features)
        
        # Regression outputs
        prob_1h = self.prob_1h_head(features)
        prob_24h = self.prob_24h_head(features)
        eviction_priority = self.eviction_priority_head(features)
        
        return pattern_logits, prob_1h.squeeze(), prob_24h.squeeze(), eviction_priority.squeeze()


class AccessPatternPredictor:
    """ML-powered predictor for embedding access patterns"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        retrain_interval_hours: int = 24,
        min_training_samples: int = 1000
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else Path("models/access_pattern_model.pt")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.retrain_interval_hours = retrain_interval_hours
        self.min_training_samples = min_training_samples
        self.last_training_time: Optional[datetime] = None
        
        # Initialize model
        self.model = AccessPatternNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Load existing model if available
        self._load_model()
        
        logger.info("AccessPatternPredictor initialized", device=self.device, model_path=str(self.model_path))
    
    def _load_model(self) -> None:
        """Load model from disk if exists"""
        try:
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.last_training_time = checkpoint.get('last_training_time')
                
                logger.info("Model loaded successfully", 
                           last_training_time=self.last_training_time,
                           model_path=str(self.model_path))
            else:
                logger.info("No existing model found, will train from scratch")
                
        except Exception as e:
            logger.error("Failed to load model", error=str(e), model_path=str(self.model_path))
            raise ModelLoadError(f"Failed to load model from {self.model_path}: {e}")
    
    def _save_model(self) -> None:
        """Save model to disk"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'last_training_time': datetime.utcnow(),
                'training_config': {
                    'retrain_interval_hours': self.retrain_interval_hours,
                    'min_training_samples': self.min_training_samples
                }
            }
            torch.save(checkpoint, self.model_path)
            logger.info("Model saved successfully", model_path=str(self.model_path))
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e), model_path=str(self.model_path))
            raise ModelLoadError(f"Failed to save model to {self.model_path}: {e}")
    
    def needs_retraining(self, current_sample_count: int) -> bool:
        """Check if model needs retraining"""
        if self.last_training_time is None:
            return current_sample_count >= self.min_training_samples
        
        time_elapsed = datetime.utcnow() - self.last_training_time
        hours_elapsed = time_elapsed.total_seconds() / 3600
        
        return (hours_elapsed >= self.retrain_interval_hours and 
                current_sample_count >= self.min_training_samples)
    
    async def train(
        self,
        training_data: List[Tuple[EmbeddingFeatures, AccessPattern]],
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 128
    ) -> Dict[str, float]:
        """Train the access pattern prediction model"""
        try:
            if len(training_data) < self.min_training_samples:
                raise TrainingError(f"Insufficient training data: {len(training_data)} < {self.min_training_samples}")
            
            logger.info("Starting model training", 
                       samples=len(training_data),
                       epochs=epochs,
                       batch_size=batch_size)
            
            # Split data
            split_idx = int(len(training_data) * (1 - validation_split))
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]
            
            # Create datasets
            train_features, train_labels = zip(*train_data)
            val_features, val_labels = zip(*val_data)
            
            train_dataset = EmbeddingDataset(list(train_features), list(train_labels))
            val_dataset = EmbeddingDataset(list(val_features), list(val_labels))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                epoch_train_loss = 0.0
                
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    pattern_logits, prob_1h, prob_24h, eviction_priority = self.model(batch_features)
                    
                    # Classification loss
                    pattern_loss = nn.CrossEntropyLoss()(pattern_logits, batch_labels)
                    
                    # For simplicity, we'll focus on classification loss
                    # In practice, you'd want to add regression losses with proper targets
                    total_loss = pattern_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_train_loss += total_loss.item()
                
                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                self.model.eval()
                epoch_val_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        pattern_logits, _, _, _ = self.model(batch_features)
                        
                        val_loss = nn.CrossEntropyLoss()(pattern_logits, batch_labels)
                        epoch_val_loss += val_loss.item()
                        
                        # Accuracy calculation
                        predictions = torch.argmax(pattern_logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                        total_predictions += batch_labels.size(0)
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                val_accuracy = correct_predictions / total_predictions
                
                self.scheduler.step(avg_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info("Training progress",
                               epoch=epoch + 1,
                               train_loss=avg_train_loss,
                               val_loss=avg_val_loss,
                               val_accuracy=val_accuracy)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_model()
            
            self.last_training_time = datetime.utcnow()
            
            training_metrics = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss,
                'final_val_accuracy': val_accuracy,
                'epochs_trained': epochs,
                'samples_used': len(training_data)
            }
            
            logger.info("Training completed", **training_metrics)
            return training_metrics
            
        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise TrainingError(f"Model training failed: {e}")
    
    async def predict(self, features: EmbeddingFeatures) -> PredictionResult:
        """Predict access pattern for a single embedding"""
        try:
            self.model.eval()
            
            # Convert features to tensor
            dataset = EmbeddingDataset([features])
            feature_tensor, _ = dataset[0]
            feature_tensor = feature_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pattern_logits, prob_1h, prob_24h, eviction_priority = self.model(feature_tensor)
                
                # Get pattern prediction
                pattern_probs = torch.softmax(pattern_logits, dim=1)
                predicted_pattern_idx = torch.argmax(pattern_probs, dim=1).item()
                predicted_pattern = list(AccessPattern)[predicted_pattern_idx]
                confidence = pattern_probs[0, predicted_pattern_idx].item()
                
                # Extract predictions
                prob_1h_val = prob_1h.item() if prob_1h.dim() == 0 else prob_1h[0].item()
                prob_24h_val = prob_24h.item() if prob_24h.dim() == 0 else prob_24h[0].item()
                eviction_val = eviction_priority.item() if eviction_priority.dim() == 0 else eviction_priority[0].item()
            
            result = PredictionResult(
                vector_id=features.vector_id,
                predicted_pattern=predicted_pattern,
                access_probability_1h=prob_1h_val,
                access_probability_24h=prob_24h_val,
                eviction_priority=eviction_val,
                confidence=confidence,
                features_used={
                    'access_count_1h': float(features.access_count_1h),
                    'access_count_24h': float(features.access_count_24h),
                    'time_since_last_access': features.time_since_last_access,
                    'cache_hit_ratio': features.cache_hit_ratio
                }
            )
            
            logger.debug("Prediction completed",
                        vector_id=features.vector_id,
                        predicted_pattern=predicted_pattern.value,
                        confidence=confidence)
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed", 
                        vector_id=features.vector_id,
                        error=str(e))
            raise PredictionError(f"Failed to predict access pattern for {features.vector_id}: {e}")
    
    async def predict_batch(self, features_list: List[EmbeddingFeatures]) -> List[PredictionResult]:
        """Predict access patterns for multiple embeddings"""
        try:
            if not features_list:
                return []
            
            self.model.eval()
            
            # Create batch dataset
            dataset = EmbeddingDataset(features_list)
            batch_size = min(128, len(features_list))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            results = []
            
            with torch.no_grad():
                for batch_idx, (batch_features, _) in enumerate(dataloader):
                    batch_features = batch_features.to(self.device)
                    
                    pattern_logits, prob_1h, prob_24h, eviction_priority = self.model(batch_features)
                    
                    # Process each item in batch
                    pattern_probs = torch.softmax(pattern_logits, dim=1)
                    predicted_patterns = torch.argmax(pattern_probs, dim=1)
                    confidences = torch.gather(pattern_probs, 1, predicted_patterns.unsqueeze(1)).squeeze()
                    
                    for i in range(batch_features.size(0)):
                        feature_idx = batch_idx * batch_size + i
                        feature = features_list[feature_idx]
                        
                        predicted_pattern = list(AccessPattern)[predicted_patterns[i].item()]
                        confidence = confidences[i].item() if confidences.dim() > 0 else confidences.item()
                        
                        result = PredictionResult(
                            vector_id=feature.vector_id,
                            predicted_pattern=predicted_pattern,
                            access_probability_1h=prob_1h[i].item(),
                            access_probability_24h=prob_24h[i].item(),
                            eviction_priority=eviction_priority[i].item(),
                            confidence=confidence,
                            features_used={
                                'access_count_1h': float(feature.access_count_1h),
                                'access_count_24h': float(feature.access_count_24h),
                                'time_since_last_access': feature.time_since_last_access,
                                'cache_hit_ratio': feature.cache_hit_ratio
                            }
                        )
                        
                        results.append(result)
            
            logger.info("Batch prediction completed", batch_size=len(features_list))
            return results
            
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e), batch_size=len(features_list))
            raise PredictionError(f"Failed to predict batch of {len(features_list)} embeddings: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'retrain_interval_hours': self.retrain_interval_hours,
            'min_training_samples': self.min_training_samples,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }