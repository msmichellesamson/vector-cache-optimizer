"""Intelligent vector cache manager with ML-driven eviction.

This module provides the main CacheManager class that coordinates
between Redis cache, ML predictors, and monitoring systems.
"""

from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from datetime import datetime, timedelta

from .cache_engine import CacheEngine
from .eviction_selector import EvictionSelector
from ..ml.hit_predictor import HitPredictor
from ..monitoring.memory_monitor import MemoryMonitor
from ..monitoring.hit_rate_tracker import HitRateTracker

logger = logging.getLogger(__name__)


class CacheManager:
    """Main cache manager coordinating ML-driven eviction and monitoring.
    
    This class serves as the central coordinator for:
    - Vector embedding storage and retrieval
    - ML-based hit prediction and eviction policies
    - Real-time performance monitoring and alerting
    - Memory pressure management
    
    Example:
        >>> manager = CacheManager(redis_url="redis://localhost:6379")
        >>> await manager.initialize()
        >>> 
        >>> # Store vector embedding
        >>> vector = [0.1, 0.2, 0.3, ...]
        >>> await manager.store("doc_123", vector, metadata={"type": "document"})
        >>> 
        >>> # Retrieve with similarity search
        >>> results = await manager.retrieve_similar(query_vector, top_k=10)
    """
    
    def __init__(self, redis_url: str, ml_model_path: Optional[str] = None):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection string
            ml_model_path: Optional path to pre-trained ML model
        """
        self.cache_engine = CacheEngine(redis_url)
        self.eviction_selector = EvictionSelector()
        self.hit_predictor = HitPredictor(ml_model_path)
        self.memory_monitor = MemoryMonitor()
        self.hit_rate_tracker = HitRateTracker()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all components and start monitoring."""
        await self.cache_engine.initialize()
        await self.hit_predictor.load_model()
        await self.memory_monitor.start()
        self._initialized = True
        logger.info("CacheManager initialized successfully")
    
    async def store(self, key: str, vector: List[float], 
                   metadata: Optional[Dict[str, Any]] = None, 
                   ttl: Optional[int] = None) -> bool:
        """Store vector embedding with optional metadata.
        
        Args:
            key: Unique identifier for the vector
            vector: Vector embedding as list of floats
            metadata: Optional metadata dict
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            True if stored successfully, False otherwise
            
        Raises:
            CacheError: If storage fails
        """
        if not self._initialized:
            raise RuntimeError("CacheManager not initialized")
            
        return await self.cache_engine.store(key, vector, metadata, ttl)
    
    async def retrieve_similar(self, query_vector: List[float], 
                              top_k: int = 10,
                              similarity_threshold: float = 0.8) -> List[Tuple[str, float, Dict]]:
        """Retrieve most similar vectors using ML-optimized search.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of (key, similarity_score, metadata) tuples
        """
        results = await self.cache_engine.similarity_search(
            query_vector, top_k, similarity_threshold
        )
        
        # Update hit rate tracking
        await self.hit_rate_tracker.record_hits(len(results))
        
        return results