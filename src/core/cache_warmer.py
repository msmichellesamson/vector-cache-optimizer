"""Cache warming strategy with ML-driven prioritization."""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from heapq import heappush, heappop
import time

from .errors import CacheError
from ..ml.hit_predictor import HitPredictor
from ..monitoring.logger import get_logger


@dataclass
class WarmingTask:
    """Cache warming task with priority."""
    priority: float  # Higher = more important
    key: str
    embedding: List[float]
    timestamp: float
    
    def __lt__(self, other):
        return self.priority > other.priority  # Max heap


class CacheWarmer:
    """Intelligent cache warming with ML-driven prioritization."""
    
    def __init__(self, cache_manager, hit_predictor: HitPredictor, max_warming_rate: int = 100):
        self.cache_manager = cache_manager
        self.hit_predictor = hit_predictor
        self.max_warming_rate = max_warming_rate
        self.warming_queue: List[WarmingTask] = []
        self.logger = get_logger(__name__)
        self._warming_active = False
        
    async def schedule_warming(self, embeddings: List[Tuple[str, List[float]]]) -> None:
        """Schedule embeddings for warming based on predicted hit probability."""
        try:
            current_time = time.time()
            
            for key, embedding in embeddings:
                # Skip if already cached
                if await self.cache_manager.exists(key):
                    continue
                    
                # Get ML prediction for hit probability
                hit_prob = await self.hit_predictor.predict_hit_probability(embedding)
                
                # Priority = hit_probability * recency_factor
                priority = hit_prob * (1.0 + 0.1 * (current_time % 3600) / 3600)
                
                task = WarmingTask(
                    priority=priority,
                    key=key,
                    embedding=embedding,
                    timestamp=current_time
                )
                
                heappush(self.warming_queue, task)
                
            self.logger.info(f"Scheduled {len(embeddings)} embeddings for warming")
            
        except Exception as e:
            self.logger.error(f"Error scheduling warming: {e}")
            raise CacheError(f"Failed to schedule warming: {e}")
    
    async def start_warming(self) -> None:
        """Start background cache warming process."""
        if self._warming_active:
            return
            
        self._warming_active = True
        self.logger.info("Starting cache warming process")
        
        try:
            while self._warming_active and self.warming_queue:
                # Process up to max_warming_rate items per batch
                batch_size = min(self.max_warming_rate, len(self.warming_queue))
                batch = [heappop(self.warming_queue) for _ in range(batch_size)]
                
                await self._warm_batch(batch)
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
        except Exception as e:
            self.logger.error(f"Error in warming process: {e}")
        finally:
            self._warming_active = False
            
    async def _warm_batch(self, batch: List[WarmingTask]) -> None:
        """Warm a batch of cache entries."""
        for task in batch:
            try:
                # Set with longer TTL for warmed entries
                await self.cache_manager.set(
                    task.key, 
                    task.embedding, 
                    ttl=3600,  # 1 hour TTL for warmed entries
                    metadata={'warmed': True, 'priority': task.priority}
                )
            except Exception as e:
                self.logger.warning(f"Failed to warm {task.key}: {e}")
                
    def stop_warming(self) -> None:
        """Stop the warming process."""
        self._warming_active = False
        self.logger.info("Stopped cache warming process")
        
    def get_queue_size(self) -> int:
        """Get current warming queue size."""
        return len(self.warming_queue)
