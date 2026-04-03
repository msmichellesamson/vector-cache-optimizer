from typing import List, Dict, Any, Set
import asyncio
import logging
from datetime import datetime

from .errors import CacheError

logger = logging.getLogger(__name__)

class BatchEvictionProcessor:
    """Processes evictions in batches for better Redis performance."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 0.5):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._pending_keys: Set[str] = set()
        self._last_flush = datetime.now()
        self._redis_client = None
        self._stats = {
            'batches_processed': 0,
            'keys_evicted': 0,
            'avg_batch_size': 0.0
        }
    
    def set_redis_client(self, client) -> None:
        """Set Redis client for batch operations."""
        self._redis_client = client
    
    async def queue_eviction(self, key: str) -> None:
        """Queue a key for batch eviction."""
        self._pending_keys.add(key)
        
        if len(self._pending_keys) >= self.batch_size:
            await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Flush pending evictions to Redis."""
        if not self._pending_keys or not self._redis_client:
            return
            
        keys_to_evict = list(self._pending_keys)
        self._pending_keys.clear()
        
        try:
            # Use pipeline for batch deletion
            pipe = self._redis_client.pipeline()
            for key in keys_to_evict:
                pipe.delete(key)
            await pipe.execute()
            
            # Update stats
            self._stats['batches_processed'] += 1
            self._stats['keys_evicted'] += len(keys_to_evict)
            self._update_avg_batch_size(len(keys_to_evict))
            
            logger.info(f"Batch evicted {len(keys_to_evict)} keys")
            
        except Exception as e:
            logger.error(f"Batch eviction failed: {e}")
            # Re-queue failed keys
            self._pending_keys.update(keys_to_evict)
            raise CacheError(f"Batch eviction failed: {e}")
        
        self._last_flush = datetime.now()
    
    def _update_avg_batch_size(self, batch_size: int) -> None:
        """Update running average of batch sizes."""
        current_avg = self._stats['avg_batch_size']
        batches = self._stats['batches_processed']
        self._stats['avg_batch_size'] = (current_avg * (batches - 1) + batch_size) / batches
    
    async def force_flush(self) -> None:
        """Force flush all pending evictions."""
        await self._flush_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            **self._stats,
            'pending_keys': len(self._pending_keys),
            'last_flush': self._last_flush.isoformat()
        }
    
    def get_pending_count(self) -> int:
        """Get number of pending keys."""
        return len(self._pending_keys)