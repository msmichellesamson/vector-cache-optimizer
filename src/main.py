import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Tuple
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import redis.asyncio as redis
import structlog
import torch
import torch.nn as nn
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import pickle
import json
import hashlib
from datetime import datetime, timedelta

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
CACHE_HITS = Counter('vector_cache_hits_total', 'Number of cache hits')
CACHE_MISSES = Counter('vector_cache_misses_total', 'Number of cache misses')
EVICTION_COUNT = Counter('vector_cache_evictions_total', 'Number of cache evictions', ['policy'])
QUERY_DURATION = Histogram('vector_query_duration_seconds', 'Time spent on vector queries')
CACHE_SIZE = Gauge('vector_cache_size_bytes', 'Current cache size in bytes')
ACTIVE_EMBEDDINGS = Gauge('vector_cache_active_embeddings', 'Number of cached embeddings')

class VectorCacheException(Exception):
    """Base exception for vector cache operations"""
    pass

class EmbeddingNotFoundError(VectorCacheException):
    """Raised when requested embedding is not found"""
    pass

class CacheFullError(VectorCacheException):
    """Raised when cache is at capacity"""
    pass

class InvalidVectorError(VectorCacheException):
    """Raised when vector dimensions don't match"""
    pass

class VectorQuery(BaseModel):
    """Vector query request model"""
    query_vector: List[float]
    top_k: int = 10
    similarity_threshold: float = 0.7
    
    @validator('query_vector')
    def validate_vector_dimensions(cls, v):
        if len(v) != 768:  # Standard embedding dimension
            raise ValueError('Vector must be 768 dimensions')
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('top_k must be between 1 and 100')
        return v

class VectorStore(BaseModel):
    """Vector storage request model"""
    key: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    ttl: Optional[int] = None
    
    @validator('vector')
    def validate_vector_dimensions(cls, v):
        if len(v) != 768:
            raise ValueError('Vector must be 768 dimensions')
        return v

class EvictionPolicy(nn.Module):
    """Neural network for learning optimal eviction policies"""
    
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict eviction probability for given features"""
        return self.network(features)

class VectorCacheManager:
    """Manages vector embeddings with ML-driven eviction policies"""
    
    def __init__(self, redis_client: redis.Redis, max_cache_size: int = 1024 * 1024 * 1024):
        self.redis = redis_client
        self.max_cache_size = max_cache_size
        self.eviction_model = EvictionPolicy()
        self.feature_history: Dict[str, List[float]] = {}
        self.access_stats: Dict[str, Dict[str, Any]] = {}
        
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        try:
            info = await self.redis.info('memory')
            keyspace_info = await self.redis.info('keyspace')
            
            total_keys = 0
            if 'db0' in keyspace_info:
                db_info = keyspace_info['db0']
                total_keys = int(db_info.split(',')[0].split('=')[1])
            
            return {
                'memory_used': info.get('used_memory', 0),
                'memory_peak': info.get('used_memory_peak', 0),
                'total_keys': total_keys,
                'cache_hit_rate': await self._calculate_hit_rate(),
                'eviction_stats': await self._get_eviction_stats()
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {}
    
    async def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from metrics"""
        try:
            hits = CACHE_HITS._value._value
            misses = CACHE_MISSES._value._value
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    async def _get_eviction_stats(self) -> Dict[str, int]:
        """Get eviction statistics by policy"""
        try:
            # Access private metric data - in production, use proper metric collection
            ml_evictions = getattr(EVICTION_COUNT.labels(policy='ml'), '_value', {}).get('_value', 0)
            lru_evictions = getattr(EVICTION_COUNT.labels(policy='lru'), '_value', {}).get('_value', 0)
            return {
                'ml_policy': ml_evictions,
                'lru_fallback': lru_evictions
            }
        except Exception:
            return {'ml_policy': 0, 'lru_fallback': 0}
    
    async def store_vector(self, key: str, vector: np.ndarray, metadata: Optional[Dict] = None, ttl: Optional[int] = None) -> bool:
        """Store vector embedding with metadata"""
        try:
            # Check cache size before storing
            current_size = await self._get_cache_size()
            vector_size = vector.nbytes + len(key.encode()) + (len(json.dumps(metadata)) if metadata else 0)
            
            if current_size + vector_size > self.max_cache_size:
                await self._trigger_eviction(vector_size)
            
            # Prepare data for storage
            vector_data = {
                'vector': vector.tobytes(),
                'shape': vector.shape,
                'dtype': str(vector.dtype),
                'metadata': metadata or {},
                'stored_at': datetime.utcnow().isoformat(),
                'access_count': 0,
                'last_accessed': datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            serialized_data = pickle.dumps(vector_data)
            
            if ttl:
                await self.redis.setex(f"vector:{key}", ttl, serialized_data)
            else:
                await self.redis.set(f"vector:{key}", serialized_data)
            
            # Update statistics
            await self._update_access_stats(key, 'store')
            ACTIVE_EMBEDDINGS.inc()
            CACHE_SIZE.set(current_size + vector_size)
            
            logger.info("Vector stored successfully", key=key, size=vector_size)
            return True
            
        except Exception as e:
            logger.error("Failed to store vector", key=key, error=str(e))
            raise VectorCacheException(f"Storage failed: {str(e)}")
    
    async def get_vector(self, key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Retrieve vector embedding and metadata"""
        try:
            data = await self.redis.get(f"vector:{key}")
            if not data:
                CACHE_MISSES.inc()
                return None
            
            vector_data = pickle.loads(data)
            
            # Reconstruct numpy array
            vector = np.frombuffer(
                vector_data['vector'], 
                dtype=vector_data['dtype']
            ).reshape(vector_data['shape'])
            
            # Update access statistics
            vector_data['access_count'] += 1
            vector_data['last_accessed'] = datetime.utcnow().isoformat()
            
            # Store updated data back to Redis
            await self.redis.set(f"vector:{key}", pickle.dumps(vector_data))
            await self._update_access_stats(key, 'get')
            
            CACHE_HITS.inc()
            
            logger.debug("Vector retrieved successfully", key=key)
            return vector, vector_data['metadata']
            
        except Exception as e:
            logger.error("Failed to retrieve vector", key=key, error=str(e))
            CACHE_MISSES.inc()
            return None
    
    async def similarity_search(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.7) -> List[Dict]:
        """Perform similarity search across cached vectors"""
        with QUERY_DURATION.time():
            try:
                results = []
                keys = await self.redis.keys("vector:*")
                
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    vector_key = key_str.replace("vector:", "")
                    
                    cached_result = await self.get_vector(vector_key)
                    if not cached_result:
                        continue
                    
                    cached_vector, metadata = cached_result
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_vector, cached_vector)
                    
                    if similarity >= threshold:
                        results.append({
                            'key': vector_key,
                            'similarity': float(similarity),
                            'metadata': metadata
                        })
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:top_k]
                
            except Exception as e:
                logger.error("Similarity search failed", error=str(e))
                raise VectorCacheException(f"Search failed: {str(e)}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _get_cache_size(self) -> int:
        """Get current cache size in bytes"""
        try:
            info = await self.redis.info('memory')
            return info.get('used_memory', 0)
        except Exception:
            return 0
    
    async def _trigger_eviction(self, required_space: int) -> None:
        """Trigger intelligent eviction to free up space"""
        try:
            keys = await self.redis.keys("vector:*")
            if not keys:
                return
            
            # Collect features for each key
            eviction_candidates = []
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                vector_key = key_str.replace("vector:", "")
                
                features = await self._extract_features(vector_key)
                if features is not None:
                    eviction_candidates.append((vector_key, features))
            
            if not eviction_candidates:
                # Fallback to LRU eviction
                await self._lru_eviction(len(keys) // 4)
                EVICTION_COUNT.labels(policy='lru').inc()
                return
            
            # Use ML model to predict eviction probabilities
            feature_matrix = torch.tensor([features for _, features in eviction_candidates], dtype=torch.float32)
            
            with torch.no_grad():
                eviction_probs = self.eviction_model(feature_matrix).squeeze()
            
            # Sort by eviction probability (highest first)
            candidates_with_probs = list(zip(eviction_candidates, eviction_probs.tolist()))
            candidates_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Evict keys until we have enough space
            freed_space = 0
            evicted_count = 0
            
            for (key, _), _ in candidates_with_probs:
                if freed_space >= required_space:
                    break
                
                # Estimate key size (approximation)
                key_size = await self._estimate_key_size(key)
                await self.redis.delete(f"vector:{key}")
                
                freed_space += key_size
                evicted_count += 1
                ACTIVE_EMBEDDINGS.dec()
            
            EVICTION_COUNT.labels(policy='ml').inc(evicted_count)
            logger.info("ML-driven eviction completed", evicted_keys=evicted_count, freed_space=freed_space)
            
        except Exception as e:
            logger.error("Eviction failed, falling back to LRU", error=str(e))
            await self._lru_eviction(len(keys) // 4)
            EVICTION_COUNT.labels(policy='lru').inc()
    
    async def _extract_features(self, key: str) -> Optional[List[float]]:
        """Extract features for eviction prediction"""
        try:
            data = await self.redis.get(f"vector:{key}")
            if not data:
                return None
            
            vector_data = pickle.loads(data)
            
            # Extract temporal features
            stored_at = datetime.fromisoformat(vector_data['stored_at'])
            last_accessed = datetime.fromisoformat(vector_data['last_accessed'])
            now = datetime.utcnow()
            
            age_hours = (now - stored_at).total_seconds() / 3600
            idle_hours = (now - last_accessed).total_seconds() / 3600
            access_count = vector_data.get('access_count', 0)
            
            # Calculate access frequency
            access_frequency = access_count / max(age_hours, 1.0)
            
            # Get vector size
            vector_size = len(vector_data['vector'])
            
            # Get hit rate for this key (simplified)
            hit_rate = min(access_count / max(age_hours / 24, 1.0), 1.0)
            
            features = [
                age_hours,
                idle_hours,
                access_count,
                access_frequency,
                vector_size / 1024,  # Size in KB
                hit_rate
            ]
            
            return features
            
        except Exception as e:
            logger.error("Feature extraction failed", key=key, error=str(e))
            return None
    
    async def _lru_eviction(self, count: int) -> None:
        """Fallback LRU eviction policy"""
        try:
            keys = await self.redis.keys("vector:*")
            lru_candidates = []
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                vector_key = key_str.replace("vector:", "")
                
                data = await self.redis.get(key)
                if data:
                    vector_data = pickle.loads(data)
                    last_accessed = datetime.fromisoformat(vector_data.get('last_accessed', datetime.utcnow().isoformat()))
                    lru_candidates.append((vector_key, last_accessed))
            
            # Sort by last accessed time (oldest first)
            lru_candidates.sort(key=lambda x: x[1])
            
            # Evict oldest keys
            for i in range(min(count, len(lru_candidates))):
                key = lru_candidates[i][0]
                await self.redis.delete(f"vector:{key}")
                ACTIVE_EMBEDDINGS.dec()
            
            logger.info("LRU eviction completed", evicted_keys=min(count, len(lru_candidates)))
            
        except Exception as e:
            logger.error("LRU eviction failed", error=str(e))
    
    async def _estimate_key_size(self, key: str) -> int:
        """Estimate the size of a key in bytes"""
        try:
            data = await self.redis.get(f"vector:{key}")
            if data:
                return len(data) + len(key.encode())
            return 0
        except Exception:
            return 1024  # Default estimate
    
    async def _update_access_stats(self, key: str, operation: str) -> None:
        """Update access statistics for ML training"""
        try:
            if key not in self.access_stats:
                self.access_stats[key] = {
                    'total_accesses': 0,
                    'last_operation': None,
                    'operation_counts': {}
                }
            
            stats = self.access_stats[key]
            stats['total_accesses'] += 1
            stats['last_operation'] = operation
            stats['operation_counts'][operation] = stats['operation_counts'].get(operation, 0) + 1
            
            # Keep stats size manageable
            if len(self.access_stats) > 10000:
                # Remove oldest entries
                oldest_keys = sorted(self.access_stats.keys())[:1000]
                for old_key in oldest_keys:
                    del self.access_stats[old_key]
                    
        except Exception as e:
            logger.error("Failed to update access stats", key=key, error=str(e))

# Global cache manager instance
cache_manager: Optional[VectorCacheManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cache_manager
    
    # Startup
    logger.info("Starting vector cache optimizer service")
    
    # Start Prometheus metrics server
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")
    
    # Initialize Redis connection
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise
    
    # Initialize cache manager
    cache_manager = VectorCacheManager(redis_client)
    logger.info("Vector cache manager initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down vector cache optimizer service")
    if cache_manager:
        await cache_manager.redis.close()

# Create FastAPI application
app = FastAPI(
    title="Vector Cache Optimizer",
    description="Intelligent embedding cache with ML-driven eviction policies",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    try:
        if cache_manager:
            await cache_manager.redis.ping()
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get cache performance metrics"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        stats = await cache_manager.get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.post("/vectors")
async def store_vector(vector_data: VectorStore, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Store a vector embedding with optional metadata"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        vector_array = np.array(vector_data.vector, dtype=np.float32)
        
        success = await cache_manager.store_vector(
            key=vector_data.key,
            vector=vector_array,
            metadata=vector_data.metadata,
            ttl=vector_data.ttl
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store vector")
        
        logger.info("Vector stored via API", key=vector_data.key)
        
        return {
            "message": "Vector stored successfully",
            "key": vector_data.key,
            "dimension": len(vector_data.vector),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except VectorCacheException as e:
        logger.error("Vector storage failed", key=vector_data.key, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error storing vector", key=vector_data.key, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/vectors/{key}")
async def get_vector(key: str) -> Dict[str, Any]:
    """Retrieve a vector embedding and metadata"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        result = await cache_manager.get_vector(key)
        
        if not result:
            raise HTTPException(status_code=404, detail="Vector not found")
        
        vector, metadata = result
        
        return {
            "key": key,
            "vector": vector.tolist(),
            "metadata": metadata,
            "dimension": len(vector),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve vector", key=key, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
async def similarity_search(query: VectorQuery) -> Dict[str, Any]:
    """Perform similarity search across cached vectors"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        query_vector = np.array(query.query_vector, dtype=np.float32)
        
        results = await cache_manager.similarity_search(
            query_vector=query_vector,
            top_k=query.top_k,
            threshold=query.similarity_threshold
        )
        
        return {
            "results": results,
            "query_dimension": len(query.query_vector),
            "total_results": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Similarity search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/vectors/{key}")
async def delete_vector(key: str) -> Dict[str, str]:
    """Delete a vector from cache"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        result = await cache_manager.redis.delete(f"vector:{key}")
        
        if result == 0:
            raise HTTPException(status_code=404, detail="Vector not found")
        
        ACTIVE_EMBEDDINGS.dec()
        logger.info("Vector deleted via API", key=key)
        
        return {
            "message": "Vector deleted successfully",
            "key": key,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete vector", key=key, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/evict")
async def trigger_eviction(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Manually trigger cache eviction process"""
    if not cache_manager:
        raise HTTPException(status_code=500, detail="Cache manager not initialized")
    
    try:
        # Trigger eviction for 10% of cache size
        background_tasks.add_task(cache_manager._trigger_eviction, cache_manager.max_cache_size // 10)
        
        return {
            "message": "Eviction process triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to trigger eviction", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Use structlog instead
        access_log=False
    )