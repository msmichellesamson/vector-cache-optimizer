import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.core.cache_engine import CacheEngine
from src.core.errors import CacheError, ConnectionError


@pytest.fixture
def mock_redis():
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = 0
    return redis_mock


@pytest.fixture
def cache_engine(mock_redis):
    with patch('src.core.cache_engine.redis.Redis', return_value=mock_redis):
        engine = CacheEngine(host='localhost', port=6379)
        return engine


@pytest.mark.asyncio
class TestCacheEngine:
    async def test_store_vector_success(self, cache_engine, mock_redis):
        vector = [0.1, 0.2, 0.3]
        metadata = {'type': 'embedding'}
        
        result = await cache_engine.store('test_key', vector, metadata)
        
        assert result is True
        mock_redis.set.assert_called_once()
        
    async def test_retrieve_vector_exists(self, cache_engine, mock_redis):
        vector_data = {'vector': [0.1, 0.2, 0.3], 'metadata': {'type': 'embedding'}}
        mock_redis.get.return_value = str(vector_data).encode()
        
        result = await cache_engine.retrieve('test_key')
        
        assert result is not None
        mock_redis.get.assert_called_once_with('test_key')
        
    async def test_retrieve_vector_missing(self, cache_engine, mock_redis):
        mock_redis.get.return_value = None
        
        result = await cache_engine.retrieve('missing_key')
        
        assert result is None
        
    async def test_delete_vector_success(self, cache_engine, mock_redis):
        mock_redis.delete.return_value = 1
        
        result = await cache_engine.delete('test_key')
        
        assert result is True
        mock_redis.delete.assert_called_once_with('test_key')
        
    async def test_delete_vector_missing(self, cache_engine, mock_redis):
        mock_redis.delete.return_value = 0
        
        result = await cache_engine.delete('missing_key')
        
        assert result is False
        
    async def test_connection_error_handling(self, cache_engine, mock_redis):
        mock_redis.get.side_effect = Exception('Connection failed')
        
        with pytest.raises(CacheError):
            await cache_engine.retrieve('test_key')
            
    async def test_exists_check(self, cache_engine, mock_redis):
        mock_redis.exists.return_value = 1
        
        result = await cache_engine.exists('test_key')
        
        assert result is True
        mock_redis.exists.assert_called_once_with('test_key')