"""Unit tests for batch processor."""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from src.core.batch_processor import BatchProcessor
from src.core.errors import CacheError


class TestBatchProcessor:
    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.get_many = AsyncMock(return_value={})
        cache.set_many = AsyncMock(return_value=True)
        cache.delete_many = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def processor(self, mock_cache):
        return BatchProcessor(cache=mock_cache, batch_size=3, flush_interval=0.1)

    @pytest.mark.asyncio
    async def test_batch_get_operations(self, processor, mock_cache):
        """Test batching of get operations."""
        mock_cache.get_many.return_value = {"key1": b"val1", "key2": b"val2"}
        
        # Start processor
        task = asyncio.create_task(processor.start())
        
        # Queue operations
        result1 = await processor.queue_get("key1")
        result2 = await processor.queue_get("key2")
        
        # Verify results
        assert result1 == b"val1"
        assert result2 == b"val2"
        mock_cache.get_many.assert_called_once_with(["key1", "key2"])
        
        processor.stop()
        await task

    @pytest.mark.asyncio
    async def test_batch_set_operations(self, processor, mock_cache):
        """Test batching of set operations."""
        task = asyncio.create_task(processor.start())
        
        # Queue operations
        await processor.queue_set("key1", b"val1", ttl=300)
        await processor.queue_set("key2", b"val2", ttl=300)
        
        # Wait for batch to flush
        await asyncio.sleep(0.15)
        
        mock_cache.set_many.assert_called_once()
        args = mock_cache.set_many.call_args[0][0]
        assert "key1" in args
        assert "key2" in args
        
        processor.stop()
        await task

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, processor, mock_cache):
        """Test that batches flush when size limit is reached."""
        task = asyncio.create_task(processor.start())
        
        # Queue 4 operations (batch_size=3)
        await processor.queue_set("key1", b"val1")
        await processor.queue_set("key2", b"val2")
        await processor.queue_set("key3", b"val3")
        await processor.queue_set("key4", b"val4")
        
        await asyncio.sleep(0.05)  # Allow batch processing
        
        # Should have 2 calls: one for first 3, one pending
        assert mock_cache.set_many.call_count >= 1
        
        processor.stop()
        await task

    @pytest.mark.asyncio
    async def test_error_handling(self, processor, mock_cache):
        """Test error handling in batch operations."""
        mock_cache.get_many.side_effect = Exception("Redis error")
        
        task = asyncio.create_task(processor.start())
        
        with pytest.raises(CacheError):
            await processor.queue_get("key1")
        
        processor.stop()
        await task