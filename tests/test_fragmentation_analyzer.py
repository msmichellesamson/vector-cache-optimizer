import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.monitoring.fragmentation_analyzer import FragmentationAnalyzer
from src.core.errors import CacheError


class TestFragmentationAnalyzer:
    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.info.return_value = {
            'used_memory': 1024 * 1024 * 100,  # 100MB
            'used_memory_rss': 1024 * 1024 * 150,  # 150MB
            'mem_fragmentation_ratio': 1.5
        }
        return mock
    
    @pytest.fixture
    def analyzer(self, mock_redis):
        return FragmentationAnalyzer(mock_redis, threshold=2.0)
    
    @pytest.mark.asyncio
    async def test_analyze_normal_fragmentation(self, analyzer, mock_redis):
        result = await analyzer.analyze()
        
        assert result['fragmentation_ratio'] == 1.5
        assert result['fragmentation_level'] == 'normal'
        assert result['memory_overhead_mb'] == 50
        assert not result['needs_defrag']
    
    @pytest.mark.asyncio
    async def test_analyze_high_fragmentation(self, analyzer, mock_redis):
        mock_redis.info.return_value = {
            'used_memory': 1024 * 1024 * 100,
            'used_memory_rss': 1024 * 1024 * 250,
            'mem_fragmentation_ratio': 2.5
        }
        
        result = await analyzer.analyze()
        
        assert result['fragmentation_ratio'] == 2.5
        assert result['fragmentation_level'] == 'high'
        assert result['needs_defrag']
    
    @pytest.mark.asyncio
    async def test_analyze_redis_error(self, analyzer, mock_redis):
        mock_redis.info.side_effect = Exception("Redis connection failed")
        
        with pytest.raises(CacheError, match="fragmentation analysis"):
            await analyzer.analyze()
    
    @pytest.mark.asyncio
    async def test_get_defrag_recommendation_low_fragmentation(self, analyzer, mock_redis):
        recommendation = await analyzer.get_defrag_recommendation()
        
        assert recommendation['action'] == 'none'
        assert 'fragmentation is within normal range' in recommendation['reason']
    
    @pytest.mark.asyncio 
    async def test_get_defrag_recommendation_high_fragmentation(self, analyzer, mock_redis):
        mock_redis.info.return_value = {
            'used_memory': 1024 * 1024 * 500,  # 500MB
            'used_memory_rss': 1024 * 1024 * 1500,  # 1.5GB
            'mem_fragmentation_ratio': 3.0
        }
        
        recommendation = await analyzer.get_defrag_recommendation()
        
        assert recommendation['action'] == 'schedule_defrag'
        assert 'HIGH' in recommendation['priority']
        assert recommendation['estimated_recovery_mb'] > 0
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, analyzer):
        """Benchmark analyzer performance with high call frequency."""
        import time
        
        start_time = time.perf_counter()
        
        # Run 100 analysis calls concurrently
        tasks = [analyzer.analyze() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        
        # Should complete 100 calls in under 5 seconds
        assert end_time - start_time < 5.0
        assert len(results) == 100
        assert all('fragmentation_ratio' in result for result in results)