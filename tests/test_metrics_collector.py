import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time

from src.metrics.collector import MetricsCollector
from src.core.errors import CacheError


class TestMetricsCollector:
    
    def setup_method(self):
        self.collector = MetricsCollector()
        
    def test_record_hit(self):
        """Test recording cache hits"""
        self.collector.record_hit("key1")
        self.collector.record_hit("key2")
        
        assert self.collector.total_hits == 2
        assert self.collector.hit_count["key1"] == 1
        assert self.collector.hit_count["key2"] == 1
        
    def test_record_miss(self):
        """Test recording cache misses"""
        self.collector.record_miss("key1")
        self.collector.record_miss("key1")
        
        assert self.collector.total_misses == 2
        assert self.collector.miss_count["key1"] == 2
        
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        # Record some hits and misses
        self.collector.record_hit("key1")
        self.collector.record_hit("key2")
        self.collector.record_miss("key3")
        
        hit_rate = self.collector.get_hit_rate()
        assert hit_rate == 2/3  # 2 hits, 1 miss
        
    def test_hit_rate_no_operations(self):
        """Test hit rate when no operations recorded"""
        hit_rate = self.collector.get_hit_rate()
        assert hit_rate == 0.0
        
    def test_record_latency(self):
        """Test recording operation latency"""
        self.collector.record_latency("get", 0.05)
        self.collector.record_latency("set", 0.02)
        self.collector.record_latency("get", 0.03)
        
        latencies = self.collector.get_latency_stats()
        assert "get" in latencies
        assert "set" in latencies
        assert latencies["get"]["count"] == 2
        assert latencies["get"]["avg"] == 0.04  # (0.05 + 0.03) / 2
        
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        self.collector.record_memory_usage(1024)
        self.collector.record_memory_usage(2048)
        
        stats = self.collector.get_memory_stats()
        assert stats["current"] == 2048
        assert stats["peak"] == 2048
        assert stats["avg"] == 1536  # (1024 + 2048) / 2
        
    @patch('time.time')
    def test_time_window_metrics(self, mock_time):
        """Test metrics collection within time windows"""
        # Mock time progression
        mock_time.side_effect = [1000, 1030, 1060, 1090]  # 30s intervals
        
        self.collector.record_hit("key1")  # t=1000
        mock_time.return_value = 1030
        self.collector.record_hit("key2")  # t=1030  
        mock_time.return_value = 1060
        self.collector.record_miss("key3")  # t=1060
        
        # Get metrics for last 60 seconds
        recent_metrics = self.collector.get_windowed_metrics(window_seconds=60)
        assert recent_metrics["hits"] == 2
        assert recent_metrics["misses"] == 1
        
    def test_error_tracking(self):
        """Test error event tracking"""
        error1 = CacheError("Connection failed")
        error2 = ValueError("Invalid key")
        
        self.collector.record_error(error1, "redis_connection")
        self.collector.record_error(error2, "key_validation")
        
        error_stats = self.collector.get_error_stats()
        assert error_stats["total_errors"] == 2
        assert "redis_connection" in error_stats["by_context"]
        assert "key_validation" in error_stats["by_context"]
        
    def test_reset_metrics(self):
        """Test resetting all metrics"""
        self.collector.record_hit("key1")
        self.collector.record_miss("key2")
        self.collector.record_latency("get", 0.05)
        
        self.collector.reset()
        
        assert self.collector.total_hits == 0
        assert self.collector.total_misses == 0
        assert len(self.collector.latencies) == 0