import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
import asyncio
import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import MetricsHandler
from http.server import HTTPServer
import threading


logger = structlog.get_logger(__name__)


class MetricsCollectionError(Exception):
    """Raised when metrics collection fails"""
    pass


class MetricsExportError(Exception):
    """Raised when metrics export fails"""
    pass


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    memory_usage_bytes: int = 0
    avg_lookup_time_ms: float = 0.0
    ml_prediction_accuracy: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics"""
    timestamp: datetime
    cache_size: int
    hit_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_pressure: float
    prediction_confidence: float


class MetricsCollector:
    """Production metrics collector for vector cache optimizer"""
    
    def __init__(self, port: int = 8000, collection_interval: float = 10.0):
        """Initialize metrics collector
        
        Args:
            port: Prometheus metrics endpoint port
            collection_interval: How often to collect metrics in seconds
        """
        self.port = port
        self.collection_interval = collection_interval
        self.registry = CollectorRegistry()
        self._running = False
        self._server: Optional[HTTPServer] = None
        self._collection_task: Optional[asyncio.Task] = None
        
        # Cache operation metrics
        self.cache_hits = Counter(
            'vector_cache_hits_total',
            'Total number of cache hits',
            ['cache_type', 'namespace'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'vector_cache_misses_total',
            'Total number of cache misses',
            ['cache_type', 'namespace'],
            registry=self.registry
        )
        
        self.cache_evictions = Counter(
            'vector_cache_evictions_total',
            'Total number of cache evictions',
            ['eviction_reason', 'namespace'],
            registry=self.registry
        )
        
        self.cache_insertions = Counter(
            'vector_cache_insertions_total',
            'Total number of cache insertions',
            ['namespace'],
            registry=self.registry
        )
        
        # Performance metrics
        self.lookup_duration = Histogram(
            'vector_cache_lookup_duration_seconds',
            'Time spent on cache lookups',
            ['operation', 'namespace'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self.registry
        )
        
        self.ml_prediction_duration = Histogram(
            'vector_cache_ml_prediction_duration_seconds',
            'Time spent on ML predictions',
            ['model_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self.registry
        )
        
        # State metrics
        self.cache_size = Gauge(
            'vector_cache_size_items',
            'Current number of items in cache',
            ['namespace'],
            registry=self.registry
        )
        
        self.cache_memory_usage = Gauge(
            'vector_cache_memory_usage_bytes',
            'Current memory usage of cache',
            ['namespace'],
            registry=self.registry
        )
        
        self.hit_rate = Gauge(
            'vector_cache_hit_rate',
            'Current cache hit rate (0-1)',
            ['namespace'],
            registry=self.registry
        )
        
        self.ml_prediction_accuracy = Gauge(
            'vector_cache_ml_prediction_accuracy',
            'Current ML prediction accuracy (0-1)',
            ['model_type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_pressure = Gauge(
            'vector_cache_memory_pressure',
            'Memory pressure indicator (0-1)',
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'vector_cache_active_connections',
            'Number of active client connections',
            registry=self.registry
        )
        
        # Historical metrics storage
        self._performance_history: List[PerformanceSnapshot] = []
        self._max_history_size = 1000
        
    async def start(self) -> None:
        """Start metrics collection and HTTP server"""
        if self._running:
            logger.warning("Metrics collector already running")
            return
            
        try:
            self._running = True
            
            # Start HTTP server for Prometheus scraping
            self._start_http_server()
            
            # Start collection task
            self._collection_task = asyncio.create_task(self._collection_loop())
            
            logger.info("Metrics collector started", port=self.port)
            
        except Exception as e:
            self._running = False
            raise MetricsCollectionError(f"Failed to start metrics collector: {e}") from e
    
    async def stop(self) -> None:
        """Stop metrics collection"""
        if not self._running:
            return
            
        self._running = False
        
        # Stop collection task
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        # Stop HTTP server
        if self._server:
            self._server.shutdown()
            self._server = None
            
        logger.info("Metrics collector stopped")
    
    def _start_http_server(self) -> None:
        """Start HTTP server for metrics endpoint"""
        try:
            class CustomMetricsHandler(MetricsHandler):
                def __init__(self, registry):
                    self.registry = registry
                
                def do_GET(self):
                    try:
                        output = generate_latest(self.registry)
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(output)
                    except Exception as e:
                        logger.error("Failed to generate metrics", error=str(e))
                        self.send_error(500)
            
            handler = lambda *args: CustomMetricsHandler(self.registry)(*args)
            self._server = HTTPServer(('', self.port), handler)
            
            # Start server in background thread
            server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True
            )
            server_thread.start()
            
        except Exception as e:
            raise MetricsExportError(f"Failed to start HTTP server: {e}") from e
    
    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            # Memory pressure (simplified - in production would check actual system memory)
            import psutil
            memory = psutil.virtual_memory()
            self.memory_pressure.set(memory.percent / 100.0)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def record_cache_hit(self, cache_type: str = "vector", namespace: str = "default") -> None:
        """Record a cache hit"""
        self.cache_hits.labels(cache_type=cache_type, namespace=namespace).inc()
        self._update_hit_rate(namespace)
    
    def record_cache_miss(self, cache_type: str = "vector", namespace: str = "default") -> None:
        """Record a cache miss"""
        self.cache_misses.labels(cache_type=cache_type, namespace=namespace).inc()
        self._update_hit_rate(namespace)
    
    def record_cache_eviction(self, reason: str, namespace: str = "default") -> None:
        """Record a cache eviction"""
        self.cache_evictions.labels(eviction_reason=reason, namespace=namespace).inc()
    
    def record_cache_insertion(self, namespace: str = "default") -> None:
        """Record a cache insertion"""
        self.cache_insertions.labels(namespace=namespace).inc()
    
    def record_lookup_duration(self, duration_seconds: float, operation: str = "get", 
                              namespace: str = "default") -> None:
        """Record cache lookup duration"""
        self.lookup_duration.labels(operation=operation, namespace=namespace).observe(duration_seconds)
    
    def record_ml_prediction_duration(self, duration_seconds: float, model_type: str) -> None:
        """Record ML prediction duration"""
        self.ml_prediction_duration.labels(model_type=model_type).observe(duration_seconds)
    
    def update_cache_size(self, size: int, namespace: str = "default") -> None:
        """Update cache size metric"""
        self.cache_size.labels(namespace=namespace).set(size)
    
    def update_memory_usage(self, bytes_used: int, namespace: str = "default") -> None:
        """Update memory usage metric"""
        self.cache_memory_usage.labels(namespace=namespace).set(bytes_used)
    
    def update_ml_accuracy(self, accuracy: float, model_type: str) -> None:
        """Update ML prediction accuracy"""
        self.ml_prediction_accuracy.labels(model_type=model_type).set(accuracy)
    
    def update_active_connections(self, count: int) -> None:
        """Update active connections count"""
        self.active_connections.set(count)
    
    def _update_hit_rate(self, namespace: str) -> None:
        """Calculate and update hit rate"""
        try:
            hits_metric = self.cache_hits.labels(cache_type="vector", namespace=namespace)
            misses_metric = self.cache_misses.labels(cache_type="vector", namespace=namespace)
            
            hits = hits_metric._value._value if hasattr(hits_metric._value, '_value') else 0
            misses = misses_metric._value._value if hasattr(misses_metric._value, '_value') else 0
            
            total = hits + misses
            if total > 0:
                hit_rate = hits / total
                self.hit_rate.labels(namespace=namespace).set(hit_rate)
                
        except Exception as e:
            logger.error("Failed to update hit rate", namespace=namespace, error=str(e))
    
    def create_performance_snapshot(self, cache_size: int, p50_latency: float,
                                  p95_latency: float, p99_latency: float,
                                  prediction_confidence: float = 0.0,
                                  namespace: str = "default") -> PerformanceSnapshot:
        """Create a performance snapshot"""
        try:
            # Get current hit rate
            current_hit_rate = 0.0
            try:
                hit_rate_metric = self.hit_rate.labels(namespace=namespace)
                current_hit_rate = hit_rate_metric._value._value if hasattr(hit_rate_metric._value, '_value') else 0.0
            except:
                pass
            
            # Get memory pressure
            memory_pressure_value = 0.0
            try:
                memory_pressure_value = self.memory_pressure._value._value if hasattr(self.memory_pressure._value, '_value') else 0.0
            except:
                pass
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cache_size=cache_size,
                hit_rate=current_hit_rate,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                memory_pressure=memory_pressure_value,
                prediction_confidence=prediction_confidence
            )
            
            # Store in history
            self._performance_history.append(snapshot)
            
            # Trim history if needed
            if len(self._performance_history) > self._max_history_size:
                self._performance_history = self._performance_history[-self._max_history_size:]
            
            return snapshot
            
        except Exception as e:
            logger.error("Failed to create performance snapshot", error=str(e))
            raise MetricsCollectionError(f"Failed to create performance snapshot: {e}") from e
    
    def get_performance_history(self, limit: Optional[int] = None) -> List[PerformanceSnapshot]:
        """Get performance history"""
        if limit:
            return self._performance_history[-limit:]
        return self._performance_history.copy()
    
    def get_current_metrics(self, namespace: str = "default") -> CacheMetrics:
        """Get current metrics snapshot"""
        try:
            # Extract current values from prometheus metrics
            hits = 0
            misses = 0
            try:
                hits_metric = self.cache_hits.labels(cache_type="vector", namespace=namespace)
                misses_metric = self.cache_misses.labels(cache_type="vector", namespace=namespace)
                hits = hits_metric._value._value if hasattr(hits_metric._value, '_value') else 0
                misses = misses_metric._value._value if hasattr(misses_metric._value, '_value') else 0
            except:
                pass
            
            return CacheMetrics(
                hits=int(hits),
                misses=int(misses),
                memory_usage_bytes=int(self._get_gauge_value(self.cache_memory_usage, namespace)),
                avg_lookup_time_ms=self._calculate_avg_lookup_time(namespace),
                ml_prediction_accuracy=self._get_gauge_value(self.ml_prediction_accuracy, "default")
            )
            
        except Exception as e:
            logger.error("Failed to get current metrics", error=str(e))
            return CacheMetrics()
    
    def _get_gauge_value(self, gauge, *labels) -> float:
        """Safely get gauge value"""
        try:
            if labels:
                metric = gauge.labels(*labels)
            else:
                metric = gauge
            return metric._value._value if hasattr(metric._value, '_value') else 0.0
        except:
            return 0.0
    
    def _calculate_avg_lookup_time(self, namespace: str) -> float:
        """Calculate average lookup time from histogram"""
        try:
            histogram = self.lookup_duration.labels(operation="get", namespace=namespace)
            if hasattr(histogram, '_sum') and hasattr(histogram, '_count'):
                count = histogram._count._value if hasattr(histogram._count, '_value') else 0
                if count > 0:
                    sum_value = histogram._sum._value if hasattr(histogram._sum, '_value') else 0
                    return (sum_value / count) * 1000  # Convert to milliseconds
            return 0.0
        except:
            return 0.0