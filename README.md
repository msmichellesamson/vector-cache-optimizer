# Vector Cache Optimizer

Intelligent embedding cache with ML-driven eviction policies and real-time performance optimization.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cache Client  │───▶│  Cache Manager  │───▶│  Redis Cluster  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ML Hit Predictor│    │ Eviction Engine │    │ Monitoring Stack│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/

# Run locally
docker build -t vector-cache .
docker run -p 8080:8080 vector-cache
```

## API Reference

### Cache Operations

#### Store Vector
```python
POST /cache/store
Content-Type: application/json

{
    "key": "doc_123",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": {"type": "document", "source": "pdf"},
    "ttl": 3600
}
```

**Response:**
```json
{
    "success": true,
    "key": "doc_123",
    "stored_at": "2024-01-15T10:30:00Z",
    "predicted_hits": 23
}
```

#### Retrieve Vector
```python
GET /cache/retrieve/{key}
```

**Response:**
```json
{
    "key": "doc_123",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": {"type": "document"},
    "hit_count": 15,
    "last_accessed": "2024-01-15T11:45:00Z"
}
```

#### Similarity Search
```python
POST /cache/search
Content-Type: application/json

{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "limit": 10,
    "threshold": 0.8,
    "filters": {"type": "document"}
}
```

**Response:**
```json
{
    "results": [
        {
            "key": "doc_123",
            "similarity": 0.92,
            "vector": [0.1, 0.2, 0.3, ...],
            "metadata": {"type": "document"}
        }
    ],
    "total_found": 1,
    "search_time_ms": 15
}
```

### Metrics & Health

#### Cache Stats
```python
GET /metrics/stats
```

**Response:**
```json
{
    "hit_rate": 0.85,
    "memory_usage": 0.72,
    "evictions_last_hour": 42,
    "total_keys": 15234,
    "avg_retrieval_time_ms": 2.3
}
```

#### Health Check
```python
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "redis_connected": true,
    "ml_model_loaded": true,
    "memory_pressure": "low",
    "uptime_seconds": 86400
}
```

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secretpassword
REDIS_CLUSTER_NODES=node1:6379,node2:6379,node3:6379

# ML Configuration
ML_MODEL_PATH=/models/hit_predictor.pkl
ML_PREDICTION_THRESHOLD=0.7
ML_RETRAIN_INTERVAL_HOURS=24

# Cache Configuration
DEFAULT_TTL_SECONDS=3600
MAX_MEMORY_USAGE=0.8
EVICTION_BATCH_SIZE=100

# Monitoring
PROMETHEUS_PORT=9090
METRICS_EXPORT_INTERVAL=30
ALERT_MEMORY_THRESHOLD=0.9
```

## Troubleshooting

### High Memory Usage
**Symptoms:** Memory usage > 90%, frequent evictions

**Solutions:**
1. Check eviction policy efficiency:
   ```bash
   curl localhost:8080/metrics/eviction-stats
   ```
2. Analyze hit rate patterns:
   ```bash
   kubectl logs -f deployment/vector-cache | grep "hit_rate"
   ```
3. Scale Redis cluster:
   ```bash
   kubectl scale statefulset redis --replicas=6
   ```

### Low Hit Rate
**Symptoms:** Hit rate < 60%, ML predictor accuracy dropping

**Solutions:**
1. Retrain ML model:
   ```bash
   curl -X POST localhost:8080/ml/retrain
   ```
2. Check similarity threshold:
   ```python
   # Adjust in config/redis_config.py
   SIMILARITY_THRESHOLD = 0.75  # Lower for more matches
   ```
3. Analyze cache access patterns:
   ```bash
   kubectl exec -it redis-0 -- redis-cli MONITOR
   ```

### Redis Connection Issues
**Symptoms:** Connection timeouts, circuit breaker open

**Solutions:**
1. Check Redis cluster health:
   ```bash
   kubectl get pods -l app=redis
   kubectl logs redis-0
   ```
2. Verify network policies:
   ```bash
   kubectl get networkpolicies
   ```
3. Reset connection pool:
   ```bash
   curl -X POST localhost:8080/admin/reset-connections
   ```

### ML Model Performance Issues
**Symptoms:** Prediction latency > 10ms, accuracy < 70%

**Solutions:**
1. Check model metrics:
   ```bash
   curl localhost:8080/ml/metrics
   ```
2. Force model reload:
   ```bash
   curl -X POST localhost:8080/ml/reload
   ```
3. Inspect feature distribution:
   ```python
   # Check logs for feature drift warnings
   kubectl logs deployment/vector-cache | grep "feature_drift"
   ```

## Monitoring

- **Grafana Dashboard:** `http://grafana.local/d/vector-cache`
- **Prometheus Metrics:** `http://prometheus.local:9090`
- **Application Logs:** `kubectl logs -f deployment/vector-cache`

## Technology Stack

- **Language:** Python 3.11+
- **Cache:** Redis Cluster
- **ML:** scikit-learn, numpy
- **Monitoring:** Prometheus, Grafana
- **Infrastructure:** Terraform, Kubernetes, GCP
- **Deployment:** Docker, GitHub Actions