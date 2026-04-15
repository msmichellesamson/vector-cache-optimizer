# Vector Cache Optimizer

Intelligent embedding cache with ML-driven eviction policies and real-time performance optimization.

## Overview

A production-ready vector caching system that uses machine learning to predict cache hits, optimize memory usage, and maintain high performance under varying workloads.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client API    │    │   ML Pipeline   │    │  Redis Cluster  │
│                 │    │                 │    │                 │
│ • REST/gRPC     │◄───┤ • Hit Predictor │◄───┤ • Sharded       │
│ • Batch Ops     │    │ • Pattern Learn │    │ • Replicated    │
│ • Health Check  │    │ • Drift Detect  │    │ • Monitored     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │  Cache Engine   │    │ Infrastructure  │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Eviction      │    │ • Terraform     │
│ • Grafana       │    │ • Warming       │    │ • Kubernetes    │
│ • Alerting      │    │ • Circuit Break │    │ • Docker        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Core Caching
- **Intelligent Eviction**: ML-driven policies based on access patterns
- **Vector Deduplication**: Semantic similarity detection to reduce memory
- **Batch Processing**: Optimized bulk operations with async support
- **Connection Pooling**: Resilient Redis connectivity with health monitoring
- **Circuit Breakers**: Fault tolerance and graceful degradation

### Machine Learning
- **Hit Prediction**: XGBoost model predicts cache hit probability
- **Pattern Learning**: Adapts to changing access patterns over time
- **Drift Detection**: Monitors model performance and triggers retraining
- **Vector Clustering**: Groups similar embeddings for optimized storage
- **Similarity Analysis**: Semantic deduplication and compression

### Monitoring & Observability
- **Real-time Metrics**: Hit rates, memory usage, performance counters
- **Structured Logging**: JSON logs with correlation IDs and context
- **Health Checks**: Endpoint monitoring and dependency validation
- **Memory Pressure**: Automatic scaling and eviction under load
- **Fragmentation Analysis**: Memory optimization and defragmentation
- **Connection Pool Health**: Pool exhaustion and failure detection

### Infrastructure
- **Multi-AZ Redis**: GCP Memorystore with automatic failover
- **Kubernetes Ready**: HPA, PDB, NetworkPolicies, ServiceMonitor
- **Terraform Managed**: Complete GCP infrastructure as code
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment

## Quick Start

### Local Development

```bash
# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Install dependencies
pip install -r requirements.txt

# Run service
python -m src.main

# Run tests
python -m pytest tests/ -v
```

### Production Deployment

```bash
# Deploy infrastructure
cd terraform/gcp
terraform init && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -l app=vector-cache
```

## Configuration

### Environment Variables

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
CACHE_SIZE_GB=4
ML_MODEL_PATH=/app/models/hit_predictor.pkl
LOG_LEVEL=INFO
METRICS_PORT=8080
HEALTH_CHECK_PORT=8081
```

### ML Model Configuration

```yaml
model:
  type: xgboost
  retrain_threshold: 0.05  # Drift detection
  feature_window: 1000     # Training samples
  prediction_cache_ttl: 300
  
eviction:
  policy: ml_guided
  fallback: lru
  batch_size: 100
```

## API Reference

### Vector Operations

```python
# Store vector
POST /vectors
{
  "id": "doc_123",
  "vector": [0.1, 0.2, ...],
  "metadata": {"type": "document"}
}

# Retrieve vector
GET /vectors/doc_123

# Batch similarity search
POST /search/batch
{
  "queries": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "k": 10,
  "threshold": 0.8
}
```

### Monitoring Endpoints

```bash
# Health check
curl http://localhost:8081/health

# Metrics (Prometheus format)
curl http://localhost:8080/metrics

# Cache statistics
curl http://localhost:8080/stats
```

## Performance

### Benchmarks (on GKE n2-standard-4)

- **Throughput**: 50,000 ops/sec (mixed read/write)
- **Latency**: P99 < 10ms for vector retrieval
- **Memory Efficiency**: 40% reduction via deduplication
- **Hit Rate**: 85%+ with ML-guided eviction
- **Availability**: 99.9% uptime with circuit breakers

### Scaling Characteristics

- **Horizontal**: Linear scaling to 100+ pods
- **Vertical**: Supports up to 32GB cache per instance
- **Redis Cluster**: Scales to TB-scale datasets
- **Model Serving**: <1ms ML prediction latency

## Architecture Decisions

### Technology Choices

**Python + asyncio**: High-performance async I/O, rich ML ecosystem
**Redis Cluster**: Proven scalability, atomic operations, persistence
**XGBoost**: Fast inference, handles mixed data types, interpretable
**Prometheus**: Industry standard metrics, excellent Kubernetes integration
**Terraform**: Infrastructure as code, GCP provider maturity

### Design Patterns

**Circuit Breaker**: Prevents cascade failures, graceful degradation
**Connection Pooling**: Resource efficiency, connection reuse
**Batch Processing**: Reduces network overhead, improves throughput
**Event-Driven ML**: Real-time adaptation to changing patterns

## Monitoring

### Key Metrics

- `cache_hit_rate`: Overall hit rate percentage
- `cache_memory_usage`: Current memory utilization
- `ml_prediction_accuracy`: Model performance score
- `redis_connection_pool_active`: Active connections
- `eviction_policy_effectiveness`: ML vs LRU comparison

### Alerts

- Hit rate drops below 70%
- Memory usage exceeds 90%
- Connection pool exhaustion
- Model drift detection
- High error rates (>1%)

## Development

### Project Structure

```
src/
├── core/           # Cache engine, connection management
├── ml/             # ML models, pattern learning
├── monitoring/     # Metrics, logging, health checks
├── config/         # Configuration management
└── main.py         # Application entry point

tests/              # Comprehensive test suite
terraform/          # Infrastructure as code
k8s/                # Kubernetes manifests
docker/             # Container definitions
```

### Contributing

1. All changes must include tests
2. Follow type hints and error handling patterns
3. Update README.md with any configuration changes
4. Ensure CI pipeline passes

### Testing

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests (requires Redis)
python -m pytest tests/integration/ -v

# Performance tests
python -m pytest tests/performance/ --benchmark-only
```

## License

MIT License - see LICENSE file for details.

---

**Skills Demonstrated**: AI/ML (XGBoost, embeddings, pattern learning), Infrastructure (Terraform, GCP, Redis), SRE (monitoring, alerting, health checks), Backend (async APIs, connection pooling), Database (Redis optimization), DevOps (K8s, CI/CD), Data Engineering (vector processing, deduplication)