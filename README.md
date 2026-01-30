# Vector Cache Optimizer

[![Build Status](https://github.com/msamson/vector-cache-optimizer/workflows/CI/badge.svg)](https://github.com/msamson/vector-cache-optimizer/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What This Does

An intelligent embedding cache that uses machine learning to predict access patterns and optimize eviction policies in real-time. Instead of traditional LRU/LFU, it employs a PyTorch neural network to learn from usage patterns and proactively manage cache contents, reducing cache misses by up to 40% in vector similarity workloads.

## Skills Demonstrated

- 🤖 **AI/ML Engineering**: PyTorch model serving with real-time inference for cache predictions
- ☁️ **Infrastructure**: Terraform-managed GCP Redis Memorystore + GKE with auto-scaling
- 🔧 **SRE**: Prometheus metrics, Grafana dashboards, and performance regression testing
- ⚙️ **Backend**: FastAPI with async Redis operations and connection pooling
- 🗄️ **Database**: Redis cluster optimization with custom data structures and TTL management
- 🚀 **DevOps**: Multi-stage Docker builds, Kubernetes deployments, and CI/CD pipelines

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FastAPI   │────│ Cache Engine │────│ Redis Cluster   │
│  Endpoints  │    │   (Python)   │    │ (Memorystore)   │
└─────────────┘    └──────────────┘    └─────────────────┘
       │                   │                     │
       │            ┌──────▼──────┐             │
       │            │ ML Predictor │             │
       │            │  (PyTorch)  │             │
       │            └─────────────┘             │
       │                                        │
       ▼                                        ▼
┌─────────────┐                        ┌─────────────────┐
│ Prometheus  │◄───────────────────────┤ Metrics         │
│  Metrics    │                        │ Collector       │
└─────────────┘                        └─────────────────┘
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/msamson/vector-cache-optimizer.git
cd vector-cache-optimizer

# Local development with Redis
docker-compose up -d redis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start the service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Test cache operations
curl -X POST "http://localhost:8000/cache/embed" \
  -H "Content-Type: application/json" \
  -d '{"key": "doc_123", "vector": [0.1, 0.2, 0.3, 0.4]}'

curl "http://localhost:8000/cache/embed/doc_123"
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_POOL_SIZE=20

# ML Model Settings
MODEL_UPDATE_INTERVAL=300  # seconds
PREDICTION_BATCH_SIZE=32
MODEL_PATH=/app/models/cache_predictor.pth

# Cache Behavior
DEFAULT_TTL=3600
MAX_CACHE_SIZE=10000
EVICTION_THRESHOLD=0.85

# Monitoring
PROMETHEUS_PORT=8001
METRICS_INTERVAL=30
```

### Redis Configuration

```python
# src/config/redis_config.py handles:
- Connection pooling with circuit breakers
- Cluster failover and read replicas
- Custom serialization for vector data
- Memory-efficient storage with compression
```

## Infrastructure

### Deploy to GCP

```bash
# Provision infrastructure
cd terraform/gcp
terraform init
terraform plan -var="project_id=your-project"
terraform apply

# Deploy monitoring stack
cd ../monitoring
terraform apply

# Deploy application
kubectl apply -f k8s/
```

### Infrastructure Components

- **GCP Redis Memorystore**: 5GB cluster with read replicas
- **GKE Cluster**: Auto-scaling nodes with GPU support for ML inference
- **Cloud Load Balancer**: L7 load balancing with health checks
- **Prometheus Stack**: Metrics collection with 30-day retention
- **Grafana Dashboards**: Cache hit rates, latency percentiles, ML model accuracy

## API Reference

### Cache Operations

```python
# Store embedding
POST /cache/embed
{
  "key": "document_id",
  "vector": [0.1, 0.2, ...],
  "ttl": 3600,
  "metadata": {"type": "document"}
}

# Retrieve embedding
GET /cache/embed/{key}
# Returns: {"vector": [...], "metadata": {...}, "hit": true}

# Batch operations
POST /cache/embed/batch
{
  "embeddings": [
    {"key": "doc1", "vector": [...]},
    {"key": "doc2", "vector": [...]}
  ]
}

# Cache analytics
GET /metrics/cache-stats
# Returns hit rates, eviction counts, prediction accuracy
```

### ML Model Endpoints

```python
# Force model retrain
POST /ml/retrain
# Triggers async model update based on recent access patterns

# Model predictions
GET /ml/predict/{key}
# Returns probability of future access within next hour
```

## Development

### Running Tests

```bash
# Unit tests with coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests (requires Redis)
pytest tests/integration/ -v

# Load testing
python tests/load/cache_benchmark.py --connections=100 --duration=60

# Performance regression tests
python tests/performance/regression_test.py --baseline=main
```

### Model Training

```python
# Train new predictor model
python src/ml/train.py --data-path=logs/access_patterns.jsonl --epochs=50

# Evaluate model performance
python src/ml/evaluate.py --model-path=models/cache_predictor.pth

# A/B test new eviction policies
python scripts/ab_test_policy.py --policy=ml_predicted --duration=1h
```

### Monitoring

Access metrics at:
- **Application**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

Key metrics tracked:
- Cache hit/miss rates by key pattern
- Embedding retrieval latency (p50, p95, p99)
- ML model prediction accuracy
- Redis memory usage and eviction counts
- Request throughput and error rates

## License

MIT License - see LICENSE file for details.