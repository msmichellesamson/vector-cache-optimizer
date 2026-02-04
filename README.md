# Vector Cache Optimizer

Intelligent embedding cache with ML-driven eviction policies and real-time performance optimization.

## Features
- ML-powered cache hit prediction
- Dynamic TTL optimization based on access patterns
- Circuit breaker for Redis failures
- Real-time memory monitoring and alerting
- Horizontal pod autoscaling based on cache performance

## Architecture
```
[Embedding API] → [Cache Engine] → [Redis Cluster]
       ↓               ↓
[ML Predictor] → [TTL Optimizer]
       ↓               ↓
[Prometheus] ← [Metrics Collector]
```

## Quick Start
```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/

# Build and run locally
docker build -f docker/Dockerfile -t vector-cache .
docker run -p 8080:8080 vector-cache
```

## Configuration
- Redis connection: `REDIS_URL`
- Cache size limit: `MAX_MEMORY_MB`
- ML model path: `MODEL_PATH`
- Metrics port: `METRICS_PORT=9090`

## Monitoring
- Prometheus metrics on `:9090/metrics`
- Grafana dashboards for cache performance
- Alerting on memory usage and hit rate degradation
- HPA scaling based on CPU, memory, and cache hit rate

## ML Features
- Hit probability prediction using LRU access patterns
- Dynamic TTL adjustment based on predicted access frequency
- Memory pressure-aware eviction policies

## Tech Stack
- **Backend**: Python, FastAPI, Redis
- **ML**: scikit-learn, pandas
- **Infrastructure**: Terraform, GCP, Kubernetes
- **Monitoring**: Prometheus, Grafana
- **DevOps**: Docker, GitHub Actions, HPA