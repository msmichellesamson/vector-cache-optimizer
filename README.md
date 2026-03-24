# Vector Cache Optimizer

Intelligent embedding cache with ML-driven eviction policies and real-time performance optimization.

## Overview

This system combines Redis clustering with machine learning to optimize vector embedding cache performance. It predicts cache hit patterns and dynamically adjusts eviction strategies to maximize hit rates while maintaining memory efficiency.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Predictor  │    │  Cache Engine   │    │  Redis Cluster  │
│   Hit Patterns  │◄──►│  Eviction Logic │◄──►│   Sharded Data  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │  Alert Engine   │    │  Memory Monitor │
│    Metrics      │    │  SLA Tracking   │    │  Circuit Breaker│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Features

- **ML-Driven Eviction**: Predicts cache hit probability using access patterns
- **Redis Clustering**: Horizontal scaling with automatic sharding
- **Circuit Breakers**: Resilience patterns for Redis connection failures
- **Real-time Monitoring**: Prometheus metrics with Grafana dashboards
- **TTL Optimization**: Dynamic TTL adjustment based on access frequency
- **Memory Management**: Intelligent memory pressure handling

## Quick Start

1. **Deploy Infrastructure**:
   ```bash
   cd terraform
   terraform init && terraform apply
   ```

2. **Deploy Application**:
   ```bash
   kubectl apply -f k8s/
   ```

3. **Access API**:
   ```bash
   curl http://localhost:8080/health
   ```

## API Documentation

See [API Documentation](docs/api.md) for complete endpoint reference.

## Configuration

### Redis Configuration
- Cluster mode with 3 nodes minimum
- Memory policy: `allkeys-lru` with ML override
- Persistence: RDB + AOF for durability

### ML Model Configuration
```python
# Hit prediction model parameters
HIT_PREDICTION_FEATURES = [
    'access_frequency',
    'time_since_last_access', 
    'embedding_similarity',
    'query_pattern_match'
]
```

## Monitoring

### Key Metrics
- Cache hit rate (target: >85%)
- Memory utilization (alert: >90%)
- Eviction effectiveness
- ML prediction accuracy

### Alerts
- High memory pressure
- Circuit breaker open
- Cache hit rate degradation
- Redis cluster node failures

## Technology Stack

**Languages**: Python 3.11+, SQL
**Infrastructure**: GCP, Terraform, Kubernetes
**Databases**: Redis Cluster, PostgreSQL (metrics)
**ML**: scikit-learn, NumPy
**Monitoring**: Prometheus, Grafana
**DevOps**: Docker, GitHub Actions

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start local Redis
docker-compose up redis

# Run application
python src/main.py
```

## Production Deployment

The system is designed for production with:
- Horizontal Pod Autoscaler (HPA)
- Resource limits and requests
- Health checks and probes
- Persistent volumes for Redis
- Network policies for security

## Performance

- **Latency**: <1ms cache operations
- **Throughput**: 10k+ ops/second
- **Memory**: Intelligent eviction maintains 85%+ hit rate
- **Availability**: 99.9% uptime with circuit breakers