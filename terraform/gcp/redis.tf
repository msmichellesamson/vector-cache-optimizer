# Redis Cluster for Vector Cache
resource "google_redis_instance" "cache_primary" {
  name           = "vector-cache-primary"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.redis_memory_gb
  
  redis_version     = "REDIS_6_X"
  display_name      = "Vector Cache Primary"
  tier             = "STANDARD_HA"
  
  auth_enabled            = true
  transit_encryption_mode = "SERVER_CLIENT"
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
      }
    }
  }
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout         = "60"
  }
  
  labels = {
    environment = var.environment
    component   = "cache"
  }
}

resource "google_redis_instance" "cache_replica" {
  name           = "vector-cache-replica"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.redis_memory_gb
  
  redis_version     = "REDIS_6_X"
  display_name      = "Vector Cache Replica"
  tier             = "STANDARD_HA"
  
  auth_enabled            = true
  transit_encryption_mode = "SERVER_CLIENT"
  
  replica_of = google_redis_instance.cache_primary.id
  
  labels = {
    environment = var.environment
    component   = "cache-replica"
  }
}

output "redis_primary_host" {
  value = google_redis_instance.cache_primary.host
}

output "redis_primary_port" {
  value = google_redis_instance.cache_primary.port
}

output "redis_auth_string" {
  value     = google_redis_instance.cache_primary.auth_string
  sensitive = true
}