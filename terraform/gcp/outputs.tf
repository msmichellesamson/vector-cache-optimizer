# GCP Infrastructure Outputs

output "redis_host" {
  description = "Redis cluster primary endpoint"
  value       = google_redis_instance.cache_cluster.host
  sensitive   = false
}

output "redis_port" {
  description = "Redis cluster port"
  value       = google_redis_instance.cache_cluster.port
  sensitive   = false
}

output "redis_auth_string" {
  description = "Redis authentication string"
  value       = google_redis_instance.cache_cluster.auth_string
  sensitive   = true
}

output "vpc_network" {
  description = "VPC network name for service deployment"
  value       = google_compute_network.vpc_network.name
  sensitive   = false
}

output "private_subnet" {
  description = "Private subnet for application deployment"
  value       = google_compute_subnetwork.private_subnet.name
  sensitive   = false
}

output "prometheus_endpoint" {
  description = "Prometheus monitoring endpoint"
  value       = "http://${google_compute_instance.monitoring.network_interface[0].access_config[0].nat_ip}:9090"
  sensitive   = false
}

output "grafana_endpoint" {
  description = "Grafana dashboard endpoint"
  value       = "http://${google_compute_instance.monitoring.network_interface[0].access_config[0].nat_ip}:3000"
  sensitive   = false
}