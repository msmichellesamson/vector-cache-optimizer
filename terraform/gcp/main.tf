terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

variable "project_id" {
  description = "GCP project ID for vector cache optimizer infrastructure"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 4
  
  validation {
    condition     = var.redis_memory_size_gb >= 1 && var.redis_memory_size_gb <= 300
    error_message = "Redis memory size must be between 1 and 300 GB."
  }
}

variable "gke_node_count" {
  description = "Initial number of GKE nodes"
  type        = number
  default     = 3
  
  validation {
    condition     = var.gke_node_count >= 1 && var.gke_node_count <= 10
    error_message = "GKE node count must be between 1 and 10."
  }
}

variable "gke_machine_type" {
  description = "GKE node machine type"
  type        = string
  default     = "e2-standard-4"
}

variable "redis_tier" {
  description = "Redis service tier (BASIC or STANDARD_HA)"
  type        = string
  default     = "STANDARD_HA"
  
  validation {
    condition     = contains(["BASIC", "STANDARD_HA"], var.redis_tier)
    error_message = "Redis tier must be BASIC or STANDARD_HA."
  }
}

locals {
  name_prefix = "vector-cache-${var.environment}"
  
  common_labels = {
    project     = "vector-cache-optimizer"
    environment = var.environment
    managed-by  = "terraform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_client_config" "default" {}

# VPC Network for secure communication
resource "google_compute_network" "vpc_network" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  
  description = "VPC network for vector cache optimizer ${var.environment}"
}

# Subnet for GKE cluster
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "${local.name_prefix}-gke-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc_network.id
  
  description = "Subnet for GKE cluster nodes"
  
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.20.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.30.0.0/16"
  }
}

# Firewall rules for Redis access from GKE
resource "google_compute_firewall" "allow_redis_from_gke" {
  name    = "${local.name_prefix}-allow-redis-from-gke"
  network = google_compute_network.vpc_network.name
  
  description = "Allow Redis access from GKE nodes"
  
  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }
  
  source_ranges = [
    google_compute_subnetwork.gke_subnet.ip_cidr_range,
    google_compute_subnetwork.gke_subnet.secondary_ip_range[0].ip_cidr_range
  ]
  
  target_tags = ["redis-server"]
}

# Redis Memorystore instance
resource "google_redis_instance" "vector_cache" {
  name               = "${local.name_prefix}-redis"
  tier               = var.redis_tier
  memory_size_gb     = var.redis_memory_size_gb
  region             = var.region
  authorized_network = google_compute_network.vpc_network.id
  
  display_name = "Vector Cache Redis ${title(var.environment)}"
  
  redis_version = "REDIS_7_0"
  
  # Enable AUTH for security
  auth_enabled = true
  
  # Performance and reliability settings
  redis_configs = {
    maxmemory-policy      = "allkeys-lru"
    notify-keyspace-events = "Ex"
    timeout               = "300"
  }
  
  # Maintenance window
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  labels = local.common_labels
}

# GKE cluster with optimized configuration
resource "google_container_cluster" "vector_cache_cluster" {
  name     = "${local.name_prefix}-gke"
  location = var.zone
  
  description = "GKE cluster for vector cache optimizer ${var.environment}"
  
  # Network configuration
  network    = google_compute_network.vpc_network.name
  subnetwork = google_compute_subnetwork.gke_subnet.name
  
  # IP allocation for pods and services
  ip_allocation_policy {
    cluster_secondary_range_name  = google_compute_subnetwork.gke_subnet.secondary_ip_range[0].range_name
    services_secondary_range_name = google_compute_subnetwork.gke_subnet.secondary_ip_range[1].range_name
  }
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Workload Identity for secure pod-to-GCP authentication
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Network policy for security
  network_policy {
    enabled  = true
    provider = "CALICO"
  }
  
  # Master auth configuration
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Monitoring and logging
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
    
    managed_prometheus {
      enabled = true
    }
  }
  
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
  }
  
  # Security settings
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  
  # Enable shielded nodes
  enable_shielded_nodes = true
  
  resource_labels = local.common_labels
}

# Primary node pool for vector cache workloads
resource "google_container_node_pool" "vector_cache_nodes" {
  name       = "${local.name_prefix}-nodes"
  location   = var.zone
  cluster    = google_container_cluster.vector_cache_cluster.name
  
  node_count = var.gke_node_count
  
  # Autoscaling configuration
  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
  
  # Node configuration
  node_config {
    preemptible  = var.environment == "dev"
    machine_type = var.gke_machine_type
    disk_type    = "pd-ssd"
    disk_size_gb = 50
    
    # Service account for nodes
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded VM settings
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    # Node labels for scheduling
    labels = merge(local.common_labels, {
      "workload-type" = "vector-cache"
    })
    
    # Node taints for dedicated workloads in production
    dynamic "taint" {
      for_each = var.environment == "prod" ? [1] : []
      content {
        key    = "vector-cache"
        value  = "dedicated"
        effect = "NO_SCHEDULE"
      }
    }
    
    tags = ["vector-cache-node", "gke-node"]
  }
  
  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
  
  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Service account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${local.name_prefix}-gke-nodes"
  display_name = "GKE Nodes Service Account - ${title(var.environment)}"
  description  = "Service account for GKE nodes in vector cache optimizer"
}

# IAM bindings for GKE nodes service account
resource "google_project_iam_member" "gke_nodes_registry" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_metrics" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Service account for vector cache application workloads
resource "google_service_account" "vector_cache_app" {
  account_id   = "${local.name_prefix}-app"
  display_name = "Vector Cache Application - ${title(var.environment)}"
  description  = "Service account for vector cache application workloads"
}

# IAM policy binding for Redis access
resource "google_project_iam_member" "app_redis_editor" {
  project = var.project_id
  role    = "roles/redis.editor"
  member  = "serviceAccount:${google_service_account.vector_cache_app.email}"
}

# Workload Identity binding
resource "google_service_account_iam_binding" "workload_identity_binding" {
  service_account_id = google_service_account.vector_cache_app.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[vector-cache/vector-cache-service]"
  ]
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = "https://${google_container_cluster.vector_cache_cluster.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.vector_cache_cluster.master_auth[0].cluster_ca_certificate)
}

# Kubernetes namespace for vector cache
resource "kubernetes_namespace" "vector_cache" {
  metadata {
    name = "vector-cache"
    
    labels = merge(local.common_labels, {
      "name" = "vector-cache"
    })
  }
  
  depends_on = [google_container_node_pool.vector_cache_nodes]
}

# Kubernetes service account with Workload Identity
resource "kubernetes_service_account" "vector_cache_service" {
  metadata {
    name      = "vector-cache-service"
    namespace = kubernetes_namespace.vector_cache.metadata[0].name
    
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.vector_cache_app.email
    }
    
    labels = local.common_labels
  }
}

# ConfigMap for Redis connection details
resource "kubernetes_config_map" "redis_config" {
  metadata {
    name      = "redis-config"
    namespace = kubernetes_namespace.vector_cache.metadata[0].name
    labels    = local.common_labels
  }
  
  data = {
    redis_host = google_redis_instance.vector_cache.host
    redis_port = tostring(google_redis_instance.vector_cache.port)
  }
}

# Secret for Redis AUTH string
resource "kubernetes_secret" "redis_auth" {
  metadata {
    name      = "redis-auth"
    namespace = kubernetes_namespace.vector_cache.metadata[0].name
    labels    = local.common_labels
  }
  
  data = {
    auth_string = google_redis_instance.vector_cache.auth_string
  }
  
  type = "Opaque"
}

# Output values for application deployment
output "redis_host" {
  description = "Redis instance host address"
  value       = google_redis_instance.vector_cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.vector_cache.port
}

output "redis_auth_string" {
  description = "Redis AUTH string for authentication"
  value       = google_redis_instance.vector_cache.auth_string
  sensitive   = true
}

output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.vector_cache_cluster.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.vector_cache_cluster.endpoint
  sensitive   = true
}

output "gke_cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.vector_cache_cluster.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "kubernetes_namespace" {
  description = "Kubernetes namespace for vector cache"
  value       = kubernetes_namespace.vector_cache.metadata[0].name
}

output "service_account_email" {
  description = "Service account email for application workloads"
  value       = google_service_account.vector_cache_app.email
}

output "gcp_project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "gcp_region" {
  description = "GCP region"
  value       = var.region
}