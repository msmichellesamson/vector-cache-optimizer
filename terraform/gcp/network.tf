# GCP VPC and networking configuration

# VPC Network
resource "google_compute_network" "vector_cache_vpc" {
  name                    = "${var.environment}-vector-cache-vpc"
  auto_create_subnetworks = false
  project                = var.project_id

  depends_on = [google_project_service.compute]
}

# Subnet for cache components
resource "google_compute_subnetwork" "cache_subnet" {
  name          = "${var.environment}-cache-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.vector_cache_vpc.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Firewall rule for Redis cluster communication
resource "google_compute_firewall" "redis_cluster" {
  name    = "${var.environment}-redis-cluster-fw"
  network = google_compute_network.vector_cache_vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["6379", "16379"]
  }

  source_ranges = ["10.0.1.0/24"]
  target_tags   = ["redis-cluster"]
}

# Firewall rule for monitoring
resource "google_compute_firewall" "monitoring" {
  name    = "${var.environment}-monitoring-fw"
  network = google_compute_network.vector_cache_vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["9090", "3000", "8080"]
  }

  source_ranges = ["10.0.1.0/24"]
  target_tags   = ["monitoring"]
}

# Router for NAT gateway
resource "google_compute_router" "cache_router" {
  name    = "${var.environment}-cache-router"
  region  = var.region
  network = google_compute_network.vector_cache_vpc.id
  project = var.project_id
}

# NAT gateway for outbound internet access
resource "google_compute_router_nat" "cache_nat" {
  name                               = "${var.environment}-cache-nat"
  router                             = google_compute_router.cache_router.name
  region                             = var.region
  project                           = var.project_id
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

output "network_name" {
  value = google_compute_network.vector_cache_vpc.name
}

output "subnet_name" {
  value = google_compute_subnetwork.cache_subnet.name
}