variable "project_id" {
  description = "GCP project ID for Prometheus monitoring stack"
  type        = string
}

variable "region" {
  description = "GCP region for monitoring resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "prometheus_retention_days" {
  description = "Prometheus data retention in days"
  type        = number
  default     = 15
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "alert_notification_email" {
  description = "Email for alert notifications"
  type        = string
}

locals {
  common_labels = {
    project     = "vector-cache-optimizer"
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "google_compute_disk" "prometheus_data" {
  name = "vco-prometheus-data-${var.environment}"
  type = "pd-ssd"
  zone = "${var.region}-a"
  size = 100

  labels = local.common_labels
}

resource "google_compute_disk" "grafana_data" {
  name = "vco-grafana-data-${var.environment}"
  type = "pd-ssd"
  zone = "${var.region}-a"
  size = 20

  labels = local.common_labels
}

resource "google_compute_instance" "prometheus" {
  name         = "vco-prometheus-${var.environment}"
  machine_type = "e2-standard-2"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 20
    }
  }

  attached_disk {
    source      = google_compute_disk.prometheus_data.id
    device_name = "prometheus-data"
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.prometheus.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init-prometheus.yaml", {
      prometheus_config = base64encode(templatefile("${path.module}/prometheus.yml", {
        project_id            = var.project_id
        alert_notification_email = var.alert_notification_email
      }))
      alert_rules = base64encode(file("${path.module}/alert-rules.yml"))
      retention_days = var.prometheus_retention_days
    })
  }

  labels = local.common_labels

  tags = ["prometheus", "monitoring"]
}

resource "google_compute_instance" "grafana" {
  name         = "vco-grafana-${var.environment}"
  machine_type = "e2-small"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 20
    }
  }

  attached_disk {
    source      = google_compute_disk.grafana_data.id
    device_name = "grafana-data"
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.grafana.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init-grafana.yaml", {
      grafana_config = base64encode(templatefile("${path.module}/grafana.ini", {
        admin_password = var.grafana_admin_password
      }))
      dashboards = base64encode(file("${path.module}/dashboards.json"))
    })
  }

  labels = local.common_labels

  tags = ["grafana", "monitoring"]
}

resource "google_service_account" "prometheus" {
  account_id   = "vco-prometheus-${var.environment}"
  display_name = "Vector Cache Optimizer Prometheus Service Account"
  description  = "Service account for Prometheus monitoring in ${var.environment}"
}

resource "google_service_account" "grafana" {
  account_id   = "vco-grafana-${var.environment}"
  display_name = "Vector Cache Optimizer Grafana Service Account"
  description  = "Service account for Grafana dashboards in ${var.environment}"
}

resource "google_project_iam_member" "prometheus_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.prometheus.email}"
}

resource "google_project_iam_member" "prometheus_compute" {
  project = var.project_id
  role    = "roles/compute.viewer"
  member  = "serviceAccount:${google_service_account.prometheus.email}"
}

resource "google_compute_firewall" "prometheus" {
  name    = "vco-prometheus-${var.environment}"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9090"]
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["prometheus"]
}

resource "google_compute_firewall" "grafana" {
  name    = "vco-grafana-${var.environment}"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["3000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["grafana"]
}

resource "google_compute_firewall" "node_exporter" {
  name    = "vco-node-exporter-${var.environment}"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9100"]
  }

  source_tags = ["prometheus"]
  target_tags = ["vector-cache-app"]
}

resource "google_compute_firewall" "app_metrics" {
  name    = "vco-app-metrics-${var.environment}"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_tags = ["prometheus"]
  target_tags = ["vector-cache-app"]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "VCO Email Notifications ${title(var.environment)}"
  type         = "email"
  labels = {
    email_address = var.alert_notification_email
  }
}

resource "google_monitoring_alert_policy" "high_cache_miss_rate" {
  display_name = "VCO High Cache Miss Rate - ${title(var.environment)}"
  combiner     = "OR"

  conditions {
    display_name = "Cache miss rate > 30%"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/miss_rate\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.3

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "high_memory_usage" {
  display_name = "VCO High Memory Usage - ${title(var.environment)}"
  combiner     = "OR"

  conditions {
    display_name = "Memory usage > 85%"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"compute.googleapis.com/instance/memory/utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

resource "google_monitoring_alert_policy" "ml_model_prediction_latency" {
  display_name = "VCO ML Model High Prediction Latency - ${title(var.environment)}"
  combiner     = "OR"

  conditions {
    display_name = "ML prediction latency > 100ms"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/ml_prediction_latency\""
      duration        = "180s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_95"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

resource "google_monitoring_alert_policy" "redis_connection_failures" {
  display_name = "VCO Redis Connection Failures - ${title(var.environment)}"
  combiner     = "OR"

  conditions {
    display_name = "Redis connection failure rate > 5%"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/redis_connection_errors\""
      duration        = "120s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

resource "google_monitoring_dashboard" "vector_cache_overview" {
  dashboard_json = jsonencode({
    displayName = "Vector Cache Optimizer - ${title(var.environment)}"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Cache Hit Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/hit_rate\""
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Hit Rate"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Cache Memory Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/memory_usage_bytes\""
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Bytes"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "ML Prediction Latency"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/ml_prediction_latency\""
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Seconds"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/vector_cache/request_rate\""
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Requests/sec"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

output "prometheus_external_ip" {
  description = "External IP address of Prometheus instance"
  value       = google_compute_instance.prometheus.network_interface[0].access_config[0].nat_ip
}

output "grafana_external_ip" {
  description = "External IP address of Grafana instance"
  value       = google_compute_instance.grafana.network_interface[0].access_config[0].nat_ip
}

output "prometheus_url" {
  description = "URL to access Prometheus"
  value       = "http://${google_compute_instance.prometheus.network_interface[0].access_config[0].nat_ip}:9090"
}

output "grafana_url" {
  description = "URL to access Grafana"
  value       = "http://${google_compute_instance.grafana.network_interface[0].access_config[0].nat_ip}:3000"
}