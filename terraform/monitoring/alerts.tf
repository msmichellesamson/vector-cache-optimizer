# Redis Cluster Monitoring and Alerting Configuration

resource "google_monitoring_alert_policy" "redis_memory_usage" {
  display_name = "Redis Memory Usage High"
  combiner     = "OR"
  
  conditions {
    display_name = "Redis memory usage > 80%"
    
    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"redis.googlecloud.com/stats/memory/usage_ratio\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
  
  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "redis_connection_count" {
  display_name = "Redis Connection Count High"
  combiner     = "OR"
  
  conditions {
    display_name = "Redis connections > 1000"
    
    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"redis.googlecloud.com/stats/connections/total\""
      duration        = "180s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 1000
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Alerts"
  type         = "email"
  
  labels = {
    email_address = var.alert_email
  }
}

# Custom metrics for cache hit rate
resource "google_monitoring_alert_policy" "cache_hit_rate_low" {
  display_name = "Vector Cache Hit Rate Low"
  combiner     = "OR"
  
  conditions {
    display_name = "Hit rate < 70%"
    
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/vector_cache/hit_rate\""
      duration        = "600s"
      comparison      = "COMPARISON_LESS_THAN"
      threshold_value = 0.7
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
}