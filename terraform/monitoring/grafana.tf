resource "grafana_dashboard" "vector_cache" {
  config_json = jsonencode({
    dashboard = {
      id       = null
      title    = "Vector Cache Optimizer"
      tags     = ["cache", "ml", "performance"]
      timezone = "browser"
      panels = [
        {
          id          = 1
          title       = "Cache Hit Rate"
          type        = "stat"
          targets     = [{
            expr         = "vector_cache_hit_rate"
            refId        = "A"
            datasource   = "prometheus"
          }]
          fieldConfig = {
            defaults = {
              min  = 0
              max  = 1
              unit = "percentunit"
            }
          }
          gridPos = { h = 8, w = 12, x = 0, y = 0 }
        },
        {
          id      = 2
          title   = "ML Prediction Accuracy"
          type    = "timeseries"
          targets = [{
            expr       = "vector_cache_ml_accuracy"
            refId      = "A"
            datasource = "prometheus"
          }]
          gridPos = { h = 8, w = 12, x = 12, y = 0 }
        },
        {
          id      = 3
          title   = "Eviction Events"
          type    = "timeseries"
          targets = [{
            expr       = "rate(vector_cache_evictions_total[5m])"
            refId      = "A"
            datasource = "prometheus"
          }]
          gridPos = { h = 8, w = 24, x = 0, y = 8 }
        }
      ]
      time = {
        from = "now-1h"
        to   = "now"
      }
      refresh = "30s"
    }
  })
}

resource "grafana_folder" "vector_cache" {
  title = "Vector Cache Monitoring"
}

resource "grafana_data_source" "prometheus" {
  type = "prometheus"
  name = "prometheus"
  url  = "http://prometheus:9090"

  json_data_encoded = jsonencode({
    httpMethod   = "POST"
    queryTimeout = "60s"
  })
}