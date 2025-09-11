"""Monitoring and metrics collection for enterprise features."""

import time
import psutil
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from src.utils.logger import logger


class MetricsCollector:
    """Prometheus metrics collector for application monitoring."""
    
    def __init__(self, port: int = 8000):
        """Initialize metrics collector."""
        self.port = port
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.query_counter = Counter(
            'youtube_insights_queries_total',
            'Total number of queries processed',
            ['client_id', 'subscription_tier', 'status'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'youtube_insights_query_duration_seconds',
            'Time spent processing queries',
            ['client_id', 'subscription_tier'],
            registry=self.registry
        )
        
        self.video_processing_duration = Histogram(
            'youtube_insights_video_processing_duration_seconds',
            'Time spent processing videos',
            ['video_id', 'language'],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'youtube_insights_active_sessions',
            'Number of active client sessions',
            registry=self.registry
        )
        
        self.api_usage = Gauge(
            'youtube_insights_api_usage',
            'API usage by client',
            ['client_id', 'subscription_tier'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'youtube_insights_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'youtube_insights_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'youtube_insights_errors_total',
            'Total number of errors',
            ['error_type', 'client_id'],
            registry=self.registry
        )
        
        self.system_metrics = Gauge(
            'youtube_insights_system_metrics',
            'System resource usage',
            ['metric_type'],
            registry=self.registry
        )
        
        # Start metrics server
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_query(self, client_id: str, subscription_tier: str, 
                    duration: float, status: str = "success"):
        """Record query metrics."""
        try:
            self.query_counter.labels(
                client_id=client_id,
                subscription_tier=subscription_tier,
                status=status
            ).inc()
            
            self.query_duration.labels(
                client_id=client_id,
                subscription_tier=subscription_tier
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
    
    def record_video_processing(self, video_id: str, language: str, duration: float):
        """Record video processing metrics."""
        try:
            self.video_processing_duration.labels(
                video_id=video_id,
                language=language
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record video processing metrics: {e}")
    
    def record_cache_hit(self, cache_type: str = "default"):
        """Record cache hit."""
        try:
            self.cache_hits.labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error(f"Failed to record cache hit: {e}")
    
    def record_cache_miss(self, cache_type: str = "default"):
        """Record cache miss."""
        try:
            self.cache_misses.labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error(f"Failed to record cache miss: {e}")
    
    def record_error(self, error_type: str, client_id: str = "unknown"):
        """Record error metrics."""
        try:
            self.error_counter.labels(
                error_type=error_type,
                client_id=client_id
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}")
    
    def update_active_sessions(self, count: int):
        """Update active sessions count."""
        try:
            self.active_sessions.set(count)
        except Exception as e:
            logger.error(f"Failed to update active sessions: {e}")
    
    def update_api_usage(self, client_id: str, subscription_tier: str, usage: int):
        """Update API usage metrics."""
        try:
            self.api_usage.labels(
                client_id=client_id,
                subscription_tier=subscription_tier
            ).set(usage)
        except Exception as e:
            logger.error(f"Failed to update API usage: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics.labels(metric_type="cpu_percent").set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics.labels(metric_type="memory_percent").set(memory.percent)
            self.system_metrics.labels(metric_type="memory_used_mb").set(memory.used / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics.labels(metric_type="disk_percent").set(disk.percent)
            self.system_metrics.labels(metric_type="disk_used_gb").set(disk.used / 1024 / 1024 / 1024)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")


class PerformanceMonitor:
    """Performance monitoring and alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor."""
        self.metrics = metrics_collector
        self.alert_thresholds = {
            'response_time': 5.0,  # seconds
            'error_rate': 0.05,    # 5%
            'cpu_usage': 80.0,     # percentage
            'memory_usage': 85.0   # percentage
        }
        self.alerts = []
    
    def check_performance(self) -> Dict[str, Any]:
        """Check system performance and generate alerts."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Check thresholds
            alerts = []
            
            if cpu_percent > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'type': 'high_cpu',
                    'message': f'High CPU usage: {cpu_percent}%',
                    'severity': 'warning'
                })
            
            if memory.percent > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'type': 'high_memory',
                    'message': f'High memory usage: {memory.percent}%',
                    'severity': 'warning'
                })
            
            # Store alerts
            self.alerts.extend(alerts)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / 1024 / 1024,
                'alerts': alerts,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            performance = self.check_performance()
            
            # Determine health status
            if performance.get('alerts'):
                health_status = 'warning'
            else:
                health_status = 'healthy'
            
            return {
                'status': health_status,
                'performance': performance,
                'alerts_count': len(self.alerts),
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}


# Global monitoring instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
performance_monitor.start_time = time.time()