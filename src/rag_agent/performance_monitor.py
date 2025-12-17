"""
Performance monitoring and response time tracking for RAG Agent
Implements comprehensive performance metrics collection and monitoring
"""

import time
import threading
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from src.rag_agent.models import AgentRequest, AgentResponse


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: datetime
    query_duration: float
    retrieval_duration: float
    generation_duration: float
    total_duration: float
    top_k: int
    retrieved_chunks: int
    query_length: int
    response_length: int
    confidence_score: float


class PerformanceMonitor:
    """
    Performance monitoring service that tracks response times and other performance metrics.
    Provides real-time monitoring and historical analysis capabilities.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the performance monitor.

        Args:
            max_history: Maximum number of historical metrics to keep
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.current_metrics: Dict[str, Any] = {}
        self.lock = threading.Lock()

        # Initialize performance trackers
        self.query_times = deque(maxlen=max_history)
        self.retrieval_times = deque(maxlen=max_history)
        self.generation_times = deque(maxlen=max_history)
        self.total_times = deque(maxlen=max_history)

    def start_monitoring(self, request: AgentRequest) -> str:
        """
        Start monitoring for a specific request.

        Args:
            request: The agent request being monitored

        Returns:
            Monitoring session ID
        """
        session_id = f"session_{datetime.now().timestamp()}_{id(request)}"

        with self.lock:
            self.current_metrics[session_id] = {
                'start_time': time.time(),
                'request': request,
                'retrieval_start': None,
                'generation_start': None,
                'retrieval_duration': 0,
                'generation_duration': 0
            }

        return session_id

    def start_retrieval(self, session_id: str):
        """
        Mark the start of the retrieval phase.

        Args:
            session_id: The monitoring session ID
        """
        with self.lock:
            if session_id in self.current_metrics:
                self.current_metrics[session_id]['retrieval_start'] = time.time()

    def end_retrieval(self, session_id: str):
        """
        Mark the end of the retrieval phase and record duration.

        Args:
            session_id: The monitoring session ID
        """
        with self.lock:
            if session_id in self.current_metrics:
                metrics = self.current_metrics[session_id]
                if metrics['retrieval_start']:
                    metrics['retrieval_duration'] = time.time() - metrics['retrieval_start']
                    self.retrieval_times.append(metrics['retrieval_duration'])

    def start_generation(self, session_id: str):
        """
        Mark the start of the generation phase.

        Args:
            session_id: The monitoring session ID
        """
        with self.lock:
            if session_id in self.current_metrics:
                self.current_metrics[session_id]['generation_start'] = time.time()

    def end_generation(self, session_id: str):
        """
        Mark the end of the generation phase and record duration.

        Args:
            session_id: The monitoring session ID
        """
        with self.lock:
            if session_id in self.current_metrics:
                metrics = self.current_metrics[session_id]
                if metrics['generation_start']:
                    metrics['generation_duration'] = time.time() - metrics['generation_start']
                    self.generation_times.append(metrics['generation_duration'])

    def end_monitoring(self, session_id: str, response: AgentResponse) -> PerformanceMetrics:
        """
        End monitoring for a specific request and record metrics.

        Args:
            session_id: The monitoring session ID
            response: The agent response

        Returns:
            PerformanceMetrics object with the recorded metrics
        """
        with self.lock:
            if session_id not in self.current_metrics:
                # If session doesn't exist, create basic metrics
                duration = time.time() - time.time()  # This will be 0, but we'll handle it
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    query_duration=0,
                    retrieval_duration=0,
                    generation_duration=0,
                    total_duration=0,
                    top_k=response.retrieved_chunks[0].rank if response.retrieved_chunks else 0,
                    retrieved_chunks=len(response.retrieved_chunks),
                    query_length=len(response.query_text),
                    response_length=len(response.answer),
                    confidence_score=response.confidence_score
                )
                return metrics

            metrics_data = self.current_metrics.pop(session_id)
            total_duration = time.time() - metrics_data['start_time']

            # Add to time series
            self.total_times.append(total_duration)
            if 'retrieval_duration' not in metrics_data:
                metrics_data['retrieval_duration'] = 0
            if 'generation_duration' not in metrics_data:
                metrics_data['generation_duration'] = 0

            perf_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                query_duration=total_duration,
                retrieval_duration=metrics_data['retrieval_duration'],
                generation_duration=metrics_data['generation_duration'],
                total_duration=total_duration,
                top_k=metrics_data['request'].top_k,
                retrieved_chunks=len(response.retrieved_chunks),
                query_length=len(response.query_text),
                response_length=len(response.answer),
                confidence_score=response.confidence_score
            )

            # Add to history
            self.metrics_history.append(perf_metrics)
            self.query_times.append(total_duration)

            return perf_metrics

    def get_current_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with current performance statistics
        """
        with self.lock:
            if not self.total_times:
                return {
                    "message": "No performance data available yet",
                    "sample_size": 0
                }

            stats = {
                "sample_size": len(self.total_times),
                "query": {
                    "avg_time": statistics.mean(self.query_times),
                    "median_time": statistics.median(self.query_times) if self.query_times else 0,
                    "min_time": min(self.query_times) if self.query_times else 0,
                    "max_time": max(self.query_times) if self.query_times else 0,
                    "p95_time": self._calculate_percentile(self.query_times, 95) if self.query_times else 0,
                    "p99_time": self._calculate_percentile(self.query_times, 99) if self.query_times else 0
                },
                "retrieval": {
                    "avg_time": statistics.mean(self.retrieval_times) if self.retrieval_times else 0,
                    "median_time": statistics.median(self.retrieval_times) if self.retrieval_times else 0,
                    "min_time": min(self.retrieval_times) if self.retrieval_times else 0,
                    "max_time": max(self.retrieval_times) if self.retrieval_times else 0,
                    "p95_time": self._calculate_percentile(self.retrieval_times, 95) if self.retrieval_times else 0
                },
                "generation": {
                    "avg_time": statistics.mean(self.generation_times) if self.generation_times else 0,
                    "median_time": statistics.median(self.generation_times) if self.generation_times else 0,
                    "min_time": min(self.generation_times) if self.generation_times else 0,
                    "max_time": max(self.generation_times) if self.generation_times else 0,
                    "p95_time": self._calculate_percentile(self.generation_times, 95) if self.generation_times else 0
                }
            }

            # Add compliance information
            stats["compliance"] = {
                "p95_within_2s": stats["query"]["p95_time"] < 2.0,
                "avg_within_1s": stats["query"]["avg_time"] < 1.0,
                "target_met": stats["query"]["p95_time"] < 2.0  # 95% of requests under 2s
            }

            return stats

    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile for a list of values.

        Args:
            data: List of float values
            percentile: Percentile to calculate (e.g., 95 for 95th percentile)

        Returns:
            Calculated percentile value
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = lower_index + 1

        if upper_index >= len(sorted_data):
            return sorted_data[lower_index]

        # Interpolate between the two values
        fraction = index - lower_index
        lower_value = sorted_data[lower_index]
        upper_value = sorted_data[upper_index]
        return lower_value + fraction * (upper_value - lower_value)

    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trend analysis for the specified time period.

        Args:
            hours: Number of hours to analyze (default 24)

        Returns:
            Dictionary with trend analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter metrics from the specified time period
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if len(recent_metrics) < 2:
            return {"message": f"Not enough data for {hours} hour trend analysis"}

        # Group by hour for trend analysis
        hourly_data = defaultdict(list)
        for metric in recent_metrics:
            hour_key = metric.timestamp.strftime("%Y-%m-%d %H")
            hourly_data[hour_key].append(metric)

        # Calculate hourly averages
        hourly_averages = {}
        for hour, metrics in hourly_data.items():
            hourly_averages[hour] = {
                "avg_query_time": statistics.mean([m.query_duration for m in metrics]),
                "avg_retrieval_time": statistics.mean([m.retrieval_duration for m in metrics]),
                "avg_generation_time": statistics.mean([m.generation_duration for m in metrics]),
                "query_count": len(metrics),
                "avg_confidence": statistics.mean([m.confidence_score for m in metrics])
            }

        # Calculate overall trend (comparing first and last periods)
        sorted_hours = sorted(hourly_averages.keys())
        if len(sorted_hours) >= 2:
            first_hour = hourly_averages[sorted_hours[0]]
            last_hour = hourly_averages[sorted_hours[-1]]

            trend = {
                "query_time_trend": last_hour["avg_query_time"] - first_hour["avg_query_time"],
                "retrieval_time_trend": last_hour["avg_retrieval_time"] - first_hour["avg_retrieval_time"],
                "generation_time_trend": last_hour["avg_generation_time"] - first_hour["avg_generation_time"],
                "confidence_trend": last_hour["avg_confidence"] - first_hour["avg_confidence"]
            }
        else:
            trend = {"message": "Not enough hourly data for trend calculation"}

        return {
            "period_hours": hours,
            "total_queries": len(recent_metrics),
            "hourly_averages": dict(hourly_averages),
            "trend": trend
        }

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """
        Get performance alerts based on threshold violations.

        Returns:
            List of performance alerts
        """
        stats = self.get_current_performance_stats()

        if "sample_size" not in stats or stats["sample_size"] == 0:
            return []

        alerts = []

        # Check if p95 response time exceeds 2 seconds
        if stats["query"]["p95_time"] >= 2.0:
            alerts.append({
                "severity": "HIGH",
                "type": "response_time",
                "message": f"P95 response time ({stats['query']['p95_time']:.3f}s) exceeds 2s threshold",
                "current_value": stats["query"]["p95_time"],
                "threshold": 2.0
            })

        # Check if average response time exceeds 1 second
        if stats["query"]["avg_time"] >= 1.0:
            alerts.append({
                "severity": "MEDIUM",
                "type": "response_time",
                "message": f"Average response time ({stats['query']['avg_time']:.3f}s) exceeds 1s target",
                "current_value": stats["query"]["avg_time"],
                "threshold": 1.0
            })

        # Check if retrieval time is excessive
        if stats["retrieval"]["avg_time"] >= 1.0:
            alerts.append({
                "severity": "MEDIUM",
                "type": "retrieval_time",
                "message": f"Average retrieval time ({stats['retrieval']['avg_time']:.3f}s) is high",
                "current_value": stats["retrieval"]["avg_time"],
                "threshold": 1.0
            })

        return alerts

    def reset_monitoring(self):
        """Reset all monitoring data."""
        with self.lock:
            self.metrics_history.clear()
            self.query_times.clear()
            self.retrieval_times.clear()
            self.generation_times.clear()
            self.total_times.clear()
            self.current_metrics.clear()


class ResponseTimeTracker:
    """
    Simple response time tracker that can be used as a decorator or context manager.
    """

    def __init__(self, operation_name: str, monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize the response time tracker.

        Args:
            operation_name: Name of the operation being tracked
            monitor: Optional PerformanceMonitor instance to record metrics
        """
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"{self.operation_name} took {duration:.4f}s")

            # If a monitor is provided, record the metric
            if self.monitor:
                # This is a simplified recording - in practice you'd want to record more details
                pass

    def time_function(self, func: Callable) -> Callable:
        """
        Decorator to time a function.

        Args:
            func: Function to time

        Returns:
            Timed function
        """
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def track_performance(request: AgentRequest) -> str:
    """
    Start tracking performance for a request.

    Args:
        request: The agent request to track

    Returns:
        Session ID for tracking
    """
    return performance_monitor.start_monitoring(request)


def record_retrieval_start(session_id: str):
    """Record the start of retrieval phase."""
    performance_monitor.start_retrieval(session_id)


def record_retrieval_end(session_id: str):
    """Record the end of retrieval phase."""
    performance_monitor.end_retrieval(session_id)


def record_generation_start(session_id: str):
    """Record the start of generation phase."""
    performance_monitor.start_generation(session_id)


def record_generation_end(session_id: str):
    """Record the end of generation phase."""
    performance_monitor.end_generation(session_id)


def record_response(session_id: str, response: AgentResponse):
    """
    Record the final response and complete performance tracking.

    Args:
        session_id: The session ID
        response: The agent response
    """
    return performance_monitor.end_monitoring(session_id, response)


def get_performance_stats():
    """Get current performance statistics."""
    return performance_monitor.get_current_performance_stats()


def get_performance_trends(hours: int = 24):
    """
    Get performance trends for the specified number of hours.

    Args:
        hours: Number of hours to analyze (default 24)

    Returns:
        Performance trend analysis
    """
    return performance_monitor.get_trend_analysis(hours)


def get_performance_alerts():
    """Get current performance alerts."""
    return performance_monitor.get_performance_alerts()


if __name__ == "__main__":
    # Example usage
    print("Performance Monitoring Example")
    print("="*40)

    # Simulate tracking a request
    req = AgentRequest(query_text="Test query for performance monitoring", top_k=5)
    session_id = track_performance(req)

    # Simulate retrieval phase
    record_retrieval_start(session_id)
    time.sleep(0.1)  # Simulate work
    record_retrieval_end(session_id)

    # Simulate generation phase
    record_generation_start(session_id)
    time.sleep(0.2)  # Simulate work
    record_generation_end(session_id)

    # Create a mock response
    from src.rag_agent.models import AgentResponse, ContentChunk
    response = AgentResponse(
        query_text="Test query for performance monitoring",
        answer="This is a test answer for performance monitoring",
        retrieved_chunks=[ContentChunk(
            content="Test content chunk",
            similarity_score=0.85,
            metadata={"url": "test", "title": "test", "chunk_index": 0, "source_metadata": {}, "created_at": "2025-12-16T05:00:00Z"},
            rank=1
        )],
        confidence_score=0.85,
        execution_time=0.3,
        timestamp=datetime.now()
    )

    # Record the response
    metrics = record_response(session_id, response)
    print(f"Recorded metrics: {metrics}")

    # Get performance stats
    stats = get_performance_stats()
    print(f"Performance stats: {stats}")

    # Get alerts
    alerts = get_performance_alerts()
    print(f"Performance alerts: {alerts}")