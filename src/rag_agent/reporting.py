"""
Agent result reporting and summary functionality
Provides tools for generating reports and summaries of agent interactions
"""

import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from io import StringIO

from src.rag_agent.models import AgentRequest, AgentResponse, AgentResult, AgentMetrics


class AgentReportingService:
    """
    Service for generating reports and summaries of agent interactions.
    Provides functionality for tracking performance, usage, and quality metrics.
    """

    def __init__(self):
        """Initialize the reporting service."""
        self.agent_results = []
        self.metrics_history = []

    def add_agent_result(self, agent_request: AgentRequest, agent_response: AgentResponse,
                        relevance_score: float = 0.0, consistency_score: float = 0.0) -> AgentResult:
        """
        Add an agent result for reporting and tracking.

        Args:
            agent_request: The original agent request
            agent_response: The agent response
            relevance_score: Human-validated relevance score (0-1)
            consistency_score: For repeated queries, similarity of results (0-1)

        Returns:
            AgentResult object with the tracked result
        """
        result = AgentResult(
            agent_request=agent_request,
            agent_response=agent_response,
            metadata_preserved=self._check_metadata_preservation(agent_response),
            relevance_score=relevance_score,
            consistency_score=consistency_score,
            validation_timestamp=datetime.now()
        )

        self.agent_results.append(result)
        return result

    def _check_metadata_preservation(self, agent_response: AgentResponse) -> bool:
        """
        Check if all required metadata fields are preserved in the response.

        Args:
            agent_response: The agent response to check

        Returns:
            True if all metadata is preserved, False otherwise
        """
        required_fields = ["url", "title", "chunk_index", "source_metadata", "created_at"]

        for chunk in agent_response.retrieved_chunks:
            for field in required_fields:
                if field not in chunk.metadata or chunk.metadata[field] is None:
                    return False
        return True

    def calculate_current_metrics(self) -> AgentMetrics:
        """
        Calculate current agent metrics based on all tracked results.

        Returns:
            AgentMetrics object with calculated metrics
        """
        if not self.agent_results:
            return AgentMetrics(
                total_queries=0,
                successful_queries=0,
                avg_response_time=0.0,
                avg_confidence_score=0.0,
                metadata_accuracy=0.0,
                relevance_accuracy=0.0,
                consistency_rate=0.0,
                timestamp=datetime.now()
            )

        total_queries = len(self.agent_results)
        successful_queries = sum(1 for result in self.agent_results
                               if result.agent_response.answer and len(result.agent_response.answer) > 10)

        response_times = [result.agent_response.execution_time for result in self.agent_results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        confidence_scores = [result.agent_response.confidence_score for result in self.agent_results]
        avg_confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        metadata_accuracy = sum(1 for result in self.agent_results if result.metadata_preserved) / total_queries * 100

        relevance_scores = [result.relevance_score for result in self.agent_results if result.relevance_score > 0]
        relevance_accuracy = (sum(relevance_scores) / len(relevance_scores) * 100) if relevance_scores else 0.0

        consistency_scores = [result.consistency_score for result in self.agent_results if result.consistency_score > 0]
        consistency_rate = (sum(consistency_scores) / len(consistency_scores) * 100) if consistency_scores else 0.0

        metrics = AgentMetrics(
            total_queries=total_queries,
            successful_queries=successful_queries,
            avg_response_time=avg_response_time,
            avg_confidence_score=avg_confidence_score,
            metadata_accuracy=metadata_accuracy,
            relevance_accuracy=relevance_accuracy,
            consistency_rate=consistency_rate,
            timestamp=datetime.now()
        )

        # Store in history for trend analysis
        self.metrics_history.append(metrics)

        return metrics

    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate a summary report for the specified number of days.

        Args:
            days: Number of days to include in the report (default 7)

        Returns:
            Dictionary with summary report data
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter results for the specified period
        recent_results = [
            result for result in self.agent_results
            if result.validation_timestamp >= cutoff_date
        ]

        if not recent_results:
            return {
                "period": f"Last {days} days",
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat(),
                "total_queries": 0,
                "message": "No data available for the specified period"
            }

        # Calculate metrics for the period
        total_queries = len(recent_results)
        successful_queries = sum(1 for result in recent_results
                               if result.agent_response.answer and len(result.agent_response.answer) > 10)

        response_times = [result.agent_response.execution_time for result in recent_results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        confidence_scores = [result.agent_response.confidence_score for result in recent_results]
        avg_confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        metadata_accuracy = sum(1 for result in recent_results if result.metadata_preserved) / total_queries * 100

        relevance_scores = [result.relevance_score for result in recent_results if result.relevance_score > 0]
        relevance_accuracy = (sum(relevance_scores) / len(relevance_scores) * 100) if relevance_scores else 0.0

        consistency_scores = [result.consistency_score for result in recent_results if result.consistency_score > 0]
        consistency_rate = (sum(consistency_scores) / len(consistency_scores) * 100) if consistency_scores else 0.0

        # Top queries by frequency (if we track query text)
        query_counts = {}
        for result in recent_results:
            query = result.agent_request.query_text.lower().strip()
            query_counts[query] = query_counts.get(query, 0) + 1

        top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        report = {
            "period": f"Last {days} days",
            "start_date": cutoff_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries * 100 if total_queries > 0 else 0,
            "avg_response_time": avg_response_time,
            "avg_confidence_score": avg_confidence_score,
            "metadata_accuracy": metadata_accuracy,
            "relevance_accuracy": relevance_accuracy,
            "consistency_rate": consistency_rate,
            "top_queries": top_queries,
            "trend_analysis": self._get_trend_analysis(days)
        }

        return report

    def _get_trend_analysis(self, days: int) -> Dict[str, Any]:
        """
        Get trend analysis for the specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(self.metrics_history) < 2:
            return {"message": "Not enough data for trend analysis"}

        # Get metrics from beginning and end of period
        recent_metrics = self.metrics_history[-1]
        if len(self.metrics_history) > days:
            earlier_metrics = self.metrics_history[-days]
        else:
            earlier_metrics = self.metrics_history[0]

        trend_analysis = {
            "total_queries_change": recent_metrics.total_queries - earlier_metrics.total_queries,
            "success_rate_change": recent_metrics.successful_queries/ max(recent_metrics.total_queries, 1)*100 -
                                earlier_metrics.successful_queries/ max(earlier_metrics.total_queries, 1)*100,
            "avg_response_time_change": recent_metrics.avg_response_time - earlier_metrics.avg_response_time,
            "avg_confidence_change": recent_metrics.avg_confidence_score - earlier_metrics.avg_confidence_score,
            "metadata_accuracy_change": recent_metrics.metadata_accuracy - earlier_metrics.metadata_accuracy,
            "relevance_accuracy_change": recent_metrics.relevance_accuracy - earlier_metrics.relevance_accuracy,
            "consistency_rate_change": recent_metrics.consistency_rate - earlier_metrics.consistency_rate
        }

        return trend_analysis

    def export_results_to_csv(self) -> str:
        """
        Export agent results to CSV format.

        Returns:
            CSV formatted string
        """
        if not self.agent_results:
            return "No results to export"

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "Query", "Answer Preview", "Confidence Score", "Execution Time",
            "Retrieved Chunks", "Metadata Preserved", "Relevance Score",
            "Consistency Score", "Timestamp"
        ])

        # Write data rows
        for result in self.agent_results:
            writer.writerow([
                result.agent_request.query_text,
                result.agent_response.answer[:100] + "..." if len(result.agent_response.answer) > 100 else result.agent_response.answer,
                result.agent_response.confidence_score,
                result.agent_response.execution_time,
                len(result.agent_response.retrieved_chunks),
                result.metadata_preserved,
                result.relevance_score,
                result.consistency_score,
                result.validation_timestamp.isoformat()
            ])

        return output.getvalue()

    def export_metrics_to_json(self) -> str:
        """
        Export metrics history to JSON format.

        Returns:
            JSON formatted string
        """
        metrics_data = []
        for metrics in self.metrics_history:
            metrics_data.append({
                "total_queries": metrics.total_queries,
                "successful_queries": metrics.successful_queries,
                "avg_response_time": metrics.avg_response_time,
                "avg_confidence_score": metrics.avg_confidence_score,
                "metadata_accuracy": metrics.metadata_accuracy,
                "relevance_accuracy": metrics.relevance_accuracy,
                "consistency_rate": metrics.consistency_rate,
                "timestamp": metrics.timestamp.isoformat()
            })

        return json.dumps(metrics_data, indent=2)

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for a performance dashboard.

        Returns:
            Dictionary with dashboard-ready data
        """
        current_metrics = self.calculate_current_metrics()

        # Prepare data for charts and graphs
        daily_metrics = self._get_daily_metrics()

        dashboard_data = {
            "current_metrics": {
                "total_queries": current_metrics.total_queries,
                "success_rate": current_metrics.successful_queries / max(current_metrics.total_queries, 1) * 100,
                "avg_response_time": current_metrics.avg_response_time,
                "avg_confidence_score": current_metrics.avg_confidence_score,
                "metadata_accuracy": current_metrics.metadata_accuracy,
                "relevance_accuracy": current_metrics.relevance_accuracy,
                "consistency_rate": current_metrics.consistency_rate
            },
            "trends": {
                "daily_performance": daily_metrics,
                "recent_improvements": self._calculate_improvements()
            },
            "alerts": self._check_for_alerts(current_metrics)
        }

        return dashboard_data

    def _get_daily_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics aggregated by day for trend visualization.

        Returns:
            List of daily metrics
        """
        if not self.agent_results:
            return []

        # Group results by date
        daily_data = {}
        for result in self.agent_results:
            date_key = result.validation_timestamp.date().isoformat()
            if date_key not in daily_data:
                daily_data[date_key] = {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "response_times": [],
                    "confidence_scores": []
                }

            daily_data[date_key]["total_queries"] += 1
            if result.agent_response.answer and len(result.agent_response.answer) > 10:
                daily_data[date_key]["successful_queries"] += 1
            daily_data[date_key]["response_times"].append(result.agent_response.execution_time)
            daily_data[date_key]["confidence_scores"].append(result.agent_response.confidence_score)

        # Calculate daily aggregates
        daily_metrics = []
        for date, data in daily_data.items():
            daily_metrics.append({
                "date": date,
                "total_queries": data["total_queries"],
                "success_rate": data["successful_queries"] / data["total_queries"] * 100 if data["total_queries"] > 0 else 0,
                "avg_response_time": sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else 0,
                "avg_confidence_score": sum(data["confidence_scores"]) / len(data["confidence_scores"]) if data["confidence_scores"] else 0
            })

        return sorted(daily_metrics, key=lambda x: x["date"], reverse=True)[:30]  # Last 30 days

    def _calculate_improvements(self) -> Dict[str, float]:
        """
        Calculate recent improvements in metrics.

        Returns:
            Dictionary with improvement percentages
        """
        if len(self.metrics_history) < 2:
            return {}

        recent = self.metrics_history[-1]
        previous = self.metrics_history[-2] if len(self.metrics_history) > 1 else recent

        improvements = {}
        if previous.total_queries > 0:
            improvements["query_volume"] = ((recent.total_queries - previous.total_queries) / previous.total_queries) * 100
        if previous.avg_response_time > 0:
            # Negative improvement is better for response time
            improvements["response_time"] = ((previous.avg_response_time - recent.avg_response_time) / previous.avg_response_time) * 100
        if previous.avg_confidence_score > 0:
            improvements["confidence"] = ((recent.avg_confidence_score - previous.avg_confidence_score) / previous.avg_confidence_score) * 100
        if previous.metadata_accuracy > 0:
            improvements["metadata"] = ((recent.metadata_accuracy - previous.metadata_accuracy) / previous.metadata_accuracy) * 100
        if previous.relevance_accuracy > 0:
            improvements["relevance"] = ((recent.relevance_accuracy - previous.relevance_accuracy) / previous.relevance_accuracy) * 100
        if previous.consistency_rate > 0:
            improvements["consistency"] = ((recent.consistency_rate - previous.consistency_rate) / previous.consistency_rate) * 100

        return improvements

    def _check_for_alerts(self, current_metrics: AgentMetrics) -> List[Dict[str, str]]:
        """
        Check for any metric thresholds that have been crossed.

        Args:
            current_metrics: Current metrics to check

        Returns:
            List of alert messages
        """
        alerts = []

        if current_metrics.avg_response_time > 5.0:
            alerts.append({
                "type": "performance",
                "severity": "high",
                "message": f"Average response time ({current_metrics.avg_response_time:.2f}s) exceeds 5s threshold"
            })

        if current_metrics.metadata_accuracy < 95.0:
            alerts.append({
                "type": "quality",
                "severity": "medium",
                "message": f"Metadata accuracy ({current_metrics.metadata_accuracy:.1f}%) below 95% threshold"
            })

        if current_metrics.relevance_accuracy < 85.0:
            alerts.append({
                "type": "quality",
                "severity": "high",
                "message": f"Relevance accuracy ({current_metrics.relevance_accuracy:.1f}%) below 85% threshold"
            })

        if current_metrics.consistency_rate < 90.0:
            alerts.append({
                "type": "quality",
                "severity": "medium",
                "message": f"Consistency rate ({current_metrics.consistency_rate:.1f}%) below 90% threshold"
            })

        return alerts


# Example usage and testing
def example_usage():
    """Example of how to use the reporting service."""
    from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk

    reporting_service = AgentReportingService()

    # Simulate adding some results
    for i in range(5):
        request = AgentRequest(query_text=f"Test query {i}")
        response = AgentResponse(
            query_text=f"Test query {i}",
            answer=f"This is the answer for query {i}",
            retrieved_chunks=[
                ContentChunk(
                    content=f"Content chunk for query {i}",
                    similarity_score=0.85,
                    metadata={"url": f"https://example.com/{i}", "title": f"Title {i}",
                            "chunk_index": i, "source_metadata": {}, "created_at": "2025-12-16T05:00:00Z"},
                    rank=1
                )
            ],
            confidence_score=0.85,
            execution_time=0.5 + i * 0.1,
            timestamp=datetime.now()
        )

        reporting_service.add_agent_result(
            agent_request=request,
            agent_response=response,
            relevance_score=0.9,
            consistency_score=0.95
        )

    # Generate a summary report
    summary = reporting_service.generate_summary_report(days=7)
    print("Summary Report:")
    print(json.dumps(summary, indent=2))

    # Get current metrics
    metrics = reporting_service.calculate_current_metrics()
    print(f"\nCurrent Metrics: {metrics.total_queries} total queries, "
          f"{metrics.avg_response_time:.2f}s avg response time")

    # Get dashboard data
    dashboard = reporting_service.get_performance_dashboard_data()
    print(f"\nDashboard has alerts: {len(dashboard['alerts']) > 0}")


if __name__ == "__main__":
    example_usage()