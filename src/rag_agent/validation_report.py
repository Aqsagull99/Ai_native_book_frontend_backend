"""
Final agent validation report generation
Creates comprehensive validation reports based on the full validation pipeline
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.rag_agent.validation_pipeline import run_full_validation_pipeline_and_verify_all_success_criteria
from src.rag_agent.reporting import AgentReportingService
from src.rag_agent.models import AgentMetrics


class ValidationReportGenerator:
    """
    Generator for comprehensive validation reports.
    Creates detailed reports based on validation pipeline results and agent metrics.
    """

    def __init__(self):
        """Initialize the validation report generator."""
        self.reporting_service = AgentReportingService()

    def generate_validation_report(self, title: str = "RAG Agent Validation Report") -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.

        Args:
            title: Title for the validation report

        Returns:
            Dictionary containing the complete validation report
        """
        print("Generating comprehensive validation report...")

        # Run the full validation pipeline
        validation_results = run_full_validation_pipeline_and_verify_all_success_criteria()

        # Get current metrics from reporting service
        current_metrics = self.reporting_service.calculate_current_metrics()

        # Get summary report
        summary_report = self.reporting_service.generate_summary_report(days=7)

        # Generate the complete report
        report = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "validation_summary": {
                "overall_status": validation_results["overall_status"],
                "all_criteria_passed": validation_results["all_criteria_passed"],
                "total_tests": validation_results["summary"]["total_tests"],
                "passed_tests": validation_results["summary"]["passed_tests"],
                "failed_tests": validation_results["summary"]["failed_tests"]
            },
            "success_criteria_evaluation": self._format_success_criteria_results(validation_results),
            "performance_metrics": {
                "total_queries": current_metrics.total_queries,
                "successful_queries": current_metrics.successful_queries,
                "success_rate": current_metrics.successful_queries / max(current_metrics.total_queries, 1) * 100,
                "avg_response_time": current_metrics.avg_response_time,
                "avg_confidence_score": current_metrics.avg_confidence_score,
                "metadata_accuracy": current_metrics.metadata_accuracy,
                "relevance_accuracy": current_metrics.relevance_accuracy,
                "consistency_rate": current_metrics.consistency_rate
            },
            "summary_statistics": summary_report,
            "detailed_test_results": validation_results["validation_results"],
            "recommendations": self._generate_recommendations(validation_results, current_metrics),
            "conformance_status": self._evaluate_conformance(validation_results)
        }

        return report

    def _format_success_criteria_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the success criteria evaluation results.

        Args:
            validation_results: Results from the validation pipeline

        Returns:
            Formatted success criteria results
        """
        validation_tests = validation_results["validation_results"]

        criteria_results = {
            "configuration_validated": validation_tests.get("configuration", {}).get("passed", False),
            "semantic_accuracy_validated": validation_tests.get("semantic_accuracy", {}).get("passed", False),
            "metadata_preservation_validated": validation_tests.get("metadata_preservation", {}).get("passed", False),
            "consistency_validated": validation_tests.get("consistency", {}).get("passed", False),
            "performance_validated": validation_tests.get("performance", {}).get("passed", False),
            "availability_validated": validation_tests.get("availability", {}).get("passed", False),
            "stress_test_validated": validation_tests.get("stress_test", {}).get("passed", False)
        }

        return criteria_results

    def _generate_recommendations(self, validation_results: Dict[str, Any],
                                 current_metrics: AgentMetrics) -> List[Dict[str, str]]:
        """
        Generate recommendations based on validation results and metrics.

        Args:
            validation_results: Results from the validation pipeline
            current_metrics: Current agent metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for performance issues
        if current_metrics.avg_response_time > 2.0:
            recommendations.append({
                "priority": "high",
                "category": "performance",
                "description": f"Average response time ({current_metrics.avg_response_time:.2f}s) exceeds target of 2s",
                "suggestion": "Investigate LLM and vector database response times, consider caching strategies"
            })

        # Check for accuracy issues
        if current_metrics.relevance_accuracy < 90.0:
            recommendations.append({
                "priority": "high",
                "category": "quality",
                "description": f"Relevance accuracy ({current_metrics.relevance_accuracy:.1f}%) below target of 90%",
                "suggestion": "Review vector embeddings, improve retrieval prompts, enhance content quality"
            })

        # Check for metadata issues
        if current_metrics.metadata_accuracy < 100.0:
            recommendations.append({
                "priority": "medium",
                "category": "data_integrity",
                "description": f"Metadata accuracy ({current_metrics.metadata_accuracy:.1f}%) below target of 100%",
                "suggestion": "Verify Qdrant collection schema, ensure all metadata fields are preserved during retrieval"
            })

        # Check for consistency issues
        if current_metrics.consistency_rate < 95.0:
            recommendations.append({
                "priority": "medium",
                "category": "reliability",
                "description": f"Consistency rate ({current_metrics.consistency_rate:.1f}%) below target of 95%",
                "suggestion": "Investigate vector search parameters, consider result caching for common queries"
            })

        # Check validation pipeline results
        validation_tests = validation_results["validation_results"]
        if not validation_tests.get("semantic_accuracy", {}).get("passed"):
            recommendations.append({
                "priority": "high",
                "category": "validation",
                "description": "Semantic accuracy validation failed",
                "suggestion": "Review content embeddings and similarity search implementation"
            })

        if not validation_tests.get("performance", {}).get("passed"):
            recommendations.append({
                "priority": "high",
                "category": "validation",
                "description": "Performance validation failed",
                "suggestion": "Optimize API endpoints, review external service response times"
            })

        if not validation_tests.get("stress_test", {}).get("passed"):
            recommendations.append({
                "priority": "high",
                "category": "validation",
                "description": "Stress test validation failed",
                "suggestion": "Scale infrastructure, implement rate limiting, optimize concurrent processing"
            })

        return recommendations

    def _evaluate_conformance(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate overall conformance to requirements.

        Args:
            validation_results: Results from the validation pipeline

        Returns:
            Conformance evaluation results
        """
        all_passed = validation_results["all_criteria_passed"]
        passed_count = validation_results["summary"]["passed_tests"]
        total_count = validation_results["summary"]["total_tests"]

        conformance_score = (passed_count / total_count) * 100 if total_count > 0 else 0

        if conformance_score >= 95:
            level = "excellent"
        elif conformance_score >= 85:
            level = "good"
        elif conformance_score >= 70:
            level = "fair"
        else:
            level = "needs_improvement"

        return {
            "conformance_level": level,
            "conformance_score": conformance_score,
            "all_criteria_met": all_passed,
            "passed_criteria_count": passed_count,
            "total_criteria_count": total_count,
            "status": "PASS" if all_passed else "NEEDS_ATTENTION"
        }

    def save_report_to_file(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the validation report to a file.

        Args:
            report: The validation report to save
            filename: Optional filename (default: auto-generated with timestamp)

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_agent_validation_report_{timestamp}.json"

        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Validation report saved to: {filepath}")
        return filepath

    def generate_validation_summary_html(self, report: Dict[str, Any]) -> str:
        """
        Generate an HTML summary of the validation report.

        Args:
            report: The validation report to convert to HTML

        Returns:
            HTML string with the report summary
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .pass {{ background-color: #d4edda; }}
        .fail {{ background-color: #f8d7da; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report['title']}</h1>
        <p>Generated: {report['generated_at']}</p>
    </div>

    <div class="section">
        <h2>Validation Summary</h2>
        <div class="metric {'pass' if report['validation_summary']['all_criteria_passed'] else 'fail'}">
            Overall Status: {report['validation_summary']['overall_status']}
        </div>
        <div class="metric">
            Tests Passed: {report['validation_summary']['passed_tests']}/{report['validation_summary']['total_tests']}
        </div>
        <div class="metric">
            Success Rate: {report['validation_summary']['passed_tests']/report['validation_summary']['total_tests']*100:.1f}%
        </div>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metric">
            Total Queries: {report['performance_metrics']['total_queries']}
        </div>
        <div class="metric">
            Success Rate: {report['performance_metrics']['success_rate']:.1f}%
        </div>
        <div class="metric">
            Avg Response Time: {report['performance_metrics']['avg_response_time']:.3f}s
        </div>
        <div class="metric">
            Relevance Accuracy: {report['performance_metrics']['relevance_accuracy']:.1f}%
        </div>
    </div>

    <div class="section">
        <h2>Conformance Status</h2>
        <div class="metric {'pass' if report['conformance_status']['all_criteria_met'] else 'fail'}">
            Level: {report['conformance_status']['conformance_level']}
        </div>
        <div class="metric">
            Score: {report['conformance_status']['conformance_score']:.1f}%
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        """

        for rec in report['recommendations']:
            html += f"""
        <div class="recommendation">
            <strong>{rec['priority'].title()} Priority ({rec['category']}):</strong> {rec['description']}<br>
            <em>Suggestion:</em> {rec['suggestion']}
        </div>
        """

        html += """
    </div>
</body>
</html>
        """

        return html

    def save_html_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the validation report as an HTML file.

        Args:
            report: The validation report to save as HTML
            filename: Optional filename (default: auto-generated with timestamp)

        Returns:
            Path to the saved HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_agent_validation_report_{timestamp}.html"

        html_content = self.generate_validation_summary_html(report)

        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML validation report saved to: {filepath}")
        return filepath


def generate_final_agent_validation_report() -> Dict[str, Any]:
    """
    Generate the final agent validation report with all required information.

    Returns:
        Complete validation report as a dictionary
    """
    generator = ValidationReportGenerator()
    report = generator.generate_validation_report()

    # Save the report to a file
    generator.save_report_to_file(report)

    # Also save as HTML for easy viewing
    generator.save_html_report(report)

    return report


if __name__ == "__main__":
    print("Generating final agent validation report...")
    final_report = generate_final_agent_validation_report()

    print(f"\nValidation report generated successfully!")
    print(f"Overall status: {final_report['validation_summary']['overall_status']}")
    print(f"Tests passed: {final_report['validation_summary']['passed_tests']}/{final_report['validation_summary']['total_tests']}")
    print(f"Conformance level: {final_report['conformance_status']['conformance_level']}")

    if final_report['recommendations']:
        print(f"\nRecommendations ({len(final_report['recommendations'])} items):")
        for i, rec in enumerate(final_report['recommendations'][:3], 1):  # Show first 3
            print(f"  {i}. {rec['description']}")

    print(f"\nReports have been saved to the 'reports' directory")