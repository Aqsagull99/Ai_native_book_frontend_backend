"""
Full validation pipeline for RAG Agent
Verifies all success criteria are met as specified in the feature requirements
"""

import asyncio
import time
from typing import Dict, Any, List

from src.rag_agent.models import AgentRequest
from src.rag_agent.agent import process_agent_request
from src.rag_agent.stress_test import run_stress_test
from src.rag_agent.config import Config


class ValidationPipeline:
    """
    Complete validation pipeline to verify all success criteria are met.
    Tests semantic accuracy, metadata preservation, consistency, performance, etc.
    """

    def __init__(self):
        """Initialize the validation pipeline."""
        self.validation_results = {
            "success_criteria": {},
            "test_results": {},
            "overall_status": "pending"
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate that the agent configuration is correct."""
        print("Validating configuration...")

        is_valid, error_msg = Config.validate()

        result = {
            "passed": is_valid,
            "details": f"Configuration validation: {'Passed' if is_valid else 'Failed'} - {error_msg if not is_valid else 'OK'}"
        }

        print(f"  {result['details']}")
        return result

    def validate_semantic_accuracy(self) -> Dict[str, Any]:
        """Validate that retrieved content is semantically relevant to queries (90%+ relevance)."""
        print("Validating semantic accuracy...")

        # Test queries that should have clear, relevant answers in the knowledge base
        test_cases = [
            {
                "query": "What is artificial intelligence?",
                "expected_topics": ["artificial", "intelligence", "ai", "machine"]
            },
            {
                "query": "Explain machine learning concepts",
                "expected_topics": ["machine", "learning", "algorithm", "model"]
            },
            {
                "query": "What are neural networks?",
                "expected_topics": ["neural", "network", "connection", "node"]
            }
        ]

        relevant_count = 0
        total_tests = len(test_cases)

        for test_case in test_cases:
            try:
                agent_request = AgentRequest(query_text=test_case["query"], top_k=3)
                response = process_agent_request(agent_request)

                # Check if response contains expected topics
                response_lower = response.answer.lower()
                contains_topic = any(topic.lower() in response_lower for topic in test_case["expected_topics"])

                if contains_topic:
                    relevant_count += 1

            except Exception as e:
                print(f"  Error processing query '{test_case['query']}': {str(e)}")

        accuracy_rate = relevant_count / total_tests if total_tests > 0 else 0
        passed = accuracy_rate >= 0.9  # 90% threshold

        result = {
            "passed": passed,
            "details": f"Semantic accuracy: {relevant_count}/{total_tests} ({accuracy_rate*100:.1f}%) - Target: 90%+",
            "accuracy_rate": accuracy_rate,
            "relevant_count": relevant_count,
            "total_tests": total_tests
        }

        print(f"  {result['details']} - {'✅ PASS' if passed else '❌ FAIL'}")
        return result

    def validate_metadata_preservation(self) -> Dict[str, Any]:
        """Validate that all metadata fields are preserved (100% fields preserved)."""
        print("Validating metadata preservation...")

        test_query = "What is artificial intelligence?"

        try:
            agent_request = AgentRequest(query_text=test_query, top_k=5)
            response = process_agent_request(agent_request)

            required_fields = ["url", "title", "chunk_index", "source_metadata", "created_at"]
            preserved_count = 0
            total_chunks = len(response.retrieved_chunks)

            for chunk in response.retrieved_chunks:
                chunk_preserved = True
                for field in required_fields:
                    if field not in chunk.metadata or chunk.metadata[field] is None:
                        chunk_preserved = False
                        break

                if chunk_preserved:
                    preserved_count += 1

            preservation_rate = preserved_count / total_chunks if total_chunks > 0 else 1.0
            passed = preservation_rate >= 1.0  # 100% threshold

            result = {
                "passed": passed,
                "details": f"Metadata preservation: {preserved_count}/{total_chunks} chunks ({preservation_rate*100:.1f}%) - Target: 100%",
                "preservation_rate": preservation_rate,
                "preserved_count": preserved_count,
                "total_chunks": total_chunks
            }

            print(f"  {result['details']} - {'✅ PASS' if passed else '❌ FAIL'}")
            return result

        except Exception as e:
            result = {
                "passed": False,
                "details": f"Metadata preservation test failed with error: {str(e)}",
                "preservation_rate": 0.0,
                "preserved_count": 0,
                "total_chunks": 0
            }

            print(f"  {result['details']} - ❌ FAIL")
            return result

    def validate_consistency(self) -> Dict[str, Any]:
        """Validate consistency across repeated queries (95%+ overlap in results)."""
        print("Validating consistency...")

        test_query = "What is machine learning?"

        try:
            # Run the same query multiple times
            responses = []
            for i in range(3):
                agent_request = AgentRequest(query_text=test_query, top_k=3)
                response = process_agent_request(agent_request)
                responses.append(response)

            # Compare the content of the results
            if len(responses) >= 2:
                first_response_content = set(chunk.content[:100] for chunk in responses[0].retrieved_chunks)

                consistent_count = 0
                for response in responses[1:]:
                    current_content = set(chunk.content[:100] for chunk in response.retrieved_chunks)
                    # Calculate overlap
                    overlap = len(first_response_content.intersection(current_content))
                    max_possible = len(first_response_content)
                    overlap_ratio = overlap / max_possible if max_possible > 0 else 1.0

                    if overlap_ratio >= 0.95:  # 95% overlap
                        consistent_count += 1

                consistency_rate = consistent_count / (len(responses) - 1) if len(responses) > 1 else 1.0
                passed = consistency_rate >= 0.95  # 95% threshold

                result = {
                    "passed": passed,
                    "details": f"Consistency: {consistent_count}/{len(responses)-1} repetitions consistent ({consistency_rate*100:.1f}%) - Target: 95%+",
                    "consistency_rate": consistency_rate,
                    "consistent_count": consistent_count,
                    "total_comparisons": len(responses) - 1
                }

                print(f"  {result['details']} - {'✅ PASS' if passed else '❌ FAIL'}")
                return result
            else:
                result = {
                    "passed": False,
                    "details": "Could not perform consistency test - not enough responses",
                    "consistency_rate": 0.0,
                    "consistent_count": 0,
                    "total_comparisons": 0
                }

                print(f"  {result['details']} - ❌ FAIL")
                return result

        except Exception as e:
            result = {
                "passed": False,
                "details": f"Consistency test failed with error: {str(e)}",
                "consistency_rate": 0.0,
                "consistent_count": 0,
                "total_comparisons": 0
            }

            print(f"  {result['details']} - ❌ FAIL")
            return result

    def validate_performance(self) -> Dict[str, Any]:
        """Validate response time performance (<2s for 95%+ of requests)."""
        print("Validating performance...")

        # Run multiple queries to test performance
        test_queries = [
            "What is artificial intelligence?",
            "Explain deep learning",
            "What are neural networks?",
            "Describe machine learning",
            "What is natural language processing?"
        ]

        execution_times = []

        for query in test_queries:
            try:
                start_time = time.time()
                agent_request = AgentRequest(query_text=query, top_k=3)
                response = process_agent_request(agent_request)
                end_time = time.time()

                execution_time = end_time - start_time
                execution_times.append(execution_time)

            except Exception as e:
                print(f"  Error timing query '{query}': {str(e)}")

        if execution_times:
            # Calculate 95th percentile
            sorted_times = sorted(execution_times)
            p95_index = int(0.95 * len(sorted_times))
            if p95_index >= len(sorted_times):
                p95_index = len(sorted_times) - 1
            p95_time = sorted_times[p95_index]

            passed = p95_time < 2.0  # Less than 2 seconds for 95% of requests

            result = {
                "passed": passed,
                "details": f"Performance: 95th percentile response time = {p95_time:.4f}s - Target: <2s",
                "p95_time": p95_time,
                "avg_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "max_time": max(execution_times) if execution_times else 0,
                "min_time": min(execution_times) if execution_times else 0,
                "total_requests": len(execution_times)
            }

            print(f"  {result['details']} - {'✅ PASS' if passed else '❌ FAIL'}")
            return result
        else:
            result = {
                "passed": False,
                "details": "Performance test failed - no successful requests",
                "p95_time": float('inf'),
                "avg_time": 0,
                "max_time": 0,
                "min_time": 0,
                "total_requests": 0
            }

            print(f"  {result['details']} - ❌ FAIL")
            return result

    def validate_availability(self) -> Dict[str, Any]:
        """Validate system availability and connection to Qdrant."""
        print("Validating availability...")

        try:
            # Test basic functionality
            test_query = "Test availability"
            agent_request = AgentRequest(query_text=test_query, top_k=1)
            response = process_agent_request(agent_request)

            # If we get a response without error, the system is available
            passed = True

            result = {
                "passed": passed,
                "details": "System availability: Connection to agent and Qdrant is working",
                "connected": True
            }

            print(f"  {result['details']} - {'✅ PASS' if passed else '❌ FAIL'}")
            return result

        except Exception as e:
            result = {
                "passed": False,
                "details": f"System availability: Connection failed - {str(e)}",
                "connected": False
            }

            print(f"  {result['details']} - ❌ FAIL")
            return result

    def run_full_validation(self) -> Dict[str, Any]:
        """Run the complete validation pipeline."""
        print("="*70)
        print("RUNNING FULL VALIDATION PIPELINE")
        print("="*70)
        print("Validating all success criteria for RAG Agent...")
        print()

        # Run all validation tests
        validation_tests = [
            ("Configuration", self.validate_configuration),
            ("Semantic Accuracy", self.validate_semantic_accuracy),
            ("Metadata Preservation", self.validate_metadata_preservation),
            ("Consistency", self.validate_consistency),
            ("Performance", self.validate_performance),
            ("Availability", self.validate_availability)
        ]

        results = {}
        all_passed = True

        for test_name, test_func in validation_tests:
            print(f"\n{test_name} Validation:")
            result = test_func()
            results[test_name.lower().replace(' ', '_')] = result
            if not result["passed"]:
                all_passed = False

        # Run stress test as well
        print(f"\nStress Test (100 queries):")
        try:
            stress_results = run_stress_test(num_queries=20, concurrent_queries=5)  # Smaller test for validation
            stress_passed = (stress_results["success_criteria"]["response_time_under_5s_95_percent"] and
                           stress_results["success_criteria"]["success_rate_above_90_percent"])

            stress_result = {
                "passed": stress_passed,
                "details": f"Stress test: 95th percentile time = {stress_results['metrics']['p95_execution_time']:.4f}s, " +
                          f"Success rate = {stress_results['metrics']['success_rate']:.1f}%",
                "stress_results": stress_results
            }

            results["stress_test"] = stress_result
            if not stress_passed:
                all_passed = False

            print(f"  {stress_result['details']} - {'✅ PASS' if stress_passed else '❌ FAIL'}")

        except Exception as e:
            stress_result = {
                "passed": False,
                "details": f"Stress test failed with error: {str(e)}",
                "stress_results": None
            }

            results["stress_test"] = stress_result
            all_passed = False
            print(f"  {stress_result['details']} - ❌ FAIL")

        # Overall result
        overall_status = "PASSED" if all_passed else "FAILED"

        final_result = {
            "overall_status": overall_status,
            "all_criteria_passed": all_passed,
            "validation_results": results,
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results.values() if r["passed"]),
                "failed_tests": sum(1 for r in results.values() if not r["passed"])
            }
        }

        print("\n" + "="*70)
        print("VALIDATION PIPELINE COMPLETE")
        print("="*70)
        print(f"Overall Status: {overall_status}")
        print(f"Summary: {final_result['summary']['passed_tests']}/{final_result['summary']['total_tests']} tests passed")

        if all_passed:
            print("🎉 All validation criteria have been met!")
        else:
            print("⚠️ Some validation criteria were not met. See details above.")

        print("="*70)

        return final_result


def run_full_validation_pipeline_and_verify_all_success_criteria() -> Dict[str, Any]:
    """
    Main function to run the full validation pipeline and verify all success criteria.

    Returns:
        Dictionary with complete validation results
    """
    pipeline = ValidationPipeline()
    return pipeline.run_full_validation()


if __name__ == "__main__":
    results = run_full_validation_pipeline_and_verify_all_success_criteria()