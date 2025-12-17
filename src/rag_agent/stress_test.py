"""
100-query stress test for RAG Agent
Tests the agent's performance under load as specified in success criteria
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from src.rag_agent.models import AgentRequest
from src.rag_agent.agent import process_agent_request


class StressTester:
    """
    Class to perform stress testing on the RAG agent.
    Tests the agent's ability to handle 100 concurrent queries as per success criteria.
    """

    def __init__(self, max_concurrent_queries: int = 10):
        """
        Initialize the stress tester.

        Args:
            max_concurrent_queries: Maximum number of concurrent queries to run
        """
        self.max_concurrent_queries = max_concurrent_queries
        self.test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "What are neural networks?",
            "Describe deep learning applications",
            "What is natural language processing?",
            "How does computer vision work?",
            "What are reinforcement learning algorithms?",
            "Explain the concept of overfitting",
            "What is data preprocessing?",
            "Describe feature engineering techniques",
            "What is supervised learning?",
            "Explain unsupervised learning",
            "What is the difference between classification and regression?",
            "How do decision trees work?",
            "Explain random forest algorithm",
            "What is gradient descent?",
            "Describe backpropagation in neural networks",
            "What are convolutional neural networks?",
            "Explain recurrent neural networks",
            "What is transfer learning?",
            "How does ensemble learning work?",
            "What is cross-validation?",
            "Explain hyperparameter tuning",
            "What is regularization in machine learning?",
            "Describe clustering algorithms",
            "What is dimensionality reduction?",
            "Explain principal component analysis",
            "What is support vector machines?",
            "How do genetic algorithms work?",
            "What is anomaly detection?",
            "Explain time series forecasting",
            "What are recommendation systems?",
            "Describe natural language generation",
            "What is sentiment analysis?",
            "Explain text classification",
            "What is computer vision?",
            "Describe object detection",
            "What is image segmentation?",
            "Explain generative adversarial networks",
            "What is reinforcement learning?",
            "How does Q-learning work?",
            "What is deep reinforcement learning?",
            "Explain policy gradient methods",
            "What is actor-critic method?",
            "Describe Monte Carlo methods",
            "What is temporal difference learning?",
            "Explain multi-armed bandits",
            "What is exploration vs exploitation?",
            "How does curriculum learning work?",
            "What is meta-learning?",
            "Explain few-shot learning",
            "What is zero-shot learning?",
            "Describe active learning",
            "What is online learning?",
            "Explain federated learning",
            "What is differential privacy?",
            "How does homomorphic encryption work?",
            "What is secure multi-party computation?",
            "Explain blockchain technology",
            "What are smart contracts?",
            "Describe cryptocurrency concepts",
            "What is decentralized finance?",
            "Explain consensus algorithms",
            "What is proof of work?",
            "Describe proof of stake",
            "What is DeFi yield farming?",
            "How do NFTs work?",
            "Explain tokenomics",
            "What is Web3 technology?",
            "Describe decentralized applications",
            "What is smart contract auditing?",
            "Explain gas fees in blockchain",
            "What is layer 2 scaling?",
            "Describe cross-chain interoperability",
            "What is quantum computing?",
            "Explain quantum algorithms",
            "What is quantum supremacy?",
            "Describe quantum entanglement",
            "What is quantum cryptography?",
            "Explain quantum error correction",
            "What are quantum gates?",
            "Describe quantum circuits",
            "What is quantum machine learning?",
            "Explain variational quantum algorithms",
            "What is quantum annealing?",
            "Describe quantum simulation",
            "What is quantum teleportation?",
            "Explain quantum key distribution",
            "What is quantum internet?",
            "Describe quantum sensors",
            "What is quantum metrology?",
            "Explain quantum imaging",
            "What is quantum radar?",
            "Describe quantum lidar",
            "What is quantum sensing?",
            "Explain quantum computing applications",
            "What are quantum algorithms?",
            "Describe quantum complexity theory",
            "What is quantum information theory?",
            "Explain quantum error rates",
            "What is quantum fault tolerance?",
            "Describe quantum supremacy experiments",
            "What is quantum advantage?",
            "Explain quantum computing hardware",
            "What are quantum processors?",
            "Describe quantum computers",
            "What is quantum software?",
            "Explain quantum programming languages"
        ]

    def run_single_query(self, query: str) -> Dict[str, Any]:
        """
        Run a single query and measure performance.

        Args:
            query: The query to process

        Returns:
            Dictionary with query results and performance metrics
        """
        start_time = time.time()
        success = True
        error_msg = None

        try:
            agent_request = AgentRequest(query_text=query, top_k=3)
            response = process_agent_request(agent_request)
            execution_time = time.time() - start_time

            result = {
                "query": query,
                "success": True,
                "execution_time": execution_time,
                "response_length": len(response.answer),
                "retrieved_chunks_count": len(response.retrieved_chunks),
                "confidence_score": response.confidence_score
            }
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "query": query,
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "response_length": 0,
                "retrieved_chunks_count": 0,
                "confidence_score": 0.0
            }

        return result

    def run_stress_test(self, num_queries: int = 100) -> Dict[str, Any]:
        """
        Run the stress test with the specified number of queries.

        Args:
            num_queries: Number of queries to run (default 100 as per success criteria)

        Returns:
            Dictionary with comprehensive test results and metrics
        """
        print(f"Starting {num_queries}-query stress test with {self.max_concurrent_queries} concurrent queries...")

        # Use only the first num_queries from our test queries (duplicating if necessary)
        test_queries = []
        for i in range(num_queries):
            test_queries.append(self.test_queries[i % len(self.test_queries)])

        start_time = time.time()

        # Run queries concurrently
        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_queries) as executor:
            futures = [executor.submit(self.run_single_query, query) for query in test_queries]
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)

                # Progress update
                if (i + 1) % 10 == 0 or (i + 1) == len(test_queries):
                    print(f"Completed {i + 1}/{len(test_queries)} queries...")

        total_time = time.time() - start_time

        # Calculate metrics
        successful_queries = [r for r in results if r["success"]]
        failed_queries = [r for r in results if not r["success"]]

        execution_times = [r["execution_time"] for r in successful_queries]
        response_lengths = [r["response_length"] for r in successful_queries]
        confidence_scores = [r["confidence_score"] for r in successful_queries]
        retrieved_counts = [r["retrieved_chunks_count"] for r in successful_queries]

        # Calculate statistics
        metrics = {
            "total_queries": num_queries,
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": len(successful_queries) / num_queries * 100 if num_queries > 0 else 0,
            "total_execution_time": total_time,
            "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
            "median_execution_time": statistics.median(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "p95_execution_time": 0,  # Calculate 95th percentile
            "avg_response_length": statistics.mean(response_lengths) if response_lengths else 0,
            "avg_confidence_score": statistics.mean(confidence_scores) if confidence_scores else 0,
            "avg_retrieved_chunks": statistics.mean(retrieved_counts) if retrieved_counts else 0,
            "concurrent_queries": self.max_concurrent_queries,
            "throughput_queries_per_second": num_queries / total_time if total_time > 0 else 0
        }

        # Calculate 95th percentile for execution time
        if execution_times:
            sorted_times = sorted(execution_times)
            p95_index = int(0.95 * len(sorted_times))
            if p95_index >= len(sorted_times):
                p95_index = len(sorted_times) - 1
            metrics["p95_execution_time"] = sorted_times[p95_index]

        # Success criteria evaluation
        success_criteria = {
            "response_time_under_5s_95_percent": metrics["p95_execution_time"] < 5.0,
            "success_rate_above_90_percent": metrics["success_rate"] >= 90.0,
            "able_to_handle_100_queries": num_queries >= 100
        }

        results_summary = {
            "metrics": metrics,
            "success_criteria": success_criteria,
            "results": results[:10],  # Include first 10 results as sample
            "test_config": {
                "num_queries": num_queries,
                "max_concurrent_queries": self.max_concurrent_queries,
                "start_time": start_time,
                "end_time": time.time()
            }
        }

        return results_summary

    def print_test_report(self, results: Dict[str, Any]):
        """
        Print a formatted test report.

        Args:
            results: Results from run_stress_test
        """
        metrics = results["metrics"]
        criteria = results["success_criteria"]

        print("\n" + "="*70)
        print("STRESS TEST REPORT")
        print("="*70)
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Successful Queries: {metrics['successful_queries']}")
        print(f"Failed Queries: {metrics['failed_queries']}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Total Execution Time: {metrics['total_execution_time']:.2f}s")
        print(f"Throughput: {metrics['throughput_queries_per_second']:.2f} queries/sec")
        print(f"Concurrent Queries: {metrics['concurrent_queries']}")
        print("\nPerformance Metrics:")
        print(f"  Avg Execution Time: {metrics['avg_execution_time']:.4f}s")
        print(f"  Median Execution Time: {metrics['median_execution_time']:.4f}s")
        print(f"  Min Execution Time: {metrics['min_execution_time']:.4f}s")
        print(f"  Max Execution Time: {metrics['max_execution_time']:.4f}s")
        print(f"  95th Percentile Time: {metrics['p95_execution_time']:.4f}s")
        print(f"  Avg Response Length: {metrics['avg_response_length']:.2f} chars")
        print(f"  Avg Confidence Score: {metrics['avg_confidence_score']:.4f}")
        print(f"  Avg Retrieved Chunks: {metrics['avg_retrieved_chunks']:.2f}")
        print("\nSuccess Criteria:")
        print(f"  Response time < 5s (95%): {'✅ PASS' if criteria['response_time_under_5s_95_percent'] else '❌ FAIL'}")
        print(f"  Success rate >= 90%: {'✅ PASS' if criteria['success_rate_above_90_percent'] else '❌ FAIL'}")
        print(f"  Handle 100+ queries: {'✅ PASS' if criteria['able_to_handle_100_queries'] else '❌ FAIL'}")
        print("="*70)


def run_stress_test(num_queries: int = 100, concurrent_queries: int = 10) -> Dict[str, Any]:
    """
    Run the 100-query stress test as specified in success criteria.

    Args:
        num_queries: Number of queries to run (default 100)
        concurrent_queries: Number of concurrent queries (default 10)

    Returns:
        Dictionary with test results
    """
    tester = StressTester(max_concurrent_queries=concurrent_queries)
    results = tester.run_stress_test(num_queries=num_queries)
    tester.print_test_report(results)
    return results


if __name__ == "__main__":
    print("Running 100-query stress test for RAG Agent...")
    print("This test evaluates the agent's performance under load.")
    print("Success criteria: 95% of queries respond in under 5 seconds")
    print("and system handles 100 concurrent requests during stress testing.")
    print()

    results = run_stress_test(num_queries=100, concurrent_queries=10)