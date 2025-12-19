# """
# Manual validation helper for response quality assessment
# Provides tools for manually evaluating the quality of agent responses
# """

# import json
# from datetime import datetime
# from typing import Dict, Any, List, Optional

# from src.rag_agent.models import AgentRequest, AgentResponse
# from src.rag_agent.agent import process_agent_request


# class ManualValidationHelper:
#     """
#     Helper class for manually validating agent response quality.
#     Provides methods to evaluate response relevance, accuracy, and helpfulness.
#     """

#     def __init__(self):
#         """Initialize the validation helper."""
#         self.validation_results = []
#         self.evaluation_criteria = {
#             "relevance": {
#                 "description": "How relevant is the response to the original query?",
#                 "scale": "1-5 (1=not relevant, 5=highly relevant)"
#             },
#             "accuracy": {
#                 "description": "How factually accurate is the response?",
#                 "scale": "1-5 (1=many inaccuracies, 5=highly accurate)"
#             },
#             "completeness": {
#                 "description": "Does the response adequately address the query?",
#                 "scale": "1-5 (1=very incomplete, 5=very complete)"
#             },
#             "helpfulness": {
#                 "description": "How helpful is the response to the user?",
#                 "scale": "1-5 (1=not helpful, 5=very helpful)"
#             },
#             "citations": {
#                 "description": "Are sources properly cited and relevant?",
#                 "scale": "1-5 (1=no citations, 5=excellent citations)"
#             }
#         }

#     def evaluate_response(self, query: str, response: AgentResponse) -> Dict[str, Any]:
#         """
#         Evaluate a single agent response using manual validation criteria.

#         Args:
#             query: The original query that was asked
#             response: The agent response to evaluate

#         Returns:
#             Dictionary containing evaluation scores and feedback
#         """
#         print(f"\n--- Manual Validation for Query: {query} ---")
#         print(f"Response: {response.answer[:500]}{'...' if len(response.answer) > 500 else ''}")
#         print("\nRetrieved Sources:")
#         for i, chunk in enumerate(response.retrieved_chunks[:3]):  # Show first 3 sources
#             print(f"  Source {i+1}: {chunk.metadata.get('url', 'N/A')[:50]}...")
#             print(f"    Content preview: {chunk.content[:100]}...")

#         print("\nPlease rate the following aspects (1-5 scale):")
#         print("(1=poor, 2=fair, 3=good, 4=very good, 5=excellent)")

#         evaluation = {
#             "query": query,
#             "response_summary": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
#             "timestamp": datetime.now().isoformat(),
#             "scores": {},
#             "feedback": {}
#         }

#         # Collect scores for each criterion
#         for criterion, details in self.evaluation_criteria.items():
#             print(f"\n{criterion.title()}: {details['description']}")
#             print(f"Scale: {details['scale']}")
#             while True:
#                 try:
#                     score = int(input(f"Enter score (1-5) for {criterion}: "))
#                     if 1 <= score <= 5:
#                         evaluation["scores"][criterion] = score
#                         break
#                     else:
#                         print("Please enter a number between 1 and 5.")
#                 except ValueError:
#                     print("Please enter a valid number.")

#             # Optional feedback
#             feedback = input(f"Optional feedback for {criterion} (or press Enter to skip): ").strip()
#             evaluation["feedback"][criterion] = feedback if feedback else "No feedback provided"

#         # Calculate overall score
#         overall_score = sum(evaluation["scores"].values()) / len(evaluation["scores"])
#         evaluation["overall_score"] = round(overall_score, 2)

#         print(f"\nOverall Score: {evaluation['overall_score']}/5.0")
#         self.validation_results.append(evaluation)

#         return evaluation

#     def batch_evaluate(self, queries: List[str], max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
#         """
#         Evaluate multiple queries using the agent and manual validation.

#         Args:
#             queries: List of queries to evaluate
#             max_samples: Maximum number of queries to process (None for all)

#         Returns:
#             List of evaluation results
#         """
#         if max_samples:
#             queries = queries[:max_samples]

#         results = []
#         print(f"Starting batch evaluation for {len(queries)} queries...")

#         for i, query in enumerate(queries):
#             print(f"\nProcessing query {i+1}/{len(queries)}: {query}")

#             # Process query with agent
#             agent_request = AgentRequest(query_text=query)
#             response = process_agent_request(agent_request)

#             # Evaluate response
#             result = self.evaluate_response(query, response)
#             results.append(result)

#             # Ask if user wants to continue
#             if i < len(queries) - 1:
#                 continue_batch = input("\nContinue to next query? (y/n): ").lower()
#                 if continue_batch != 'y':
#                     print("Batch evaluation paused by user.")
#                     break

#         return results

#     def save_validation_results(self, filename: str):
#         """
#         Save validation results to a JSON file.

#         Args:
#             filename: Path to save the validation results
#         """
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
#         print(f"Validation results saved to {filename}")

#     def load_validation_results(self, filename: str):
#         """
#         Load validation results from a JSON file.

#         Args:
#             filename: Path to load the validation results from
#         """
#         with open(filename, 'r', encoding='utf-8') as f:
#             self.validation_results = json.load(f)
#         print(f"Validation results loaded from {filename}")

#     def get_validation_summary(self) -> Dict[str, Any]:
#         """
#         Get a summary of all validation results.

#         Returns:
#             Dictionary containing summary statistics
#         """
#         if not self.validation_results:
#             return {"message": "No validation results available"}

#         # Calculate average scores for each criterion
#         avg_scores = {}
#         for criterion in self.evaluation_criteria.keys():
#             scores = [result["scores"][criterion] for result in self.validation_results]
#             avg_scores[criterion] = sum(scores) / len(scores) if scores else 0

#         # Calculate overall average
#         overall_scores = [result["overall_score"] for result in self.validation_results]
#         avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0

#         summary = {
#             "total_evaluations": len(self.validation_results),
#             "average_scores": {k: round(v, 2) for k, v in avg_scores.items()},
#             "average_overall_score": round(avg_overall, 2),
#             "evaluation_criteria": self.evaluation_criteria
#         }

#         return summary

#     def print_validation_summary(self):
#         """Print a formatted validation summary to console."""
#         summary = self.get_validation_summary()

#         if "message" in summary:
#             print(summary["message"])
#             return

#         print("\n" + "="*50)
#         print("VALIDATION SUMMARY")
#         print("="*50)
#         print(f"Total Evaluations: {summary['total_evaluations']}")
#         print(f"Average Overall Score: {summary['average_overall_score']}/5.0")
#         print("\nAverage Scores by Criterion:")
#         for criterion, avg_score in summary['average_scores'].items():
#             print(f"  {criterion.title()}: {avg_score}/5.0")
#         print("="*50)


# def run_manual_validation_session():
#     """
#     Run an interactive manual validation session.
#     """
#     print("Starting manual validation session for RAG Agent responses...")
#     print("This tool will help you evaluate the quality of agent responses.")
#     print("You will be asked to rate responses on several criteria.")

#     helper = ManualValidationHelper()

#     while True:
#         print("\nOptions:")
#         print("1. Evaluate a single response")
#         print("2. Run batch evaluation")
#         print("3. View validation summary")
#         print("4. Save validation results")
#         print("5. Load validation results")
#         print("6. Exit")

#         choice = input("\nEnter your choice (1-6): ").strip()

#         if choice == "1":
#             query = input("Enter the query to test: ").strip()
#             if query:
#                 # Process query with agent
#                 agent_request = AgentRequest(query_text=query)
#                 response = process_agent_request(agent_request)
#                 helper.evaluate_response(query, response)
#             else:
#                 print("Query cannot be empty.")

#         elif choice == "2":
#             print("Enter queries one per line. Type 'done' on a new line when finished:")
#             queries = []
#             while True:
#                 query = input().strip()
#                 if query.lower() == 'done':
#                     break
#                 if query:
#                     queries.append(query)

#             if queries:
#                 max_samples_input = input(f"Process all {len(queries)} queries or enter max number: ").strip()
#                 max_samples = None
#                 if max_samples_input and max_samples_input.isdigit():
#                     max_samples = int(max_samples_input)
#                 helper.batch_evaluate(queries, max_samples)
#             else:
#                 print("No queries entered.")

#         elif choice == "3":
#             helper.print_validation_summary()

#         elif choice == "4":
#             filename = input("Enter filename to save results (e.g., validation_results.json): ").strip()
#             if filename:
#                 helper.save_validation_results(filename)
#             else:
#                 print("Filename cannot be empty.")

#         elif choice == "5":
#             filename = input("Enter filename to load results from: ").strip()
#             if filename:
#                 try:
#                     helper.load_validation_results(filename)
#                 except FileNotFoundError:
#                     print(f"File {filename} not found.")
#                 except json.JSONDecodeError:
#                     print(f"Invalid JSON in file {filename}.")
#             else:
#                 print("Filename cannot be empty.")

#         elif choice == "6":
#             print("Exiting manual validation session.")
#             break

#         else:
#             print("Invalid choice. Please enter 1-6.")


# if __name__ == "__main__":
#     run_manual_validation_session()




"""
Manual validation helper for RAG response quality assessment
(Human-in-the-loop evaluation tool)
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.rag_agent.models import AgentRequest, AgentResponse
from src.rag_agent.agent import create_rag_agent


class ManualValidationHelper:
    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []
        self.agent = create_rag_agent()  # ✅ create ONCE

        self.evaluation_criteria = {
            "relevance": "How relevant is the response to the query?",
            "accuracy": "How factually correct is the response?",
            "completeness": "Does it fully answer the question?",
            "helpfulness": "Is the response helpful to the user?",
            "citations": "Are sources relevant and meaningful?",
        }

    def evaluate_response(self, query: str, response: AgentResponse) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"QUERY:\n{query}")
        print("=" * 60)

        print("\nANSWER:\n")
        print(response.answer)

        print("\n--- Retrieval Info ---")
        print(f"Chunks Retrieved: {len(response.retrieved_chunks)}")
        print(f"Confidence Score: {response.confidence_score:.3f}")

        if response.retrieved_chunks:
            print("\nSample Sources:")
            for i, chunk in enumerate(response.retrieved_chunks[:3]):
                print(f"\n[{i+1}] {chunk.metadata.get('url', 'N/A')}")
                print(chunk.content[:120] + "...")
        else:
            print("⚠️ No retrieved content found")

        evaluation = {
            "query": query,
            "answer_preview": response.answer[:300],
            "timestamp": datetime.now().isoformat(),
            "scores": {},
            "feedback": {},
        }

        print("\nRate each category (1=poor → 5=excellent):")

        for criterion, description in self.evaluation_criteria.items():
            while True:
                try:
                    score = int(input(f"{criterion.title()} ({description}): "))
                    if 1 <= score <= 5:
                        evaluation["scores"][criterion] = score
                        break
                except ValueError:
                    pass
                print("Enter a number between 1 and 5.")

            comment = input("Optional feedback: ").strip()
            evaluation["feedback"][criterion] = comment or "—"

        evaluation["overall_score"] = round(
            sum(evaluation["scores"].values()) / len(evaluation["scores"]), 2
        )

        print(f"\n✅ Overall Score: {evaluation['overall_score']}/5")
        self.validation_results.append(evaluation)

        return evaluation

    def evaluate_query(self, query: str):
        response = self.agent.process_query(query)
        return self.evaluate_response(query, response)

    def save_results(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

    def summary(self) -> Dict[str, Any]:
        if not self.validation_results:
            return {"message": "No evaluations yet"}

        avg = {}
        for key in self.evaluation_criteria:
            avg[key] = round(
                sum(r["scores"][key] for r in self.validation_results)
                / len(self.validation_results),
                2,
            )

        return {
            "total": len(self.validation_results),
            "average_scores": avg,
            "overall_avg": round(
                sum(r["overall_score"] for r in self.validation_results)
                / len(self.validation_results),
                2,
            ),
        }
