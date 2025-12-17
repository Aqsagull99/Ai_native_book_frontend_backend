#!/usr/bin/env python3
"""
Test script to validate that book content has been successfully embedded in Qdrant
and can be retrieved using semantic similarity search.
This script now includes comprehensive validation functionality
for the vector retrieval validation feature.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_comprehensive_validation():
    """Run comprehensive validation of the vector retrieval system."""
    print("🚀 Starting comprehensive vector retrieval validation...")

    # Import our validation modules
    try:
        from src.content_embedding.retrieval_service import run_full_validation_pipeline_and_verify_all_success_criteria
        from src.content_embedding.qdrant_service import validate_qdrant_connection
        from src.content_embedding.retrieval_service import run_basic_validation_test
        from src.content_embedding.retrieval_service import create_100_query_stress_test
    except ImportError as e:
        print(f"❌ Error importing validation modules: {e}")
        return False

    print("\n" + "="*80)
    print("VECTOR RETRIEVAL VALIDATION PIPELINE")
    print("="*80)

    # Step 1: Validate Qdrant connection
    print("\n🔍 Step 1: Validating Qdrant connection...")
    qdrant_valid = validate_qdrant_connection()
    print(f"   Qdrant connection: {'✅ Valid' if qdrant_valid else '❌ Invalid'}")

    if not qdrant_valid:
        print("❌ Cannot proceed with validation - Qdrant connection failed")
        return False

    # Step 2: Run basic validation tests
    print("\n🧪 Step 2: Running basic validation tests...")
    basic_test_results = run_basic_validation_test()
    print(f"   Basic tests: {basic_test_results['successful_queries']}/{basic_test_results['total_queries']} successful")
    print(f"   Success rate: {basic_test_results['successful_queries']/basic_test_results['total_queries']*100:.1f}%" if basic_test_results['total_queries'] > 0 else "   Success rate: 0%")

    # Step 3: Run 100-query stress test
    print("\n💪 Step 3: Running 100-query stress test...")
    print("   This may take a few minutes...")
    stress_test_results = create_100_query_stress_test()
    print(f"   Stress test: {stress_test_results['successful_queries']}/{stress_test_results['total_queries']} successful")
    print(f"   Success rate: {stress_test_results['success_rate_percentage']:.1f}%")
    print(f"   Avg response time: {stress_test_results['average_response_time']:.4f}s")
    print(f"   Queries under 2s: {stress_test_results['percent_queries_under_2s']:.1f}%")

    # Step 4: Run full validation pipeline
    print("\n🎯 Step 4: Running full validation pipeline...")
    full_validation_results = run_full_validation_pipeline_and_verify_all_success_criteria()

    print(f"\n📋 Validation Summary:")
    print(f"   Overall Success: {'✅ YES' if full_validation_results['overall_success'] else '❌ NO'}")
    print(f"   Connection Valid: {'✅ YES' if full_validation_results['connection_valid'] else '❌ NO'}")
    print(f"   Completed Steps: {len(full_validation_results['completed_steps'])}")

    print(f"\n✅ Success Criteria Verification:")
    for criteria, met in full_validation_results['success_criteria_met'].items():
        status = '✅ MET' if met else '❌ NOT MET'
        print(f"   {criteria}: {status}")

    print("\n" + "="*80)
    if full_validation_results['overall_success']:
        print("🎉 COMPREHENSIVE VALIDATION SUCCESSFUL!")
        print("✅ All vector retrieval validation tests passed")
        print("✅ The RAG retrieval layer is reliable and ready for agent integration")
        print("✅ Semantic search returns relevant content chunks")
        print("✅ Metadata (URL, section, chunk index) is preserved")
        print("✅ Retrieval works consistently across multiple queries")
    else:
        print("💥 COMPREHENSIVE VALIDATION FAILED!")
        print("❌ Some validation tests did not pass")
        print("❌ The RAG retrieval layer may not be ready for agent integration")

        # Print specific failures
        print("\n❌ Failed Criteria:")
        for criteria, met in full_validation_results['success_criteria_met'].items():
            if not met:
                print(f"   - {criteria}")

    print("="*80)

    return full_validation_results['overall_success']


def run_sample_queries():
    """Run sample queries to demonstrate the validation functionality."""
    print("\n🔍 Running sample queries to demonstrate validation...")

    try:
        from src.content_embedding.retrieval_service import validate_semantic_search, manual_validation_helper
    except ImportError as e:
        print(f"❌ Error importing query modules: {e}")
        return

    sample_queries = [
        "AI and machine learning fundamentals",
        "neural network architectures",
        "natural language processing"
    ]

    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        try:
            # Use manual validation helper for formatted results
            result = manual_validation_helper(query, top_k=3)
            print(f"  Results: {len(result['results'])} retrieved")
            for j, res in enumerate(result['results'], 1):
                print(f"    {j}. Score: {res['similarity_score']:.3f}, URL: {res['url'][:50]}...")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def main():
    """Main validation function."""
    print("🚀 Vector Retrieval & Pipeline Validation System")
    print("This system validates the RAG retrieval layer before agent integration\n")

    # Run comprehensive validation
    success = run_comprehensive_validation()

    # Run sample queries for demonstration
    run_sample_queries()

    print(f"\n🏁 Validation process completed.")
    print(f"Overall result: {'SUCCESS' if success else 'FAILURE'}")

    return success


if __name__ == "__main__":
    main()