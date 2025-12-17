#!/usr/bin/env python3
"""
Command-line interface for the content embedding pipeline.
This script provides CLI access to the content embedding functionality.
"""

import asyncio
import sys
import os
import argparse
from typing import Optional

# Add the backend/src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from content_embedding.embedding_service import main, run_pipeline_with_progress
from content_embedding.utils import setup_logging, validate_config, Config


def create_cli_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Content Embedding Pipeline - Crawl, extract, chunk, and store website content as embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --run                    # Run the full pipeline with progress tracking
  %(prog)s --validate-config        # Validate configuration without running the pipeline
  %(prog)s --show-config            # Show current configuration values
        """
    )

    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the full content embedding pipeline'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration settings'
    )

    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display current configuration values'
    )

    parser.add_argument(
        '--book-url',
        type=str,
        help='Override the base URL for the book (default from config)'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        help='Override the maximum crawling depth (default from config)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Override the chunk size (default from config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser


def show_config():
    """Display the current configuration values."""
    print("Current Configuration:")
    print(f"  BOOK_BASE_URL: {Config.BOOK_BASE_URL}")
    print(f"  TARGET_SITE: {Config.TARGET_SITE}")
    print(f"  SITEMAP_URL: {Config.SITEMAP_URL}")
    print(f"  CHUNK_SIZE: {Config.CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP: {Config.CHUNK_OVERLAP}")
    print(f"  MAX_DEPTH: {Config.MAX_DEPTH}")
    print(f"  REQUEST_DELAY: {Config.REQUEST_DELAY}")
    print(f"  COLLECTION_NAME: {Config.COLLECTION_NAME}")
    print(f"  COHERE_MODEL: {Config.COHERE_MODEL}")


async def run_pipeline_cli(args: argparse.Namespace) -> bool:
    """Run the content embedding pipeline based on CLI arguments."""
    # Set up logging
    setup_logging()

    # Override configuration values if provided
    if args.book_url:
        Config.BOOK_BASE_URL = args.book_url
    if args.max_depth is not None:
        Config.MAX_DEPTH = args.max_depth
    if args.chunk_size is not None:
        Config.CHUNK_SIZE = args.chunk_size

    try:
        if args.validate_config:
            validate_config()
            print("✓ Configuration is valid")
            return True

        if args.show_config:
            show_config()
            return True

        if args.run:
            print("Starting content embedding pipeline...")
            await run_pipeline_with_progress()
            print("Content embedding pipeline completed successfully!")
            return True

        # If no specific action was requested, show help
        print("No action specified. Use --run to execute the pipeline or --help for options.")
        return False

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


async def main_cli():
    """Main CLI entry point."""
    parser = create_cli_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    args = parser.parse_args()

    success = await run_pipeline_cli(args)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main_cli())
    sys.exit(exit_code)