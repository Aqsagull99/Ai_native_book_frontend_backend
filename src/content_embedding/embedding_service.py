import asyncio
import logging
from typing import List, Dict, Any
from .crawler import get_all_url
from .text_extractor import extract_text_from_url
from .chunker import chunk_text_with_metadata
from .qdrant_service import create_collection, batch_save_chunks_to_qdrant
from .utils import Config, setup_logging, validate_config

logger = logging.getLogger(__name__)


async def main():
    """
    Main execution function to orchestrate the entire pipeline from URL crawling to Qdrant storage.
    """
    # Set up logging
    setup_logging()
    logger.info("Starting content embedding pipeline...")

    # Validate configuration
    validate_config()
    logger.info("Configuration validation passed")

    try:
        # Step 1: Create Qdrant collection
        logger.info("Creating Qdrant collection...")
        await create_collection(Config.COLLECTION_NAME)
        logger.info(f"Qdrant collection '{Config.COLLECTION_NAME}' created successfully")

        # Step 2: Get all URLs from the target site
        logger.info(f"Starting to crawl URLs from {Config.BOOK_BASE_URL}...")
        urls = await get_all_url(Config.BOOK_BASE_URL, max_depth=Config.MAX_DEPTH)
        logger.info(f"Found {len(urls)} URLs to process")

        # Step 3: Process each URL: extract content, chunk, and save to Qdrant
        total_chunks = 0
        processed_urls = 0

        for i, url in enumerate(urls):
            try:
                logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

                # Extract content from the URL
                content_data = await extract_text_from_url(url)
                logger.info(f"Extracted content from {url} (title: {content_data['title']})")

                # Chunk the content with metadata
                chunks = chunk_text_with_metadata(
                    text=content_data['content'],
                    metadata=content_data['metadata'],
                    chunk_size=Config.CHUNK_SIZE,
                    overlap=Config.CHUNK_OVERLAP
                )
                logger.info(f"Chunked content into {len(chunks)} chunks")

                # Save chunks to Qdrant
                success = await batch_save_chunks_to_qdrant(
                    chunks=chunks,
                    collection_name=Config.COLLECTION_NAME,
                    batch_size=10
                )

                if success:
                    total_chunks += len(chunks)
                    processed_urls += 1
                    logger.info(f"Successfully saved {len(chunks)} chunks for {url}")
                else:
                    logger.error(f"Failed to save chunks for {url}")

            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue  # Continue with the next URL

        logger.info(f"Pipeline completed! Processed {processed_urls}/{len(urls)} URLs, "
                   f"saved {total_chunks} total chunks to Qdrant collection '{Config.COLLECTION_NAME}'")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


async def run_pipeline_with_progress():
    """
    Run the embedding pipeline with detailed progress tracking and status updates.
    """
    # Set up logging
    setup_logging()
    logger.info("Starting content embedding pipeline with progress tracking...")

    try:
        # Step 1: Create Qdrant collection
        logger.info("Creating Qdrant collection...")
        await create_collection(Config.COLLECTION_NAME)
        logger.info(f"Qdrant collection '{Config.COLLECTION_NAME}' created successfully")

        # Step 2: Get all URLs from the target site
        logger.info(f"Starting to crawl URLs from {Config.BOOK_BASE_URL}...")
        urls = await get_all_url(Config.BOOK_BASE_URL, max_depth=Config.MAX_DEPTH)
        logger.info(f"Found {len(urls)} URLs to process")

        # Step 3: Process each URL with progress tracking
        total_urls = len(urls)
        processed_urls = 0
        total_chunks = 0

        for i, url in enumerate(urls):
            try:
                logger.info(f"Processing URL {i+1}/{total_urls}: {url}")

                # Extract content from the URL
                content_data = await extract_text_from_url(url)
                logger.info(f"Extracted content from {url} (title: {content_data['title']})")

                # Chunk the content with metadata
                chunks = chunk_text_with_metadata(
                    text=content_data['content'],
                    metadata=content_data['metadata'],
                    chunk_size=Config.CHUNK_SIZE,
                    overlap=Config.CHUNK_OVERLAP
                )
                logger.info(f"Chunked content into {len(chunks)} chunks")

                # Save chunks to Qdrant
                success = await batch_save_chunks_to_qdrant(
                    chunks=chunks,
                    collection_name=Config.COLLECTION_NAME,
                    batch_size=10
                )

                if success:
                    total_chunks += len(chunks)
                    processed_urls += 1
                    logger.info(f"Successfully saved {len(chunks)} chunks for {url}")
                else:
                    logger.error(f"Failed to save chunks for {url}")

                # Log progress
                progress = (i + 1) / total_urls * 100
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{total_urls} URLs processed)")

            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue  # Continue with the next URL

        logger.info(f"Pipeline completed! Processed {processed_urls}/{total_urls} URLs, "
                   f"saved {total_chunks} total chunks to Qdrant collection '{Config.COLLECTION_NAME}'")

        # Final summary
        print(f"\nPipeline Summary:")
        print(f"- Total URLs found: {total_urls}")
        print(f"- URLs processed successfully: {processed_urls}")
        print(f"- Total text chunks created and saved: {total_chunks}")
        print(f"- Qdrant collection: {Config.COLLECTION_NAME}")
        print(f"- All content is now available for semantic search!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_pipeline_with_progress())