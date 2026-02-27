#!/usr/bin/env python3
"""Main entry point for Cloud Run Job."""
import sys
import traceback
import os
import logging

# Configure logging to ensure output is captured
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== Starting main.py ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

try:
    logger.info("Importing pipeline...")
    from pipeline import main
    logger.info("Pipeline imported successfully")
    if __name__ == '__main__':
        logger.info("Calling main()...")
        main()
        logger.info("main() completed successfully")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)
