#!/usr/bin/env python3
import sys

from minselfflow.utils.log import logger
from minselfflow.utils.timing import segment_timer


def main() -> int:
    logger.info("Starting minselfflow application")
    logger.debug("Debug logging is enabled")

    try:
        # Example usage of timing utility
        with segment_timer("Example work execution"):
            logger.info("Performing some example work")
            # Simulate some work
            result = "Hello from minselfflow!"
            logger.debug("Work completed with result: %s", result)

        logger.info("Application completed successfully: %s", result)
        return 0

    except Exception as e:
        logger.exception("Application failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
