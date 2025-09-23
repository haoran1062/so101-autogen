# -*- coding: utf-8 -*-
"""
Unified Logging Management Utility
Provides functions for setting up logging, extracted from the main script.
"""

import logging
import os
import sys


class LoggerManager:
    """Unified Logging Manager"""
    
    @staticmethod
    def setup_logging(level="INFO"):
        """Sets up the logging configuration.
        
        Args:
            level (str): The logging level, e.g., "INFO", "DEBUG".
        """
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        print(f"✅ Logging system initialized with level: {level}")
    
    @staticmethod
    def get_logger(name):
        """Gets a logger with the specified name.
        
        Args:
            name (str): The name of the logger.
            
        Returns:
            logging.Logger: An instance of the logger.
        """
        return logging.getLogger(name)


def setup_logging(level="INFO"):
    """Compatibility function to maintain the same interface as the main script."""
    LoggerManager.setup_logging(level)
