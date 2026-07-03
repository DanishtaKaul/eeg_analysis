# scripts/__init__.py
"""shared logger setup for the pipeline scripts"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
