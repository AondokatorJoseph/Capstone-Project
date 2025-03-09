"""
Test suite for CyclotekAI waste classification system.

This package contains all test modules including:
- Unit tests for data loading and preprocessing
- Model architecture and training tests
- Integration tests for the complete pipeline
"""

import os
import sys

# Add project root to Python path for test imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import test modules
from tests.unit.test_dataset import TestDataset
from tests.unit.test_model import TestModel

__all__ = ['TestDataset', 'TestModel']