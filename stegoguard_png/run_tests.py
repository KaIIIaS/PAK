#!/usr/bin/env python3
"""Запуск всех автотестов: python run_tests.py"""
import sys
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
