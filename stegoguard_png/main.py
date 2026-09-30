#!/usr/bin/env python3
"""Точка входа: python main.py <команда> ... (без аргументов запускается графический интерфейс)."""
import sys

from stegoguard.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["gui"]))
