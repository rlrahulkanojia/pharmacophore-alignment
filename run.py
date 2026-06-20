#!/usr/bin/env python3
"""
Run the pharmacophore alignment solver.

Usage:
    python run.py
    python run.py --input targets.json --output poses.sdf
    python run.py --verbose
"""

import runpy
import sys
import os

# Ensure project root is on sys.path so `src` package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

runpy.run_module("src", run_name="__main__", alter_sys=True)
