#!/usr/bin/env python3
"""
YouTube Transcription MCP Server - Main Entry Point

This is the main entry point for the YouTube Transcription MCP server.
It imports and runs the server from the src module.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.server import main

if __name__ == "__main__":
    main()
