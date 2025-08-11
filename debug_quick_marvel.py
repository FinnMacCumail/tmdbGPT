#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import build_app_graph

graph = build_app_graph()
result = graph.invoke({"input": "Movies by Marvel Studios"})
print("Done")