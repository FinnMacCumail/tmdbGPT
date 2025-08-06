#!/usr/bin/env python3
"""
Script to automatically clean debug statements for production merge.
Usage: python scripts/clean_for_production.py
"""

import os
import re
from pathlib import Path

def clean_debug_statements(file_path):
    """Remove debug print statements while preserving 🧠 DEBUGGING SUMMARY REPORT."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Preserve 🧠 DEBUGGING SUMMARY REPORT
    if '🧠 DEBUGGING SUMMARY REPORT' in content:
        return content  # Don't modify log_summary.py or files with the report
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip standalone debug print statements
        if re.match(r'^\s*(print\s*\(|f".*"\)|f\'.*\'\))', line.strip()):
            cleaned_lines.append(re.sub(r'^(\s*)', r'\1# Debug output removed', line))
        # Fix empty except blocks
        elif line.strip() == 'except Exception as e:':
            cleaned_lines.append(line)
            cleaned_lines.append(re.sub(r'^(\s*)', r'\1    # Debug output removed\n\1    pass', line))
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_repository():
    """Clean all Python files in the repository."""
    
    for py_file in Path('.').rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        try:
            cleaned_content = clean_debug_statements(py_file)
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"✅ Cleaned {py_file}")
        except Exception as e:
            print(f"❌ Failed to clean {py_file}: {e}")

if __name__ == "__main__":
    print("🧹 Cleaning debug statements for production...")
    clean_repository()
    print("✅ Production cleaning complete!")