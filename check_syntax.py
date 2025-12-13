#!/usr/bin/env python
"""
Syntax and logic checker for run_makefile.py
"""
import ast
import sys

def check_syntax(filename):
    """Check Python file for syntax errors."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Try to parse the file
        ast.parse(code)
        print(f"OK: {filename}: No syntax errors found")
        return True
    except SyntaxError as e:
        print(f"ERROR: {filename}: Syntax error at line {e.lineno}")
        print(f"  {e.msg}")
        print(f"  {e.text}")
        return False
    except Exception as e:
        print(f"ERROR: {filename}: {e}")
        return False

if __name__ == "__main__":
    result = check_syntax("run_makefile.py")
    sys.exit(0 if result else 1)
