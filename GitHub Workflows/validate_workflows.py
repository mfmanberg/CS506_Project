#!/usr/bin/env python3
"""
Validate GitHub Actions workflow files for syntax and reproducibility.
Tests workflows to ensure they work on any computer.
"""

import sys
from pathlib import Path
import re


def validate_yaml_syntax(file_path):
    """Validate YAML syntax."""
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, "Valid YAML syntax"
    except ImportError:
        # Fallback: basic validation without yaml module
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Basic checks
                if not content.strip():
                    return False, "Empty file"
                if content.count(':') < 3:
                    return False, "Doesn't look like YAML (too few colons)"
                return True, "Basic YAML structure OK (install PyYAML for full validation)"
        except Exception as e:
            return False, f"Read error: {e}"
    except Exception as e:
        return False, f"YAML syntax error: {e}"


def check_hardcoded_paths(file_path):
    """Check for hardcoded paths that won't work on other computers."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Patterns to check
    hardcoded_patterns = [
        (r'/home/[^/]+/', 'Hardcoded home directory'),
        (r'C:\\Users\\[^\\]+\\', 'Hardcoded Windows user path'),
        (r'/mnt/c/Users/[^/]+/', 'Hardcoded WSL path'),
        (r'cd /[a-z]/Users/', 'Hardcoded user directory'),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, description in hardcoded_patterns:
            if re.search(pattern, line):
                issues.append(f"  Line {i}: {description}: {line.strip()}")
    
    return issues


def check_requirements_path(file_path):
    """Verify requirements.txt path is correct."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for requirements.txt references
    if 'requirements.txt' in content:
        if '-r requirements.txt' in content:
            # Old path (missing Dependencies/)
            issues.append("  ⚠️  Uses 'requirements.txt' instead of 'Dependencies/requirements.txt'")
        elif 'Dependencies/requirements.txt' in content:
            issues.append("  ✅ Correctly uses 'Dependencies/requirements.txt'")
    
    return issues


def check_extract_results_path(file_path):
    """Check if extract_results.py path is correct."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'extract_results.py' in content:
        if 'python extract_results.py' in content:
            issues.append("  ⚠️  Uses 'python extract_results.py' (should be 'python Build/extract_results.py')")
        elif 'python Build/extract_results.py' in content:
            issues.append("  ✅ Correctly uses 'python Build/extract_results.py'")
    
    return issues


def check_test_execution_path(file_path):
    """Check if test_execution.py path is correct."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'test_execution.py' in content:
        if 'python test_execution.py' in content:
            issues.append("  ⚠️  Uses 'python test_execution.py' (should be 'python Build/test_execution.py')")
        elif 'python Build/test_execution.py' in content:
            issues.append("  ✅ Correctly uses 'python Build/test_execution.py'")
    
    return issues


def check_model_results_log(file_path):
    """Check if model_results.log path is correct."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'model_results.log' in content:
        # Should reference Build/model_results.log
        if re.search(r'(?<!Build/)model_results\.log', content):
            issues.append("  ⚠️  References 'model_results.log' (should be 'Build/model_results.log')")
        if 'Build/model_results.log' in content:
            issues.append("  ✅ Correctly uses 'Build/model_results.log'")
    
    return issues


def validate_workflow(file_path):
    """Comprehensive workflow validation."""
    print(f"\n{'='*70}")
    print(f"Validating: {file_path.name}")
    print(f"{'='*70}")
    
    all_issues = []
    
    # 1. YAML Syntax
    valid, msg = validate_yaml_syntax(file_path)
    if valid:
        print(f"✅ YAML Syntax: {msg}")
    else:
        print(f"❌ YAML Syntax: {msg}")
        all_issues.append(msg)
    
    # 2. Hardcoded paths
    hardcoded = check_hardcoded_paths(file_path)
    if hardcoded:
        print(f"❌ Hardcoded Paths Found ({len(hardcoded)}):")
        for issue in hardcoded:
            print(issue)
            all_issues.append(issue)
    else:
        print("✅ No hardcoded paths detected")
    
    # 3. requirements.txt path
    req_issues = check_requirements_path(file_path)
    if req_issues:
        print("📋 Requirements Path:")
        for issue in req_issues:
            print(issue)
            if '⚠️' in issue:
                all_issues.append(issue)
    
    # 4. extract_results.py path
    extract_issues = check_extract_results_path(file_path)
    if extract_issues:
        print("📋 Extract Results Path:")
        for issue in extract_issues:
            print(issue)
            if '⚠️' in issue:
                all_issues.append(issue)
    
    # 5. test_execution.py path
    test_issues = check_test_execution_path(file_path)
    if test_issues:
        print("📋 Test Execution Path:")
        for issue in test_issues:
            print(issue)
            if '⚠️' in issue:
                all_issues.append(issue)
    
    # 6. model_results.log path
    log_issues = check_model_results_log(file_path)
    if log_issues:
        print("📋 Model Results Log Path:")
        for issue in log_issues:
            print(issue)
            if '⚠️' in issue:
                all_issues.append(issue)
    
    return len([i for i in all_issues if '❌' in i or '⚠️' in i]) == 0


def main():
    """Main validation function."""
    script_dir = Path(__file__).parent
    
    # Find all workflow YAML files
    workflow_files = list(script_dir.glob('workflows_*.yml'))
    
    if not workflow_files:
        print("❌ No workflow files found (looking for workflows_*.yml)")
        return 1
    
    print(f"Found {len(workflow_files)} workflow files to validate")
    
    all_passed = True
    for workflow_file in sorted(workflow_files):
        passed = validate_workflow(workflow_file)
        if not passed:
            all_passed = False
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if all_passed:
        print(f"✅ All {len(workflow_files)} workflow files passed validation")
        print("✅ Workflows should be reproducible on other computers")
        return 0
    else:
        print(f"⚠️  Some workflow files have issues")
        print("⚠️  Review warnings above to ensure reproducibility")
        return 1


if __name__ == '__main__':
    sys.exit(main())
