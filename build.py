import subprocess
import sys
from pathlib import Path

def build_hook(config, **kwargs):
    """Run tests before building package."""
    print("Running tests before build...")
    
    # Run pytest
    result = subprocess.run(
        ["pytest", "--cov=skext", "--cov-report=term-missing", str(Path(__file__).parent / "tests")],
        capture_output=True,
        text=True
    )
    
    # Print test output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Fail build if tests fail
    if result.returncode != 0:
        raise Exception("Tests failed, aborting build")
    
    print("Tests passed successfully!")
    return config
