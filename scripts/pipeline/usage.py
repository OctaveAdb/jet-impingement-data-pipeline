"""Redirect shim — usage.py has moved to the project root."""
import subprocess
import sys
import os

if __name__ == "__main__":
    root_usage = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'usage.py')
    subprocess.run([sys.executable, root_usage] + sys.argv[1:])
