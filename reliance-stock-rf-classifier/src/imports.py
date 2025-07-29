import subprocess
import sys
import os

req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("✅ All requirements installed successfully.")
except subprocess.CalledProcessError as e:
    print("❌ Failed to install some packages.")
    print(f"Error code: {e.returncode}")
    print("Check the full pip output above ⬆️ for details.")
