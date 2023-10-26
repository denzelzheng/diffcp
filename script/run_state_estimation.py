import subprocess
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
subprocess.run(["python", current_dir + "/../src/state_estimation_main.py"])