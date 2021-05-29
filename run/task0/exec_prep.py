import subprocess

print("Splitting original data...")
subprocess.run(["python", "run/task0/split.py"])

print("Cropping splitted data...")
subprocess.run(["python", "run/task0/crop_mask.py"])

print("Preprocessing cropped data...")
subprocess.run(["python", "run/task0/pre_processing.py"])

print("Dataset preparation ended!")
