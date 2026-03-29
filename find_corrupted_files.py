import os
import bz2
import pickle

#Script that detects potential errors in opening graph_exports files

graph_directory = "data/graph_exports"

files = sorted([
    f for f in os.listdir(graph_directory)
    if os.path.isfile(os.path.join(graph_directory, f))
    and f.endswith(".pbz2")
])


files2 = sorted([
    f for f in os.listdir(graph_directory)
    if os.path.isfile(os.path.join(graph_directory, f))
    and f.endswith(".pbz2")
])

print(len(files))
print(len(files2))

corrupted = []
checked = 0

for file in files:
    path = os.path.join(graph_directory, file)
    try:
        with bz2.open(path, "rb") as f:
            pickle.load(f)
    except Exception as e:
        print("\no CORRUPTED FILE:", file)
        print("   Error:", e)
        corrupted.append(file)
    checked += 1
    if checked % 100 == 0:
        print(f"Checked {checked}/{len(files)} files...")

print("\n===== SCAN COMPLETE =====")
print(f"Total files checked: {checked}")
print(f"Corrupted files found: {len(corrupted)}")

if corrupted:
    print("\nList of corrupted files:")
    for f in corrupted:
        print(" -", f)
else:
    print("No corrupted file)")
