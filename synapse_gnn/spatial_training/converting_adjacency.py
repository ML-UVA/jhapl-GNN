import numpy as np
import os
import time
import ijson

# CONFIGURATION
INPUT_FILE = "adjacency.json"
OUTPUT_DIR = "processed_edges"
CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB chunks
CHUNK_SIZE = 5_000_000  # 100 MB chunks


os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_adjacency_list():
    print(f"Starting Adjacency List Processing of {INPUT_FILE}...")
    start_time = time.time()
    
    edge_buffer = []
    chunk_idx = 0
    total_edges = 0

    with open(INPUT_FILE, 'rb') as f:
        try:
            # ijson.kvitems returns (key, value) tuples from the root object
            # 'key' is the Source Node (String)
            # 'value' is the List of Target Nodes (Array of Ints)
            
            parser = ijson.kvitems(f, '')
            
            for source_id_str, target_list in parser:
                # 1. Convert Source to Int
                try:
                    src_id = int(source_id_str)
                except ValueError:
                    continue # Skip if key is not a number
                
                if not target_list:
                    continue

                # 2. Convert Targets to Numpy Array (Fast)
                targets = np.array(target_list, dtype=np.uint64)
                
                # 3. Create Source Column (Repeat src_id N times)
                sources = np.full(len(targets), src_id, dtype=np.uint64)
                
                # 4. Stack into Edges (N, 2)
                # Shape: [[src, tgt1], [src, tgt2], ...]
                new_edges = np.column_stack((sources, targets))
                
                # 5. Add to buffer
                edge_buffer.append(new_edges)
                
                # Check buffer size (approximate edge count)
                # We check purely based on list length to avoid overhead
                if len(edge_buffer) >= 1000: 
                    # Consolidate and check real size
                    batch = np.concatenate(edge_buffer)
                    if len(batch) >= CHUNK_SIZE:
                        save_chunk(batch, chunk_idx)
                        total_edges += len(batch)
                        chunk_idx += 1
                        edge_buffer = [] # Clear
                    else:
                        # Put it back if not big enough (optimization)
                        edge_buffer = [batch]

        except Exception as e:
            print(f"\n--- Stopped at File Corruption (Expected) ---")
            print(f"Error: {e}")

    # Save leftovers
    if edge_buffer:
        final_batch = np.concatenate(edge_buffer)
        if len(final_batch) > 0:
            save_chunk(final_batch, chunk_idx)
            total_edges += len(final_batch)

    print("-" * 30)
    print(f"DONE! Processed {total_edges} edges.")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

def save_chunk(arr, idx):
    filename = os.path.join(OUTPUT_DIR, f"edges_dict_part_{idx}.npy")
    np.save(filename, arr)
    print(f"Saved chunk {idx}: {len(arr)} edges")
if __name__ == "__main__":
    with open(INPUT_FILE, 'rb') as f:
        process_adjacency_list()