import json
import os
import glob

def cleanup_opponent_pool():
    metadata_path = os.path.abspath("models/opponent_pool/pool_metadata.json")
    pool_dir = os.path.dirname(metadata_path)

    print(f"Reading metadata from: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        print("Metadata file not found!")
        return

    with open(metadata_path, "r") as f:
        data = json.load(f)

    # Extract all paths from metadata and normalize them
    kept_files = set()
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if "path" in value:
            # Normalize path to absolute path and normalize case
            normalized_path = os.path.normcase(os.path.abspath(value["path"]))
            kept_files.add(normalized_path)

    print(f"Found {len(kept_files)} files referenced in metadata.")

    # List all files in the pool directory
    all_files = glob.glob(os.path.join(pool_dir, "*"))
    
    deleted_count = 0
    kept_count = 0

    for file_path in all_files:
        abs_path = os.path.normcase(os.path.abspath(file_path))
        
        # Skip the metadata file itself
        if abs_path == os.path.normcase(metadata_path):
            continue

        if abs_path not in kept_files:
            print(f"Deleting unused file: {file_path}")
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        else:
            kept_count += 1

    print(f"Cleanup complete. Deleted {deleted_count} files. Kept {kept_count} files.")

if __name__ == "__main__":
    cleanup_opponent_pool()
