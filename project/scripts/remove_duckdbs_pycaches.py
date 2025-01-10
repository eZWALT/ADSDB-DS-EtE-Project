import os
import shutil

def remove_duckdb_and_pycaches():
    # Recursively go through the directory structure and delete .duckdb files and __pycache__ directories and pkl files

    for root, dirs, files in os.walk('..'):
        # Remove all .duckdb files
        for file in files:
            if file.endswith(".duckdb") or file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

        # Remove all __pycache__ directories
        if '__pycache__' in dirs:
            pycache_dir = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_dir)
            print(f"Deleted: {pycache_dir}")

    print(".duckdb files and __pycache__ directories have been removed.")

# Run the function to clean up .duckdb files and __pycache__ directories
remove_duckdb_and_pycaches()
