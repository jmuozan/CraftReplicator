import os
import argparse
from pathlib import Path

def rename_files(folder_path, dry_run=False):
    """
    Rename all files in the specified folder to sequential numbers starting from 0.
    
    Args:
        folder_path (str): Path to the folder containing files to rename
        dry_run (bool): If True, only preview changes without executing them
    """
    # Convert to Path object and verify folder exists
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"'{folder_path}' is not a valid directory")
    
    # Get list of files and sort them
    files = [f for f in folder.iterdir() if f.is_file()]
    files.sort()
    
    # Create list of rename operations
    operations = []
    for i, file_path in enumerate(files):
        new_name = f"{i}{file_path.suffix}"
        new_path = folder / new_name
        operations.append((file_path, new_path))
    
    # Print preview
    print(f"{'Preview of changes:' if dry_run else 'Executing changes:'}")
    print("-" * 50)
    for old_path, new_path in operations:
        print(f"{old_path.name} -> {new_path.name}")
    
    # Execute renaming if not a dry run
    if not dry_run:
        # First, rename all files to temporary names to avoid conflicts
        temp_operations = [(old, old.with_name(f"temp_{i}{old.suffix}")) 
                         for i, (old, _) in enumerate(operations)]
        
        # Do the temporary renames
        for old_path, temp_path in temp_operations:
            old_path.rename(temp_path)
            
        # Now rename to final names
        for i, (_, new_path) in enumerate(operations):
            temp_path = folder / f"temp_{i}{new_path.suffix}"
            temp_path.rename(new_path)
            
        print("\nFiles renamed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a folder to sequential numbers")
    parser.add_argument("folder", help="Path to the folder containing files to rename")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing them")
    
    args = parser.parse_args()
    
    try:
        rename_files(args.folder, args.dry_run)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())