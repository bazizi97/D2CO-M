#!/usr/bin/env bash
# Script to download the Ingraham CATH dataset into a specified directory.
#
# Usage:
#   ./load_Ingraham.sh [TARGET_DIR]
#   OR
#   ./load_Ingraham.sh -o [TARGET_DIR]

set -e

# Default target directory
TARGET_DIR="./data/cath"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir|--out-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

echo "Creating target directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

echo "Downloading Ingraham CATH dataset files into ${TARGET_DIR}..."
wget -c http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl -P "${TARGET_DIR}"
wget -c http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json -P "${TARGET_DIR}"

echo "Dataset downloaded successfully to ${TARGET_DIR}"