#!/bin/bash

# prepare_cls_dataset.sh
# Script to merge two datasets and create a labels.jsonl file
# with label 0 for images from the first dataset and label 1 for images from the second dataset

# Function to print usage information
print_usage() {
    echo "Usage: $0 <destination_folder> <source_folder1>[:<percentage>] <source_folder2>[:<percentage>]"
    echo "Example: $0 /path/to/merged_dataset /path/to/dataset1:100 /path/to/dataset2:50"
    echo "         This will include 100% of images from dataset1 (label 0) and 50% of images from dataset2 (label 1)"
    echo "Note: If no percentage is specified, 100% is assumed"
}

# Check if exactly three arguments are provided (one destination and two sources)
if [ $# -ne 3 ]; then
    print_usage
    exit 1
fi

# Get the destination folder from the first argument
DEST_FOLDER="$1"
shift  # Remove the first argument, leaving only source folders

# Create destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

echo "Merging datasets into: $DEST_FOLDER"

# Initialize the labels.jsonl file
LABELS_FILE="$DEST_FOLDER/labels.jsonl"
> "$LABELS_FILE"  # Create or truncate the file

# Process each source folder
LABEL=0  # Start with label 0 for the first dataset
for SOURCE_ARG in "$@"; do
    # Split the argument into folder and percentage
    SOURCE_FOLDER=$(echo "$SOURCE_ARG" | cut -d: -f1)
    PERCENTAGE=$(echo "$SOURCE_ARG" | grep -q ":" && echo "$SOURCE_ARG" | cut -d: -f2 || echo "100")
    
    # Validate percentage
    if ! [[ "$PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$PERCENTAGE" -gt 100 ]; then
        echo "Error: Invalid percentage '$PERCENTAGE' for $SOURCE_FOLDER. Using 100% instead."
        PERCENTAGE=100
    fi
    
    echo "Processing source folder: $SOURCE_FOLDER (using $PERCENTAGE% of images, label: $LABEL)"
    
    # Function to process images from a directory
    process_images() {
        local SRC_DIR="$1"
        local IMGS=()
        
        # Collect all image files
        while IFS= read -r -d $'\0' img; do
            IMGS+=("$img")
        done < <(find "$SRC_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o -name "*.bmp" -o -name "*.webp" \) -print0)
        
        # Calculate how many images to take
        local TOTAL=${#IMGS[@]}
        local TO_TAKE=$(( TOTAL * PERCENTAGE / 100 ))
        
        echo "Found $TOTAL images, taking $TO_TAKE ($PERCENTAGE%)"
        
        # If no images to take, return
        if [ "$TO_TAKE" -eq 0 ]; then
            return
        fi
        
        # Shuffle the array and take the first TO_TAKE elements
        if [ "$PERCENTAGE" -lt 100 ]; then
            # Only shuffle if we're not taking all images
            local SHUFFLED=( $(shuf -e "${IMGS[@]}") )
            IMGS=( "${SHUFFLED[@]:0:$TO_TAKE}" )
        fi
        
        # Create symlinks for the selected images and add entries to labels.jsonl
        for img in "${IMGS[@]}"; do
            local DEST_FILENAME="$(basename "$img")"
            # Ensure unique filenames by prefixing with dataset label if needed
            if [ -e "$DEST_FOLDER/$DEST_FILENAME" ]; then
                DEST_FILENAME="dataset${LABEL}_${DEST_FILENAME}"
            fi
            
            ln -sf "$(realpath "$img")" "$DEST_FOLDER/$DEST_FILENAME"
            
            # Add entry to labels.jsonl
            echo "{\"image_file\": \"$DEST_FILENAME\", \"label\": $LABEL}" >> "$LABELS_FILE"
        done
    }
    
    # Symlink selected images to the destination folder
    if [ -d "$SOURCE_FOLDER/images" ]; then
        # If there's an images subfolder, symlink from there
        echo "Symlinking images from $SOURCE_FOLDER/images"
        process_images "$SOURCE_FOLDER/images"
    else
        # Otherwise symlink image files directly from the source folder
        echo "Symlinking images from $SOURCE_FOLDER"
        process_images "$SOURCE_FOLDER"
    fi
    
    # Increment the label for the next dataset
    LABEL=$((LABEL + 1))
done

# Count the number of images in the destination folder
IMAGE_COUNT=$(find "$DEST_FOLDER" -type l | wc -l)
echo "Merged dataset contains $IMAGE_COUNT images (symlinks)"
echo "Created labels.jsonl file with binary classification labels (0 for first dataset, 1 for second dataset)"

echo "Dataset merging complete!"
