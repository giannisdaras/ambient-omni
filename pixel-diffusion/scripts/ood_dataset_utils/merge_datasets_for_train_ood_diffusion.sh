#!/bin/bash

# merge_datasets_for_train_ood_diffusion.sh
# Script to merge multiple datasets by symlinking all images to a target folder
# and concatenating any annotations.jsonl files
# Supports selecting a percentage of images from each dataset

# Function to print usage information
print_usage() {
    echo "Usage: $0 <destination_folder> <source_folder1>[:<percentage>] [<source_folder2>[:<percentage>] ...]"
    echo "Example: $0 /path/to/merged_dataset /path/to/dataset1:100 /path/to/dataset2:50"
    echo "         This will include 100% of images from dataset1 and 50% of images from dataset2"
    echo "Note: If no percentage is specified, 100% is assumed"
}

# Check if at least two arguments are provided (at least one source and one destination)
if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

# Get the destination folder from the first argument
DEST_FOLDER="$1"
shift  # Remove the first argument, leaving only source folders

# Create destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

echo "Merging datasets into: $DEST_FOLDER"

# Initialize a temporary file for concatenating annotations
TEMP_ANNOTATIONS=$(mktemp)

# Process each source folder
for SOURCE_ARG in "$@"; do
    # Split the argument into folder and percentage
    SOURCE_FOLDER=$(echo "$SOURCE_ARG" | cut -d: -f1)
    PERCENTAGE=$(echo "$SOURCE_ARG" | grep -q ":" && echo "$SOURCE_ARG" | cut -d: -f2 || echo "100")
    
    # Validate percentage
    if ! [[ "$PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$PERCENTAGE" -gt 100 ]; then
        echo "Error: Invalid percentage '$PERCENTAGE' for $SOURCE_FOLDER. Using 100% instead."
        PERCENTAGE=100
    fi
    
    echo "Processing source folder: $SOURCE_FOLDER (using $PERCENTAGE% of images)"
    
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
        
        # Create symlinks for the selected images
        for img in "${IMGS[@]}"; do
            ln -sf "$(realpath "$img")" "$DEST_FOLDER/$(basename "$img")"
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
    
    # Append annotations.jsonl if it exists
    if [ -f "$SOURCE_FOLDER/annotations.jsonl" ]; then
        echo "Appending annotations from $SOURCE_FOLDER/annotations.jsonl"
        
        # If we're taking all images, just append the whole file
        if [ "$PERCENTAGE" -eq 100 ]; then
            cat "$SOURCE_FOLDER/annotations.jsonl" >> "$TEMP_ANNOTATIONS"
        else
            # Otherwise, we need to filter the annotations to match the selected images
            echo "Filtering annotations to match selected images..."
            
            # Get the list of selected image filenames (without path)
            local SELECTED_IMAGES=$(find "$DEST_FOLDER" -type l -name "$(basename "$SOURCE_FOLDER")*" -exec basename {} \; | sort)
            
            # Filter annotations to only include those for selected images
            while IFS= read -r line; do
                # Extract the image filename from the annotation
                local IMG_FILE=$(echo "$line" | grep -o '"image_file": "[^"]*"' | cut -d'"' -f4 | xargs basename)
                
                # If the image is in our selected list, include the annotation
                if echo "$SELECTED_IMAGES" | grep -q "^$IMG_FILE$"; then
                    echo "$line" >> "$TEMP_ANNOTATIONS"
                fi
            done < "$SOURCE_FOLDER/annotations.jsonl"
        fi
    fi
done

# Move the concatenated annotations to the destination folder if not empty
if [ -s "$TEMP_ANNOTATIONS" ]; then
    mv "$TEMP_ANNOTATIONS" "$DEST_FOLDER/annotations.jsonl"
    echo "Created merged annotations.jsonl file"
else
    echo "No annotations.jsonl files found in source folders"
    rm "$TEMP_ANNOTATIONS"
fi

# Count the number of images in the destination folder
IMAGE_COUNT=$(find "$DEST_FOLDER" -type l | wc -l)
echo "Merged dataset contains $IMAGE_COUNT images (symlinks)"

echo "Dataset merging complete!"
