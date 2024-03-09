#!/bin/bash

# Directory containing the result files
results_dir="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/TESTS/results"

# Pattern to match the files
file_pattern="MERGE*"

# Loop through the files
for file in $results_dir/$file_pattern; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Clean the file by removing empty lines and lines starting with a character
        sed -i '/^[[:space:]]*$/d; /^[[:alpha:]]/d' "$file"
        echo "Cleaned $file"

        # Order the lines numerically based on the first two fields
        sort -t ',' -k1,2n -o "$file" "$file"
        echo "Ordered $file"
    fi
done

echo > BEST_RESULTS
# now for each file print on the screen for each group of lines (with the first two fields equal) the line with lowest value in the last field
for file in $results_dir/$file_pattern; do
    # Check if the file exists
    if [ -f "$file" ]; then
        echo "-----------------------------------------------------" >> BEST_RESULTS
        echo "-----------------File: $file-------------------------" >> BEST_RESULTS
        echo "-----------------------------------------------------" >> BEST_RESULTS
        # Print the lines with the lowest value in the last field for each group of lines with the same first two fields and save the result in a new file
        if [[ "$file" == *_gen* ]]; then
            awk -F ',' 'NR==1 {print $0; next} $1!=p1 || $2!=p2 {print $0; p1=$1; p2=$2} $1==p1 && $2==p2 && $6<min {min=$6; line=$0} END {print line}' "$file" >> BEST_RESULTS
        elif [[ "$file" == *_gen*NCCL* ]]; then
            awk -F ',' 'NR==1 {print $0; next} $1!=p1 || $2!=p2 {print $0; p1=$1; p2=$2} $1==p1 && $2==p2 && $7<min {min=$7; line=$0} END {print line}' "$file" >> BEST_RESULTS
        else
            awk -F ',' 'NR==1 {print $0; next} $1!=p1 || $2!=p2 {print $0; p1=$1; p2=$2} $1==p1 && $2==p2 && $3<min {min=$3; line=$0} END {print line}' "$file" >> BEST_RESULTS
        fi
    fi
done
