#!/bin/bash

dir_preset="configs"

test_script="parallel_runner.py"

if [ ! -d "$dir_preset" ]; then
    echo "Error: folder $dir_preset does not exists."
    exit 1
fi

if [ ! -f "$test_script" ]; then
    echo "Error: file $test_script does not exists."
    exit 1
fi

for file in "$dir_preset"/*; do
    if [ -f "$file" ]; then
        echo "Running $test_script with $file"
        python3 "$test_script" "$file"
        if [ $? -ne 0 ]; then
            echo "Error in $test_script with $file"
            exit 1
        fi
    fi
done

echo "Completed."
