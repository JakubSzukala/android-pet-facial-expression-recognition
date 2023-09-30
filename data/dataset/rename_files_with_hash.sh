#!/bin/bash

# fail the entire script if any of commands fails or if unset variable will be expanded
set -e
set -u

SOURCE_DIR=$1
DST_DIR=$2

for FILE in "${SOURCE_DIR}"/*;
do
    echo "Processing: $FILE..."
    # Calculate md5sum, pipe it into cut that will split into fields by the
    # delimiter (-d ' ') and pick field number 1 (-f 1) to display
    HASH=$(md5sum "${FILE}" | cut -d ' ' -f 1)

    # Remove everything up to the last occurence of dot
    EXT=${FILE##*.}

    cp -v "$FILE" "${DST_DIR}/${HASH}.${EXT}"
done
