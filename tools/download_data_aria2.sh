#!/bin/bash

# Configuration
COMPETITION="happy-whale-and-dolphin"
DEST_DIR="happywhale_data"
THREADS=16

# 1. Install Dependencies
echo "[1/4] Installing dependencies..."
if ! command -v aria2c &> /dev/null; then
    apt-get update && apt-get install -y aria2
fi
pip install -q kaggle

# 2. Setup Kaggle Config (if needed)
# Assumes kaggle.json is present in ~/.kaggle/kaggle.json or current dir
if [ -f "kaggle.json" ]; then
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
fi

# 3. Get Download URL
echo "[2/4] Getting download URL via Kaggle API..."
# This Python snippet uses the kaggle lib to get the signed URL
URL=$(python3 -c "import kaggle; kaggle.api.authenticate(); print(kaggle.api.competitions_data_download_files_url('${COMPETITION}'))")

if [ -z "$URL" ]; then
    echo "Error: Failed to get download URL. Check your kaggle.json configuration."
    exit 1
fi

# 4. Download with Aria2 (16 Threads)
echo "[3/4] Downloading dataset with Aria2 (16 threads)..."
aria2c -x ${THREADS} -s ${THREADS} -k 1M -c -o ${COMPETITION}.zip "${URL}"

# 5. Unzip
echo "[4/4] Unzipping dataset..."
mkdir -p ${DEST_DIR}
unzip -q ${COMPETITION}.zip -d ${DEST_DIR}
rm ${COMPETITION}.zip

echo "Done! Dataset is ready in ${DEST_DIR}"
