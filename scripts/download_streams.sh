#!/bin/bash

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/test-audio"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download MSNBC streams with timing offset
echo "Starting staggered downloads of MSNBC streams..."

# Start ICEcast download first
echo "Starting Icecast download..."
ffmpeg -i "http://stream1.skyviewnetworks.com:8010/MSNBC" \
    -t 30 \
    -ar 44100 \
    -ac 2 \
    -y \
    "$OUTPUT_DIR/msnbc_icecast_30s.wav" \
    </dev/null \
    >"$OUTPUT_DIR/icecast_download.log" 2>&1 &
PID1=$!

echo "ICEcast download started (PID: $PID1)"
echo "Waiting 20 seconds before starting HLS download..."
sleep 20

echo "Starting HLS download..."
ffmpeg -i "https://tni-drct-msnbc-int-jg89w.fast.nbcuni.com/live/master.m3u8" \
    -t 30 \
    -ar 44100 \
    -ac 2 \
    -y \
    "$OUTPUT_DIR/msnbc_hls_30s.wav" \
    </dev/null \
    >"$OUTPUT_DIR/hls_download.log" 2>&1 &
PID2=$!

echo "HLS download started (PID: $PID2)"
echo "Waiting for downloads to complete..."

# Wait for both to finish
wait $PID1
RESULT1=$?
echo "Icecast download finished (exit code: $RESULT1)"

wait $PID2
RESULT2=$?
echo "HLS download finished (exit code: $RESULT2)"

# Check results
echo ""
echo "Download Summary:"
if [ $RESULT1 -eq 0 ]; then
    echo "✅ Icecast download successful"
    if [ -f "$OUTPUT_DIR/msnbc_icecast_30s.wav" ]; then
        SIZE1=$(ls -lh "$OUTPUT_DIR/msnbc_icecast_30s.wav" | awk '{print $5}')
        echo "   File: $OUTPUT_DIR/msnbc_icecast_30s.wav ($SIZE1)"
    fi
else
    echo "❌ Icecast download failed (check $OUTPUT_DIR/icecast_download.log)"
fi

if [ $RESULT2 -eq 0 ]; then
    echo "✅ HLS download successful"
    if [ -f "$OUTPUT_DIR/msnbc_hls_30s.wav" ]; then
        SIZE2=$(ls -lh "$OUTPUT_DIR/msnbc_hls_30s.wav" | awk '{print $5}')
        echo "   File: $OUTPUT_DIR/msnbc_hls_30s.wav ($SIZE2)"
    fi
else
    echo "❌ HLS download failed (check $OUTPUT_DIR/hls_download.log)"
fi

# Show file details if both succeeded
if [ $RESULT1 -eq 0 ] && [ $RESULT2 -eq 0 ]; then
    echo ""
    echo "File details:"
    ls -la "$OUTPUT_DIR"/*.wav
    
    echo ""
    echo "Timing strategy: ICEcast started 10 seconds before HLS"
    echo "Expected offset reduction: ~10 seconds (from ~32s to ~22s)"
    echo ""
    echo "Ready to test! Run from project root:"
    echo "./cdn-benchmark-cli fingerprint-test file://\$(pwd)/test-audio/msnbc_icecast_30s.wav file://\$(pwd)/test-audio/msnbc_hls_30s.wav --debug --verbose --content-type news --max-offset 40"
fi

echo ""
echo "Log files created:"
echo "  - $OUTPUT_DIR/icecast_download.log"
echo "  - $OUTPUT_DIR/hls_download.log"
