#!/bin/bash

# CDN Latency Simulation Download Script
# Downloads source streams immediately, then CDN streams with realistic delays
# This simulates the actual user experience with CDN propagation delays

set -euo pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/test-audio"
CONFIG_FILE="$PROJECT_ROOT/configs/examples/test-broadcasts.yaml"

# Default parameters
DURATION=30
SAMPLE_RATE=44100
CHANNELS=2
ICECAST_CDN_DELAY=120  # 2 minutes delay for Icecast CDN
HLS_CDN_DELAY=180      # 3 minutes delay for HLS CDN (2 + 1 additional)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --icecast-delay)
            ICECAST_CDN_DELAY="$2"
            shift 2
            ;;
        --hls-delay)
            HLS_CDN_DELAY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --duration SECONDS    Recording duration (default: 30)"
            echo "  -o, --output DIR          Output directory"
            echo "  -c, --config FILE         Config file path"
            echo "  --icecast-delay SECONDS   Delay before downloading Icecast CDN (default: 120)"
            echo "  --hls-delay SECONDS       Delay before downloading HLS CDN (default: 180)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🎵 CDN LATENCY SIMULATION DOWNLOAD"
echo "=================================="
echo "Duration: ${DURATION}s"
echo "Icecast CDN delay: ${ICECAST_CDN_DELAY}s"
echo "HLS CDN delay: ${HLS_CDN_DELAY}s"
echo ""

# Function to add ad-skipping parameters to URLs
add_ad_skip_params() {
    local url="$1"
    local separator
    
    if [[ "$url" == *"?"* ]]; then
        separator="&"
    else
        separator="?"
    fi
    
    echo "${url}${separator}aw_0_1st.premium=true&partnerID=BotTIStream&playerid=BotTIStream&aw_0_1st.ads_partner_alias=bot.TIStream"
}

# Arrays for stream data (Bash 3.x compatible)
STREAM_NAMES=()
STREAM_URLS=()
STREAM_TYPES=()
STREAM_ROLES=()

echo "📋 Parsing configuration..."

# Parse YAML to extract stream information
current_stream=""
current_url=""
current_type=""
current_role=""
current_enabled="true"

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    
    if [[ "$line" =~ ^[[:space:]]*([a-z_]+):[[:space:]]*$ ]]; then
        # Save previous stream if enabled
        if [[ -n "$current_stream" && "$current_enabled" == "true" && -n "$current_url" ]]; then
            STREAM_NAMES+=("$current_stream")
            STREAM_URLS+=($(add_ad_skip_params "$current_url"))
            STREAM_TYPES+=("$current_type")
            STREAM_ROLES+=("$current_role")
        fi
        
        current_stream="${BASH_REMATCH[1]}"
        current_url=""
        current_type=""
        current_role=""
        current_enabled="true"
        
    elif [[ "$line" =~ ^[[:space:]]*url:[[:space:]]*\"(.+)\"[[:space:]]*$ ]]; then
        current_url="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*type:[[:space:]]*\"(.+)\"[[:space:]]*$ ]]; then
        current_type="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*role:[[:space:]]*\"(.+)\"[[:space:]]*$ ]]; then
        current_role="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*enabled:[[:space:]]*false[[:space:]]*$ ]]; then
        current_enabled="false"
    fi
done < "$CONFIG_FILE"

# Don't forget the last stream
if [[ -n "$current_stream" && "$current_enabled" == "true" && -n "$current_url" ]]; then
    STREAM_NAMES+=("$current_stream")
    STREAM_URLS+=($(add_ad_skip_params "$current_url"))
    STREAM_TYPES+=("$current_type")
    STREAM_ROLES+=("$current_role")
fi

echo "Found streams:"
for i in "${!STREAM_NAMES[@]}"; do
    stream_name="${STREAM_NAMES[$i]}"
    stream_type="${STREAM_TYPES[$i]}"
    stream_role="${STREAM_ROLES[$i]}"
    echo "  📡 $stream_name ($stream_type/$stream_role)"
done
echo ""

# Helper function to find stream index by name
find_stream_index() {
    local stream_name="$1"
    for i in "${!STREAM_NAMES[@]}"; do
        if [[ "${STREAM_NAMES[$i]}" == "$stream_name" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

# Function to download a stream
download_stream() {
    local stream_name="$1"
    local delay_seconds="$2"
    local start_msg="$3"
    
    local stream_index
    if ! stream_index=$(find_stream_index "$stream_name"); then
        echo "  ❌ Stream '$stream_name' not found"
        return 1
    fi
    
    local url="${STREAM_URLS[$stream_index]}"
    local type="${STREAM_TYPES[$stream_index]}"
    local output_file="$OUTPUT_DIR/msnbc_${stream_name}_${DURATION}s.wav"
    local log_file="$OUTPUT_DIR/${stream_name}_download.log"
    
    echo "$start_msg"
    
    if [[ $delay_seconds -gt 0 ]]; then
        echo "  ⏳ Waiting ${delay_seconds}s to simulate CDN delay..."
        sleep $delay_seconds
    fi
    
    echo "  🎬 Starting download: $stream_name"
    
    if [[ "$type" == "hls" ]]; then
        ffmpeg -f hls \
            -live_start_index -1 \
            -allowed_extensions ALL \
            -i "$url" \
            -t $DURATION \
            -ar $SAMPLE_RATE \
            -ac $CHANNELS \
            -acodec pcm_s16le \
            -avoid_negative_ts make_zero \
            -y \
            "$output_file" \
            </dev/null \
            >"$log_file" 2>&1
    else
        ffmpeg -reconnect 1 \
            -reconnect_at_eof 1 \
            -reconnect_streamed 1 \
            -reconnect_delay_max 10 \
            -i "$url" \
            -t $DURATION \
            -ar $SAMPLE_RATE \
            -ac $CHANNELS \
            -acodec pcm_s16le \
            -avoid_negative_ts make_zero \
            -y \
            "$output_file" \
            </dev/null \
            >"$log_file" 2>&1
    fi
    
    if [[ $? -eq 0 ]]; then
        echo "  ✅ $stream_name completed successfully"
    else
        echo "  ❌ $stream_name failed (check $log_file)"
    fi
}

# Record the exact start time for reference
START_TIME=$(date '+%H:%M:%S')
START_TIMESTAMP=$(date '+%s')

echo "🚀 STARTING PHASED DOWNLOADS"
echo "============================"
echo "Start time: $START_TIME"
echo ""

# Phase 1: Download source streams immediately and simultaneously
echo "📡 PHASE 1: SOURCE STREAMS (IMMEDIATE)"
echo "======================================"

# Start both source streams in parallel
if find_stream_index "hls_source" >/dev/null 2>&1; then
    download_stream "hls_source" 0 "🎯 Starting HLS source (no delay)" &
    HLS_SOURCE_PID=$!
fi

if find_stream_index "icecast_source" >/dev/null 2>&1; then
    download_stream "icecast_source" 0 "🎯 Starting Icecast source (no delay)" &
    ICECAST_SOURCE_PID=$!
fi

# Wait for source downloads to complete
if [[ -n "${HLS_SOURCE_PID:-}" ]]; then
    wait $HLS_SOURCE_PID
fi
if [[ -n "${ICECAST_SOURCE_PID:-}" ]]; then
    wait $ICECAST_SOURCE_PID
fi

echo ""
echo "📡 PHASE 2: CDN STREAMS (WITH DELAYS)"
echo "====================================="

# Phase 2: Download CDN streams with appropriate delays
# Start both CDN downloads in parallel, each with their own delay
if find_stream_index "icecast_cdn" >/dev/null 2>&1; then
    download_stream "icecast_cdn" $ICECAST_CDN_DELAY "🎯 Starting Icecast CDN (${ICECAST_CDN_DELAY}s delay)" &
    ICECAST_CDN_PID=$!
fi

if find_stream_index "hls_cdn" >/dev/null 2>&1; then
    download_stream "hls_cdn" $HLS_CDN_DELAY "🎯 Starting HLS CDN (${HLS_CDN_DELAY}s delay)" &
    HLS_CDN_PID=$!
fi

# Wait for CDN downloads to complete
if [[ -n "${ICECAST_CDN_PID:-}" ]]; then
    wait $ICECAST_CDN_PID
fi
if [[ -n "${HLS_CDN_PID:-}" ]]; then
    wait $HLS_CDN_PID
fi

echo ""
echo "📊 DOWNLOAD SUMMARY"
echo "==================="

END_TIME=$(date '+%H:%M:%S')
TOTAL_DURATION=$(($(date '+%s') - START_TIMESTAMP))

echo "Start time: $START_TIME"
echo "End time: $END_TIME"
echo "Total duration: ${TOTAL_DURATION}s"
echo ""

# Check results
SUCCESS_COUNT=0
TOTAL_COUNT=${#STREAM_NAMES[@]}

for i in "${!STREAM_NAMES[@]}"; do
    stream_name="${STREAM_NAMES[$i]}"
    output_file="$OUTPUT_DIR/msnbc_${stream_name}_${DURATION}s.wav"
    
    if [[ -f "$output_file" ]]; then
        size=$(ls -lh "$output_file" | awk '{print $5}')
        duration_info=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_file" 2>/dev/null || echo "unknown")
        echo "✅ $stream_name: $output_file ($size, ${duration_info}s)"
        ((SUCCESS_COUNT++))
    else
        echo "❌ $stream_name: Failed"
    fi
done

echo ""
echo "Success rate: $SUCCESS_COUNT/$TOTAL_COUNT"

if [[ $SUCCESS_COUNT -eq $TOTAL_COUNT ]]; then
    echo ""
    echo "🎯 ALL DOWNLOADS SUCCESSFUL!"
    echo ""
    echo "📈 TIMING SIMULATION:"
    echo "  • Source streams captured live content immediately"
    echo "  • Icecast CDN captured content from ${ICECAST_CDN_DELAY}s ago"
    echo "  • HLS CDN captured content from ${HLS_CDN_DELAY}s ago"
    echo "  • This simulates real user experience with CDN delays"
    echo ""
    echo "🧪 Now you can compare content that should align temporally!"
else
    echo ""
    echo "⚠️  Some downloads failed. Check log files in $OUTPUT_DIR/"
fi

echo ""
echo "📝 Log files: $OUTPUT_DIR/"
echo "🎵 Audio files: $OUTPUT_DIR/"
