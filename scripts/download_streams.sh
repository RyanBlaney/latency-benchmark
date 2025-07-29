#!/bin/bash

# Simultaneous Multi-Stream Audio Download Script
# Downloads all streams from config simultaneously to capture same temporal window
# Compatible with Bash 3.x+ (macOS default)

set -euo pipefail

# Check Bash version and warn if old
if [[ ${BASH_VERSION%%.*} -lt 4 ]]; then
    echo "⚠️  Warning: Using Bash ${BASH_VERSION}. For best results, use Bash 4+ or install via 'brew install bash'"
    echo ""
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/test-audio"
CONFIG_FILE="$PROJECT_ROOT/configs/examples/test-broadcasts.yaml"

# Default parameters
DURATION=30
SAMPLE_RATE=44100
CHANNELS=2
SYNC_OFFSET=5  # Seconds to wait for all processes to start before beginning capture

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
        -s|--sync-offset)
            SYNC_OFFSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --duration SECONDS    Recording duration (default: 30)"
            echo "  -o, --output DIR          Output directory (default: PROJECT_ROOT/test-audio)"
            echo "  -c, --config FILE         Config file path (default: configs/examples/test-broadcasts.yaml)"
            echo "  -s, --sync-offset SECONDS Sync offset for simultaneous start (default: 5)"
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

echo "🎵 SIMULTANEOUS STREAM DOWNLOAD"
echo "==============================="
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Duration: ${DURATION}s"
echo "Sync offset: ${SYNC_OFFSET}s"
echo "Bash version: $BASH_VERSION"
echo ""

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Function to add ad-skipping parameters to URLs
add_ad_skip_params() {
    local url="$1"
    local separator
    
    # Determine if URL already has query parameters
    if [[ "$url" == *"?"* ]]; then
        separator="&"
    else
        separator="?"
    fi
    
    # Add the ad-skipping and bot identification parameters
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
    # Skip comments and empty lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    
    # Extract stream name, URL, type, and role
    if [[ "$line" =~ ^[[:space:]]*([a-z_]+):[[:space:]]*$ ]]; then
        # Save previous stream if it was enabled
        if [[ -n "$current_stream" && "$current_enabled" == "true" && -n "$current_url" ]]; then
            STREAM_NAMES+=("$current_stream")
            STREAM_URLS+=($(add_ad_skip_params "$current_url"))
            STREAM_TYPES+=("$current_type")
            STREAM_ROLES+=("$current_role")
        fi
        
        # Start new stream
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

# Validate we found streams
if [[ ${#STREAM_NAMES[@]} -eq 0 ]]; then
    echo "❌ No enabled streams found in config file"
    exit 1
fi

echo "🚫 AD-SKIPPING ENABLED"
echo "======================"
echo "Added parameters to all URLs:"
echo "  • aw_0_1st.premium=true (skip preroll ads)"
echo "  • partnerID=BotTIStream (mark as bot traffic)"  
echo "  • playerid=BotTIStream (bot player identification)"
echo "  • aw_0_1st.ads_partner_alias=bot.TIStream (adOps bot marker)"
echo ""

echo "Found ${#STREAM_NAMES[@]} enabled streams:"
for i in "${!STREAM_NAMES[@]}"; do
    stream_name="${STREAM_NAMES[$i]}"
    url="${STREAM_URLS[$i]}"
    type="${STREAM_TYPES[$i]:-unknown}"
    role="${STREAM_ROLES[$i]:-unknown}"
    echo "  📡 $stream_name ($type/$role): ${url:0:80}..."
done
echo ""

# Prepare synchronized download using compatible date command
if date -v+1S >/dev/null 2>&1; then
    # macOS date command
    SYNC_TIME=$(date -v+${SYNC_OFFSET}S '+%s')
    SYNC_TIME_DISPLAY=$(date -v+${SYNC_OFFSET}S '+%H:%M:%S')
else
    # GNU date command
    SYNC_TIME=$(date -d "+${SYNC_OFFSET} seconds" '+%s')
    SYNC_TIME_DISPLAY=$(date -d "@$SYNC_TIME" '+%H:%M:%S')
fi

echo "⏰ SYNCHRONIZED DOWNLOAD SETUP"
echo "=============================="
echo "Current time: $(date '+%H:%M:%S')"
echo "Start time: $SYNC_TIME_DISPLAY"
echo "Sync offset: ${SYNC_OFFSET}s"
echo ""

PIDS=()
LOG_FILES=()
OUTPUT_FILES=()

echo "🚀 Starting synchronized downloads..."

# Start all downloads simultaneously
for i in "${!STREAM_NAMES[@]}"; do
    stream_name="${STREAM_NAMES[$i]}"
    url="${STREAM_URLS[$i]}"
    type="${STREAM_TYPES[$i]}"
    role="${STREAM_ROLES[$i]}"
    
    # Generate output filename
    output_file="$OUTPUT_DIR/msnbc_${stream_name}_${DURATION}s.wav"
    log_file="$OUTPUT_DIR/${stream_name}_download.log"
    
    OUTPUT_FILES+=("$output_file")
    LOG_FILES+=("$log_file")
    
    echo "  🎯 Scheduling: $stream_name"
    
    # Use background process with sleep (most compatible approach)
    (
        sleep_time=$((SYNC_TIME - $(date '+%s')))
        [[ $sleep_time -gt 0 ]] && sleep $sleep_time
        
        # Add stream-specific optimizations
        if [[ "$type" == "hls" ]]; then
            # HLS-specific optimizations
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
            # ICEcast/MP3 stream optimizations
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
    ) &
    PIDS+=($!)
done

echo ""
echo "⏳ Waiting for synchronized start at $SYNC_TIME_DISPLAY..."

# Wait for sync time
current_time=$(date '+%s')
wait_time=$((SYNC_TIME - current_time))
if [[ $wait_time -gt 0 ]]; then
    sleep $wait_time
fi

echo "🎬 Downloads started simultaneously!"
echo ""

if date -v+${DURATION}S >/dev/null 2>&1; then
    # macOS date
    echo "⏱️  Estimated completion: $(date -v+${DURATION}S '+%H:%M:%S')"
else
    # GNU date
    echo "⏱️  Estimated completion: $(date -d "+${DURATION} seconds" '+%H:%M:%S')"
fi

# Wait for background processes
echo "Waiting for ${#PIDS[@]} download processes..."

# Monitor progress
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    stream_name="${STREAM_NAMES[$i]}"
    
    if wait "$pid"; then
        echo "  ✅ $stream_name completed successfully"
    else
        echo "  ❌ $stream_name failed (check log: ${LOG_FILES[$i]})"
    fi
done

echo ""
echo "📊 DOWNLOAD SUMMARY"
echo "==================="

# Check results and generate summary
SUCCESS_COUNT=0
TOTAL_COUNT=${#OUTPUT_FILES[@]}

for i in "${!OUTPUT_FILES[@]}"; do
    output_file="${OUTPUT_FILES[$i]}"
    log_file="${LOG_FILES[$i]}"
    stream_name="${STREAM_NAMES[$i]}"
    
    if [[ -f "$output_file" ]]; then
        size=$(ls -lh "$output_file" | awk '{print $5}')
        duration_info=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_file" 2>/dev/null || echo "unknown")
        echo "✅ $stream_name: $output_file ($size, ${duration_info}s)"
        ((SUCCESS_COUNT++))
        
        # Check for potential ad contamination in logs
        if grep -q "adw_ad\|adswizzContext\|proxidigital" "$log_file" 2>/dev/null; then
            echo "    ⚠️  Potential ad content detected in download"
        fi
    else
        echo "❌ $stream_name: Failed (check $log_file)"
    fi
done

echo ""
echo "Success rate: $SUCCESS_COUNT/$TOTAL_COUNT"

if [[ $SUCCESS_COUNT -eq $TOTAL_COUNT ]]; then
    echo ""
    echo "🎯 ALL DOWNLOADS SUCCESSFUL!"
    echo ""
    echo "File details:"
    ls -la "$OUTPUT_DIR"/*.wav 2>/dev/null || echo "No .wav files found"
    
    echo ""
    echo "📈 Temporal Synchronization Benefits:"
    echo "  • All streams captured from same temporal window"
    echo "  • Ad-skipping parameters should prevent preroll ads"
    echo "  • Bot traffic markers prevent affecting real metrics"
    echo "  • Eliminates CDN propagation delay artifacts"
    echo "  • Accounts for HLS vs ICEcast timing differences"
    echo "  • Reduces alignment detection complexity"
    
    echo ""
    echo "🧪 Ready to test with synchronized audio!"
    echo ""
    echo "Example fingerprint test commands:"
    
    # Generate test commands for all stream pairs (Bash 3.x compatible)
    for i in "${!STREAM_NAMES[@]}"; do
        for j in "${!STREAM_NAMES[@]}"; do
            # Only process pairs where i < j to avoid duplicates
            if [[ $i -lt $j ]]; then
                stream1="${STREAM_NAMES[$i]}"
                stream2="${STREAM_NAMES[$j]}"
                file1="$OUTPUT_DIR/msnbc_${stream1}_${DURATION}s.wav"
                file2="$OUTPUT_DIR/msnbc_${stream2}_${DURATION}s.wav"
                
                if [[ -f "$file1" && -f "$file2" ]]; then
                    echo "  ./cdn-benchmark-cli fingerprint-test \\"
                    echo "    \"file://\$(pwd)/${file1#$PROJECT_ROOT/}\" \\"
                    echo "    \"file://\$(pwd)/${file2#$PROJECT_ROOT/}\" \\"
                    echo "    --content-type=news --max-offset=10 --verbose"
                    echo ""
                fi
            fi
        done
    done
    
else
    echo ""
    echo "⚠️  Some downloads failed. Check log files:"
    for i in "${!LOG_FILES[@]}"; do
        echo "  📄 ${LOG_FILES[$i]}"
    done
fi

echo ""
echo "📝 Log files location: $OUTPUT_DIR/"
echo "🎵 Audio files location: $OUTPUT_DIR/"
