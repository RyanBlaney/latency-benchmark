#!/bin/bash

# CDN Latency Simulation Download Script
# Downloads all streams for a single broadcast simultaneously for temporal alignment
# Uses GNU parallel for true simultaneous execution

set -euo pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/test-audio"
CONFIG_FILE="$PROJECT_ROOT/configs/examples/test-broadcasts.yaml"

# Default parameters
BROADCAST=""
DURATION=30
SAMPLE_RATE=44100
CHANNELS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--broadcast)
            BROADCAST="$2"
            shift 2
            ;;
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
        -h|--help)
            echo "Usage: $0 -b BROADCAST [OPTIONS]"
            echo "Required:"
            echo "  -b, --broadcast NAME      Broadcast name (e.g., msnbc_news, cnn, fox)"
            echo "Options:"
            echo "  -d, --duration SECONDS    Recording duration (default: 30)"
            echo "  -o, --output DIR          Output directory"
            echo "  -c, --config FILE         Config file path"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$BROADCAST" ]]; then
    echo "❌ Error: Broadcast name is required. Use -b or --broadcast to specify it."
    echo "   Example: $0 -b msnbc_news -d 60"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🎵 CDN LATENCY SIMULATION DOWNLOAD"
echo "=================================="
echo "Broadcast: $BROADCAST"
echo "Duration: ${DURATION}s"
echo "TRUE SIMULTANEOUS START: ENABLED"
echo ""

# Function to add ad-skipping parameters to URLs (for SoundStack/AdZwizz)
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

# Arrays for stream data
STREAM_NAMES=()
STREAM_URLS=()
STREAM_TYPES=()
STREAM_ROLES=()

echo "📋 Parsing configuration for broadcast: $BROADCAST"

# Parse YAML - same parsing logic as before
in_target_broadcast=false
in_streams_section=false
current_stream=""
current_url=""
current_type="icecast"
current_role=""
current_enabled="true"

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    
    if [[ "$line" =~ ^([[:space:]]*) ]]; then
        current_indent=${#BASH_REMATCH[1]}
    else
        current_indent=0
    fi
    
    if [[ "$line" =~ ^[[:space:]]*${BROADCAST}:[[:space:]]*$ ]]; then
        in_target_broadcast=true
        in_streams_section=false
        broadcast_indent=$current_indent
        continue
    fi
    
    if [[ "$in_target_broadcast" == true ]]; then
        if [[ $current_indent -le $broadcast_indent && "$line" =~ ^[[:space:]]*[a-z_]+:[[:space:]]*$ ]] && [[ "$line" != *"${BROADCAST}:"* ]] && [[ "$line" != *"streams:"* ]]; then
            in_target_broadcast=false
            in_streams_section=false
            continue
        fi
        
        if [[ "$line" =~ ^[[:space:]]*streams:[[:space:]]*$ ]]; then
            in_streams_section=true
            streams_indent=$current_indent
            continue
        fi
        
        if [[ "$in_streams_section" == true ]]; then
            if [[ $current_indent -le $broadcast_indent && "$line" =~ ^[[:space:]]*[a-z_]+:[[:space:]]*$ ]] && [[ "$line" != *"streams:"* ]]; then
                in_streams_section=false
                in_target_broadcast=false
                continue
            fi
            
            if [[ $current_indent -gt $streams_indent && "$line" =~ ^[[:space:]]*([a-z_]+):[[:space:]]*$ ]]; then
                if [[ -n "$current_stream" && "$current_enabled" == "true" && -n "$current_url" ]]; then
                    STREAM_NAMES+=("$current_stream")
                    STREAM_URLS+=($(add_ad_skip_params "$current_url"))
                    STREAM_TYPES+=("$current_type")
                    STREAM_ROLES+=("$current_role")
                fi
                
                current_stream="${BASH_REMATCH[1]}"
                current_url=""
                current_type="icecast"
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
        fi
    fi
done < "$CONFIG_FILE"

if [[ "$in_streams_section" == true && -n "$current_stream" && "$current_enabled" == "true" && -n "$current_url" ]]; then
    STREAM_NAMES+=("$current_stream")
    STREAM_URLS+=($(add_ad_skip_params "$current_url"))
    STREAM_TYPES+=("$current_type")
    STREAM_ROLES+=("$current_role")
fi

if [[ ${#STREAM_NAMES[@]} -eq 0 ]]; then
    echo "❌ Error: No enabled streams found for broadcast '$BROADCAST'"
    exit 1
fi

echo "Found ${#STREAM_NAMES[@]} enabled streams for $BROADCAST:"
for i in "${!STREAM_NAMES[@]}"; do
    echo "  📡 ${STREAM_NAMES[$i]} (${STREAM_TYPES[$i]}/${STREAM_ROLES[$i]})"
done
echo ""

# Create a job file for GNU parallel
JOBS_FILE="$OUTPUT_DIR/.parallel_jobs"
rm -f "$JOBS_FILE"

echo "🚀 PREPARING SIMULTANEOUS DOWNLOADS"
echo "==================================="

for i in "${!STREAM_NAMES[@]}"; do
    stream_name="${STREAM_NAMES[$i]}"
    url="${STREAM_URLS[$i]}"
    type="${STREAM_TYPES[$i]}"
    output_file="$OUTPUT_DIR/${BROADCAST}_${stream_name}_${DURATION}s.wav"
    log_file="$OUTPUT_DIR/${stream_name}_download.log"
    
    if [[ "$type" == "hls" ]]; then
        ffmpeg_cmd="ffmpeg -f hls -live_start_index -1 -allowed_extensions ALL -i '$url' -t $DURATION -ar $SAMPLE_RATE -ac $CHANNELS -acodec pcm_s16le -avoid_negative_ts make_zero -y '$output_file' 2>'$log_file'"
    else
        ffmpeg_cmd="ffmpeg -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1 -reconnect_delay_max 10 -i '$url' -t $DURATION -ar $SAMPLE_RATE -ac $CHANNELS -acodec pcm_s16le -avoid_negative_ts make_zero -y '$output_file' 2>'$log_file'"
    fi
    
    echo "$ffmpeg_cmd" >> "$JOBS_FILE"
    echo "  📡 Queued: $stream_name"
done

echo ""
echo "🎯 STARTING ALL ${#STREAM_NAMES[@]} STREAMS SIMULTANEOUSLY"
echo "========================================"

START_TIME=$(date '+%H:%M:%S')

# Check if GNU parallel is available
if command -v parallel >/dev/null 2>&1; then
    echo "Using GNU parallel for perfect synchronization..."
    parallel -j0 --bar < "$JOBS_FILE"
    PARALLEL_EXIT=$?
else
    echo "GNU parallel not found, using shell background jobs with sleep synchronization..."
    
    # Alternative: use shell jobs with a very short delay to minimize timing differences
    PIDS=()
    
    # Start all jobs with minimal delay
    while IFS= read -r cmd; do
        eval "$cmd" &
        PIDS+=($!)
        sleep 0.01  # Minimal delay to avoid overwhelming the system
    done < "$JOBS_FILE"
    
    # Wait for all to complete
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    PARALLEL_EXIT=0
fi

END_TIME=$(date '+%H:%M:%S')

# Clean up
rm -f "$JOBS_FILE"

echo ""
echo "📊 DOWNLOAD SUMMARY"
echo "==================="
echo "Start time: $START_TIME"
echo "End time: $END_TIME"
echo ""

if [[ $PARALLEL_EXIT -eq 0 ]]; then
    echo "📁 FILES CREATED:"
    echo "=================="
    
    SUCCESS_COUNT=0
    for i in "${!STREAM_NAMES[@]}"; do
        stream_name="${STREAM_NAMES[$i]}"
        output_file="$OUTPUT_DIR/${BROADCAST}_${stream_name}_${DURATION}s.wav"
        
        if [[ -f "$output_file" ]]; then
            size=$(ls -lh "$output_file" | awk '{print $5}')
            duration_info=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_file" 2>/dev/null | cut -d. -f1 || echo "unknown")
            echo "✅ $stream_name: $output_file ($size, ${duration_info}s)"
            ((SUCCESS_COUNT++))
        else
            echo "❌ $stream_name: Failed"
        fi
    done
    
    echo ""
    echo "Success rate: $SUCCESS_COUNT/${#STREAM_NAMES[@]} streams"
    
    if [[ $SUCCESS_COUNT -eq ${#STREAM_NAMES[@]} ]]; then
        echo ""
        echo "🎯 ALL DOWNLOADS SUCCESSFUL!"
        echo ""
        echo "🕰️ SIMULTANEOUS DOWNLOAD ACHIEVED:"
        echo "  • All ${#STREAM_NAMES[@]} streams started as close to simultaneously as possible"
        echo "  • Minimal timing variation between streams"
        echo "  • Ready for temporal alignment analysis"
    fi
else
    echo "❌ Some downloads may have failed. Check individual log files."
fi

echo ""
echo "📝 Individual logs: $OUTPUT_DIR/*_download.log"
echo "🎵 Audio files: $OUTPUT_DIR/"
