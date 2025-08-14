#!/bin/bash
echo "Starting HLS test..."
ffmpeg -live_start_index -1 -i "https://188a2110c0c0aa3d.mediapackage.us-east-2.amazonaws.com/out/v1/67fcd733de1c4728ac189e5dec958a18/master_1_0.m3u8" -t 120 -c copy fox_source_mp3_compare.ts -y
echo "HLS test completed at $(date)"
