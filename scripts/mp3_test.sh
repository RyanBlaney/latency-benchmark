#!/bin/bash
echo "Starting MP3 test..."
# ffmpeg -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1 -reconnect_delay_max 2 -fflags +genpts+igndts -rw_timeout 30000000 -timeout 60000000 -i "http://cdn.tunein.com/v1/broadcast/newsfree/foxfree/stream.mp3" -t 90 -c copy fox_ti_mp3_test.mp3 -y
ffmpeg -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1 -i "http://cdn.tunein.com/v1/broadcast/newsfree/foxfree/stream.mp3" -t 120 -c copy fox_mp3_minimal.mp3 -y
# ffmpeg -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1 -i "http://stream1.skyviewnetworks.com:8010/FOX" -t 90 -c copy fox_mp3_minimal.mp3 -y
echo "MP3 test completed at $(date)"
