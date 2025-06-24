package hls

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// DetectFromURL matches the URL with common HLS patterns
func DetectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URl: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)

	if strings.HasSuffix(path, ".m3u8") ||
		strings.Contains(path, "/playlist.m3u8") ||
		strings.Contains(path, "/index.m3u8") ||
		strings.Contains(u.RawQuery, "m3u8") {
		return common.StreamTypeHLS
	}
	return common.StreamTypeUnsupported
}

// DetectFromHeaders matches the HTTP headers with common HLS patterns
func DetectFromHeaders(ctx context.Context, client *http.Client, streamURL string) common.StreamType {
	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		fmt.Printf("{DEBUG}: Error creating request to HTTP headers: %v", err)
		return common.StreamTypeUnsupported
	}

	// Set user agent to avoid blocking
	req.Header.Set("User-Agent", "TuneIN-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "*/*")

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("{DEBUG}: Error getting response from client: %v", err)
		return common.StreamTypeUnsupported
	}
	defer resp.Body.Close()

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))

	if strings.Contains(contentType, "application/vnd.apple.mpegurl") ||
		strings.Contains(contentType, "application/x-mpegurl") ||
		strings.Contains(contentType, "vnd.apple.mpegurl") {
		return common.StreamTypeHLS
	}
	return common.StreamTypeUnsupported
}

// DetectFromM3U8Content attempts to validate and parse M3U8 content
func DetectFromM3U8Content(ctx context.Context, client *http.Client, url string) (*M3U8Playlist, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch playlist: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Store response headers
	headers := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			headers[strings.ToLower(key)] = values[0]
		}
	}

	// Parse the M3U8 content
	playlist, err := parseM3U8Content(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse M3U8: %w", err)
	}

	playlist.Headers = headers
	playlist.Metadata = extractMetadataFromPlaylist(playlist, url)

	return playlist, nil
}

// parseM3U8Content parses M3U8 playlist content from an io.Reader
func parseM3U8Content(reader io.Reader) (*M3U8Playlist, error) {
	playlist := &M3U8Playlist{
		Segments: make([]M3U8Segment, 0),
		Variants: make([]M3U8Variant, 0),
	}

	scanner := bufio.NewScanner(reader)
	var currentSegment *M3U8Segment
	var currentVariant *M3U8Variant

	if !scanner.Scan() {
		return nil, fmt.Errorf("empty playlist")
	}

	// First line must be #EXTM3U
	firstLine := strings.TrimSpace(scanner.Text())
	if firstLine != "#EXTM3U" {
		return nil, fmt.Errorf("invalid M3U8 format: missing #EXTM3U header")
	}

	playlist.IsValid = true

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments (except M3U8 tags)
		if line == "" || (strings.HasPrefix(line, "#") && !strings.HasPrefix(line, "#EXT")) {
			continue
		}

		// Parse M3U8 tags
		if strings.HasPrefix(line, "#EXT") {
			if err := parseM3U8Tag(line, playlist, &currentSegment, &currentVariant); err != nil {
				return nil, fmt.Errorf("failed to parse tag %s: %w", line, err)
			}
		} else {
			// This is a URI line
			if currentSegment != nil {
				// Complete the current segment
				currentSegment.URI = line
				playlist.Segments = append(playlist.Segments, *currentSegment)
				currentSegment = nil
			} else if currentVariant != nil {
				// Complete the current variant
				currentVariant.URI = line
				playlist.Variants = append(playlist.Variants, *currentVariant)
				currentVariant = nil
				playlist.IsMaster = true
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading playlist: %w", err)
	}

	// Determine if this is a live stream
	playlist.IsLive = !hasEndListTag(playlist)

	return playlist, nil
}

// parseM3U8Tag parses individual M3U8 tags
func parseM3U8Tag(line string, playlist *M3U8Playlist, currentSegment **M3U8Segment, currentVariant **M3U8Variant) error {
	parts := strings.SplitN(line, ":", 2)
	tag := parts[0]
	value := ""
	if len(parts) > 1 {
		value = parts[1]
	}

	switch tag {
	case "#EXT-X-VERSION":
		if v, err := strconv.Atoi(value); err == nil {
			playlist.Version = v
		}

	case "#EXT-X-TARGETDURATION":
		if v, err := strconv.Atoi(value); err == nil {
			playlist.TargetDuration = v
		}

	case "#EXT-X-MEDIA-SEQUENCE":
		if v, err := strconv.Atoi(value); err == nil {
			playlist.MediaSequence = v
		}

	case "#EXT-X-DISCONTINUITY-SEQUENCE":
		// TuneIn specific - track discontinuity sequence
		if playlist.Headers == nil {
			playlist.Headers = make(map[string]string)
		}
		playlist.Headers["discontinuity_sequence"] = value

	case "#EXT-X-START":
		// Parse TIME-OFFSET
		if strings.Contains(value, "TIME-OFFSET=") {
			offsetStr := strings.TrimPrefix(value, "TIME-OFFSET=")
			if playlist.Headers == nil {
				playlist.Headers = make(map[string]string)
			}
			playlist.Headers["start_time_offset"] = offsetStr
		}

	case "#EXT-X-COM-TUNEIN-AVAIL-DUR":
		// TuneIn specific - available duration
		if playlist.Headers == nil {
			playlist.Headers = make(map[string]string)
		}
		playlist.Headers["tunein_available_duration"] = value

	case "#EXT-X-PROGRAM-DATE-TIME":
		// Store program date time with current segment
		if *currentSegment != nil {
			if (*currentSegment).Title == "" {
				(*currentSegment).Title = "PDT:" + value
			} else {
				(*currentSegment).Title += ",PDT:" + value
			}
		}

	case "#EXTINF":
		// Start a new segment
		*currentSegment = &M3U8Segment{}

		// Parse duration and title - handle content_cat attribute
		parts := strings.SplitN(value, ",", 2)
		if len(parts) > 0 {
			if duration, err := strconv.ParseFloat(parts[0], 64); err == nil {
				(*currentSegment).Duration = duration
			}
		}
		if len(parts) > 1 {
			title := parts[1]
			(*currentSegment).Title = title

			// Extract content category for metadata
			if strings.Contains(title, "content_cat=") {
				// Parse content_cat="news" format
				if start := strings.Index(title, `content_cat="`); start != -1 {
					start += len(`content_cat="`)
					if end := strings.Index(title[start:], `"`); end != -1 {
						category := title[start : start+end]
						// Store category in a structured way (will be used in metadata)
						if (*currentSegment).Title == title {
							(*currentSegment).Title = title + ",CATEGORY:" + category
						}
					}
				}
			}
		}

	case "#EXT-X-BYTERANGE":
		if *currentSegment != nil {
			(*currentSegment).ByteRange = value
		}

	case "#EXT-X-CUE-OUT":
		// Ad break start - mark in current segment
		if *currentSegment != nil {
			if (*currentSegment).Title == "" {
				(*currentSegment).Title = "AD_BREAK_START"
			} else {
				(*currentSegment).Title += ",AD_BREAK_START"
			}
		}

	case "#EXT-X-CUE-IN":
		// Ad break end - mark in current segment
		if *currentSegment != nil {
			if (*currentSegment).Title == "" {
				(*currentSegment).Title = "AD_BREAK_END"
			} else {
				(*currentSegment).Title += ",AD_BREAK_END"
			}
		}

	case "#EXT-X-DISCONTINUITY":
		// Content discontinuity - mark in current segment or create marker
		if *currentSegment != nil {
			if (*currentSegment).Title == "" {
				(*currentSegment).Title = "DISCONTINUITY"
			} else {
				(*currentSegment).Title += ",DISCONTINUITY"
			}
		}

	case "#EXT-X-STREAM-INF":
		// Start a new variant stream
		*currentVariant = &M3U8Variant{}

		// Parse stream info attributes
		attrs := parseAttributes(value)
		if bandwidth, exists := attrs["BANDWIDTH"]; exists {
			if b, err := strconv.Atoi(bandwidth); err == nil {
				(*currentVariant).Bandwidth = b
			}
		}
		if codecs, exists := attrs["CODECS"]; exists {
			(*currentVariant).Codecs = strings.Trim(codecs, "\"")
		}
		if resolution, exists := attrs["RESOLUTION"]; exists {
			(*currentVariant).Resolution = resolution
		}
		if frameRate, exists := attrs["FRAME-RATE"]; exists {
			if f, err := strconv.ParseFloat(frameRate, 64); err == nil {
				(*currentVariant).FrameRate = f
			}
		}

	case "#EXT-X-ENDLIST":
		// This indicates the end of the playlist (not live)
		playlist.IsLive = false

	default:
		// Store unknown/custom tags in headers for debugging
		if playlist.Headers == nil {
			playlist.Headers = make(map[string]string)
		}
		if strings.HasPrefix(tag, "#EXT-X-") {
			cleanTag := strings.TrimPrefix(tag, "#EXT-X-")
			playlist.Headers["custom_"+strings.ToLower(cleanTag)] = value
		}
	}

	return nil
}

// parseAttributes parses M3U8 attribute strings like 'BANDWIDTH=1280000,CODECS="avc1.42e00a,mp4a.40.2"'
func parseAttributes(attrString string) map[string]string {
	attrs := make(map[string]string)

	// Split by comma, but be careful with quoted values
	var parts []string
	var current strings.Builder
	inQuotes := false

	for _, char := range attrString {
		switch char {
		case '"':
			inQuotes = !inQuotes
			current.WriteRune(char)
		case ',':
			if inQuotes {
				current.WriteRune(char)
			} else {
				parts = append(parts, current.String())
				current.Reset()
			}
		default:
			current.WriteRune(char)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	// Parse key=value pairs
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) == 2 {
			attrs[kv[0]] = kv[1]
		}
	}

	return attrs
}

// hasEndListTag checks if the playlist has an EXT-X-ENDLIST tag
func hasEndListTag(playlist *M3U8Playlist) bool {
	// TODO: hasEndListTag
	// This would be set during parsing - simplified for now
	// In a real implementation, you'd track this during parsing
	return !playlist.IsLive // Inverse relationship for now
}

// extractMetadataFromPlaylist extracts stream metadata from the parsed playlist
func extractMetadataFromPlaylist(playlist *M3U8Playlist, url string) *common.StreamMetadata {
	metadata := &common.StreamMetadata{
		URL:       url,
		Type:      common.StreamTypeHLS,
		Headers:   make(map[string]string),
		Timestamp: time.Now(),
	}

	// Copy playlist headers to metadata
	maps.Copy(metadata.Headers, playlist.Headers)

	// Extract bitrate from URL pattern (TuneIn specific)
	// URLs like: .../aac_adts/96k/... or .../96/media.m3u8
	if strings.Contains(url, "/96k/") || strings.Contains(url, "/96/") {
		metadata.Bitrate = 96
	} else if strings.Contains(url, "/128k/") || strings.Contains(url, "/128/") {
		metadata.Bitrate = 128
	} else if strings.Contains(url, "/192k/") || strings.Contains(url, "/192/") {
		metadata.Bitrate = 192
	} else if strings.Contains(url, "/256k/") || strings.Contains(url, "/256/") {
		metadata.Bitrate = 256
	}

	// Extract codec from URL pattern
	if strings.Contains(url, "aac_adts") {
		metadata.Codec = "aac"
		metadata.Format = "adts"
	} else if strings.Contains(url, "/aac/") {
		metadata.Codec = "aac"
	} else if strings.Contains(url, "/mp3/") {
		metadata.Codec = "mp3"
	}

	// Extract additional info from segments
	if len(playlist.Segments) > 0 {
		// Analyze content categories and ad breaks
		categories := make(map[string]int)
		hasAdBreaks := false

		for i, segment := range playlist.Segments {
			if i > 10 { // Check first 10 segments for patterns
				break
			}

			// Extract categories
			if strings.Contains(segment.Title, "CATEGORY:") {
				if start := strings.Index(segment.Title, "CATEGORY:"); start != -1 {
					start += len("CATEGORY:")
					if end := strings.Index(segment.Title[start:], ","); end != -1 {
						category := segment.Title[start : start+end]
						categories[category]++
					} else {
						category := segment.Title[start:]
						categories[category]++
					}
				}
			}

			// Check for ad breaks
			if strings.Contains(segment.Title, "AD_BREAK_START") || strings.Contains(segment.Title, "AD_BREAK_END") {
				hasAdBreaks = true
			}
		}

		// Set primary category (most common)
		var primaryCategory string
		maxCount := 0
		for category, count := range categories {
			if count > maxCount {
				maxCount = count
				primaryCategory = category
			}
		}

		if primaryCategory != "" {
			metadata.Genre = primaryCategory
			// Store all categories in headers
			categoryList := make([]string, 0, len(categories))
			for category := range categories {
				categoryList = append(categoryList, category)
			}
			metadata.Headers["content_categories"] = strings.Join(categoryList, ",")
		}

		if hasAdBreaks {
			metadata.Headers["has_ad_breaks"] = "true"
		}

		// Extract station info from URL pattern
		if strings.Contains(url, "/bloomberg/") {
			metadata.Station = "Bloomberg"
		} else if strings.Contains(url, "/news/") && metadata.Station == "" {
			metadata.Station = "News"
		}
	}

	// Extract codec information from variants (if master playlist)
	if len(playlist.Variants) > 0 {
		// Use the highest bandwidth variant for metadata
		var bestVariant *M3U8Variant
		for i := range playlist.Variants {
			variant := &playlist.Variants[i]
			if bestVariant == nil || variant.Bandwidth > bestVariant.Bandwidth {
				bestVariant = variant
			}
		}

		if bestVariant != nil {
			metadata.Bitrate = bestVariant.Bandwidth / 1000 // Convert to kbps

			// Parse codecs
			if bestVariant.Codecs != "" {
				codecs := strings.Split(bestVariant.Codecs, ",")
				for _, codec := range codecs {
					codec = strings.TrimSpace(codec)
					if strings.HasPrefix(codec, "mp4a") {
						metadata.Codec = "aac"
					} else if strings.HasPrefix(codec, "avc1") {
						// Video codec - not relevant for audio analysis
					}
				}
			}
		}
	}

	// Set defaults if not determined
	if metadata.Codec == "" {
		metadata.Codec = "aac" // Common default for HLS
	}
	if metadata.Channels == 0 {
		metadata.Channels = 2 // Stereo default
	}
	if metadata.SampleRate == 0 {
		metadata.SampleRate = 44100 // Common default
	}

	return metadata
}

// IsValidHLSContent checks if the content appears to be valid HLS
func IsValidHLSContent(ctx context.Context, client *http.Client, url string) bool {
	// Set a short timeout for detection
	detectCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	playlist, err := DetectFromM3U8Content(detectCtx, client, url)
	if err != nil {
		return false
	}

	return playlist.IsValid && (len(playlist.Segments) > 0 || len(playlist.Variants) > 0)
}
