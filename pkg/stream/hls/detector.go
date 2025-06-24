package hls

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// DetectFromURL matches the URL with common HLS patterns
func DetectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URL: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)

	if strings.HasSuffix(path, ".m3u8") ||
		strings.Contains(path, "/playlist.m3u8") ||
		strings.Contains(path, "/index.m3u8") ||
		strings.Contains(path, "/master.m3u8") ||
		strings.Contains(u.RawQuery, "m3u8") {
		return common.StreamTypeHLS
	}
	return common.StreamTypeUnsupported
}

// DetectFromHeaders matches the HTTP headers with common HLS patterns
func DetectFromHeaders(ctx context.Context, client *http.Client, streamURL string) common.StreamType {
	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		fmt.Printf("{DEBUG}: Error creating request for HTTP headers: %v", err)
		return common.StreamTypeUnsupported
	}

	// Set user agent to avoid blocking
	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
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
func DetectFromM3U8Content(ctx context.Context, client *http.Client, streamURL string) (*M3U8Playlist, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", streamURL, nil)
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

	// Parse the M3U8 content using the new parser
	parser := NewParser()
	playlist, err := parser.ParseM3U8Content(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse M3U8: %w", err)
	}

	// Merge HTTP headers with playlist headers
	if playlist.Headers == nil {
		playlist.Headers = make(map[string]string)
	}
	maps.Copy(playlist.Headers, headers)

	// Extract metadata using the new metadata extractor
	metadataExtractor := NewMetadataExtractor()
	playlist.Metadata = metadataExtractor.ExtractMetadata(playlist, streamURL)

	return playlist, nil
}

// IsValidHLSContent checks if the content appears to be valid HLS
func IsValidHLSContent(ctx context.Context, client *http.Client, streamURL string) bool {
	// Set a short timeout for detection
	detectCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	playlist, err := DetectFromM3U8Content(detectCtx, client, streamURL)
	if err != nil {
		return false
	}

	return playlist.IsValid && (len(playlist.Segments) > 0 || len(playlist.Variants) > 0)
}

