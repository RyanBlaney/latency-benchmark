package hls

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

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
