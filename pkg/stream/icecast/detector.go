package icecast

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// DetectType determines if the stream is ICEcast using URL patterns and headers
func DetectType(ctx context.Context, client *http.Client, streamURL string) (common.StreamType, error) {
	// First try URL-based detection (fastest)
	if DetectFromURL(streamURL) == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	// Fall back to header-based detection
	if DetectFromHeaders(ctx, client, streamURL) == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	// Not ICEcast
	return common.StreamTypeUnsupported, nil
}

// ProbeStream performs a lightweight probe to gather ICEcast stream metadata
func ProbeStream(ctx context.Context, client *http.Client, streamURL string) (*common.StreamMetadata, error) {
	// Perform HEAD request to get metadata from headers
	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers for ICEcast compatibility
	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Icy-MetaData", "1") // Request ICY metadata

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to probe ICEcast stream: %w", err)
	}
	defer resp.Body.Close()

	// Create metadata from response headers
	metadata := &common.StreamMetadata{
		URL:       streamURL,
		Type:      common.StreamTypeICEcast,
		Headers:   make(map[string]string),
		Timestamp: time.Now(),
	}

	// Extract ICEcast-specific metadata from headers
	if name := resp.Header.Get("icy-name"); name != "" {
		metadata.Station = name
	}
	if genre := resp.Header.Get("icy-genre"); genre != "" {
		metadata.Genre = genre
	}
	if desc := resp.Header.Get("icy-description"); desc != "" {
		metadata.Title = desc
	}
	if url := resp.Header.Get("icy-url"); url != "" {
		metadata.Headers["icy-url"] = url
	}

	// Extract technical metadata
	if bitrate := resp.Header.Get("icy-br"); bitrate != "" {
		if br, err := strconv.Atoi(bitrate); err == nil {
			metadata.Bitrate = br
		}
		metadata.Headers["icy-br"] = bitrate
	}
	if sampleRate := resp.Header.Get("icy-sr"); sampleRate != "" {
		if sr, err := strconv.Atoi(sampleRate); err == nil {
			metadata.SampleRate = sr
		}
		metadata.Headers["icy-sr"] = sampleRate
	}

	// Determine format/codec from content type
	contentType := resp.Header.Get("Content-Type")
	metadata.ContentType = contentType

	switch {
	case strings.Contains(strings.ToLower(contentType), "mpeg"):
		metadata.Codec = "mp3"
		metadata.Format = "mp3"
	case strings.Contains(strings.ToLower(contentType), "aac"):
		metadata.Codec = "aac"
		metadata.Format = "aac"
	case strings.Contains(strings.ToLower(contentType), "ogg"):
		metadata.Codec = "ogg"
		metadata.Format = "ogg"
	default:
		metadata.Format = "unknown"
	}

	// Store all relevant ICEcast headers
	relevantHeaders := []string{
		"content-type", "server", "icy-name", "icy-genre", "icy-description",
		"icy-url", "icy-br", "icy-sr", "icy-pub", "icy-notice1", "icy-notice2",
		"icy-metaint", "icy-version",
	}

	for _, header := range relevantHeaders {
		if value := resp.Header.Get(header); value != "" {
			metadata.Headers[strings.ToLower(header)] = value
		}
	}

	return metadata, nil
}

// DetectFromURL matches URL for common ICEcast patterns
func DetectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URl: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)

	if strings.HasSuffix(path, ".mp3") ||
		strings.HasSuffix(path, ".aac") ||
		strings.HasSuffix(path, ".ogg") ||
		strings.HasSuffix(path, "/stream") ||
		strings.Contains(path, "/listen") ||
		u.Port() == "8000" || // Common ICEcast port
		u.Port() == "8080" {
		return common.StreamTypeICEcast
	}
	return common.StreamTypeUnsupported
}

// DetectFromHeader matches HTTP headers for common ICEcast patterns
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
	server := strings.ToLower(resp.Header.Get("Server"))

	if strings.Contains(contentType, "audio/mpeg") ||
		strings.Contains(contentType, "audio/aac") ||
		strings.Contains(contentType, "audio/ogg") ||
		strings.Contains(contentType, "application/ogg") ||
		strings.Contains(server, "icecast") ||
		resp.Header.Get("icy-name") != "" ||
		resp.Header.Get("icy-description") != "" {
		return common.StreamTypeICEcast
	}
	return common.StreamTypeUnsupported
}
