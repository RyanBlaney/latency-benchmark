package stream

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

type Detector struct {
	client *http.Client
}

func NewDetector() *Detector {
	return &Detector{
		client: &http.Client{
			Timeout: 10 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				// Follow up to 3 redirects
				if len(via) >= 3 {
					return http.ErrUseLastResponse
				}
				return nil
			},
		},
	}
}

func (sd *Detector) DetectType(ctx context.Context, streamURL string) (common.StreamType, error) {
	// URL-based detection
	if streamType := sd.detectFromURL(streamURL); streamType != common.StreamTypeUnsupported {
		return streamType, nil
	}

	// Fall back to HTTP header-based detection
	return sd.detectFromHeaders(ctx, streamURL)
}

// detectFromURL attempts to detect stream type from URL patterns
func (sd *Detector) detectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URl: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)

	// HLS detection patterns
	if strings.HasSuffix(path, ".m3u8") ||
		strings.Contains(path, "/playlist.m3u8") ||
		strings.Contains(path, "/index.m3u8") ||
		strings.Contains(u.RawQuery, "m3u8") {
		return common.StreamTypeHLS
	}

	// ICEcast detection patterns
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

// detectFromHeaders performs HTTP HEAD request to detect stream type from headers
func (sd *Detector) detectFromHeaders(ctx context.Context, streamURL string) (common.StreamType, error) {
	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		fmt.Printf("{DEBUG}: Error creating request to HTTP headers: %v", err)
		return common.StreamTypeUnsupported, err
	}

	// Set user agent to avoid blocking
	req.Header.Set("User-Agent", "TuneIN-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "*/*")

	resp, err := sd.client.Do(req)
	if err != nil {
		fmt.Printf("{DEBUG}: Error getting response from client: %v", err)
		return common.StreamTypeUnsupported, err
	}
	defer resp.Body.Close()

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	server := strings.ToLower(resp.Header.Get("Server"))

	// HLS detection
	if strings.Contains(contentType, "application/vnd.apple.mpegurl") ||
		strings.Contains(contentType, "application/x-mpegurl") ||
		strings.Contains(contentType, "vnd.apple.mpegurl") {
		return common.StreamTypeHLS, nil
	}

	// ICEcast detection
	if strings.Contains(contentType, "audio/mpeg") ||
		strings.Contains(contentType, "audio/aac") ||
		strings.Contains(contentType, "audio/ogg") ||
		strings.Contains(contentType, "application/ogg") ||
		strings.Contains(server, "icecast") ||
		resp.Header.Get("icy-name") != "" ||
		resp.Header.Get("icy-description") != "" {
		return common.StreamTypeICEcast, nil
	}

	return common.StreamTypeUnsupported, common.NewStreamError(
		common.StreamTypeUnsupported, streamURL, common.ErrCodeUnsupported,
		"unable to determine stream type from URL or headers",
		nil,
	)
}

// ProbeStream performs a lightweight probe to gather basic info
func (d *Detector) ProbeStream(ctx context.Context, streamURL string) (*common.StreamMetadata, error) {
	streamType, err := d.DetectType(ctx, streamURL)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "*/*")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	metadata := &common.StreamMetadata{
		URL:       streamURL,
		Type:      streamType,
		Headers:   make(map[string]string),
		Timestamp: time.Now(),
	}

	// Extract basic metadata from headers
	if name := resp.Header.Get("icy-name"); name != "" {
		metadata.Station = name
	}
	if genre := resp.Header.Get("icy-genre"); genre != "" {
		metadata.Genre = genre
	}
	if desc := resp.Header.Get("icy-description"); desc != "" {
		metadata.Title = desc
	}

	// Store relevant headers
	relevantHeaders := []string{
		"content-type", "server", "icy-name", "icy-genre",
		"icy-description", "icy-bitrate", "icy-samplerate",
	}
	for _, header := range relevantHeaders {
		if value := resp.Header.Get(header); value != "" {
			metadata.Headers[header] = value
		}
	}

	return metadata, nil
}
