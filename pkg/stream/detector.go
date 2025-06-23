package stream

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/icecast"
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
	var st common.StreamType

	// HLS detection patterns
	st = hls.DetectFromURL(streamURL)
	if st == common.StreamTypeHLS {
		return common.StreamTypeHLS
	}

	// ICEcast detection patterns
	st = icecast.DetectFromURL(streamURL)
	if st == common.StreamTypeICEcast {
		return common.StreamTypeICEcast
	}

	return common.StreamTypeUnsupported
}

// detectFromHeaders performs HTTP HEAD request to detect stream type from headers
func (sd *Detector) detectFromHeaders(ctx context.Context, streamURL string) (common.StreamType, error) {
	var st common.StreamType

	// HLS detection
	st = hls.DetectFromHeaders(ctx, sd.client, streamURL)
	if st == common.StreamTypeHLS {
		return common.StreamTypeHLS, nil
	}

	// ICEcast detection
	st = icecast.DetectFromHeaders(ctx, sd.client, streamURL)
	if st == common.StreamTypeICEcast {
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
