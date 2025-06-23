package icecast

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

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
