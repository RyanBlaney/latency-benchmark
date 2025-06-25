package hls

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Detector handles HLS stream detection with configurable options
type Detector struct {
	config *DetectionConfig
	client *http.Client
}

// NewDetector creates a new HLS detector with default configuration
func NewDetector() *Detector {
	return NewDetectorWithConfig(nil)
}

// NewDetectorWithConfig creates a new HLS detector with custom configuration
func NewDetectorWithConfig(config *DetectionConfig) *Detector {
	if config == nil {
		config = DefaultConfig().Detection
	}

	client := &http.Client{
		Timeout: time.Duration(config.TimeoutSeconds) * time.Second,
	}

	return &Detector{
		config: config,
		client: client,
	}
}

// DetectType determines if the stream is HLS using URL patterns and headers
func DetectType(ctx context.Context, client *http.Client, streamURL string) (common.StreamType, error) {
	// First try URL-based detection (fastest)
	if DetectFromURL(streamURL) == common.StreamTypeHLS {
		return common.StreamTypeHLS, nil
	}

	// Fall back to header-based detection
	if DetectFromHeaders(ctx, client, streamURL) == common.StreamTypeHLS {
		return common.StreamTypeHLS, nil
	}

	// Not HLS
	return common.StreamTypeUnsupported, nil
}

// ProbeStream performs a lightweight probe to gather HLS stream metadata
func ProbeStream(ctx context.Context, client *http.Client, streamURL string) (*common.StreamMetadata, error) {
	// Use the existing DetectFromM3U8Content function which already extracts metadata
	playlist, err := DetectFromM3U8Content(ctx, client, streamURL)
	if err != nil {
		return nil, fmt.Errorf("failed to probe HLS stream: %w", err)
	}

	// Return the metadata that was extracted during parsing
	if playlist.Metadata != nil {
		return playlist.Metadata, nil
	}

	// Fallback: create basic metadata if parsing didn't extract it
	return &common.StreamMetadata{
		URL:       streamURL,
		Type:      common.StreamTypeHLS,
		Format:    "hls",
		Headers:   playlist.Headers,
		Timestamp: time.Now(),
	}, nil
}

// DetectFromURL matches the URL with configured HLS patterns
func (d *Detector) DetectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URL: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)
	query := strings.ToLower(u.RawQuery)

	// Check against configured URL patterns
	for _, pattern := range d.config.URLPatterns {
		if matched, err := regexp.MatchString(pattern, path); err == nil && matched {
			return common.StreamTypeHLS
		}
		if matched, err := regexp.MatchString(pattern, query); err == nil && matched {
			return common.StreamTypeHLS
		}
	}

	return common.StreamTypeUnsupported
}

// DetectFromHeaders matches the HTTP headers with configured HLS patterns
func (d *Detector) DetectFromHeaders(ctx context.Context, streamURL string, httpConfig *HTTPConfig) common.StreamType {
	// Use provided client or create one with timeout
	client := d.client
	if httpConfig != nil {
		client = &http.Client{
			Timeout: httpConfig.ConnectionTimeout + httpConfig.ReadTimeout,
		}
	}

	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		fmt.Printf("{DEBUG}: Error creating request for HTTP headers: %v", err)
		return common.StreamTypeUnsupported
	}

	// Set headers from configuration
	if httpConfig != nil {
		headers := httpConfig.GetHTTPHeaders()
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else {
		// Fallback to default headers
		req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
		req.Header.Set("Accept", "*/*")
	}

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("{DEBUG}: Error getting response from client: %v", err)
		return common.StreamTypeUnsupported
	}
	defer resp.Body.Close()

	// Check required headers if configured
	for _, requiredHeader := range d.config.RequiredHeaders {
		if resp.Header.Get(requiredHeader) == "" {
			return common.StreamTypeUnsupported
		}
	}

	// Check content type against configured patterns
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	for _, pattern := range d.config.ContentTypes {
		if strings.Contains(contentType, strings.ToLower(pattern)) {
			return common.StreamTypeHLS
		}
	}

	return common.StreamTypeUnsupported
}

// DetectFromM3U8Content attempts to validate and parse M3U8 content with configuration
func (d *Detector) DetectFromM3U8Content(ctx context.Context, streamURL string, httpConfig *HTTPConfig, parserConfig *ParserConfig) (*M3U8Playlist, error) {
	// Use configured timeout
	detectCtx, cancel := context.WithTimeout(ctx, time.Duration(d.config.TimeoutSeconds)*time.Second)
	defer cancel()

	// Use provided client or create one with timeout
	client := d.client
	if httpConfig != nil {
		client = &http.Client{
			Timeout: httpConfig.ConnectionTimeout + httpConfig.ReadTimeout,
		}
	}

	req, err := http.NewRequestWithContext(detectCtx, "GET", streamURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers from configuration
	if httpConfig != nil {
		headers := httpConfig.GetHTTPHeaders()
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else {
		// Fallback to default headers
		req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
		req.Header.Set("Accept", "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain")
	}

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

	// Parse the M3U8 content using configured parser
	var parser *Parser
	if parserConfig != nil {
		parser = NewConfigurableParser(parserConfig).Parser
	} else {
		parser = NewParser()
	}

	playlist, err := parser.ParseM3U8Content(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse M3U8: %w", err)
	}

	// Merge HTTP headers with playlist headers
	if playlist.Headers == nil {
		playlist.Headers = make(map[string]string)
	}
	maps.Copy(playlist.Headers, headers)

	// Extract metadata using configured extractor
	metadataExtractor := NewMetadataExtractor()
	playlist.Metadata = metadataExtractor.ExtractMetadata(playlist, streamURL)

	return playlist, nil
}

// IsValidHLSContent checks if the content appears to be valid HLS with configuration
func (d *Detector) IsValidHLSContent(ctx context.Context, streamURL string, httpConfig *HTTPConfig, parserConfig *ParserConfig) bool {
	playlist, err := d.DetectFromM3U8Content(ctx, streamURL, httpConfig, parserConfig)
	if err != nil {
		return false
	}

	return playlist.IsValid && (len(playlist.Segments) > 0 || len(playlist.Variants) > 0)
}

// GetHTTPHeaders returns configured HTTP headers for this detector
func (httpConfig *HTTPConfig) GetHTTPHeaders() map[string]string {
	headers := make(map[string]string)

	// Set standard headers
	headers["User-Agent"] = httpConfig.UserAgent
	headers["Accept"] = httpConfig.AcceptHeader

	// Add custom headers
	for k, v := range httpConfig.CustomHeaders {
		headers[k] = v
	}

	return headers
}

// Package-level convenience functions that use default configuration
// These maintain backward compatibility with existing code

// DetectFromURL matches the URL with common HLS patterns (backward compatibility)
func DetectFromURL(streamURL string) common.StreamType {
	detector := NewDetector()
	return detector.DetectFromURL(streamURL)
}

// DetectFromHeaders matches the HTTP headers with common HLS patterns (backward compatibility)
func DetectFromHeaders(ctx context.Context, client *http.Client, streamURL string) common.StreamType {
	detector := NewDetector()

	// Create a temporary HTTP config from the client timeout
	httpConfig := &HTTPConfig{
		UserAgent:    "TuneIn-CDN-Benchmark/1.0",
		AcceptHeader: "*/*",
	}

	if client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.DetectFromHeaders(ctx, streamURL, httpConfig)
}

// DetectFromM3U8Content attempts to validate and parse M3U8 content (backward compatibility)
func DetectFromM3U8Content(ctx context.Context, client *http.Client, streamURL string) (*M3U8Playlist, error) {
	detector := NewDetector()

	// Create a temporary HTTP config from the client
	httpConfig := &HTTPConfig{
		UserAgent:    "TuneIn-CDN-Benchmark/1.0",
		AcceptHeader: "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain",
	}

	if client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.DetectFromM3U8Content(ctx, streamURL, httpConfig, nil)
}

// IsValidHLSContent checks if the content appears to be valid HLS (backward compatibility)
func IsValidHLSContent(ctx context.Context, client *http.Client, streamURL string) bool {
	detector := NewDetector()

	// Create a temporary HTTP config from the client
	httpConfig := &HTTPConfig{
		UserAgent:    "TuneIn-CDN-Benchmark/1.0",
		AcceptHeader: "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain",
	}

	if client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.IsValidHLSContent(ctx, streamURL, httpConfig, nil)
}

// ConfigurableDetection provides a complete detection suite with configuration
func ConfigurableDetection(ctx context.Context, streamURL string, config *Config) (common.StreamType, *M3U8Playlist, error) {
	if config == nil {
		config = DefaultConfig()
	}

	detector := NewDetectorWithConfig(config.Detection)

	// Step 1: URL pattern detection
	if streamType := detector.DetectFromURL(streamURL); streamType == common.StreamTypeHLS {
		// Step 2: Try to parse content for verification
		playlist, err := detector.DetectFromM3U8Content(ctx, streamURL, config.HTTP, config.Parser)
		if err == nil && playlist.IsValid {
			return common.StreamTypeHLS, playlist, nil
		}
	}

	// Step 3: Header detection as fallback
	if streamType := detector.DetectFromHeaders(ctx, streamURL, config.HTTP); streamType == common.StreamTypeHLS {
		// Try to parse content
		playlist, err := detector.DetectFromM3U8Content(ctx, streamURL, config.HTTP, config.Parser)
		if err == nil {
			return common.StreamTypeHLS, playlist, nil
		}
		// Return HLS type even if parsing failed
		return common.StreamTypeHLS, nil, fmt.Errorf("detected HLS from headers but failed to parse content: %w", err)
	}

	// Step 4: Content detection as last resort
	playlist, err := detector.DetectFromM3U8Content(ctx, streamURL, config.HTTP, config.Parser)
	if err == nil && playlist.IsValid {
		return common.StreamTypeHLS, playlist, nil
	}

	return common.StreamTypeUnsupported, nil, fmt.Errorf("stream type not supported or invalid")
}

