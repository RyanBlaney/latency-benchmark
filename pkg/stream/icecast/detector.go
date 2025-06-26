package icecast

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Detector handles ICEcast stream detection with configurable options
type Detector struct {
	config *DetectionConfig
	client *http.Client
}

// NewDetector creates a new ICEcast detector with default configuration
func NewDetector() *Detector {
	return NewDetectorWithConfig(nil)
}

// NewDetectorWithConfig creates a new ICEcast detector with custom configuration
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

// DetectTypeWithConfig determines if the stream is ICEcast using full configuration
// DetectTypeWithConfig determines if the stream is ICEcast using full configuration
func DetectTypeWithConfig(ctx context.Context, streamURL string, config *Config) (common.StreamType, error) {
	if config == nil {
		config = DefaultConfig()
	}

	detector := NewDetectorWithConfig(config.Detection)

	// Create a context with configured timeout
	detectCtx, cancel := context.WithTimeout(ctx, time.Duration(config.Detection.TimeoutSeconds)*time.Second)
	defer cancel()

	// Step 1: URL pattern detection (fastest, no network calls)
	if streamType := detector.DetectFromURL(streamURL); streamType == common.StreamTypeICEcast {
		// Verify with a lightweight HEAD request if configured to do so
		if len(config.Detection.RequiredHeaders) > 0 || len(config.Detection.ContentTypes) > 0 {
			if detector.DetectFromHeaders(detectCtx, streamURL, config.HTTP) == common.StreamTypeICEcast {
				return common.StreamTypeICEcast, nil
			}
		} else {
			// Trust URL pattern if no header validation required
			return common.StreamTypeICEcast, nil
		}
	}

	// Step 2: Header-based detection
	if streamType := detector.DetectFromHeaders(detectCtx, streamURL, config.HTTP); streamType == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	// Not ICEcast or detection failed
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
	req.Header.Set("Accept", "audio/*")
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
	extractMetadataFromHeaders(resp.Header, metadata)

	return metadata, nil
}

// ProbeStreamWithConfig performs a probe with full configuration control
func ProbeStreamWithConfig(ctx context.Context, streamURL string, config *Config) (*common.StreamMetadata, error) {
	if config == nil {
		config = DefaultConfig()
	}

	detector := NewDetectorWithConfig(config.Detection)

	// Use configured timeout
	detectCtx, cancel := context.WithTimeout(ctx, time.Duration(config.Detection.TimeoutSeconds)*time.Second)
	defer cancel()

	// Use detector's client or create one with HTTP config timeout
	client := detector.client
	if config.HTTP != nil {
		client = &http.Client{
			Timeout: config.HTTP.ConnectionTimeout + config.HTTP.ReadTimeout,
		}
	}

	req, err := http.NewRequestWithContext(detectCtx, "HEAD", streamURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers from configuration
	if config.HTTP != nil {
		headers := config.HTTP.GetHTTPHeaders()
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else {
		// Fallback to default headers
		req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
		req.Header.Set("Accept", "audio/*")
		req.Header.Set("Icy-MetaData", "1")
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to probe stream: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Create metadata using configured extractor
	metadata := &common.StreamMetadata{
		URL:       streamURL,
		Type:      common.StreamTypeICEcast,
		Headers:   make(map[string]string),
		Timestamp: time.Now(),
	}

	// Extract metadata from headers
	extractMetadataFromHeaders(resp.Header, metadata)

	// Apply default values from configuration if available
	if config.MetadataExtractor != nil && config.MetadataExtractor.DefaultValues != nil {
		applyDefaultValues(metadata, config.MetadataExtractor.DefaultValues)
	}

	return metadata, nil
}

// DetectFromURL matches the URL with configured ICEcast patterns
func (d *Detector) DetectFromURL(streamURL string) common.StreamType {
	u, err := url.Parse(streamURL)
	if err != nil {
		fmt.Printf("{DEBUG}: Error in parsing URL: %v", err)
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)
	port := u.Port()

	// Check port-based detection first
	for _, commonPort := range d.config.CommonPorts {
		if port == commonPort {
			return common.StreamTypeICEcast
		}
	}

	// Check against configured URL patterns
	for _, pattern := range d.config.URLPatterns {
		if matched, err := regexp.MatchString(pattern, path); err == nil && matched {
			return common.StreamTypeICEcast
		}
	}

	return common.StreamTypeUnsupported
}

// DetectFromHeaders matches the HTTP headers with configured ICEcast patterns
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
		headers := httpConfig.CustomHeaders
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else {
		// Fallback to default headers
		req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
		req.Header.Set("Accept", "audio/*")
		req.Header.Set("Icy-MetaData", "1")
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
			return common.StreamTypeICEcast
		}
	}

	// Check for ICEcast-specific headers
	server := strings.ToLower(resp.Header.Get("Server"))
	if strings.Contains(server, "icecast") ||
		resp.Header.Get("icy-name") != "" ||
		resp.Header.Get("icy-description") != "" ||
		resp.Header.Get("icy-genre") != "" ||
		resp.Header.Get("icy-br") != "" ||
		resp.Header.Get("icy-metaint") != "" {
		return common.StreamTypeICEcast
	}

	return common.StreamTypeUnsupported
}

// extractMetadataFromHeaders extracts ICEcast metadata from HTTP headers
func extractMetadataFromHeaders(headers http.Header, metadata *common.StreamMetadata) {
	// Extract ICEcast-specific headers
	if name := headers.Get("icy-name"); name != "" {
		metadata.Station = name
		metadata.Headers["icy-name"] = name
	}
	if genre := headers.Get("icy-genre"); genre != "" {
		metadata.Genre = genre
		metadata.Headers["icy-genre"] = genre
	}
	if desc := headers.Get("icy-description"); desc != "" {
		metadata.Title = desc
		metadata.Headers["icy-description"] = desc
	}
	if url := headers.Get("icy-url"); url != "" {
		metadata.Headers["icy-url"] = url
	}

	// Extract technical metadata
	if bitrate := headers.Get("icy-br"); bitrate != "" {
		if br, err := strconv.Atoi(bitrate); err == nil {
			metadata.Bitrate = br
		}
		metadata.Headers["icy-br"] = bitrate
	}
	if sampleRate := headers.Get("icy-sr"); sampleRate != "" {
		if sr, err := strconv.Atoi(sampleRate); err == nil {
			metadata.SampleRate = sr
		}
		metadata.Headers["icy-sr"] = sampleRate
	}
	if channels := headers.Get("icy-channels"); channels != "" {
		if ch, err := strconv.Atoi(channels); err == nil {
			metadata.Channels = ch
		}
		metadata.Headers["icy-channels"] = channels
	}

	// Determine format/codec from content type
	contentType := headers.Get("Content-Type")
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
		"icy-url", "icy-br", "icy-sr", "icy-channels", "icy-pub", "icy-notice1",
		"icy-notice2", "icy-metaint", "icy-version",
	}

	for _, header := range relevantHeaders {
		if value := headers.Get(header); value != "" {
			metadata.Headers[strings.ToLower(header)] = value
		}
	}
}

// applyDefaultValues applies default values from configuration
func applyDefaultValues(metadata *common.StreamMetadata, defaults map[string]interface{}) {
	for field, defaultValue := range defaults {
		switch field {
		case "codec":
			if metadata.Codec == "" {
				if codec, ok := defaultValue.(string); ok {
					metadata.Codec = codec
				}
			}
		case "channels":
			if metadata.Channels == 0 {
				if channels, ok := defaultValue.(int); ok {
					metadata.Channels = channels
				} else if channelsFloat, ok := defaultValue.(float64); ok {
					metadata.Channels = int(channelsFloat)
				}
			}
		case "sample_rate":
			if metadata.SampleRate == 0 {
				if rate, ok := defaultValue.(int); ok {
					metadata.SampleRate = rate
				} else if rateFloat, ok := defaultValue.(float64); ok {
					metadata.SampleRate = int(rateFloat)
				}
			}
		case "bitrate":
			if metadata.Bitrate == 0 {
				if bitrate, ok := defaultValue.(int); ok {
					metadata.Bitrate = bitrate
				} else if bitrateFloat, ok := defaultValue.(float64); ok {
					metadata.Bitrate = int(bitrateFloat)
				}
			}
		case "format":
			if metadata.Format == "" || metadata.Format == "unknown" {
				if format, ok := defaultValue.(string); ok {
					metadata.Format = format
				}
			}
		}
	}
}

// Package-level convenience functions that use default configuration
// These maintain backward compatibility with existing code

// DetectFromURL matches the URL with common ICEcast patterns (backward compatibility)
func DetectFromURL(streamURL string) common.StreamType {
	detector := NewDetector()
	return detector.DetectFromURL(streamURL)
}

// DetectFromHeaders matches the HTTP headers with common ICEcast patterns (backward compatibility)
func DetectFromHeaders(ctx context.Context, client *http.Client, streamURL string) common.StreamType {
	detector := NewDetector()

	// Create a temporary HTTP config from the client timeout
	httpConfig := &HTTPConfig{
		UserAgent:      "TuneIn-CDN-Benchmark/1.0",
		AcceptHeader:   "audio/*",
		RequestICYMeta: true,
	}

	if client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.DetectFromHeaders(ctx, streamURL, httpConfig)
}

// ConfigurableDetection provides a complete detection suite with configuration
func ConfigurableDetection(ctx context.Context, streamURL string, config *Config) (common.StreamType, error) {
	if config == nil {
		config = DefaultConfig()
	}

	detector := NewDetectorWithConfig(config.Detection)

	// Step 1: URL pattern detection
	if streamType := detector.DetectFromURL(streamURL); streamType == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	// Step 2: Header detection
	if streamType := detector.DetectFromHeaders(ctx, streamURL, config.HTTP); streamType == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	return common.StreamTypeUnsupported, fmt.Errorf("stream type not supported or invalid")
}
