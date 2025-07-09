package icecast

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
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
		Transport: &http.Transport{
			MaxIdleConns:       10,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
		},
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

	// Step 3: Content validation as last resort (check if audio stream is actually streaming)
	if detector.IsValidICEcastContent(detectCtx, streamURL, config.HTTP) {
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
		return nil, common.NewStreamError(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, "failed to create request", err)
	}

	// Set headers for ICEcast compatibility
	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "audio/*")
	req.Header.Set("Icy-MetaData", "1") // Request ICY metadata

	resp, err := client.Do(req)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, "failed to probe ICEcast stream", err)
	}
	defer resp.Body.Close()

	// Check if the response status indicates success
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, common.NewStreamError(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, fmt.Sprintf("HTTP error: %d %s", resp.StatusCode, resp.Status), nil)
	}

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
			Transport: &http.Transport{
				MaxIdleConns:       10,
				IdleConnTimeout:    30 * time.Second,
				DisableCompression: false,
			},
		}
	}

	req, err := http.NewRequestWithContext(detectCtx, "HEAD", streamURL, nil)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, "failed to create request", err)
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
		return nil, common.NewStreamError(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, "failed to probe stream", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, common.NewStreamErrorWithFields(common.StreamTypeICEcast, streamURL,
			common.ErrCodeConnection, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, resp.Status), nil,
			logging.Fields{
				"status_code": resp.StatusCode,
				"status_text": resp.Status,
			})
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
		logging.Debug("Error parsing URL", logging.Fields{"url": streamURL, "error": err.Error()})
		return common.StreamTypeUnsupported
	}

	// ICEcast streams must use HTTP or HTTPS
	if u.Scheme != "http" && u.Scheme != "https" {
		return common.StreamTypeUnsupported
	}

	path := strings.ToLower(u.Path)
	query := strings.ToLower(u.RawQuery)
	port := u.Port()

	// Check port-based detection first
	if slices.Contains(d.config.CommonPorts, port) {
		return common.StreamTypeICEcast
	}

	// Check against configured URL patterns
	for _, pattern := range d.config.URLPatterns {
		if matched, err := regexp.MatchString(pattern, path); err == nil && matched {
			return common.StreamTypeICEcast
		}
		// Also check query string for edge cases
		if matched, err := regexp.MatchString(pattern, query); err == nil && matched {
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
			Transport: &http.Transport{
				MaxIdleConns:       10,
				IdleConnTimeout:    30 * time.Second,
				DisableCompression: false,
			},
		}
	}

	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		logging.Debug("Error creating request for HTTP headers", logging.Fields{
			"url":   streamURL,
			"error": err.Error(),
		})
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
		req.Header.Set("Accept", "audio/*")
		req.Header.Set("Icy-MetaData", "1")
	}

	resp, err := client.Do(req)
	if err != nil {
		logging.Debug("Error getting response from client", logging.Fields{
			"url":   streamURL,
			"error": err.Error(),
		})
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

// IsValidICEcastContent checks if the content appears to be valid ICEcast audio stream
func (d *Detector) IsValidICEcastContent(ctx context.Context, streamURL string, httpConfig *HTTPConfig) bool {
	// Use configured timeout for content validation
	detectCtx, cancel := context.WithTimeout(ctx, time.Duration(d.config.TimeoutSeconds)*time.Second)
	defer cancel()

	// Use provided client or create one with timeout
	client := d.client
	if httpConfig != nil {
		client = &http.Client{
			Timeout: httpConfig.ConnectionTimeout + httpConfig.ReadTimeout,
			Transport: &http.Transport{
				MaxIdleConns:       10,
				IdleConnTimeout:    30 * time.Second,
				DisableCompression: false,
			},
		}
	}

	req, err := http.NewRequestWithContext(detectCtx, "GET", streamURL, nil)
	if err != nil {
		return false
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
		req.Header.Set("Accept", "audio/*")
		req.Header.Set("Icy-MetaData", "1")
	}

	// Set range header to only read a small amount of data
	req.Header.Set("Range", "bytes=0-1023")

	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	// Accept both 200 (full content) and 206 (partial content)
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return false
	}

	// Verify content type indicates audio
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if contentType != "" && !strings.HasPrefix(contentType, "audio/") {
		// Check if it's a known ICEcast content type
		isValidContentType := false
		for _, validType := range d.config.ContentTypes {
			if strings.Contains(contentType, strings.ToLower(validType)) {
				isValidContentType = true
				break
			}
		}
		if !isValidContentType {
			return false
		}
	}

	// Read a small amount of data to verify it's actually streaming
	buffer := make([]byte, 1024)
	n, err := resp.Body.Read(buffer)
	if err != nil && n == 0 {
		return false
	}

	if n > 0 {
		return isValidMP3Data(buffer[:n])
	}
	return false
}

func isValidMP3Data(data []byte) bool {
	if len(data) < 4 {
		return false
	}

	// Look for MP3 frame sync
	// MP3 frames start with 11 bits set (0xFFE)
	for i := 0; i < len(data)-1; i++ {
		if data[i] == 0xFF && (data[i+1]&0xE0) == 0xE0 {
			// Found potential MP3 frame header
			if i+3 < len(data) {
				// Additional validation of the MP3 frame header
				if isValidMP3FrameHeader(data[i : i+4]) {
					return true
				}
			}
		}
	}

	return false
}

func isValidMP3FrameHeader(header []byte) bool {
	if len(header) < 4 {
		return false
	}

	// First 11 bits should be 1 (frame sync)
	if header[0] != 0xFF || (header[1]&0xE0) != 0xE0 {
		return false
	}

	// MPEG version (bits 19-20)
	version := (header[1] >> 3) & 0x03
	if version == 0x01 { // Reserved version
		return false
	}

	// Layer (bits 17-18)
	layer := (header[1] >> 1) & 0x03
	if layer == 0x00 { // Reserved layer
		return false
	}

	// Bitrate (bits 12-15)
	bitrate := (header[2] >> 4) & 0x0F
	if bitrate == 0x00 || bitrate == 0x0F { // Free or reserved bitrate
		return false
	}

	// Sample rate (bits 10-11)
	sampleRate := (header[2] >> 2) & 0x03
	if sampleRate == 0x03 { // Reserved sample rate
		return false
	}

	return true
}

// extractMetadataFromHeaders extracts ICEcast metadata from HTTP headers
func extractMetadataFromHeaders(headers http.Header, metadata *common.StreamMetadata) {
	// Store all headers in lowercase for reference using maps.Copy
	headerMap := make(map[string]string)
	for key, values := range headers {
		if len(values) > 0 {
			headerMap[strings.ToLower(key)] = values[0]
		}
	}
	maps.Copy(metadata.Headers, headerMap)

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

	// Store all relevant ICEcast headers efficiently
	relevantHeaders := map[string]string{
		"content-type":    headers.Get("content-type"),
		"server":          headers.Get("server"),
		"icy-name":        headers.Get("icy-name"),
		"icy-genre":       headers.Get("icy-genre"),
		"icy-description": headers.Get("icy-description"),
		"icy-url":         headers.Get("icy-url"),
		"icy-br":          headers.Get("icy-br"),
		"icy-sr":          headers.Get("icy-sr"),
		"icy-channels":    headers.Get("icy-channels"),
		"icy-pub":         headers.Get("icy-pub"),
		"icy-notice1":     headers.Get("icy-notice1"),
		"icy-notice2":     headers.Get("icy-notice2"),
		"icy-metaint":     headers.Get("icy-metaint"),
		"icy-version":     headers.Get("icy-version"),
	}

	// Only add non-empty headers using maps.Copy
	filteredHeaders := make(map[string]string)
	for key, value := range relevantHeaders {
		if value != "" {
			filteredHeaders[key] = value
		}
	}
	maps.Copy(metadata.Headers, filteredHeaders)
}

// applyDefaultValues applies default values from configuration
func applyDefaultValues(metadata *common.StreamMetadata, defaults map[string]any) {
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

	// Step 3: Content validation as last resort
	if detector.IsValidICEcastContent(ctx, streamURL, config.HTTP) {
		return common.StreamTypeICEcast, nil
	}

	return common.StreamTypeUnsupported, common.NewStreamError(common.StreamTypeICEcast, streamURL,
		common.ErrCodeUnsupported, "stream type not supported or invalid", nil)
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
		CustomHeaders:  make(map[string]string),
	}

	if client != nil && client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.DetectFromHeaders(ctx, streamURL, httpConfig)
}

// IsValidICEcastContent checks if the content appears to be valid ICEcast (backward compatibility)
func IsValidICEcastContent(ctx context.Context, client *http.Client, streamURL string) bool {
	detector := NewDetector()

	// Create HTTP config from client
	httpConfig := &HTTPConfig{
		UserAgent:      "TuneIn-CDN-Benchmark/1.0",
		AcceptHeader:   "audio/*",
		RequestICYMeta: true,
		CustomHeaders:  make(map[string]string),
	}
	if client != nil && client.Timeout > 0 {
		httpConfig.ConnectionTimeout = client.Timeout / 2
		httpConfig.ReadTimeout = client.Timeout / 2
	}

	return detector.IsValidICEcastContent(ctx, streamURL, httpConfig)
}
