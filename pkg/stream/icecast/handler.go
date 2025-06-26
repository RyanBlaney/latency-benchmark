package icecast

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
)

// Handler implements StreamHandler for ICEcast streams with full configuration support
type Handler struct {
	client            *http.Client
	url               string
	metadata          *common.StreamMetadata
	stats             *common.StreamStats
	connected         bool
	response          *http.Response
	metadataExtractor *MetadataExtractor
	config            *Config
	icyMetaInt        int    // ICY metadata interval
	icyTitle          string // Current ICY title
	bytesRead         int64  // Bytes read since last metadata
}

// NewHandler creates a new ICEcast stream handler with default configuration
func NewHandler() *Handler {
	return NewHandlerWithConfig(nil)
}

// NewHandlerWithConfig creates a new ICEcast stream handler with custom configuration
func NewHandlerWithConfig(config *Config) *Handler {
	if config == nil {
		config = DefaultConfig()
	}

	// Create HTTP client with configured timeouts
	client := &http.Client{
		Timeout: config.HTTP.ConnectionTimeout + config.HTTP.ReadTimeout,
		Transport: &http.Transport{
			MaxIdleConns:       10,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
		},
	}

	return &Handler{
		client:            client,
		stats:             &common.StreamStats{},
		metadataExtractor: NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor,
		config:            config,
	}
}

// Type returns the stream type for this handler
func (h *Handler) Type() common.StreamType {
	return common.StreamTypeICEcast
}

// CanHandle determines if this handler can process the given URL
func (h *Handler) CanHandle(ctx context.Context, url string) bool {
	// Use configurable detection for better accuracy
	streamType, err := ConfigurableDetection(ctx, url, h.config)
	if err == nil && streamType == common.StreamTypeICEcast {
		return true
	}

	// Fallback to individual detection methods for backward compatibility
	detector := NewDetectorWithConfig(h.config.Detection)

	if st := detector.DetectFromURL(url); st == common.StreamTypeICEcast {
		return true
	}

	if st := detector.DetectFromHeaders(ctx, url, h.config.HTTP); st == common.StreamTypeICEcast {
		return true
	}

	return false
}

// Connect establishes connection to the ICEcast stream
func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "already connected", nil)
	}

	h.url = url
	startTime := time.Now()

	// Create context with configured timeout
	ctx, cancel := context.WithTimeout(ctx, h.config.HTTP.ConnectionTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "failed to create request", err)
	}

	// Set headers from configuration
	headers := h.config.GetHTTPHeaders()
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "connection failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, fmt.Sprintf("HTTP %d", resp.StatusCode), nil)
	}

	h.response = resp

	// Parse ICY metadata interval if present
	if metaInt := resp.Header.Get("icy-metaint"); metaInt != "" {
		if interval, err := strconv.Atoi(metaInt); err == nil {
			h.icyMetaInt = interval
		}
	}

	// Extract initial metadata using the configured extractor
	h.metadata = h.metadataExtractor.ExtractMetadata(resp.Header, url)

	h.stats.ConnectionTime = time.Since(startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime
	h.connected = true

	return nil
}

// GetMetadata retrieves stream metadata
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Return the metadata extracted during connection or updated during streaming
	if h.metadata == nil {
		// Fallback metadata if extraction failed, using configured defaults
		h.metadata = &common.StreamMetadata{
			URL:       h.url,
			Type:      common.StreamTypeICEcast,
			Headers:   make(map[string]string),
			Timestamp: time.Now(),
		}

		// Apply default values from configuration
		h.applyDefaultMetadata()
	}

	// Update with current ICY title if available
	if h.icyTitle != "" {
		h.metadata.Title = h.icyTitle
		h.metadata.Timestamp = time.Now()
	}

	return h.metadata, nil
}

// applyDefaultMetadata applies default values from configuration
func (h *Handler) applyDefaultMetadata() {
	if defaults := h.config.MetadataExtractor.DefaultValues; defaults != nil {
		if codec, ok := defaults["codec"].(string); ok {
			h.metadata.Codec = codec
		}
		if sampleRate, ok := defaults["sample_rate"].(int); ok {
			h.metadata.SampleRate = sampleRate
		} else if sampleRateFloat, ok := defaults["sample_rate"].(float64); ok {
			h.metadata.SampleRate = int(sampleRateFloat)
		}
		if channels, ok := defaults["channels"].(int); ok {
			h.metadata.Channels = channels
		} else if channelsFloat, ok := defaults["channels"].(float64); ok {
			h.metadata.Channels = int(channelsFloat)
		}
		if bitrate, ok := defaults["bitrate"].(int); ok {
			h.metadata.Bitrate = bitrate
		} else if bitrateFloat, ok := defaults["bitrate"].(float64); ok {
			h.metadata.Bitrate = int(bitrateFloat)
		}
		if format, ok := defaults["format"].(string); ok {
			h.metadata.Format = format
		}
	}
}

// ReadAudio reads audio data from the ICEcast stream
func (h *Handler) ReadAudio(ctx context.Context) (*common.AudioData, error) {
	if !h.connected || h.response == nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Use configured buffer size for reading
	bufferSize := h.config.Audio.BufferSize
	if bufferSize <= 0 {
		bufferSize = 4096 // Fallback default
	}

	buffer := make([]byte, bufferSize)

	// Set read deadline based on context or configured timeout
	if deadline, ok := ctx.Deadline(); ok {
		if conn, ok := h.response.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(deadline)
		}
	} else if h.config.Audio.ReadTimeout > 0 {
		deadline := time.Now().Add(h.config.Audio.ReadTimeout)
		if conn, ok := h.response.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(deadline)
		}
	}

	// Handle ICY metadata if present and configured
	var audioBytes []byte
	if h.icyMetaInt > 0 && h.config.Audio.HandleICYMeta {
		var err error
		audioBytes, err = h.readWithICYMetadata(buffer)
		if err != nil {
			return nil, err
		}
		if len(audioBytes) == 0 {
			return nil, io.EOF
		}
	} else {
		// Simple read without ICY metadata handling
		n, err := h.response.Body.Read(buffer)
		if err != nil && err != io.EOF {
			if err == context.DeadlineExceeded || err == context.Canceled {
				return nil, err
			}
			return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
				common.ErrCodeDecoding, "failed to read stream data", err)
		}
		if n == 0 {
			return nil, io.EOF
		}
		audioBytes = buffer[:n]
	}

	// Update statistics
	h.stats.BytesReceived += int64(len(audioBytes))

	// Calculate average bitrate based on configured duration window
	if h.stats.BytesReceived > 0 {
		elapsed := time.Since(time.Now().Add(-h.config.Audio.SampleDuration))
		if elapsed > 0 {
			h.stats.AverageBitrate = float64(h.stats.BytesReceived*8) / elapsed.Seconds() / 1000
		}
	}

	// Convert to PCM using configured audio parameters
	pcmSamples, err := h.convertToPCM(audioBytes)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "failed to convert audio to PCM", err)
	}

	// Get current metadata
	metadata, err := h.GetMetadata()
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeMetadata, "failed to get metadata", err)
	}

	// Create a copy of metadata to avoid concurrent modification
	metadataCopy := *metadata
	metadataCopy.Timestamp = time.Now()

	audioData := &common.AudioData{
		PCM:        pcmSamples,
		SampleRate: metadataCopy.SampleRate,
		Channels:   metadataCopy.Channels,
		Duration:   time.Duration(len(pcmSamples)) * time.Second / time.Duration(metadataCopy.SampleRate),
		Timestamp:  time.Now(),
		Metadata:   &metadataCopy,
	}

	return audioData, nil
}

// readWithICYMetadata handles reading audio data with embedded ICY metadata
func (h *Handler) readWithICYMetadata(buffer []byte) ([]byte, error) {
	var audioData []byte
	maxRetries := h.config.Audio.MaxReadAttempts

	for len(audioData) < len(buffer) && maxRetries > 0 {
		// Calculate how many audio bytes to read before next metadata block
		audioToRead := h.icyMetaInt - int(h.bytesRead%int64(h.icyMetaInt))
		audioToRead = min(audioToRead, len(buffer)-len(audioData))

		// Read audio data
		audioChunk := make([]byte, audioToRead)
		n, err := h.response.Body.Read(audioChunk)
		if err != nil && err != io.EOF {
			maxRetries--
			if maxRetries <= 0 {
				return nil, err
			}
			continue
		}
		if n == 0 {
			break
		}

		audioData = append(audioData, audioChunk[:n]...)
		h.bytesRead += int64(n)

		// Check if we need to read metadata
		if h.bytesRead%int64(h.icyMetaInt) == 0 {
			if err := h.readICYMetadata(); err != nil {
				logging.Warn("Failed to read ICY metadata", logging.Fields{
					"url":   h.url,
					"error": err.Error(),
				})
			}
		}

		if n < audioToRead {
			break // End of stream or partial read
		}
	}

	return audioData, nil
}

// readICYMetadata reads and parses ICY metadata from the stream
func (h *Handler) readICYMetadata() error {
	// Read metadata length byte
	lengthByte := make([]byte, 1)
	_, err := io.ReadFull(h.response.Body, lengthByte)
	if err != nil {
		return err
	}

	metadataLength := int(lengthByte[0]) * 16
	if metadataLength == 0 {
		return nil // No metadata
	}

	// Read metadata block
	metadataBlock := make([]byte, metadataLength)
	_, err = io.ReadFull(h.response.Body, metadataBlock)
	if err != nil {
		return err
	}

	// Parse metadata (typically in format "StreamTitle='Artist - Title';")
	metadataStr := string(metadataBlock)
	metadataStr = strings.TrimRight(metadataStr, "\x00") // Remove null padding

	if strings.Contains(metadataStr, "StreamTitle=") {
		start := strings.Index(metadataStr, "StreamTitle='")
		if start != -1 {
			start += len("StreamTitle='")
			end := strings.Index(metadataStr[start:], "'")
			if end != -1 {
				newTitle := metadataStr[start : start+end]
				if newTitle != h.icyTitle {
					h.icyTitle = newTitle
					// Update metadata with new title
					if h.metadata != nil {
						h.metadataExtractor.UpdateWithICYMetadata(h.metadata, h.icyTitle)
					}
				}
			}
		}
	}

	return nil
}

// convertToPCM converts raw audio bytes to PCM samples
func (h *Handler) convertToPCM(audioBytes []byte) ([]float64, error) {
	// TODO: decode ICEcast
	// This is a placeholder implementation
	// In a real implementation, you would use LibAV or similar to decode the audio
	// For now, we'll simulate PCM conversion based on the assumed format

	if h.metadata == nil || h.metadata.Codec == "" {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "no codec information available for PCM conversion", nil)
	}

	// For demonstration, assume 16-bit samples and convert to normalized float64
	// Real implementation would use proper audio decoding libraries
	channels := h.metadata.Channels
	if channels <= 0 {
		channels = 2 // Default to stereo
	}

	sampleCount := len(audioBytes) / 2 // Assuming 16-bit samples
	if channels > 1 {
		sampleCount /= channels
	}

	pcmSamples := make([]float64, sampleCount)
	for i := 0; i < sampleCount && i*2+1 < len(audioBytes); i++ {
		// Convert little-endian 16-bit to normalized float64
		sample := int16(audioBytes[i*2]) | int16(audioBytes[i*2+1])<<8
		pcmSamples[i] = float64(sample) / 32768.0
	}

	return pcmSamples, nil
}

// GetStats returns current streaming statistics
func (h *Handler) GetStats() *common.StreamStats {
	// Update buffer health based on connection state and configuration
	if h.connected && h.response != nil {
		h.stats.BufferHealth = 1.0 // Simplified - real implementation would check buffer levels
	} else {
		h.stats.BufferHealth = 0.0
	}

	return h.stats
}

// Close closes the stream connection
func (h *Handler) Close() error {
	h.connected = false
	if h.response != nil {
		err := h.response.Body.Close()
		h.response = nil
		return err
	}
	return nil
}

// GetClient returns the HTTP Client
func (h *Handler) GetClient() *http.Client {
	return h.client
}

// GetConfig returns the current configuration
func (h *Handler) GetConfig() *Config {
	return h.config
}

// UpdateConfig updates the handler configuration
func (h *Handler) UpdateConfig(config *Config) {
	if config == nil {
		return
	}

	h.config = config

	// Update HTTP client timeout if changed
	totalTimeout := config.HTTP.ConnectionTimeout + config.HTTP.ReadTimeout
	h.client.Timeout = totalTimeout

	// Recreate metadata extractor with new config
	h.metadataExtractor = NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor
}

// IsConfigured returns true if the handler has a non-default configuration
func (h *Handler) IsConfigured() bool {
	return h.config != nil && h.config != DefaultConfig()
}

// GetConfiguredUserAgent returns the configured user agent
func (h *Handler) GetConfiguredUserAgent() string {
	if h.config != nil && h.config.HTTP != nil {
		return h.config.HTTP.UserAgent
	}
	return "TuneIn-CDN-Benchmark/1.0"
}

// GetCurrentICYTitle returns the current ICY title from metadata
func (h *Handler) GetCurrentICYTitle() string {
	return h.icyTitle
}

// GetICYMetadataInterval returns the ICY metadata interval
func (h *Handler) GetICYMetadataInterval() int {
	return h.icyMetaInt
}

// HasICYMetadata returns true if the stream supports ICY metadata
func (h *Handler) HasICYMetadata() bool {
	return h.icyMetaInt > 0
}

// RefreshMetadata manually refreshes metadata from headers (useful for live streams)
func (h *Handler) RefreshMetadata() error {
	if !h.connected || h.response == nil {
		return common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Re-extract metadata from current response headers
	h.metadata = h.metadataExtractor.ExtractMetadata(h.response.Header, h.url)

	return nil
}

// GetStreamInfo returns detailed information about the current stream
func (h *Handler) GetStreamInfo() map[string]any {
	info := make(map[string]any)

	if h.metadata != nil {
		info["station"] = h.metadata.Station
		info["genre"] = h.metadata.Genre
		info["bitrate"] = h.metadata.Bitrate
		info["sample_rate"] = h.metadata.SampleRate
		info["channels"] = h.metadata.Channels
		info["codec"] = h.metadata.Codec
		info["format"] = h.metadata.Format
	}

	info["icy_metadata_interval"] = h.icyMetaInt
	info["current_title"] = h.icyTitle
	info["has_icy_metadata"] = h.HasICYMetadata()
	info["bytes_read"] = h.bytesRead
	info["connected"] = h.connected

	if h.stats != nil {
		info["bytes_received"] = h.stats.BytesReceived
		info["average_bitrate"] = h.stats.AverageBitrate
		info["connection_time"] = h.stats.ConnectionTime
		info["buffer_health"] = h.stats.BufferHealth
	}

	return info
}
