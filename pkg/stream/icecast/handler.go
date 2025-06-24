// pkg/stream/icecast/handler.go
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
)

// Handler implements StreamHandler for ICEcast streams
type Handler struct {
	client    *http.Client
	url       string
	metadata  *common.StreamMetadata
	stats     *common.StreamStats
	connected bool
	response  *http.Response
}

// NewHandler creates a new ICEcast stream handler
func NewHandler() *Handler {
	return &Handler{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		stats: &common.StreamStats{},
	}
}

// Type returns the stream type this handler supports
func (h *Handler) Type() common.StreamType {
	return common.StreamTypeICEcast
}

// CanHandle determines if this handler can process the given URL
func (h *Handler) CanHandle(ctx context.Context, url string) bool {
	if st := DetectFromURL(url); st == common.StreamTypeICEcast {
		return true
	}
	if st := DetectFromHeaders(ctx, h.client, url); st == common.StreamTypeICEcast {
		return true
	}
	// TODO: add additional ICEcast-specific detection logic
	return false
}

// Connect establishes connection to the ICEcast stream
func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return fmt.Errorf("already connected")
	}

	h.url = url
	startTime := time.Now()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "failed to create request", err)
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "audio/*")
	req.Header.Set("Icy-MetaData", "1") // Request ICEcast metadata

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
	h.stats.ConnectionTime = time.Since(startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime
	h.connected = true

	return nil
}

// GetMetadata retrieves stream metadata from ICEcast headers
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected || h.response == nil {
		return nil, fmt.Errorf("not connected")
	}

	if h.metadata == nil {
		headers := h.response.Header

		h.metadata = &common.StreamMetadata{
			URL:       h.url,
			Type:      common.StreamTypeICEcast,
			Headers:   make(map[string]string),
			Timestamp: time.Now(),
		}

		// Extract metadata from ICEcast headers
		if name := headers.Get("icy-name"); name != "" {
			h.metadata.Station = name
		}
		if genre := headers.Get("icy-genre"); genre != "" {
			h.metadata.Genre = genre
		}
		if desc := headers.Get("icy-description"); desc != "" {
			h.metadata.Title = desc
		}
		if bitrate := headers.Get("icy-br"); bitrate != "" {
			if br, err := strconv.Atoi(bitrate); err == nil {
				h.metadata.Bitrate = br
			}
		}
		if sampleRate := headers.Get("icy-sr"); sampleRate != "" {
			if sr, err := strconv.Atoi(sampleRate); err == nil {
				h.metadata.SampleRate = sr
			}
		}

		// Determine codec from content type
		contentType := headers.Get("Content-Type")
		if strings.Contains(contentType, "mpeg") {
			h.metadata.Codec = "mp3"
		} else if strings.Contains(contentType, "aac") {
			h.metadata.Codec = "aac"
		} else if strings.Contains(contentType, "ogg") {
			h.metadata.Codec = "ogg"
		} else {
			h.metadata.Codec = "unknown"
		}

		// Default audio properties if not specified in headers
		if h.metadata.SampleRate == 0 {
			h.metadata.SampleRate = 44100 // Common default
		}
		if h.metadata.Channels == 0 {
			h.metadata.Channels = 2 // Stereo default
		}

		// Store relevant headers
		relevantHeaders := []string{
			"content-type", "icy-name", "icy-genre", "icy-description",
			"icy-br", "icy-sr", "icy-channels", "server", "icy-url",
			"icy-pub", "icy-metaint",
		}
		for _, header := range relevantHeaders {
			if value := headers.Get(header); value != "" {
				h.metadata.Headers[header] = value
			}
		}
	}

	return h.metadata, nil
}

// ReadAudio reads audio data from the ICEcast stream
func (h *Handler) ReadAudio(ctx context.Context) (*common.AudioData, error) {
	if !h.connected || h.response == nil {
		return nil, fmt.Errorf("not connected")
	}

	// TODO: Implement actual audio decoding using LibAV
	// For now, simulate reading raw stream data
	buffer := make([]byte, 4096)

	// Set a read deadline based on context
	if deadline, ok := ctx.Deadline(); ok {
		if conn, ok := h.response.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(deadline)
		}
	}

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

	// Update statistics
	h.stats.BytesReceived += int64(n)

	// Calculate approximate bitrate based on received data
	if h.stats.BytesReceived > 0 {
		elapsed := time.Since(time.Now().Add(-time.Second)) // Simplified calculation
		if elapsed > 0 {
			h.stats.AverageBitrate = float64(h.stats.BytesReceived*8) / elapsed.Seconds() / 1000
		}
	}

	// Mock PCM conversion (replace with actual LibAV decoding)
	// This is a placeholder that simulates converting raw audio bytes to PCM samples
	pcmSamples := make([]float64, min(1024, n/2)) // Assume 16-bit samples

	for i := 0; i < len(pcmSamples) && i*2+1 < n; i++ {
		// Convert raw bytes to normalized float64 samples (-1.0 to 1.0)
		// Assuming little-endian 16-bit PCM
		sample := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		pcmSamples[i] = float64(sample) / 32768.0
	}

	// Get current metadata for the audio data
	metadata, err := h.GetMetadata()
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata: %w", err)
	}

	audioData := &common.AudioData{
		PCM:        pcmSamples,
		SampleRate: metadata.SampleRate,
		Channels:   metadata.Channels,
		Duration:   time.Duration(len(pcmSamples)) * time.Second / time.Duration(metadata.SampleRate),
		Timestamp:  time.Now(),
		Metadata:   metadata,
	}

	return audioData, nil
}

// GetStats returns current streaming statistics
func (h *Handler) GetStats() *common.StreamStats {
	// Update buffer health based on connection state
	if h.connected && h.response != nil {
		h.stats.BufferHealth = 1.0 // Simplified - actual implementation would check buffer levels
	} else {
		h.stats.BufferHealth = 0.0
	}

	return h.stats
}

// GetClient returns the HTTP Client
func (h *Handler) GetClient() *http.Client {
	return h.client
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

// Helper function for min calculation
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
