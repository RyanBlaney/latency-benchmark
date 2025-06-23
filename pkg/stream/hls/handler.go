package hls

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Handler implements StreamHandler for HLS streams
type Handler struct {
	client    *http.Client
	url       string
	metadata  *common.StreamMetadata
	stats     *common.StreamStats
	connected bool
}

// NewHandler creates a new HLS stream handler
func NewHandler() *Handler {
	return &Handler{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		stats: &common.StreamStats{},
	}
}

// Type returns the stream type for this handler
func (h *Handler) Type() common.StreamType {
	return common.StreamTypeHLS
}

// CanHandle determines if this handler can process the given URL
func (h *Handler) CanHandle(ctx context.Context, url string) bool {
	if st := DetectFromURL(url); st == common.StreamTypeHLS {
		return true
	}

	if st := DetectFromHeaders(ctx, h.client, url); st == common.StreamTypeHLS {
		return true
	}

	// TODO: add m3u8 parsing

	return false
}

func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return fmt.Errorf("already connected")
	}

	h.url = url
	startTime := time.Now()

	// TODO: Implement actual HLS playlist parsing and validation
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeConnection, "failed to create request", err)
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "application/vnd.apple.mpegurl,application/x-mpegurl")

	resp, err := h.client.Do(req)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeConnection, "connection failed", err)
	}
	defer resp.Body.Close()

	h.stats.ConnectionTime = time.Since(startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime // TODO: more advanced end-to-end latency
	h.connected = true

	return nil
}

// GetMetadata retrieves stream metadata
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected {
		return nil, fmt.Errorf("not connected")
	}

	if h.metadata == nil {
		h.metadata = &common.StreamMetadata{
			URL:       h.url,
			Type:      common.StreamTypeHLS,
			Headers:   make(map[string]string),
			Timestamp: time.Now(),
			// TODO: Parse actual metadata from m3u8 playlist
			Codec:      "aac", // Default assumption
			SampleRate: 44100,
			Channels:   2,
		}
	}

	return h.metadata, nil
}

// ReadAudio reads audio data from the HLS stream
func (h *Handler) ReadAudio(ctx context.Context) (*common.AudioData, error) {
	if !h.connected {
		return nil, fmt.Errorf("not connected")
	}

	// TODO: implement actual HLS segment downloading and audio extraction
	// This is a placeholder that simulates audio data
	audioData := &common.AudioData{
		PCM:        make([]float64, 1024), // Mock PCM data
		SampleRate: 44100,
		Channels:   2,
		Duration:   time.Second,
		Timestamp:  time.Now(),
		Metadata:   h.metadata,
	}

	h.stats.SegmentsReceived++
	h.stats.BytesReceived += 1024 * 4 // Simulate bytes

	return audioData, nil
}

// GetStats returns current streaming statistics
func (h *Handler) GetStats() *common.StreamStats {
	return h.stats
}

// Close closes the stream connection
func (h *Handler) Close() error {
	h.connected = false
	return nil
}
