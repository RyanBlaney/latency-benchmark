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
	playlist  *M3U8Playlist
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

	// M3U8 content parsing
	return IsValidHLSContent(ctx, h.client, url)
}

func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return fmt.Errorf("already connected")
	}

	h.url = url
	startTime := time.Now()

	// Parse the M3U8 playlist
	playlist, err := DetectFromM3U8Content(ctx, h.client, url)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeConnection, "failed to parse M3U8 playlist", err)
	}

	if !playlist.IsValid {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeInvalidFormat, "invalid M3U8 playlist format", nil)
	}

	// Store the parsed playlist
	h.playlist = playlist

	// Extract metadata from the parsed playlist
	h.metadata = playlist.Metadata

	h.stats.ConnectionTime = time.Since(startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime
	h.connected = true

	return nil
}

// GetMetadata retrieves stream metadata
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected {
		return nil, fmt.Errorf("not connected")
	}

	// Return the metadata extracted from playlist parsing
	if h.metadata == nil {
		// Fallback metadata if parsing failed
		h.metadata = &common.StreamMetadata{
			URL:        h.url,
			Type:       common.StreamTypeHLS,
			Headers:    make(map[string]string),
			Timestamp:  time.Now(),
			Codec:      "aac",
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

// GetPlaylist returns the parsed playlist
func (h *Handler) GetPlaylist() *M3U8Playlist {
	return h.playlist
}

// GetClient returns the HTTP Client
func (h *Handler) GetClient() *http.Client {
	return h.client
}
