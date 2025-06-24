package hls

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Handler implements StreamHandler for HLS streams
type Handler struct {
	client            *http.Client
	url               string
	metadata          *common.StreamMetadata
	stats             *common.StreamStats
	connected         bool
	playlist          *M3U8Playlist
	parser            *Parser
	metadataExtractor *MetadataExtractor
	config            *Config
}

// NewHandler creates a new HLS stream handler with default configuration
func NewHandler() *Handler {
	return NewHandlerWithConfig(nil)
}

// NewHandlerWithConfig creates a new HLS stream handler with custom configuration
func NewHandlerWithConfig(config *Config) *Handler {
	if config == nil {
		config = DefaultConfig()
	}

	return &Handler{
		client: &http.Client{
			Timeout: time.Duration(config.Detection.TimeoutSeconds) * time.Second,
		},
		stats:             &common.StreamStats{},
		parser:            NewConfigurableParser(config.Parser).Parser,
		metadataExtractor: NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor,
		config:            config,
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

// Connect establishes connection to the HLS stream
func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return fmt.Errorf("already connected")
	}

	h.url = url
	startTime := time.Now()

	// Parse the M3U8 playlist using the configured parser and extractor
	playlist, err := h.parsePlaylist(ctx, url)
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

	// Extract metadata using the configured extractor
	h.metadata = h.metadataExtractor.ExtractMetadata(playlist, url)

	h.stats.ConnectionTime = time.Since(startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime
	h.connected = true

	return nil
}

// parsePlaylist parses the M3U8 playlist with current configuration
func (h *Handler) parsePlaylist(ctx context.Context, url string) (*M3U8Playlist, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain")

	resp, err := h.client.Do(req)
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

	// Parse the M3U8 content
	playlist, err := h.parser.ParseM3U8Content(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse M3U8: %w", err)
	}

	// Merge HTTP headers with playlist headers
	if playlist.Headers == nil {
		playlist.Headers = make(map[string]string)
	}
	maps.Copy(playlist.Headers, headers)

	return playlist, nil
}

// GetMetadata retrieves stream metadata
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected {
		return nil, fmt.Errorf("not connected")
	}

	// Return the metadata extracted during connection
	if h.metadata == nil {
		// Fallback metadata if parsing failed
		h.metadata = &common.StreamMetadata{
			URL:        h.url,
			Type:       common.StreamTypeHLS,
			Headers:    make(map[string]string),
			Timestamp:  time.Now(),
			Codec:      h.config.MetadataExtractor.DefaultValues["codec"].(string),
			SampleRate: h.config.MetadataExtractor.DefaultValues["sample_rate"].(int),
			Channels:   h.config.MetadataExtractor.DefaultValues["channels"].(int),
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
		SampleRate: h.metadata.SampleRate,
		Channels:   h.metadata.Channels,
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

	// Update timeout if changed
	if config.Detection.TimeoutSeconds > 0 {
		h.client.Timeout = time.Duration(config.Detection.TimeoutSeconds) * time.Second
	}

	// Recreate parser and metadata extractor with new config
	h.parser = NewConfigurableParser(config.Parser).Parser
	h.metadataExtractor = NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor
}

// RefreshPlaylist refreshes the playlist for live streams
func (h *Handler) RefreshPlaylist(ctx context.Context) error {
	if !h.connected {
		return fmt.Errorf("not connected")
	}

	// Only refresh for live streams
	if h.playlist != nil && !h.playlist.IsLive {
		return fmt.Errorf("not a live stream")
	}

	newPlaylist, err := h.parsePlaylist(ctx, h.url)
	if err != nil {
		return fmt.Errorf("failed to refresh playlist: %w", err)
	}

	// Update playlist and re-extract metadata
	h.playlist = newPlaylist
	h.metadata = h.metadataExtractor.ExtractMetadata(newPlaylist, h.url)

	return nil
}

// GetSegmentURLs returns URLs for all segments in the playlist
func (h *Handler) GetSegmentURLs() ([]string, error) {
	if h.playlist == nil {
		return nil, fmt.Errorf("no playlist available")
	}

	urls := make([]string, 0, len(h.playlist.Segments))
	for _, segment := range h.playlist.Segments {
		// Resolve relative URLs if needed
		segmentURL := h.resolveURL(segment.URI)
		urls = append(urls, segmentURL)
	}

	return urls, nil
}

// GetVariantURLs returns URLs for all variants in the playlist
func (h *Handler) GetVariantURLs() ([]string, error) {
	if h.playlist == nil {
		return nil, fmt.Errorf("no playlist available")
	}

	urls := make([]string, 0, len(h.playlist.Variants))
	for _, variant := range h.playlist.Variants {
		// Resolve relative URLs if needed
		variantURL := h.resolveURL(variant.URI)
		urls = append(urls, variantURL)
	}

	return urls, nil
}

// resolveURL resolves relative URLs against the base playlist URL
func (h *Handler) resolveURL(uri string) string {
	// If it's already an absolute URL, return as-is
	if strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://") {
		return uri
	}

	// Parse base URL
	baseURL, err := url.Parse(h.url)
	if err != nil {
		return uri // Return original if can't parse
	}

	// Parse relative URI
	relativeURL, err := url.Parse(uri)
	if err != nil {
		return uri // Return original if can't parse
	}

	// Resolve relative to base
	resolvedURL := baseURL.ResolveReference(relativeURL)
	return resolvedURL.String()
}

