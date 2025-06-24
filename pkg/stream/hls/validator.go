package hls

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Validator implements StreamValidator interface for HLS streams
type Validator struct {
	client *http.Client
	config *Config
}

// NewValidator creates a new HLS validator
func NewValidator() *Validator {
	return NewValidatorWithConfig(nil)
}

// NewValidatorWithConfig creates a new HLS validator with custom configuration
func NewValidatorWithConfig(config *Config) *Validator {
	if config == nil {
		config = DefaultConfig()
	}

	return &Validator{
		client: &http.Client{
			Timeout: time.Duration(config.Detection.TimeoutSeconds) * time.Second,
		},
		config: config,
	}
}

// ValidateURL validates if URL is accessible and valid for HLS
func (v *Validator) ValidateURL(ctx context.Context, streamURL string) error {
	// Parse URL
	parsedURL, err := url.Parse(streamURL)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeInvalidFormat, "invalid URL format", err)
	}

	// Check scheme
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeUnsupported, "unsupported URL scheme", nil)
	}

	// Check if URL looks like HLS
	if DetectFromURL(streamURL) != common.StreamTypeHLS {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeInvalidFormat, "URL does not appear to be HLS", nil)
	}

	// Perform HEAD request to check accessibility
	req, err := http.NewRequestWithContext(ctx, "HEAD", streamURL, nil)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeConnection, "failed to create request", err)
	}

	req.Header.Set("User-Agent", "TuneIn-CDN-Benchmark/1.0")
	req.Header.Set("Accept", "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain")

	resp, err := v.client.Do(req)
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeConnection, "connection failed", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode >= 400 {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeConnection, fmt.Sprintf("HTTP %d", resp.StatusCode), nil)
	}

	// Validate content type for HLS
	contentType := resp.Header.Get("Content-Type")
	if contentType != "" && !v.isValidHLSContentType(contentType) {
		return common.NewStreamError(common.StreamTypeHLS, streamURL,
			common.ErrCodeInvalidFormat, fmt.Sprintf("invalid content type: %s", contentType), nil)
	}

	return nil
}

// ValidateStream validates HLS stream content and format
func (v *Validator) ValidateStream(ctx context.Context, handler common.StreamHandler) error {
	// Ensure this is an HLS handler
	if handler.Type() != common.StreamTypeHLS {
		return common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeUnsupported, "handler is not for HLS streams", nil)
	}

	// Get metadata
	metadata, err := handler.GetMetadata()
	if err != nil {
		return common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeMetadata, "failed to get metadata", err)
	}

	// Validate essential HLS metadata
	if err := v.validateMetadata(metadata); err != nil {
		return err
	}

	// If this is an HLS handler, get the playlist for validation
	if hlsHandler, ok := handler.(*Handler); ok {
		playlist := hlsHandler.GetPlaylist()
		if playlist == nil {
			return common.NewStreamError(common.StreamTypeHLS, metadata.URL,
				common.ErrCodeInvalidFormat, "no playlist available", nil)
		}

		// Validate playlist structure
		if err := v.validatePlaylist(playlist, metadata.URL); err != nil {
			return err
		}

		// Validate playlist content
		if err := v.validatePlaylistContent(playlist, metadata.URL); err != nil {
			return err
		}
	}

	return nil
}

// ValidateAudio validates audio data quality for HLS
func (v *Validator) ValidateAudio(data *common.AudioData) error {
	if data == nil {
		return fmt.Errorf("audio data is nil")
	}

	if len(data.PCM) == 0 {
		return fmt.Errorf("empty PCM data")
	}

	if data.SampleRate <= 0 {
		return fmt.Errorf("invalid sample rate: %d", data.SampleRate)
	}

	if data.Channels <= 0 || data.Channels > 8 {
		return fmt.Errorf("invalid channel count: %d", data.Channels)
	}

	if data.Duration <= 0 {
		return fmt.Errorf("invalid duration: %v", data.Duration)
	}

	// HLS-specific audio validation
	if data.Metadata != nil && data.Metadata.Type == common.StreamTypeHLS {
		// Check if sample rate matches expected values for HLS
		validSampleRates := []int{22050, 44100, 48000}
		validRate := slices.Contains(validSampleRates, data.SampleRate)
		if !validRate {
			return fmt.Errorf("unusual sample rate for HLS: %d", data.SampleRate)
		}

		// Check for reasonable audio levels (not all silence)
		if v.isAllSilence(data.PCM) {
			return fmt.Errorf("audio data contains only silence")
		}
	}

	return nil
}

// validateMetadata validates HLS metadata
func (v *Validator) validateMetadata(metadata *common.StreamMetadata) error {
	if metadata.Type != common.StreamTypeHLS {
		return common.NewStreamError(common.StreamTypeHLS, metadata.URL,
			common.ErrCodeInvalidFormat, "stream type is not HLS", nil)
	}

	if metadata.Codec == "" {
		return common.NewStreamError(common.StreamTypeHLS, metadata.URL,
			common.ErrCodeInvalidFormat, "codec not detected", nil)
	}

	// Validate HLS-appropriate codecs
	validCodecs := []string{"aac", "mp3", "ac3"}
	validCodec := false
	for _, codec := range validCodecs {
		if strings.EqualFold(metadata.Codec, codec) {
			validCodec = true
			break
		}
	}
	if !validCodec {
		return common.NewStreamError(common.StreamTypeHLS, metadata.URL,
			common.ErrCodeInvalidFormat, fmt.Sprintf("unusual codec for HLS: %s", metadata.Codec), nil)
	}

	return nil
}

// validatePlaylist validates M3U8 playlist structure
func (v *Validator) validatePlaylist(playlist *M3U8Playlist, url string) error {
	if !playlist.IsValid {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeInvalidFormat, "invalid M3U8 playlist", nil)
	}

	// Check playlist has content
	if len(playlist.Segments) == 0 && len(playlist.Variants) == 0 {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeInvalidFormat, "empty playlist - no segments or variants", nil)
	}

	// Validate playlist version
	if playlist.Version > 0 && (playlist.Version < 1 || playlist.Version > 10) {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeInvalidFormat, fmt.Sprintf("invalid playlist version: %d", playlist.Version), nil)
	}

	// For media playlists, validate target duration
	if !playlist.IsMaster && playlist.TargetDuration <= 0 {
		return common.NewStreamError(common.StreamTypeHLS, url,
			common.ErrCodeInvalidFormat, "media playlist missing target duration", nil)
	}

	return nil
}

// validatePlaylistContent validates playlist content quality
func (v *Validator) validatePlaylistContent(playlist *M3U8Playlist, streamURL string) error {
	// Validate segments
	for i, segment := range playlist.Segments {
		if segment.URI == "" {
			return common.NewStreamError(common.StreamTypeHLS, streamURL,
				common.ErrCodeInvalidFormat, fmt.Sprintf("segment %d missing URI", i), nil)
		}

		if segment.Duration < 0 {
			return common.NewStreamError(common.StreamTypeHLS, streamURL,
				common.ErrCodeInvalidFormat, fmt.Sprintf("segment %d has negative duration", i), nil)
		}

		// Validate URI format if configured
		if v.config.Parser.ValidateURIs {
			if _, err := url.Parse(segment.URI); err != nil {
				return common.NewStreamError(common.StreamTypeHLS, streamURL,
					common.ErrCodeInvalidFormat, fmt.Sprintf("segment %d has invalid URI", i), err)
			}
		}
	}

	// Validate variants
	for i, variant := range playlist.Variants {
		if variant.URI == "" {
			return common.NewStreamError(common.StreamTypeHLS, streamURL,
				common.ErrCodeInvalidFormat, fmt.Sprintf("variant %d missing URI", i), nil)
		}

		if variant.Bandwidth <= 0 {
			return common.NewStreamError(common.StreamTypeHLS, streamURL,
				common.ErrCodeInvalidFormat, fmt.Sprintf("variant %d has invalid bandwidth", i), nil)
		}

		// Validate URI format if configured
		if v.config.Parser.ValidateURIs {
			if _, err := url.Parse(variant.URI); err != nil {
				return common.NewStreamError(common.StreamTypeHLS, streamURL,
					common.ErrCodeInvalidFormat, fmt.Sprintf("variant %d has invalid URI", i), err)
			}
		}
	}

	// Check for reasonable bandwidth progression in variants
	if len(playlist.Variants) > 1 {
		if err := v.validateBandwidthProgression(playlist.Variants, streamURL); err != nil {
			return err
		}
	}

	return nil
}

// validateBandwidthProgression checks if variant bandwidths make sense
func (v *Validator) validateBandwidthProgression(variants []M3U8Variant, url string) error {
	if len(variants) < 2 {
		return nil
	}

	// Check for duplicate bandwidths
	bandwidthMap := make(map[int]bool)
	for _, variant := range variants {
		if bandwidthMap[variant.Bandwidth] {
			return common.NewStreamError(common.StreamTypeHLS, url,
				common.ErrCodeInvalidFormat, fmt.Sprintf("duplicate bandwidth: %d", variant.Bandwidth), nil)
		}
		bandwidthMap[variant.Bandwidth] = true
	}

	// Check for reasonable progression (each step should be meaningful)
	bandwidths := make([]int, len(variants))
	for i, variant := range variants {
		bandwidths[i] = variant.Bandwidth
	}

	// Simple bubble sort
	for i := 0; i < len(bandwidths)-1; i++ {
		for j := 0; j < len(bandwidths)-i-1; j++ {
			if bandwidths[j] > bandwidths[j+1] {
				bandwidths[j], bandwidths[j+1] = bandwidths[j+1], bandwidths[j]
			}
		}
	}

	// Check progression ratios
	for i := 1; i < len(bandwidths); i++ {
		ratio := float64(bandwidths[i]) / float64(bandwidths[i-1])
		if ratio < 1.1 || ratio > 10.0 {
			return common.NewStreamError(common.StreamTypeHLS, url,
				common.ErrCodeInvalidFormat, "unreasonable bandwidth progression in variants", nil)
		}
	}

	return nil
}

// isValidHLSContentType checks if content type is valid for HLS
func (v *Validator) isValidHLSContentType(contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))

	// Remove parameters
	if idx := strings.Index(contentType, ";"); idx != -1 {
		contentType = contentType[:idx]
	}

	validTypes := []string{
		"application/vnd.apple.mpegurl",
		"application/x-mpegurl",
		"text/plain",
	}

	return slices.Contains(validTypes, contentType)
}

// isAllSilence checks if audio data is all silence
func (v *Validator) isAllSilence(pcm []float64) bool {
	threshold := 0.001 // Very small threshold for silence detection

	for _, sample := range pcm {
		if sample > threshold || sample < -threshold {
			return false
		}
	}

	return true
}
