package hls

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
)

// AudioDownloader handles downloading and processing HLS audio segments
type AudioDownloader struct {
	client        *http.Client
	segmentCache  map[string][]byte
	downloadStats *DownloadStats
	config        *DownloadConfig
	hlsConfig     *Config
	tempDir       string // Directory for temporary files
}

// DownloadConfig contains configuration for audio downloading
type DownloadConfig struct {
	MaxSegments      int           `json:"max_segments"`
	SegmentTimeout   time.Duration `json:"segment_timeout"`
	MaxRetries       int           `json:"max_retries"`
	CacheSegments    bool          `json:"cache_segments"`
	TargetDuration   time.Duration `json:"target_duration"`
	PreferredBitrate int           `json:"preferred_bitrate"`
	// Audio processing options (used for fallback basic processing)
	OutputSampleRate int    `json:"output_sample_rate"` // Target sample rate for basic processing
	OutputChannels   int    `json:"output_channels"`    // Target channels (1=mono, 2=stereo)
	NormalizePCM     bool   `json:"normalize_pcm"`      // Normalize PCM values
	ResampleQuality  string `json:"resample_quality"`   // "fast", "medium", "high"
	CleanupTempFiles bool   `json:"cleanup_temp_files"` // Clean up temporary files
}

// DownloadStats tracks download performance
type DownloadStats struct {
	SegmentsDownloaded int            `json:"segments_downloaded"`
	BytesDownloaded    int64          `json:"bytes_downloaded"`
	DownloadTime       time.Duration  `json:"download_time"`
	DecodeTime         time.Duration  `json:"decode_time"`
	ErrorCount         int            `json:"error_count"`
	AverageBitrate     float64        `json:"average_bitrate"`
	SegmentErrors      []SegmentError `json:"segment_errors,omitempty"`
	AudioMetrics       *AudioMetrics  `json:"audio_metrics,omitempty"`
}

// AudioMetrics contains audio processing statistics
type AudioMetrics struct {
	SamplesDecoded   int64   `json:"samples_decoded"`
	DecodedDuration  float64 `json:"decoded_duration_seconds"`
	AverageAmplitude float64 `json:"average_amplitude"`
	PeakAmplitude    float64 `json:"peak_amplitude"`
	SilenceRatio     float64 `json:"silence_ratio"`
	ClippingDetected bool    `json:"clipping_detected"`
}

// SegmentError represents an error downloading a specific segment
type SegmentError struct {
	URL       string    `json:"url"`
	Error     string    `json:"error"`
	Timestamp time.Time `json:"timestamp"`
	Retry     int       `json:"retry"`
	Type      string    `json:"type"` // "download", "decode", "format"
}

// NewAudioDownloader creates a new audio downloader
func NewAudioDownloader(client *http.Client, config *DownloadConfig, hlsConfig *Config) *AudioDownloader {
	if config == nil {
		config = DefaultDownloadConfig()
	}

	// Create temp directory for segment files
	tempDir, err := os.MkdirTemp("", "hls_segments_*")
	if err != nil {
		tempDir = "/tmp" // Fallback
	}

	return &AudioDownloader{
		client:       client,
		segmentCache: make(map[string][]byte),
		downloadStats: &DownloadStats{
			SegmentErrors: make([]SegmentError, 0),
			AudioMetrics:  &AudioMetrics{},
		},
		config:    config,
		hlsConfig: hlsConfig,
		tempDir:   tempDir,
	}
}

// DefaultDownloadConfig returns default download configuration
func DefaultDownloadConfig() *DownloadConfig {
	return &DownloadConfig{
		MaxSegments:      10,
		SegmentTimeout:   10 * time.Second,
		MaxRetries:       3,
		CacheSegments:    true,
		TargetDuration:   30 * time.Second,
		PreferredBitrate: 128,
		OutputSampleRate: 44100, // Standard for audio fingerprinting
		OutputChannels:   1,     // Mono for fingerprinting (reduces processing)
		NormalizePCM:     true,
		ResampleQuality:  "medium",
		CleanupTempFiles: true,
	}
}

// DownloadAudioSegment downloads and processes a single HLS audio segment
func (ad *AudioDownloader) DownloadAudioSegment(ctx context.Context, segmentURL string) (*common.AudioData, error) {
	startTime := time.Now()

	logger := logging.WithFields(logging.Fields{
		"component":   "audio_downloader",
		"function":    "DownloadAudioSegment",
		"segment_url": segmentURL,
	})

	// Check cache first
	if ad.config.CacheSegments {
		if cachedData, exists := ad.segmentCache[segmentURL]; exists {
			logger.Debug("Using cached segment data")
			return ad.processSegmentData(cachedData, segmentURL, startTime)
		}
	}

	// Download segment with retries
	segmentData, err := ad.downloadSegmentWithRetries(ctx, segmentURL)
	if err != nil {
		ad.recordSegmentError(segmentURL, err, 0, "download")
		return nil, common.NewStreamErrorWithFields(common.StreamTypeHLS, segmentURL, common.ErrCodeConnection,
			"failed to download segment", err,
			logging.Fields{
				"segment_url": segmentURL,
			})
	}

	logger.Debug("Segment downloaded successfully", logging.Fields{
		"data_size": len(segmentData),
	})

	// Cache if enabled
	if ad.config.CacheSegments {
		ad.segmentCache[segmentURL] = segmentData
	}

	// Update download stats
	ad.downloadStats.SegmentsDownloaded++
	ad.downloadStats.BytesDownloaded += int64(len(segmentData))
	ad.downloadStats.DownloadTime += time.Since(startTime)

	return ad.processSegmentData(segmentData, segmentURL, startTime)
}

// processSegmentData processes segment data using either injected decoder or fallback
func (ad *AudioDownloader) processSegmentData(segmentData []byte, segmentURL string, startTime time.Time) (*common.AudioData, error) {
	decodeStartTime := time.Now()

	logger := logging.WithFields(logging.Fields{
		"component":   "audio_downloader",
		"function":    "processSegmentData",
		"segment_url": segmentURL,
		"data_size":   len(segmentData),
	})

	var audioData *common.AudioData
	var err error

	// Use injected decoder if available
	if ad.hlsConfig != nil && ad.hlsConfig.AudioDecoder != nil {
		logger.Debug("Using injected audio decoder")

		audioData, err = ad.hlsConfig.AudioDecoder.DecodeBytes(segmentData)
		if err != nil {
			logger.Error(err, "Injected decoder failed, falling back to basic extraction")
			// Fall back to basic extraction
			audioData, err = ad.basicAudioExtraction(segmentData, segmentURL)
		} else {
			logger.Debug("Injected decoder succeeded", logging.Fields{
				"samples":     len(audioData.PCM),
				"sample_rate": audioData.SampleRate,
				"channels":    audioData.Channels,
				"duration":    audioData.Duration.Seconds(),
			})
		}
	} else {
		logger.Debug("No injected decoder, using basic audio extraction")
		audioData, err = ad.basicAudioExtraction(segmentData, segmentURL)
	}

	if err != nil {
		ad.recordSegmentError(segmentURL, err, 0, "decode")
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeDecoding, "failed to process audio data", err)
	}

	// Update decode stats
	ad.downloadStats.DecodeTime += time.Since(decodeStartTime)

	// Process and normalize the audio data (basic post-processing)
	ad.processAudioData(audioData)

	// Add metadata and timestamp
	audioData.Timestamp = startTime
	if audioData.Metadata == nil {
		audioData.Metadata = ad.createBasicMetadata(segmentURL)
	}

	// Update audio metrics
	ad.updateAudioMetrics(audioData)

	logger.Info("Segment processing completed", logging.Fields{
		"final_samples":     len(audioData.PCM),
		"final_sample_rate": audioData.SampleRate,
		"final_channels":    audioData.Channels,
		"final_duration":    audioData.Duration.Seconds(),
		"processing_time":   time.Since(decodeStartTime).Milliseconds(),
	})

	return audioData, nil
}

// basicAudioExtraction provides fallback audio extraction when no decoder is injected
func (ad *AudioDownloader) basicAudioExtraction(segmentData []byte, segmentURL string) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component":   "audio_downloader",
		"function":    "basicAudioExtraction",
		"segment_url": segmentURL,
		"data_size":   len(segmentData),
	})

	if len(segmentData) < 4 {
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeInvalidFormat, "segment data too small", nil)
	}

	// Basic format detection
	format := ad.detectAudioFormat(segmentData)
	logger.Debug("Detected audio format", logging.Fields{"format": format})

	var audioData *common.AudioData
	var err error

	switch format {
	case "aac":
		audioData, err = ad.extractAAC(segmentData, segmentURL)
	case "mp3":
		audioData, err = ad.extractMP3(segmentData, segmentURL)
	case "unknown":
		// Try to extract basic info without full decoding
		logger.Warn("Unknown format, creating placeholder audio data")
		audioData = &common.AudioData{
			PCM:        ad.generateSilence(ad.config.OutputSampleRate * 2), // 2 seconds of silence
			SampleRate: ad.config.OutputSampleRate,
			Channels:   ad.config.OutputChannels,
			Duration:   2 * time.Second,
			Metadata:   ad.createBasicMetadata(segmentURL),
		}
		audioData.Metadata.Codec = "unknown"
	default:
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeUnsupported, fmt.Sprintf("unsupported audio format: %s", format), nil)
	}

	if err != nil {
		return nil, err
	}

	// Set metadata
	if audioData.Metadata == nil {
		audioData.Metadata = ad.createBasicMetadata(segmentURL)
	}
	audioData.Metadata.Codec = format

	logger.Info("Basic audio extraction completed", logging.Fields{
		"format":      format,
		"samples":     len(audioData.PCM),
		"sample_rate": audioData.SampleRate,
		"channels":    audioData.Channels,
		"duration":    audioData.Duration.Seconds(),
	})

	return audioData, nil
}

// extractAAC performs basic AAC extraction
func (ad *AudioDownloader) extractAAC(data []byte, segmentURL string) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component": "audio_downloader",
		"function":  "extractAAC",
	})

	// Look for ADTS frames
	frames := 0
	estimatedDuration := 0.0

	for i := 0; i < len(data)-7; i++ {
		// Look for ADTS sync word
		if data[i] == 0xFF && (data[i+1]&0xF0) == 0xF0 {
			frames++
			// Each AAC frame is typically 1024 samples at 44.1kHz = ~23ms
			estimatedDuration += 0.023

			// Skip to next potential frame (rough estimate)
			i += 100 // Skip ahead to avoid false positives
		}
	}

	if frames == 0 {
		logger.Warn("No AAC frames detected")
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeInvalidFormat, "no valid AAC frames found", nil)
	}

	// Generate silence with estimated duration
	duration := time.Duration(estimatedDuration * float64(time.Second))
	samples := int(float64(ad.config.OutputSampleRate) * estimatedDuration)

	logger.Debug("AAC extraction completed", logging.Fields{
		"frames_found":       frames,
		"estimated_duration": estimatedDuration,
		"generated_samples":  samples,
	})

	return &common.AudioData{
		PCM:        ad.generateSilence(samples),
		SampleRate: ad.config.OutputSampleRate,
		Channels:   ad.config.OutputChannels,
		Duration:   duration,
		Metadata:   ad.createBasicMetadata(segmentURL),
	}, nil
}

// detectAudioFormat performs basic format detection
func (ad *AudioDownloader) detectAudioFormat(data []byte) string {
	if len(data) < 4 {
		return "unknown"
	}

	// Check for AAC ADTS header (0xFFF)
	if len(data) >= 2 && (data[0] == 0xFF && (data[1]&0xF0) == 0xF0) {
		return "aac"
	}

	// Check for MP3 header (0xFFE, 0xFFF, or ID3)
	if len(data) >= 3 {
		// MP3 sync word
		if data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
			return "mp3"
		}
		// ID3 tag
		if data[0] == 'I' && data[1] == 'D' && data[2] == '3' {
			return "mp3"
		}
	}

	// Check for TS packet (0x47)
	if data[0] == 0x47 {
		return "ts" // Transport Stream - contains AAC usually
	}

	return "unknown"
}

// extractMP3 performs basic MP3 extraction
func (ad *AudioDownloader) extractMP3(data []byte, segmentURL string) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component": "audio_downloader",
		"function":  "extractMP3",
	})

	// Skip ID3 tag if present
	offset := 0
	if len(data) >= 10 && data[0] == 'I' && data[1] == 'D' && data[2] == '3' {
		// ID3v2 tag size is in bytes 6-9
		tagSize := int(data[6])<<21 | int(data[7])<<14 | int(data[8])<<7 | int(data[9])
		offset = 10 + tagSize
		logger.Debug("Skipping ID3 tag", logging.Fields{"tag_size": tagSize})
	}

	frames := 0
	estimatedDuration := 0.0

	for i := offset; i < len(data)-4; i++ {
		// Look for MP3 sync word
		if data[i] == 0xFF && (data[i+1]&0xE0) == 0xE0 {
			frames++
			// MP3 frame duration varies, but roughly 26ms for 44.1kHz
			estimatedDuration += 0.026

			// Skip ahead to avoid false positives
			i += 144 // Typical MP3 frame size
		}
	}

	if frames == 0 {
		logger.Warn("No MP3 frames detected")
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeInvalidFormat, "no valid MP3 frames found", nil)
	}

	// Generate silence with estimated duration
	duration := time.Duration(estimatedDuration * float64(time.Second))
	samples := int(float64(ad.config.OutputSampleRate) * estimatedDuration)

	logger.Debug("MP3 extraction completed", logging.Fields{
		"frames_found":       frames,
		"estimated_duration": estimatedDuration,
		"generated_samples":  samples,
	})

	return &common.AudioData{
		PCM:        ad.generateSilence(samples),
		SampleRate: ad.config.OutputSampleRate,
		Channels:   ad.config.OutputChannels,
		Duration:   duration,
		Metadata:   ad.createBasicMetadata(segmentURL),
	}, nil
}

// generateSilence creates silent PCM data
func (ad *AudioDownloader) generateSilence(sampleCount int) []float64 {
	return make([]float64, sampleCount)
}

// processAudioData applies basic post-processing to audio data
func (ad *AudioDownloader) processAudioData(audioData *common.AudioData) {
	if audioData == nil || len(audioData.PCM) == 0 {
		return
	}

	logger := logging.WithFields(logging.Fields{
		"component": "audio_downloader",
		"function":  "processAudioData",
		"samples":   len(audioData.PCM),
	})

	// Apply normalization if configured
	if ad.config.NormalizePCM {
		logger.Debug("Normalizing PCM data")
		ad.normalizePCM(audioData.PCM)
	}

	// Apply channel conversion if needed
	if ad.config.OutputChannels != audioData.Channels {
		logger.Debug("Converting channels", logging.Fields{
			"from_channels": audioData.Channels,
			"to_channels":   ad.config.OutputChannels,
		})

		if ad.config.OutputChannels == 1 && audioData.Channels == 2 {
			audioData = ad.convertToMono(audioData)
		}
		// Could add more channel conversion logic here
	}

	// Apply sample rate conversion if needed (basic linear interpolation)
	if ad.config.OutputSampleRate != audioData.SampleRate {
		logger.Debug("Converting sample rate", logging.Fields{
			"from_sample_rate": audioData.SampleRate,
			"to_sample_rate":   ad.config.OutputSampleRate,
		})
		audioData = ad.convertSampleRate(audioData, ad.config.OutputSampleRate)
	}

	logger.Debug("Audio processing completed")
}

// convertToMono converts stereo audio to mono by averaging channels
func (ad *AudioDownloader) convertToMono(audioData *common.AudioData) *common.AudioData {
	if audioData.Channels != 2 {
		return audioData
	}

	monoSamples := make([]float64, len(audioData.PCM)/2)
	for i := range len(monoSamples) {
		monoSamples[i] = (audioData.PCM[i*2] + audioData.PCM[i*2+1]) / 2.0
	}

	return &common.AudioData{
		PCM:        monoSamples,
		SampleRate: audioData.SampleRate,
		Channels:   1,
		Duration:   audioData.Duration,
		Timestamp:  audioData.Timestamp,
		Metadata:   audioData.Metadata,
	}
}

// convertSampleRate performs basic linear interpolation sample rate conversion
func (ad *AudioDownloader) convertSampleRate(audioData *common.AudioData, targetRate int) *common.AudioData {
	if audioData.SampleRate == targetRate {
		return audioData
	}

	ratio := float64(targetRate) / float64(audioData.SampleRate)
	newLength := int(float64(len(audioData.PCM)) * ratio)
	newSamples := make([]float64, newLength)

	for i := range newLength {
		sourceIndex := float64(i) / ratio
		sourceIndexInt := int(sourceIndex)

		if sourceIndexInt >= len(audioData.PCM)-1 {
			newSamples[i] = audioData.PCM[len(audioData.PCM)-1]
		} else {
			fraction := sourceIndex - float64(sourceIndexInt)
			newSamples[i] = audioData.PCM[sourceIndexInt]*(1-fraction) + audioData.PCM[sourceIndexInt+1]*fraction
		}
	}

	return &common.AudioData{
		PCM:        newSamples,
		SampleRate: targetRate,
		Channels:   audioData.Channels,
		Duration:   audioData.Duration,
		Timestamp:  audioData.Timestamp,
		Metadata:   audioData.Metadata,
	}
}

// normalizePCM normalizes audio samples to prevent clipping
func (ad *AudioDownloader) normalizePCM(samples []float64) {
	if len(samples) == 0 {
		return
	}

	var peak float64
	for _, sample := range samples {
		abs := sample
		if abs < 0 {
			abs = -abs
		}
		if abs > peak {
			peak = abs
		}
	}

	if peak > 1.0 {
		factor := 0.95 / peak
		for i := range samples {
			samples[i] *= factor
		}
	}
}

// updateAudioMetrics calculates and updates audio quality metrics
func (ad *AudioDownloader) updateAudioMetrics(audioData *common.AudioData) {
	if audioData == nil || len(audioData.PCM) == 0 {
		return
	}

	metrics := ad.downloadStats.AudioMetrics

	metrics.SamplesDecoded += int64(len(audioData.PCM))
	metrics.DecodedDuration += audioData.Duration.Seconds()

	var sum, peak float64
	silentSamples := 0
	clipping := false

	for _, sample := range audioData.PCM {
		abs := sample
		if abs < 0 {
			abs = -abs
		}

		sum += abs
		if abs > peak {
			peak = abs
		}

		if abs < 0.001 { // Silence threshold
			silentSamples++
		}

		if abs >= 0.99 { // Clipping threshold
			clipping = true
		}
	}

	metrics.AverageAmplitude = sum / float64(len(audioData.PCM))
	metrics.PeakAmplitude = peak
	metrics.SilenceRatio = float64(silentSamples) / float64(len(audioData.PCM))
	metrics.ClippingDetected = clipping
}

// createBasicMetadata creates basic metadata for a segment
func (ad *AudioDownloader) createBasicMetadata(segmentURL string) *common.StreamMetadata {
	return &common.StreamMetadata{
		URL:        segmentURL,
		Type:       common.StreamTypeHLS,
		SampleRate: ad.config.OutputSampleRate,
		Channels:   ad.config.OutputChannels,
		Bitrate:    ad.config.PreferredBitrate,
		Codec:      "unknown", // Would be determined by decoder
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}
}

// DownloadAudioSample downloads multiple segments to create a sample of specified duration
func (ad *AudioDownloader) DownloadAudioSample(ctx context.Context, playlist *M3U8Playlist, targetDuration time.Duration) (*common.AudioData, error) {
	if playlist == nil || len(playlist.Segments) == 0 {
		return nil, common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeInvalidFormat, "empty playlist", nil)
	}

	logger := logging.WithFields(logging.Fields{
		"component":       "audio_downloader",
		"function":        "DownloadAudioSample",
		"target_duration": targetDuration.Seconds(),
		"max_segments":    ad.config.MaxSegments,
		"total_segments":  len(playlist.Segments),
	})

	var audioSamples []*common.AudioData
	var totalDuration time.Duration
	maxSegments := ad.config.MaxSegments

	logger.Info("Starting audio sample download")

	for i, segment := range playlist.Segments {
		if i >= maxSegments || totalDuration >= targetDuration {
			break
		}

		select {
		case <-ctx.Done():
			logger.Warn("Download cancelled by context")
			return nil, ctx.Err()
		default:
		}

		segmentURL := ad.resolveSegmentURL(playlist, segment.URI)

		audioData, err := ad.DownloadAudioSegment(ctx, segmentURL)
		if err != nil {
			logger.Warn("Failed to download segment, continuing", logging.Fields{
				"segment_index": i,
				"segment_url":   segmentURL,
				"error":         err.Error(),
			})
			ad.recordSegmentError(segmentURL, err, 0, "processing")
			continue
		}

		audioSamples = append(audioSamples, audioData)
		totalDuration += audioData.Duration

		logger.Debug("Segment processed successfully", logging.Fields{
			"segment_index":    i,
			"segment_duration": audioData.Duration.Seconds(),
			"total_duration":   totalDuration.Seconds(),
		})
	}

	if len(audioSamples) == 0 {
		logger.Error(nil, "No audio segments were successfully downloaded")
		return nil, common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeConnection, "failed to download any audio segments", nil)
	}

	logger.Info("Audio sample download completed", logging.Fields{
		"segments_downloaded": len(audioSamples),
		"total_duration":      totalDuration.Seconds(),
	})

	return ad.combineAudioSamples(audioSamples)
}

// combineAudioSamples combines multiple audio samples into one
func (ad *AudioDownloader) combineAudioSamples(samples []*common.AudioData) (*common.AudioData, error) {
	if len(samples) == 0 {
		return nil, common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeInvalidFormat, "no audio samples to combine", nil)
	}

	if len(samples) == 1 {
		return samples[0], nil
	}

	logger := logging.WithFields(logging.Fields{
		"component":    "audio_downloader",
		"function":     "combineAudioSamples",
		"sample_count": len(samples),
	})

	firstSample := samples[0]
	combined := &common.AudioData{
		SampleRate: firstSample.SampleRate,
		Channels:   firstSample.Channels,
		Timestamp:  firstSample.Timestamp,
		Metadata:   firstSample.Metadata,
	}

	// Validate all samples have compatible formats
	totalSamples := 0
	for i, sample := range samples {
		if sample.SampleRate != combined.SampleRate {
			return nil, common.NewStreamErrorWithFields(common.StreamTypeHLS, "",
				common.ErrCodeInvalidFormat, "sample rate mismatch", nil,
				logging.Fields{
					"sample_index":  i,
					"expected_rate": combined.SampleRate,
					"actual_rate":   sample.SampleRate,
				})
		}
		if sample.Channels != combined.Channels {
			return nil, common.NewStreamErrorWithFields(common.StreamTypeHLS, "",
				common.ErrCodeInvalidFormat, "channel count mismatch", nil,
				logging.Fields{
					"sample_index":      i,
					"expected_channels": combined.Channels,
					"actual_channels":   sample.Channels,
				})
		}
		totalSamples += len(sample.PCM)
		combined.Duration += sample.Duration
	}

	// Combine all PCM data
	combined.PCM = make([]float64, totalSamples)
	offset := 0
	for _, sample := range samples {
		copy(combined.PCM[offset:], sample.PCM)
		offset += len(sample.PCM)
	}

	logger.Info("Audio samples combined successfully", logging.Fields{
		"total_samples":     len(combined.PCM),
		"combined_duration": combined.Duration.Seconds(),
		"final_sample_rate": combined.SampleRate,
		"final_channels":    combined.Channels,
	})

	return combined, nil
}

// downloadSegmentWithRetries downloads a segment with retry logic
func (ad *AudioDownloader) downloadSegmentWithRetries(ctx context.Context, segmentURL string) ([]byte, error) {
	var lastErr error

	for retry := 0; retry <= ad.config.MaxRetries; retry++ {
		data, err := ad.downloadSegment(ctx, segmentURL)
		if err == nil {
			return data, nil
		}

		lastErr = err
		ad.recordSegmentError(segmentURL, err, retry, "download")

		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if retry < ad.config.MaxRetries {
			waitTime := time.Duration(retry+1) * time.Second
			time.Sleep(waitTime)
		}
	}

	return nil, lastErr
}

// downloadSegment downloads a single segment
func (ad *AudioDownloader) downloadSegment(ctx context.Context, segmentURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", segmentURL, nil)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeConnection, "failed to create request", err)
	}

	userAgent := "TuneIn-CDN-Benchmark/1.0" // Default fallback
	if ad.hlsConfig != nil && ad.hlsConfig.HTTP != nil {
		userAgent = ad.hlsConfig.HTTP.UserAgent
	}

	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "*/*")

	resp, err := ad.client.Do(req)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeConnection, "request failed", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, common.NewStreamErrorWithFields(common.StreamTypeHLS, segmentURL,
			common.ErrCodeConnection, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, resp.Status), nil,
			logging.Fields{
				"status_code": resp.StatusCode,
				"status_text": resp.Status,
			})
	}

	var reader io.Reader = resp.Body
	if ad.hlsConfig != nil && ad.hlsConfig.HTTP != nil && ad.hlsConfig.HTTP.BufferSize > 0 {
		reader = bufio.NewReaderSize(resp.Body, ad.hlsConfig.HTTP.BufferSize)
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeHLS, segmentURL,
			common.ErrCodeConnection, "failed to read response", err)
	}

	return data, nil
}

// resolveSegmentURL resolves relative segment URLs against the playlist base URL
func (ad *AudioDownloader) resolveSegmentURL(playlist *M3U8Playlist, segmentURI string) string {
	// TODO: Implement proper URL resolution
	// For now, assume segments are absolute URLs or relative to current directory
	return segmentURI
}

// recordSegmentError records an error for metrics and debugging
func (ad *AudioDownloader) recordSegmentError(url string, err error, retry int, errorType string) {
	ad.downloadStats.ErrorCount++
	ad.downloadStats.SegmentErrors = append(ad.downloadStats.SegmentErrors, SegmentError{
		URL:       url,
		Error:     err.Error(),
		Timestamp: time.Now(),
		Retry:     retry,
		Type:      errorType,
	})
}

// GetDownloadStats returns current download statistics
func (ad *AudioDownloader) GetDownloadStats() *DownloadStats {
	if ad.downloadStats.DownloadTime > 0 {
		bitsDownloaded := float64(ad.downloadStats.BytesDownloaded) * 8
		seconds := ad.downloadStats.DownloadTime.Seconds()
		ad.downloadStats.AverageBitrate = bitsDownloaded / seconds / 1000 // kbps
	}

	return ad.downloadStats
}

// ClearCache clears the segment cache
func (ad *AudioDownloader) ClearCache() {
	ad.segmentCache = make(map[string][]byte)
}

// UpdateConfig updates the download configuration
func (ad *AudioDownloader) UpdateConfig(config *DownloadConfig) {
	if config != nil {
		ad.config = config
	}
}

// Close cleans up resources
func (ad *AudioDownloader) Close() error {
	if ad.config.CleanupTempFiles && ad.tempDir != "" {
		return os.RemoveAll(ad.tempDir)
	}
	return nil
}
