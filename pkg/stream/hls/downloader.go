package hls

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/go-transcoding/v10/transcode"
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
	// Audio processing options
	OutputSampleRate int    `json:"output_sample_rate"` // Target sample rate for fingerprinting
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
		config:  config,
		tempDir: tempDir,
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

	// Check cache first
	if ad.config.CacheSegments {
		if cachedData, exists := ad.segmentCache[segmentURL]; exists {
			return ad.processSegmentData(cachedData, segmentURL, startTime)
		}
	}

	// Download segment with retries
	segmentData, err := ad.downloadSegmentWithRetries(ctx, segmentURL)
	if err != nil {
		ad.recordSegmentError(segmentURL, err, 0, "download")
		return nil, fmt.Errorf("failed to download segment %s: %w", segmentURL, err)
	}

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

// processSegmentData decodes segment data using TuneIn transcoding library
func (ad *AudioDownloader) processSegmentData(segmentData []byte, segmentURL string, startTime time.Time) (*common.AudioData, error) {
	decodeStartTime := time.Now()

	// Write to temp file
	tempFile := filepath.Join(ad.tempDir, fmt.Sprintf("segment_%d.tmp", time.Now().UnixNano()))
	err := os.WriteFile(tempFile, segmentData, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	defer os.Remove(tempFile)

	// Open input format
	inputFormat, err := transcode.OpenInput(tempFile, map[string]interface{}{})
	if err != nil {
		return nil, fmt.Errorf("failed to open input format: %w", err)
	}
	defer inputFormat.Close()

	// Just get the first (and likely only) stream
	stream := inputFormat.GetStream(0)
	if stream == nil {
		return nil, fmt.Errorf("no streams found in segment")
	}

	// Open decoder directly - no need for "best" stream selection
	decoder, err := transcode.OpenDecoder(stream)
	if err != nil {
		return nil, fmt.Errorf("failed to open decoder: %w", err)
	}
	defer decoder.Close()

	// Configure decoder for better error handling
	decoder.SetMaxErrors(10)

	// Decode all frames
	audioData, err := ad.decodeAllFrames(decoder, stream, segmentURL)
	if err != nil {
		ad.recordSegmentError(segmentURL, err, 0, "decode")
		return nil, fmt.Errorf("failed to decode frames: %w", err)
	}

	// Update decode stats
	ad.downloadStats.DecodeTime += time.Since(decodeStartTime)

	// Process and normalize the audio data
	ad.processAudioData(audioData)

	// Add metadata
	audioData.Timestamp = startTime
	audioData.Metadata = ad.createMetadata(segmentURL, stream)

	return audioData, nil
}

// decodeAllFrames decodes all audio frames from the decoder
func (ad *AudioDownloader) decodeAllFrames(decoder *transcode.Decoder, stream *transcode.Stream, segmentURL string) (*common.AudioData, error) {
	var allSamples []float64
	var totalDuration time.Duration

	sampleRate := 44100
	if ad.config != nil && ad.config.OutputSampleRate > 0 {
		sampleRate = ad.config.OutputSampleRate
	}

	channels := 2
	if ad.config != nil && ad.config.OutputChannels > 0 {
		channels = ad.config.OutputChannels
	}

	// Listen for decoder errors in a separate goroutine
	go func() {
		for err := range decoder.Error() {
			if err != nil && err != io.EOF {
				fmt.Printf("Decoder error for %s: %v\n", segmentURL, err)
			}
		}
	}()

	frameCount := 0
	for {
		frame, err := decoder.ReadFrame()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading frame %d: %w", frameCount, err)
		}

		// Extract audio data from frame
		frameSamples, frameDuration, err := common.ExtractAudioFromFrame(frame, sampleRate, channels)
		if err != nil {
			frame.Finalize()
			return nil, fmt.Errorf("error extracting audio from frame %d: %w", frameCount, err)
		}

		// Append samples
		allSamples = append(allSamples, frameSamples...)
		totalDuration += frameDuration

		frame.Finalize()
		frameCount++
	}

	if len(allSamples) == 0 {
		return nil, fmt.Errorf("no audio samples decoded")
	}

	// Create AudioData
	audioData := &common.AudioData{
		PCM:        allSamples,
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   totalDuration,
	}

	// Resample/process if needed
	if ad.config.OutputSampleRate != sampleRate || ad.config.OutputChannels != channels {
		audioData = ad.resampleAudio(audioData)
	}

	// Update audio metrics
	ad.updateAudioMetrics(audioData)

	return audioData, nil
}

// Simplified audio processing functions
func (ad *AudioDownloader) resampleAudio(audioData *common.AudioData) *common.AudioData {
	if ad.config.OutputChannels == 1 && audioData.Channels == 2 {
		audioData = ad.convertToMono(audioData)
	}

	if ad.config.OutputSampleRate != audioData.SampleRate {
		audioData = ad.convertSampleRate(audioData, ad.config.OutputSampleRate)
	}

	return audioData
}

func (ad *AudioDownloader) convertToMono(audioData *common.AudioData) *common.AudioData {
	if audioData.Channels != 2 {
		return audioData
	}

	monoSamples := make([]float64, len(audioData.PCM)/2)
	for i := 0; i < len(monoSamples); i++ {
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

func (ad *AudioDownloader) convertSampleRate(audioData *common.AudioData, targetRate int) *common.AudioData {
	if audioData.SampleRate == targetRate {
		return audioData
	}

	ratio := float64(targetRate) / float64(audioData.SampleRate)
	newLength := int(float64(len(audioData.PCM)) * ratio)
	newSamples := make([]float64, newLength)

	for i := 0; i < newLength; i++ {
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

func (ad *AudioDownloader) processAudioData(audioData *common.AudioData) {
	if ad.config.NormalizePCM {
		ad.normalizePCM(audioData.PCM)
	}
}

func (ad *AudioDownloader) normalizePCM(samples []float64) {
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

func (ad *AudioDownloader) updateAudioMetrics(audioData *common.AudioData) {
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

		if abs < 0.001 {
			silentSamples++
		}

		if abs >= 0.99 {
			clipping = true
		}
	}

	metrics.AverageAmplitude = sum / float64(len(audioData.PCM))
	metrics.PeakAmplitude = peak
	metrics.SilenceRatio = float64(silentSamples) / float64(len(audioData.PCM))
	metrics.ClippingDetected = clipping
}

func (ad *AudioDownloader) createMetadata(segmentURL string, stream *transcode.Stream) *common.StreamMetadata {
	return &common.StreamMetadata{
		URL:        segmentURL,
		Type:       common.StreamTypeHLS,
		SampleRate: 44100, // Default - extract from stream when API is known
		Channels:   2,     // Default - extract from stream when API is known
		Bitrate:    128,   // Default - extract from stream when API is known
		Codec:      "aac", // Default - extract from stream when API is known
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}
}

// DownloadAudioSample downloads multiple segments to create a sample of specified duration
func (ad *AudioDownloader) DownloadAudioSample(ctx context.Context, playlist *M3U8Playlist, targetDuration time.Duration) (*common.AudioData, error) {
	if playlist == nil || len(playlist.Segments) == 0 {
		return nil, fmt.Errorf("empty playlist")
	}

	var audioSamples []*common.AudioData
	var totalDuration time.Duration
	maxSegments := ad.config.MaxSegments

	for i, segment := range playlist.Segments {
		if i >= maxSegments || totalDuration >= targetDuration {
			break
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		segmentURL := ad.resolveSegmentURL(playlist, segment.URI)

		audioData, err := ad.DownloadAudioSegment(ctx, segmentURL)
		if err != nil {
			ad.recordSegmentError(segmentURL, err, 0, "processing")
			continue
		}

		audioSamples = append(audioSamples, audioData)
		totalDuration += audioData.Duration
	}

	if len(audioSamples) == 0 {
		return nil, fmt.Errorf("failed to download any audio segments")
	}

	return ad.combineAudioSamples(audioSamples)
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

func (ad *AudioDownloader) downloadSegment(ctx context.Context, segmentURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", segmentURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	userAgent := "TuneIn-CDN-Benchmark/1.0" // Default fallback
	if ad.hlsConfig != nil && ad.hlsConfig.HTTP != nil {
		userAgent = ad.hlsConfig.HTTP.UserAgent
	}

	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "*/*")

	resp, err := ad.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	var reader io.Reader = resp.Body
	if ad.hlsConfig != nil && ad.hlsConfig.HTTP != nil && ad.hlsConfig.HTTP.BufferSize > 0 {
		reader = bufio.NewReaderSize(resp.Body, ad.hlsConfig.HTTP.BufferSize)
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return data, nil
}

func (ad *AudioDownloader) combineAudioSamples(samples []*common.AudioData) (*common.AudioData, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("no audio samples to combine")
	}

	if len(samples) == 1 {
		return samples[0], nil
	}

	firstSample := samples[0]
	combined := &common.AudioData{
		SampleRate: firstSample.SampleRate,
		Channels:   firstSample.Channels,
		Timestamp:  firstSample.Timestamp,
		Metadata:   firstSample.Metadata,
	}

	totalSamples := 0
	for _, sample := range samples {
		if sample.SampleRate != combined.SampleRate {
			return nil, fmt.Errorf("sample rate mismatch: %d vs %d", sample.SampleRate, combined.SampleRate)
		}
		if sample.Channels != combined.Channels {
			return nil, fmt.Errorf("channel count mismatch: %d vs %d", sample.Channels, combined.Channels)
		}
		totalSamples += len(sample.PCM)
		combined.Duration += sample.Duration
	}

	combined.PCM = make([]float64, totalSamples)
	offset := 0
	for _, sample := range samples {
		copy(combined.PCM[offset:], sample.PCM)
		offset += len(sample.PCM)
	}

	return combined, nil
}

func (ad *AudioDownloader) resolveSegmentURL(playlist *M3U8Playlist, segmentURI string) string {
	return segmentURI // Placeholder - implement URL resolution
}

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

func (ad *AudioDownloader) GetDownloadStats() *DownloadStats {
	if ad.downloadStats.DownloadTime > 0 {
		bitsDownloaded := float64(ad.downloadStats.BytesDownloaded) * 8
		seconds := ad.downloadStats.DownloadTime.Seconds()
		ad.downloadStats.AverageBitrate = bitsDownloaded / seconds / 1000
	}

	return ad.downloadStats
}

func (ad *AudioDownloader) ClearCache() {
	ad.segmentCache = make(map[string][]byte)
}

func (ad *AudioDownloader) UpdateConfig(config *DownloadConfig) {
	if config != nil {
		ad.config = config
	}
}

// Close cleans up resources
func (ad *AudioDownloader) Close() error {
	if ad.config.CleanupTempFiles {
		return os.RemoveAll(ad.tempDir)
	}
	return nil
}

// Utility function
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
