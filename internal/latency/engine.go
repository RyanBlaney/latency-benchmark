package latency

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/transcode"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// MeasurementEngine handles latency measurements for streams
type MeasurementEngine struct {
	logger                 logging.Logger
	operationTimeout       time.Duration
	segmentDuration        time.Duration
	minAlignmentConf       float64
	maxAlignmentOffset     float64
	minSimilarity          float64
	enableDetailedAnalysis bool
	userAgent              string
}

// EngineConfig contains configuration for the measurement engine
type EngineConfig struct {
	OperationTimeout       time.Duration
	AudioSegmentDuration   time.Duration
	MinAlignmentConfidence float64
	MaxAlignmentOffset     float64
	MinSimilarity          float64
	EnableDetailedAnalysis bool
	UserAgent              string
	Logger                 logging.Logger
}

// NewMeasurementEngine creates a new measurement engine
func NewMeasurementEngine(config *EngineConfig) *MeasurementEngine {
	logger := config.Logger
	if logger == nil {
		logger = logging.NewDefaultLogger()
	}

	return &MeasurementEngine{
		logger:                 logger,
		operationTimeout:       config.OperationTimeout,
		segmentDuration:        config.AudioSegmentDuration,
		minAlignmentConf:       config.MinAlignmentConfidence,
		maxAlignmentOffset:     config.MaxAlignmentOffset,
		minSimilarity:          config.MinSimilarity,
		enableDetailedAnalysis: config.EnableDetailedAnalysis,
		userAgent:              config.UserAgent,
	}
}

// MeasureStream measures a single stream endpoint
func (e *MeasurementEngine) MeasureStream(ctx context.Context, endpoint *StreamEndpoint) *StreamMeasurement {
	measurement := &StreamMeasurement{
		Endpoint:  endpoint,
		Timestamp: time.Now(),
	}

	e.logger.Info("Starting stream measurement", map[string]any{
		"url":          endpoint.URL,
		"type":         endpoint.Type,
		"role":         endpoint.Role,
		"content_type": endpoint.ContentType,
	})

	totalStart := time.Now()

	// Step 1: Load audio and measure TTFB
	ttfbStart := time.Now()
	audioData, err := e.loadAudio(ctx, endpoint)
	if err != nil {
		measurement.Error = fmt.Errorf("failed to load audio: %w", err)
		return measurement
	}
	measurement.TimeToFirstByte = time.Since(ttfbStart)
	measurement.AudioData = audioData

	// Step 2: Validate stream
	measurement.StreamValidation = e.validateStream(endpoint, audioData)

	// Step 3: Generate fingerprint
	fingerprintStart := time.Now()
	fingerprint, err := e.generateFingerprint(audioData, endpoint)
	if err != nil {
		measurement.Error = fmt.Errorf("failed to generate fingerprint: %w", err)
		return measurement
	}
	measurement.FingerprintTime = time.Since(fingerprintStart)
	measurement.Fingerprint = fingerprint

	measurement.TotalProcessingTime = time.Since(totalStart)

	e.logger.Info("Stream measurement completed", map[string]any{
		"url":                 endpoint.URL,
		"ttfb_ms":             measurement.TimeToFirstByte.Milliseconds(),
		"fingerprint_time_ms": measurement.FingerprintTime.Milliseconds(),
		"total_processing_ms": measurement.TotalProcessingTime.Milliseconds(),
		"audio_duration_s":    audioData.Duration.Seconds(),
		"is_valid":            measurement.StreamValidation.IsValid,
	})

	return measurement
}

// MeasureAlignment measures temporal alignment between two streams
func (e *MeasurementEngine) MeasureAlignment(ctx context.Context, stream1, stream2 *StreamMeasurement) *AlignmentMeasurement {
	measurement := &AlignmentMeasurement{
		Stream1:   stream1,
		Stream2:   stream2,
		Timestamp: time.Now(),
	}

	if stream1.Error != nil || stream2.Error != nil {
		measurement.Error = fmt.Errorf("cannot measure alignment: one or both streams have errors")
		return measurement
	}

	e.logger.Info("Starting alignment measurement", map[string]any{
		"stream1_url": stream1.Endpoint.URL,
		"stream2_url": stream2.Endpoint.URL,
	})

	compStart := time.Now()

	// Extract alignment features
	contentType := e.parseContentType(stream1.Endpoint.ContentType)
	featureConfig := e.createFeatureConfig(contentType, stream1.AudioData.SampleRate)
	alignmentConfig := config.AlignmentConfigForContent(contentType)

	alignmentExtractor := extractors.NewAlignmentExtractorWithMaxLag(
		featureConfig, alignmentConfig, e.maxAlignmentOffset)

	alignmentFeatures, err := alignmentExtractor.ExtractAlignmentFeatures(
		stream1.Fingerprint.Features, stream2.Fingerprint.Features,
		stream1.AudioData.PCM, stream2.AudioData.PCM,
		stream1.AudioData.SampleRate)

	measurement.ComparisonTime = time.Since(compStart)

	if err != nil {
		measurement.Error = fmt.Errorf("alignment extraction failed: %w", err)
		return measurement
	}

	measurement.AlignmentResult = alignmentFeatures
	measurement.IsValidAlignment = alignmentFeatures.OffsetConfidence >= e.minAlignmentConf

	if measurement.IsValidAlignment {
		measurement.LatencySeconds = alignmentFeatures.TemporalOffset
	}

	e.logger.Info("Alignment measurement completed", map[string]any{
		"stream1_url":        stream1.Endpoint.URL,
		"stream2_url":        stream2.Endpoint.URL,
		"offset_seconds":     alignmentFeatures.TemporalOffset,
		"confidence":         alignmentFeatures.OffsetConfidence,
		"is_valid":           measurement.IsValidAlignment,
		"comparison_time_ms": measurement.ComparisonTime.Milliseconds(),
	})

	return measurement
}

// CompareFingerprintSimilarity compares fingerprint similarity between two streams
func (e *MeasurementEngine) CompareFingerprintSimilarity(ctx context.Context, stream1, stream2 *StreamMeasurement, alignmentFeatures *extractors.AlignmentFeatures) *FingerprintComparison {
	comparison := &FingerprintComparison{
		Stream1:           stream1,
		Stream2:           stream2,
		AlignmentFeatures: alignmentFeatures,
		Timestamp:         time.Now(),
	}

	if stream1.Error != nil || stream2.Error != nil {
		comparison.Error = fmt.Errorf("cannot compare fingerprints: one or both streams have errors")
		return comparison
	}

	e.logger.Info("Starting fingerprint comparison", map[string]any{
		"stream1_url":   stream1.Endpoint.URL,
		"stream2_url":   stream2.Endpoint.URL,
		"has_alignment": alignmentFeatures != nil,
	})

	compStart := time.Now()

	// Create comparator
	contentType := e.parseContentType(stream1.Endpoint.ContentType)
	comparisonConfig := fingerprint.ContentOptimizedComparisonConfig(contentType)
	comparator := fingerprint.NewFingerprintComparator(comparisonConfig)

	var result *fingerprint.SimilarityResult
	var err error

	// Compare with or without alignment
	if alignmentFeatures != nil && alignmentFeatures.OffsetConfidence >= e.minAlignmentConf {
		alignmentConfig := config.AlignmentConfigForContent(contentType)
		result, err = comparator.CompareWithAlignment(
			stream1.Fingerprint, stream2.Fingerprint, alignmentFeatures, alignmentConfig)
	} else {
		result, err = comparator.Compare(stream1.Fingerprint, stream2.Fingerprint)
	}

	comparison.ComparisonTime = time.Since(compStart)

	if err != nil {
		comparison.Error = fmt.Errorf("fingerprint comparison failed: %w", err)
		return comparison
	}

	comparison.SimilarityResult = result
	comparison.IsValidMatch = result.OverallSimilarity >= e.minSimilarity

	e.logger.Info("Fingerprint comparison completed", map[string]any{
		"stream1_url":        stream1.Endpoint.URL,
		"stream2_url":        stream2.Endpoint.URL,
		"overall_similarity": result.OverallSimilarity,
		"confidence":         result.Confidence,
		"is_valid_match":     comparison.IsValidMatch,
		"comparison_time_ms": comparison.ComparisonTime.Milliseconds(),
	})

	return comparison
}

// loadAudio loads audio from a stream endpoint
func (e *MeasurementEngine) loadAudio(ctx context.Context, endpoint *StreamEndpoint) (*common.AudioData, error) {
	if e.isLocalFile(endpoint.URL) {
		return e.loadLocalFile(endpoint.URL, endpoint.ContentType)
	}
	return e.loadStreamURL(ctx, endpoint.URL, endpoint.ContentType)
}

// isLocalFile checks if the input is a local file
func (e *MeasurementEngine) isLocalFile(input string) bool {
	return strings.HasPrefix(input, "file://") ||
		strings.HasPrefix(input, "/") ||
		strings.HasPrefix(input, "./") ||
		strings.HasPrefix(input, "../") ||
		(!strings.HasPrefix(input, "http://") && !strings.HasPrefix(input, "https://"))
}

// loadLocalFile loads audio from a local file
func (e *MeasurementEngine) loadLocalFile(filePath, contentType string) (*common.AudioData, error) {
	// Implementation similar to your working fingerprint-test.go
	cleanPath := strings.TrimPrefix(filePath, "file://")

	decoder := transcode.NewNormalizingDecoder(contentType)
	anyData, err := decoder.DecodeFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to decode audio file: %w", err)
	}

	audioData := common.ConvertToAudioData(anyData)
	if audioData == nil {
		return nil, fmt.Errorf("decoder returned unexpected type: %T", anyData)
	}

	// Truncate if needed
	if e.segmentDuration > 0 {
		maxSamples := int(e.segmentDuration.Seconds() * float64(audioData.SampleRate))
		if len(audioData.PCM) > maxSamples {
			audioData.PCM = audioData.PCM[:maxSamples]
			audioData.Duration = time.Duration(float64(len(audioData.PCM))/float64(audioData.SampleRate)) * time.Second
		}
	}

	return audioData, nil
}

// loadStreamURL loads audio from a stream URL
func (e *MeasurementEngine) loadStreamURL(ctx context.Context, url, contentType string) (*common.AudioData, error) {
	managerConfig := &stream.ManagerConfig{
		StreamTimeout:        e.operationTimeout,
		OverallTimeout:       e.operationTimeout + (10 * time.Second),
		MaxConcurrentStreams: 1,
		ResultBufferSize:     1,
	}
	manager := stream.NewManagerWithConfig(managerConfig)

	results, err := manager.ExtractAudioSequential(ctx, []string{url}, e.segmentDuration)
	if err != nil {
		return nil, err
	}

	if len(results.Results) == 0 || results.Results[0].Error != nil {
		if len(results.Results) > 0 {
			return nil, results.Results[0].Error
		}
		return nil, fmt.Errorf("no results from stream extraction")
	}

	return results.Results[0].AudioData, nil
}

// generateFingerprint generates a fingerprint for audio data
func (e *MeasurementEngine) generateFingerprint(audioData *common.AudioData, endpoint *StreamEndpoint) (*fingerprint.AudioFingerprint, error) {
	// Use the EXACT working configuration from your fingerprint-test.go
	contentType := e.parseContentType(endpoint.ContentType)
	fingerprintConfig := fingerprint.ContentOptimizedFingerprintConfig(contentType)

	// Apply the working parameters
	fingerprintConfig.WindowSize = 1024
	fingerprintConfig.HopSize = 256
	fingerprintConfig.FeatureConfig.WindowSize = 1024
	fingerprintConfig.FeatureConfig.HopSize = 256
	fingerprintConfig.FeatureConfig.SampleRate = audioData.SampleRate

	// Content-specific optimizations
	if endpoint.ContentType == "news" || endpoint.ContentType == "talk" {
		fingerprintConfig.FeatureConfig.EnableSpeechFeatures = true
		fingerprintConfig.FeatureConfig.EnableHarmonicFeatures = false
		fingerprintConfig.FeatureConfig.EnableChroma = false
		fingerprintConfig.FeatureConfig.MFCCCoefficients = 12
		fingerprintConfig.FeatureConfig.FreqRange = [2]float64{80.0, 8000.0}
	}

	fingerprintGenerator := fingerprint.NewFingerprintGenerator(fingerprintConfig)

	// Convert to transcode.AudioData
	transcodeAudioData := &transcode.AudioData{
		PCM:        audioData.PCM,
		SampleRate: audioData.SampleRate,
		Channels:   audioData.Channels,
		Metadata: &transcode.StreamMetadata{
			URL:         endpoint.URL,
			Type:        string(endpoint.Type),
			ContentType: endpoint.ContentType,
			SampleRate:  audioData.SampleRate,
			Channels:    audioData.Channels,
			Timestamp:   time.Now(),
		},
	}

	return fingerprintGenerator.GenerateFingerprint(transcodeAudioData)
}

// validateStream validates a stream's health and structure
func (e *MeasurementEngine) validateStream(endpoint *StreamEndpoint, audioData *common.AudioData) *StreamValidation {
	validation := &StreamValidation{
		IsValid:           true,
		ValidationErrors:  []string{},
		AudioFormat:       true,
		BitrateConsistent: true,
	}

	// Basic audio format validation
	if audioData.SampleRate <= 0 {
		validation.IsValid = false
		validation.AudioFormat = false
		validation.ValidationErrors = append(validation.ValidationErrors, "invalid sample rate")
	}

	if audioData.Channels <= 0 {
		validation.IsValid = false
		validation.AudioFormat = false
		validation.ValidationErrors = append(validation.ValidationErrors, "invalid channel count")
	}

	if len(audioData.PCM) == 0 {
		validation.IsValid = false
		validation.AudioFormat = false
		validation.ValidationErrors = append(validation.ValidationErrors, "no audio data")
	}

	// Duration validation
	expectedDuration := e.segmentDuration
	actualDuration := audioData.Duration
	durationDiff := actualDuration - expectedDuration
	if durationDiff < -time.Second || durationDiff > time.Second*5 {
		validation.ValidationErrors = append(validation.ValidationErrors,
			fmt.Sprintf("unexpected audio duration: got %.1fs, expected %.1fs",
				actualDuration.Seconds(), expectedDuration.Seconds()))
	}

	// TODO: Add HLS playlist structure validation
	if endpoint.Type == StreamTypeHLS {
		validation.PlaylistStructure = true // Placeholder - implement HLS validation
	}

	// TODO: Add ICEcast metadata validation
	if endpoint.Type == StreamTypeICEcast {
		validation.HTTPHeaders = true // Placeholder - implement ICEcast validation
	}

	return validation
}

// parseContentType converts string to config.ContentType
func (e *MeasurementEngine) parseContentType(contentTypeStr string) config.ContentType {
	switch strings.ToLower(contentTypeStr) {
	case "music":
		return config.ContentMusic
	case "news":
		return config.ContentNews
	case "talk":
		return config.ContentTalk
	case "sports":
		return config.ContentSports
	case "mixed":
		return config.ContentMixed
	default:
		return config.ContentNews
	}
}

// createFeatureConfig creates feature configuration for alignment
func (e *MeasurementEngine) createFeatureConfig(contentType config.ContentType, sampleRate int) *config.FeatureConfig {
	return &config.FeatureConfig{
		WindowSize: 1024,
		HopSize:    256,
		SampleRate: sampleRate,
		FreqRange:  [2]float64{80.0, 8000.0},

		EnableChroma:           false,
		EnableMFCC:             true,
		EnableSpectralContrast: false,
		EnableTemporalFeatures: true,
		EnableSpeechFeatures:   true,
		EnableHarmonicFeatures: false,

		MFCCCoefficients: 12,
		ChromaBins:       12,

		SimilarityWeights: map[string]float64{
			"mfcc":     0.50,
			"spectral": 0.25,
			"temporal": 0.15,
			"speech":   0.10,
		},
		MatchThreshold: 0.70,
	}
}
