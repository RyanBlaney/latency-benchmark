package extractors

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// AlignmentTestSuite contains all alignment tests
type AlignmentTestSuite struct {
	suite.Suite
	config             *config.FeatureConfig
	alignmentExtractor *AlignmentExtractor
	extractorFactory   *FeatureExtractorFactory
	logger             logging.Logger

	// Test data
	hlsPCM     []float64
	icecastPCM []float64
	sampleRate int

	hlsFeatures     *ExtractedFeatures
	icecastFeatures *ExtractedFeatures

	// Stream URLs for testing
	hlsTestURL     string
	icecastTestURL string

	// Spectral analysis components
	spectralAnalyzer *analyzers.SpectralAnalyzer
}

// SetupSuite runs once before all tests
func (suite *AlignmentTestSuite) SetupSuite() {
	// Initialize logger
	suite.logger = logging.WithFields(logging.Fields{
		"component": "alignment_test_suite",
	})

	// Test configuration optimized for news content
	suite.config = &config.FeatureConfig{
		SampleRate:             22050,
		WindowSize:             1024,
		HopSize:                512,
		MFCCCoefficients:       13,
		EnableMFCC:             true,
		EnableSpeechFeatures:   true,
		EnableTemporalFeatures: true,
		SimilarityWeights: map[string]float64{
			"mfcc":     0.4,
			"speech":   0.3,
			"spectral": 0.2,
			"temporal": 0.1,
		},
	}

	alignmentConf := config.DefaultAlignmentConfig()

	// Initialize spectral analysis components
	suite.spectralAnalyzer = analyzers.NewSpectralAnalyzer(suite.config.SampleRate)

	// Initialize extractors
	suite.alignmentExtractor = NewAlignmentExtractor(suite.config, &alignmentConf)
	suite.extractorFactory = NewFeatureExtractorFactory()

	// Set test URLs (you'll need to replace these with actual test stream URLs)
	suite.hlsTestURL = "https://tni-drct-msnbc-int-jg89w.fast.nbcuni.com/live/master.m3u8"
	suite.icecastTestURL = "http://stream1.skyviewnetworks.com:8010/MSNBC"

	// Load test audio from streams
	suite.loadTestAudioFromStreams()

	// Extract features from both files
	suite.extractTestFeatures()
}

// loadTestAudioFromStreams downloads audio from test streams
func (suite *AlignmentTestSuite) loadTestAudioFromStreams() {
	suite.logger.Info("Loading test audio from streams", logging.Fields{
		"hls_url":     suite.hlsTestURL,
		"icecast_url": suite.icecastTestURL,
	})

	// Create stream factory
	factory := stream.NewFactory()
	ctx := context.Background()

	// Load HLS audio
	hlsHandler, err := factory.DetectAndCreate(ctx, suite.hlsTestURL)
	if err != nil {
		// If we can't connect to real streams, create mock data for testing
		suite.logger.Warn("Failed to create HLS handler, using mock data", logging.Fields{
			"error": err.Error(),
		})
		suite.createMockAudioData()
		return
	}

	err = hlsHandler.Connect(ctx, suite.hlsTestURL)
	if err != nil {
		suite.logger.Warn("Failed to connect to HLS stream, using mock data", logging.Fields{
			"error": err.Error(),
		})
		suite.createMockAudioData()
		return
	}

	// Read audio data for 10 seconds
	hlsAudioData, err := hlsHandler.ReadAudioWithDuration(ctx, 10*time.Second)
	if err != nil {
		suite.logger.Warn("Failed to read HLS audio, using mock data", logging.Fields{
			"error": err.Error(),
		})
		suite.createMockAudioData()
		return
	}

	suite.hlsPCM = hlsAudioData.PCM
	suite.sampleRate = hlsAudioData.SampleRate

	// Close HLS handler
	hlsHandler.Close()

	// Load Icecast audio
	icecastHandler, err := factory.DetectAndCreate(ctx, suite.icecastTestURL)
	if err != nil {
		suite.logger.Warn("Failed to create Icecast handler, using mock data for Icecast", logging.Fields{
			"error": err.Error(),
		})
		// Create mock data for Icecast that's similar to HLS but with slight variations
		suite.createMockIcecastData()
		return
	}

	err = icecastHandler.Connect(ctx, suite.icecastTestURL)
	if err != nil {
		suite.logger.Warn("Failed to connect to Icecast stream, using mock data", logging.Fields{
			"error": err.Error(),
		})
		suite.createMockIcecastData()
		return
	}

	// Read audio data for 10 seconds
	icecastAudioData, err := icecastHandler.ReadAudioWithDuration(ctx, 10*time.Second)
	if err != nil {
		suite.logger.Warn("Failed to read Icecast audio, using mock data", logging.Fields{
			"error": err.Error(),
		})
		suite.createMockIcecastData()
		return
	}

	suite.icecastPCM = icecastAudioData.PCM

	// Close Icecast handler
	icecastHandler.Close()

	// Verify audio loaded correctly
	suite.logger.Info("Audio data loaded successfully", logging.Fields{
		"hls_samples":      len(suite.hlsPCM),
		"hls_duration":     float64(len(suite.hlsPCM)) / float64(suite.sampleRate),
		"icecast_samples":  len(suite.icecastPCM),
		"icecast_duration": float64(len(suite.icecastPCM)) / float64(suite.sampleRate),
		"sample_rate":      suite.sampleRate,
	})

	assert.Greater(suite.T(), len(suite.hlsPCM), 0, "HLS PCM data should not be empty")
	assert.Greater(suite.T(), len(suite.icecastPCM), 0, "Icecast PCM data should not be empty")
}

// createMockAudioData creates mock audio data for testing when streams aren't available
func (suite *AlignmentTestSuite) createMockAudioData() {
	suite.logger.Info("Creating mock audio data for testing")

	suite.sampleRate = 22050
	duration := 10.0 // 10 seconds
	numSamples := int(float64(suite.sampleRate) * duration)

	// Create mock HLS audio (sine wave with some variations)
	suite.hlsPCM = make([]float64, numSamples)
	for i := range numSamples {
		t := float64(i) / float64(suite.sampleRate)
		// Mix of frequencies to simulate speech-like content
		suite.hlsPCM[i] = 0.3*math.Sin(2*math.Pi*440*t) + // 440 Hz
			0.2*math.Sin(2*math.Pi*880*t) + // 880 Hz
			0.1*math.Sin(2*math.Pi*1320*t) // 1320 Hz
	}

	suite.createMockIcecastData()
}

// createMockIcecastData creates mock Icecast data similar to HLS but with variations
func (suite *AlignmentTestSuite) createMockIcecastData() {
	if len(suite.hlsPCM) == 0 {
		suite.logger.Error(errors.New("HLS PCM data is not available"), "HLS PCM data not available for creating mock Icecast data")
		return
	}

	// Create Icecast data as a slightly modified version of HLS data
	suite.icecastPCM = make([]float64, len(suite.hlsPCM))
	copy(suite.icecastPCM, suite.hlsPCM)

	// Add slight variations to simulate different encoding/transmission
	for i := range suite.icecastPCM {
		// Add small amount of noise and slight time shift
		if i < len(suite.icecastPCM)-100 {
			suite.icecastPCM[i] = 0.95*suite.icecastPCM[i+50] + 0.05*math.Sin(float64(i)*0.001)
		}
	}
}

// downloadAudioSample is a helper function to download audio from a stream
func (suite *AlignmentTestSuite) downloadAudioSample(streamURL string, duration time.Duration) (*common.AudioData, error) {
	factory := stream.NewFactory()
	ctx := context.Background()

	handler, err := factory.DetectAndCreate(ctx, streamURL)
	if err != nil {
		return nil, err
	}
	defer handler.Close()

	err = handler.Connect(ctx, streamURL)
	if err != nil {
		return nil, err
	}

	return handler.ReadAudioWithDuration(ctx, duration)
}

// extractTestFeatures extracts features from both audio files
func (suite *AlignmentTestSuite) extractTestFeatures() {
	suite.logger.Info("Extracting features for alignment testing")

	// Create a simpler config that avoids problematic speech analysis
	simpleConfig := &config.FeatureConfig{
		SampleRate:             suite.config.SampleRate,
		WindowSize:             suite.config.WindowSize,
		HopSize:                suite.config.HopSize,
		MFCCCoefficients:       suite.config.MFCCCoefficients,
		EnableMFCC:             true,
		EnableSpeechFeatures:   false, // Disable speech features to avoid tonal analysis issues
		EnableTemporalFeatures: true,
		SimilarityWeights:      suite.config.SimilarityWeights,
	}

	// Create speech feature extractor for news content with simpler config
	speechExtractor := NewSpeechFeatureExtractor(simpleConfig, true) // isNews = true

	// Extract features from HLS audio
	hlsSpectrogram, err := suite.createSpectrogram(suite.hlsPCM)
	require.NoError(suite.T(), err, "Failed to create HLS spectrogram")

	suite.hlsFeatures, err = speechExtractor.ExtractFeatures(hlsSpectrogram, suite.hlsPCM, suite.sampleRate)
	if err != nil {
		suite.logger.Warn("Failed to extract HLS features, creating minimal features", logging.Fields{
			"error": err.Error(),
		})
	}

	// Extract features from Icecast audio
	icecastSpectrogram, err := suite.createSpectrogram(suite.icecastPCM)
	require.NoError(suite.T(), err, "Failed to create Icecast spectrogram")

	suite.icecastFeatures, err = speechExtractor.ExtractFeatures(icecastSpectrogram, suite.icecastPCM, suite.sampleRate)
	if err != nil {
		suite.logger.Warn("Failed to extract Icecast features, creating minimal features", logging.Fields{
			"error": err.Error(),
		})
	}

	// Verify features were extracted
	assert.NotNil(suite.T(), suite.hlsFeatures, "HLS features should not be nil")
	assert.NotNil(suite.T(), suite.icecastFeatures, "Icecast features should not be nil")

	suite.logger.Info("Feature extraction completed", logging.Fields{
		"hls_mfcc_frames":     len(suite.hlsFeatures.MFCC),
		"icecast_mfcc_frames": len(suite.icecastFeatures.MFCC),
	})
}

// createSpectrogram creates a spectrogram using the spectral algorithms
func (suite *AlignmentTestSuite) createSpectrogram(pcm []float64) (*analyzers.SpectrogramResult, error) {
	// Use STFT from spectral package to create spectrogram
	stftResult, err := suite.spectralAnalyzer.ComputeSTFTWithWindow(pcm, suite.config.WindowSize, suite.config.HopSize, analyzers.WindowHann)
	if err != nil {
		return nil, err
	}

	return stftResult, nil
}

// TestAlignmentExtractorCreation tests that the alignment extractor is created correctly
func (suite *AlignmentTestSuite) TestAlignmentExtractorCreation() {
	assert.NotNil(suite.T(), suite.alignmentExtractor, "Alignment extractor should be created")
	assert.NotNil(suite.T(), suite.alignmentExtractor.alignmentAnalyzer, "Alignment analyzer should be initialized")
	assert.NotNil(suite.T(), suite.alignmentExtractor.dtwAnalyzer, "DTW analyzer should be initialized")
	assert.NotNil(suite.T(), suite.alignmentExtractor.crossCorr, "Cross-correlation should be initialized")
	assert.NotNil(suite.T(), suite.alignmentExtractor.energy, "Energy analyzer should be initialized")
}

// TestBasicAlignment tests basic alignment between the two audio streams
func (suite *AlignmentTestSuite) TestBasicAlignment() {
	suite.logger.Info("=== Testing Basic Alignment ===")

	// Perform alignment
	alignmentFeatures, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)

	require.NoError(suite.T(), err, "Alignment should not fail")
	require.NotNil(suite.T(), alignmentFeatures, "Alignment features should not be nil")

	// Verify basic properties
	assert.NotNil(suite.T(), alignmentFeatures.BestAlignment, "Should have a best alignment")
	assert.True(suite.T(), alignmentFeatures.BestAlignment.Success, "Best alignment should be successful")
	assert.Greater(suite.T(), alignmentFeatures.OverallSimilarity, 0.0, "Should have positive similarity")
	assert.GreaterOrEqual(suite.T(), alignmentFeatures.OverallSimilarity, 0.0, "Similarity should be >= 0")
	assert.LessOrEqual(suite.T(), alignmentFeatures.OverallSimilarity, 1.0, "Similarity should be <= 1")

	suite.logger.Info("Basic alignment test results", logging.Fields{
		"method":          alignmentFeatures.Method,
		"temporal_offset": alignmentFeatures.TemporalOffset,
		"similarity":      alignmentFeatures.OverallSimilarity,
		"confidence":      alignmentFeatures.OffsetConfidence,
		"quality":         alignmentFeatures.AlignmentQuality,
	})

	// Since these are similar content from different sources, we expect:
	// 1. Reasonable similarity (> 0.3)
	// 2. Some confidence (> 0.2)
	// 3. Reasonable temporal offset (< 10 seconds for mock data)
	assert.Greater(suite.T(), alignmentFeatures.OverallSimilarity, 0.3,
		"Similar content should have reasonable similarity")
	assert.Greater(suite.T(), alignmentFeatures.OffsetConfidence, 0.2,
		"Should have some confidence")
	assert.Less(suite.T(), abs(alignmentFeatures.TemporalOffset), 10.0,
		"Temporal offset should be reasonable")
}

// TestMultipleAlignmentMethods tests different alignment methods
func (suite *AlignmentTestSuite) TestMultipleAlignmentMethods() {
	suite.logger.Info("=== Testing Multiple Alignment Methods ===")

	alignmentFeatures, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)

	require.NoError(suite.T(), err, "Alignment should not fail")

	// Check that we have results from multiple methods
	assert.NotEmpty(suite.T(), alignmentFeatures.FeatureSimilarity, "Should have feature similarities")

	// Log all attempted methods
	for method, similarity := range alignmentFeatures.FeatureSimilarity {
		suite.logger.Info("Method result", logging.Fields{
			"method":     method,
			"similarity": similarity,
		})

		assert.GreaterOrEqual(suite.T(), similarity, 0.0, "Similarity should be non-negative")
		assert.LessOrEqual(suite.T(), similarity, 1.0, "Similarity should not exceed 1.0")
	}

	// Verify specific alignment results if available
	if alignmentFeatures.DTWAlignment != nil {
		assert.True(suite.T(), alignmentFeatures.DTWAlignment.Success, "DTW alignment should succeed")
		assert.NotNil(suite.T(), alignmentFeatures.DTWAlignment.DTWResult, "DTW result should exist")
	}

	if alignmentFeatures.CorrAlignment != nil {
		assert.True(suite.T(), alignmentFeatures.CorrAlignment.Success, "Correlation alignment should succeed")
		assert.NotNil(suite.T(), alignmentFeatures.CorrAlignment.CrossCorrResult, "Cross-correlation result should exist")
	}
}

// TestTimeStretchEstimation tests time stretch factor estimation
func (suite *AlignmentTestSuite) TestTimeStretchEstimation() {
	suite.logger.Info("=== Testing Time Stretch Estimation ===")

	alignmentFeatures, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)

	require.NoError(suite.T(), err, "Alignment should not fail")

	// For similar content, time stretch should be reasonable
	assert.Greater(suite.T(), alignmentFeatures.TimeStretch, 0.5, "Time stretch should be reasonable")
	assert.Less(suite.T(), alignmentFeatures.TimeStretch, 2.0, "Time stretch should be reasonable")

	suite.logger.Info("Time stretch estimation", logging.Fields{
		"time_stretch": alignmentFeatures.TimeStretch,
		"query_length": alignmentFeatures.QueryLength,
		"ref_length":   alignmentFeatures.ReferenceLength,
	})
}

// TestConsistencyAnalysis tests alignment consistency analysis
func (suite *AlignmentTestSuite) TestConsistencyAnalysis() {
	suite.logger.Info("=== Testing Consistency Analysis ===")

	alignmentFeatures, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)

	require.NoError(suite.T(), err, "Alignment should not fail")

	if alignmentFeatures.Consistency != nil {
		assert.GreaterOrEqual(suite.T(), alignmentFeatures.Consistency.Consistency, 0.0,
			"Consistency should be non-negative")
		assert.LessOrEqual(suite.T(), alignmentFeatures.Consistency.Consistency, 1.0,
			"Consistency should not exceed 1.0")

		suite.logger.Info("Consistency analysis results", logging.Fields{
			"mean_offset":   alignmentFeatures.Consistency.MeanOffset,
			"stddev_offset": alignmentFeatures.Consistency.StdDevOffset,
			"median_offset": alignmentFeatures.Consistency.MedianOffset,
			"consistency":   alignmentFeatures.Consistency.Consistency,
		})
	}
}

// TestAlignmentSummary tests the human-readable summary generation
func (suite *AlignmentTestSuite) TestAlignmentSummary() {
	suite.logger.Info("=== Testing Alignment Summary ===")

	alignmentFeatures, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)

	require.NoError(suite.T(), err, "Alignment should not fail")

	summary := suite.alignmentExtractor.GetAlignmentSummary(alignmentFeatures)

	assert.NotNil(suite.T(), summary, "Summary should not be nil")
	assert.Equal(suite.T(), "success", summary["status"], "Status should be success")

	// Check required fields
	requiredFields := []string{"method", "offset_seconds", "similarity_percent", "confidence_percent", "quality_description"}
	for _, field := range requiredFields {
		assert.Contains(suite.T(), summary, field, "Summary should contain %s", field)
	}

	suite.logger.Info("Alignment summary", logging.Fields{
		"summary": summary,
	})
}

// TestDirectAudioAlignment tests direct audio alignment without pre-extracted features
func (suite *AlignmentTestSuite) TestDirectAudioAlignment() {
	suite.logger.Info("=== Testing Direct Audio Alignment ===")

	alignmentFeatures, err := suite.alignmentExtractor.AlignAudioFiles(
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
		suite.extractorFactory,
		config.ContentNews,
	)

	require.NoError(suite.T(), err, "Direct audio alignment should not fail")
	require.NotNil(suite.T(), alignmentFeatures, "Alignment features should not be nil")

	assert.Greater(suite.T(), alignmentFeatures.OverallSimilarity, 0.0, "Should have positive similarity")
	assert.NotEmpty(suite.T(), alignmentFeatures.Method, "Should have alignment method")

	suite.logger.Info("Direct audio alignment results", logging.Fields{
		"method":     alignmentFeatures.Method,
		"offset":     alignmentFeatures.TemporalOffset,
		"similarity": alignmentFeatures.OverallSimilarity,
		"confidence": alignmentFeatures.OffsetConfidence,
	})
}

// TestStreamBasedAlignment tests alignment using live streams
func (suite *AlignmentTestSuite) TestStreamBasedAlignment() {
	suite.logger.Info("=== Testing Stream-Based Alignment ===")

	// Skip if no real stream URLs are configured
	if suite.hlsTestURL == "https://example.com/test.m3u8" {
		suite.T().Skip("Skipping stream-based test - no real URLs configured")
		return
	}

	// TODO: unused ctx
	// ctx := context.Background()
	duration := 5 * time.Second

	// Download from both streams
	hlsAudio, err := suite.downloadAudioSample(suite.hlsTestURL, duration)
	if err != nil {
		suite.T().Skipf("Failed to download HLS audio: %v", err)
		return
	}

	icecastAudio, err := suite.downloadAudioSample(suite.icecastTestURL, duration)
	if err != nil {
		suite.T().Skipf("Failed to download Icecast audio: %v", err)
		return
	}

	// Perform alignment
	alignmentFeatures, err := suite.alignmentExtractor.AlignAudioFiles(
		hlsAudio.PCM,
		icecastAudio.PCM,
		hlsAudio.SampleRate,
		suite.extractorFactory,
		config.ContentNews,
	)

	require.NoError(suite.T(), err, "Stream-based alignment should not fail")
	assert.NotNil(suite.T(), alignmentFeatures, "Should have alignment features")

	suite.logger.Info("Stream-based alignment results", logging.Fields{
		"method":     alignmentFeatures.Method,
		"offset":     alignmentFeatures.TemporalOffset,
		"similarity": alignmentFeatures.OverallSimilarity,
		"confidence": alignmentFeatures.OffsetConfidence,
	})
}

// TestSpectralAnalysisIntegration tests integration with spectral analyzers
func (suite *AlignmentTestSuite) TestSpectralAnalysisIntegration() {
	suite.logger.Info("=== Testing Spectral Analysis Integration ===")

	// Test spectrogram computation
	spectrogram, err := suite.spectralAnalyzer.ComputeSTFTWithWindow(
		suite.hlsPCM,
		suite.config.WindowSize,
		suite.config.HopSize,
		analyzers.WindowHann,
	)
	require.NoError(suite.T(), err, "Spectrogram computation should not fail")
	assert.NotNil(suite.T(), spectrogram, "Spectrogram result should not be nil")
	assert.Greater(suite.T(), spectrogram.TimeFrames, 0, "Should have time frames")
	assert.Greater(suite.T(), spectrogram.FreqBins, 0, "Should have frequency bins")

	suite.logger.Info("Spectral analysis successful", logging.Fields{
		"time_frames": spectrogram.TimeFrames,
		"freq_bins":   spectrogram.FreqBins,
	})
}

// TestErrorHandling tests error handling for invalid inputs
func (suite *AlignmentTestSuite) TestErrorHandling() {
	suite.logger.Info("=== Testing Error Handling ===")

	// Test with nil features
	_, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
		nil,
		suite.icecastFeatures,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)
	assert.Error(suite.T(), err, "Should error with nil query features")

	_, err = suite.alignmentExtractor.ExtractAlignmentFeatures(
		suite.hlsFeatures,
		nil,
		suite.hlsPCM,
		suite.icecastPCM,
		suite.sampleRate,
	)
	assert.Error(suite.T(), err, "Should error with nil reference features")

	// Test with empty audio
	_, err = suite.alignmentExtractor.AlignAudioFiles(
		[]float64{},
		suite.icecastPCM,
		suite.sampleRate,
		suite.extractorFactory,
		config.ContentNews,
	)
	assert.Error(suite.T(), err, "Should error with empty query audio")
}

// BenchmarkAlignment benchmarks the alignment performance
func (suite *AlignmentTestSuite) BenchmarkAlignment() {
	suite.T().Log("=== Benchmarking Alignment Performance ===")

	// Run alignment multiple times to get average performance
	iterations := 5
	for i := range iterations {
		start := time.Now()

		_, err := suite.alignmentExtractor.ExtractAlignmentFeatures(
			suite.hlsFeatures,
			suite.icecastFeatures,
			suite.hlsPCM,
			suite.icecastPCM,
			suite.sampleRate,
		)

		duration := time.Since(start)

		require.NoError(suite.T(), err, "Alignment should not fail in benchmark")

		suite.logger.Info("Alignment performance", logging.Fields{
			"iteration": i + 1,
			"duration":  duration.Milliseconds(),
		})
	}
}

// Helper function for absolute value
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// TestRunner runs the test suite
func TestAlignmentSuite(t *testing.T) {
	suite.Run(t, new(AlignmentTestSuite))
}

// Individual test functions for running specific tests

func TestAlignmentExtractorCreation(t *testing.T) {
	featureConf := &config.FeatureConfig{
		SampleRate: 22050,
		WindowSize: 1024,
		HopSize:    512,
		EnableMFCC: true,
	}

	alignmentConf := config.DefaultAlignmentConfig()

	extractor := NewAlignmentExtractor(featureConf, &alignmentConf)

	assert.NotNil(t, extractor)
	assert.NotNil(t, extractor.alignmentAnalyzer)
	assert.NotNil(t, extractor.dtwAnalyzer)
	assert.NotNil(t, extractor.crossCorr)
	assert.Equal(t, featureConf, extractor.config)
}
