package extractors

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
)

// FeatureExtractorTestSuite defines the test suite for all feature extractors
type FeatureExtractorTestSuite struct {
	suite.Suite
	factory         *FeatureExtractorFactory
	testConfig      config.FeatureConfig
	testSpectrogram *analyzers.SpectrogramResult
	testPCM         []float64
	testSampleRate  int
}

// SetupSuite runs once before all tests
func (s *FeatureExtractorTestSuite) SetupSuite() {
	s.factory = NewFeatureExtractorFactory()

	// Create test configuration
	s.testConfig = config.FeatureConfig{
		EnableMFCC:             true,
		EnableChroma:           true,
		EnableTemporalFeatures: true,
		EnableSpeechFeatures:   true,
		EnableSpectralContrast: true,
		MFCCCoefficients:       13,
		ChromaBins:             12,
		ContrastBands:          6,
		SimilarityWeights:      nil, // Use default weights
	}

	// Create test data
	s.setupTestData()
}

// setupTestData creates realistic test audio data
func (s *FeatureExtractorTestSuite) setupTestData() {
	s.testSampleRate = 44100

	// Generate 1 second of test audio (sine wave + noise)
	duration := 1.0 // seconds
	samples := int(float64(s.testSampleRate) * duration)
	s.testPCM = make([]float64, samples)

	// Create a complex test signal
	for i := range samples {
		t := float64(i) / float64(s.testSampleRate)

		// Fundamental frequency (440 Hz - A4)
		fundamental := 0.5 * math.Sin(2*math.Pi*440*t)

		// Harmonic (880 Hz)
		harmonic := 0.3 * math.Sin(2*math.Pi*880*t)

		// Some noise
		noise := 0.1 * (math.Sin(2*math.Pi*1000*t) + math.Sin(2*math.Pi*1500*t))

		// Envelope (to simulate natural audio)
		envelope := math.Exp(-t * 2) // Exponential decay

		s.testPCM[i] = (fundamental + harmonic + noise) * envelope
	}

	// Create test spectrogram using the actual SpectralAnalyzer
	analyzer := analyzers.NewSpectralAnalyzer(s.testSampleRate)

	// Generate real spectrogram from the test PCM data
	var err error
	s.testSpectrogram, err = analyzer.ComputeFFT(s.testPCM)
	if err != nil {
		// If STFT fails, fall back to manual creation for testing
		s.Suite.T().Logf("STFT failed, using manual spectrogram: %v", err)
		s.createManualSpectrogram()
		return
	}

	s.Suite.T().Logf("Created spectrogram with %d time frames, %d freq bins",
		s.testSpectrogram.TimeFrames, s.testSpectrogram.FreqBins)
}

// createManualSpectrogram creates a manual spectrogram as fallback
func (s *FeatureExtractorTestSuite) createManualSpectrogram() {
	s.testSpectrogram = &analyzers.SpectrogramResult{
		TimeFrames:     100,
		FreqBins:       1025,
		SampleRate:     s.testSampleRate,
		Magnitude:      make([][]float64, 100),
		WindowSize:     2048,
		HopSize:        512,
		FreqResolution: float64(s.testSampleRate) / float64(2048),
		TimeResolution: float64(512) / float64(s.testSampleRate),
	}

	// Fill with realistic magnitude data
	for t := 0; t < s.testSpectrogram.TimeFrames; t++ {
		s.testSpectrogram.Magnitude[t] = make([]float64, s.testSpectrogram.FreqBins)

		for f := 0; f < s.testSpectrogram.FreqBins; f++ {
			freq := float64(f) * float64(s.testSampleRate) / float64((s.testSpectrogram.FreqBins-1)*2)

			// Simulate realistic frequency response
			if freq >= 440 && freq <= 450 { // Fundamental
				s.testSpectrogram.Magnitude[t][f] = 0.8 * math.Exp(-float64(t)*0.05)
			} else if freq >= 880 && freq <= 890 { // Harmonic
				s.testSpectrogram.Magnitude[t][f] = 0.5 * math.Exp(-float64(t)*0.05)
			} else if freq >= 1000 && freq <= 1600 { // Noise band
				s.testSpectrogram.Magnitude[t][f] = 0.2 * math.Exp(-float64(t)*0.02)
			} else {
				s.testSpectrogram.Magnitude[t][f] = 0.05 * math.Exp(-float64(t)*0.01) // Background
			}
		}
	}
}

// Test Factory Creation
func (s *FeatureExtractorTestSuite) TestFactoryCreation() {
	factory := NewFeatureExtractorFactory()
	s.NotNil(factory)
	s.NotNil(factory.logger)
}

// Test Music Feature Extractor
func (s *FeatureExtractorTestSuite) TestMusicFeatureExtractor() {
	extractor, err := s.factory.CreateExtractor(config.ContentMusic, s.testConfig)
	s.Require().NoError(err)
	s.Require().NotNil(extractor)

	// Test interface methods
	s.Equal("MusicFeatureExtractor", extractor.GetName())
	s.Equal(config.ContentMusic, extractor.GetContentType())

	// Test feature weights
	weights := extractor.GetFeatureWeights()
	s.NotEmpty(weights)
	s.Contains(weights, "chroma")
	s.Contains(weights, "spectral_contrast")
	s.Contains(weights, "mfcc")

	// Test feature extraction
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Verify extracted features
	s.validateExtractedFeatures(features)

	// Music-specific validations
	s.NotNil(features.SpectralFeatures)
	s.NotNil(features.MFCC)
	s.NotNil(features.ChromaFeatures)
	s.NotNil(features.HarmonicFeatures)
	s.NotNil(features.TemporalFeatures)
	s.NotNil(features.EnergyFeatures)

	// Check chroma features dimensions
	s.Len(features.ChromaFeatures, s.testSpectrogram.TimeFrames)
	if len(features.ChromaFeatures) > 0 {
		s.Len(features.ChromaFeatures[0], 12) // 12 chroma bins
	}
}

// Test Speech Feature Extractor
func (s *FeatureExtractorTestSuite) TestSpeechFeatureExtractor() {
	extractor, err := s.factory.CreateExtractor(config.ContentNews, s.testConfig)
	s.Require().NoError(err)
	s.Require().NotNil(extractor)

	// Test interface methods
	s.Equal("SpeechFeatureExtractor", extractor.GetName())
	s.Equal(config.ContentNews, extractor.GetContentType())

	// Test feature weights
	weights := extractor.GetFeatureWeights()
	s.NotEmpty(weights)
	s.Contains(weights, "mfcc")
	s.Contains(weights, "speech")
	s.Contains(weights, "spectral")

	// Test feature extraction
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Verify extracted features
	s.validateExtractedFeatures(features)

	// Speech-specific validations
	s.NotNil(features.SpeechFeatures)
	s.NotNil(features.SpectralFeatures)

	// Check speech features
	if features.SpeechFeatures != nil {
		s.Len(features.SpeechFeatures.FormantFrequencies, s.testSpectrogram.TimeFrames)
		s.Len(features.SpeechFeatures.VoicingProbability, s.testSpectrogram.TimeFrames)
		s.GreaterOrEqual(features.SpeechFeatures.SpeechRate, 0.0)
	}
}

// Test Sports Feature Extractor
func (s *FeatureExtractorTestSuite) TestSportsFeatureExtractor() {
	extractor, err := s.factory.CreateExtractor(config.ContentSports, s.testConfig)
	s.Require().NoError(err)
	s.Require().NotNil(extractor)

	// Test interface methods
	s.Equal("SportsFeatureExtractor", extractor.GetName())
	s.Equal(config.ContentSports, extractor.GetContentType())

	// Test feature weights
	weights := extractor.GetFeatureWeights()
	s.NotEmpty(weights)
	s.Contains(weights, "energy")
	s.Contains(weights, "spectral_contrast")
	s.Contains(weights, "temporal")

	// Test feature extraction
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Verify extracted features
	s.validateExtractedFeatures(features)

	// Sports-specific validations
	s.NotNil(features.EnergyFeatures)
	s.NotNil(features.SpectralFeatures)
	s.NotNil(features.TemporalFeatures)

	// Check crowd features in metadata
	s.Contains(features.ExtractionMetadata, "crowd_features")
	s.Equal(true, features.ExtractionMetadata["energy_focus"])
	s.Equal(true, features.ExtractionMetadata["crowd_analysis"])
}

// Test Mixed Feature Extractor
func (s *FeatureExtractorTestSuite) TestMixedFeatureExtractor() {
	extractor, err := s.factory.CreateExtractor(config.ContentMixed, s.testConfig)
	s.Require().NoError(err)
	s.Require().NotNil(extractor)

	// Test interface methods
	s.Equal("MixedFeatureExtractor", extractor.GetName())
	s.Equal(config.ContentMixed, extractor.GetContentType())

	// Test feature weights
	weights := extractor.GetFeatureWeights()
	s.NotEmpty(weights)
	s.Contains(weights, "spectral")
	s.Contains(weights, "energy")
	s.Contains(weights, "mfcc")

	// Test feature extraction
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Verify extracted features
	s.validateExtractedFeatures(features)

	// Mixed-specific validations
	s.Contains(features.ExtractionMetadata, "content_analysis")
	s.Equal(true, features.ExtractionMetadata["robust_features"])
	s.Equal(true, features.ExtractionMetadata["adaptive_approach"])

	// Check content analysis
	contentAnalysis := features.ExtractionMetadata["content_analysis"]
	s.NotNil(contentAnalysis)
}

// Test General Feature Extractor
func (s *FeatureExtractorTestSuite) TestGeneralFeatureExtractor() {
	extractor, err := s.factory.CreateExtractor(config.ContentUnknown, s.testConfig)
	s.Require().NoError(err)
	s.Require().NotNil(extractor)

	// Test interface methods
	s.Equal("GeneralFeatureExtractor", extractor.GetName())
	s.Equal(config.ContentUnknown, extractor.GetContentType())

	// Test feature weights
	weights := extractor.GetFeatureWeights()
	s.NotEmpty(weights)
	s.Contains(weights, "spectral")
	s.Contains(weights, "mfcc")
	s.Contains(weights, "temporal")
	s.Contains(weights, "energy")
	s.Contains(weights, "chroma")

	// Verify balanced weights
	s.InDelta(0.25, weights["spectral"], 0.01)
	s.InDelta(0.25, weights["mfcc"], 0.01)

	// Test feature extraction
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Verify extracted features
	s.validateExtractedFeatures(features)

	// General-specific validations
	s.Equal(true, features.ExtractionMetadata["balanced_approach"])
	s.Equal(true, features.ExtractionMetadata["universal_features"])
}

// Test with different configurations
func (s *FeatureExtractorTestSuite) TestWithDisabledFeatures() {
	// Test with features disabled
	minimalConfig := config.FeatureConfig{
		EnableMFCC:             false,
		EnableChroma:           false,
		EnableTemporalFeatures: false,
		EnableSpeechFeatures:   false,
		EnableSpectralContrast: false,
		MFCCCoefficients:       13,
		ChromaBins:             12,
		ContrastBands:          6,
	}

	extractor, err := s.factory.CreateExtractor(config.ContentMusic, minimalConfig)
	s.Require().NoError(err)

	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Should still have spectral and energy features
	s.NotNil(features.SpectralFeatures)
	s.NotNil(features.EnergyFeatures)

	// Should not have optional features
	s.Nil(features.MFCC)
	s.Nil(features.ChromaFeatures)
}

// Test error handling
func (s *FeatureExtractorTestSuite) TestErrorHandling() {
	extractor, err := s.factory.CreateExtractor(config.ContentMusic, s.testConfig)
	s.Require().NoError(err)

	// Test with empty PCM data
	emptyPCM := []float64{}
	features, err := extractor.ExtractFeatures(s.testSpectrogram, emptyPCM, s.testSampleRate)
	s.Error(err)
	s.Nil(features)

	// Test with nil spectrogram
	features, err = extractor.ExtractFeatures(nil, s.testPCM, s.testSampleRate)
	s.Error(err)
	s.Nil(features)

	// Test with zero sample rate
	features, err = extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, 0)
	s.Error(err)
	s.Nil(features)
}

// Test custom similarity weights
func (s *FeatureExtractorTestSuite) TestCustomWeights() {
	customWeights := map[string]float64{
		"custom_weight_1": 0.5,
		"custom_weight_2": 0.3,
		"custom_weight_3": 0.2,
	}

	configWithWeights := s.testConfig
	configWithWeights.SimilarityWeights = customWeights

	extractor, err := s.factory.CreateExtractor(config.ContentMusic, configWithWeights)
	s.Require().NoError(err)

	weights := extractor.GetFeatureWeights()
	s.Equal(customWeights, weights)
}

// Test all content types
func (s *FeatureExtractorTestSuite) TestAllContentTypes() {
	contentTypes := []config.ContentType{
		config.ContentMusic,
		config.ContentNews,
		config.ContentTalk,
		config.ContentSports,
		config.ContentMixed,
		config.ContentUnknown,
	}

	for _, contentType := range contentTypes {
		s.Run(string(contentType), func() {
			extractor, err := s.factory.CreateExtractor(contentType, s.testConfig)
			s.Require().NoError(err, "Failed to create extractor for %s", contentType)
			s.Require().NotNil(extractor)

			// Test basic functionality
			s.NotEmpty(extractor.GetName())
			s.NotNil(extractor.GetFeatureWeights())

			// For talk content type, it maps to speech extractor which returns ContentNews
			expectedContentType := contentType
			s.Equal(expectedContentType, extractor.GetContentType())

			// Test feature extraction
			features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
			s.Require().NoError(err, "Failed to extract features for %s", contentType)
			s.Require().NotNil(features)

			s.validateExtractedFeatures(features)
		})
	}
}

// validateExtractedFeatures performs common validations on extracted features
func (s *FeatureExtractorTestSuite) validateExtractedFeatures(features *ExtractedFeatures) {
	// Basic structure validation
	s.NotNil(features.ExtractionMetadata)

	// Be more flexible with extractor type validation
	actualType, exists := features.ExtractionMetadata["extractor_type"]
	s.True(exists, "extractor_type should exist in metadata")
	s.NotEmpty(actualType, "extractor_type should not be empty")

	// Spectral features should always be present and not nil
	if !s.NotNil(features.SpectralFeatures, "SpectralFeatures should not be nil") {
		return // Skip further validation if spectral features are nil
	}

	s.Len(features.SpectralFeatures.SpectralCentroid, s.testSpectrogram.TimeFrames)
	s.Len(features.SpectralFeatures.SpectralRolloff, s.testSpectrogram.TimeFrames)
	s.Len(features.SpectralFeatures.SpectralBandwidth, s.testSpectrogram.TimeFrames)
	s.Len(features.SpectralFeatures.SpectralFlatness, s.testSpectrogram.TimeFrames)

	// Validate spectral feature values are reasonable
	for i, centroid := range features.SpectralFeatures.SpectralCentroid {
		s.GreaterOrEqual(centroid, 0.0, "Spectral centroid should be non-negative at frame %d", i)
		s.LessOrEqual(centroid, float64(s.testSampleRate/2), "Spectral centroid should be <= Nyquist frequency at frame %d", i)
	}

	for i, rolloff := range features.SpectralFeatures.SpectralRolloff {
		s.GreaterOrEqual(rolloff, 0.0, "Spectral rolloff should be non-negative at frame %d", i)
		s.LessOrEqual(rolloff, float64(s.testSampleRate/2), "Spectral rolloff should be <= Nyquist frequency at frame %d", i)
	}

	for i, flatness := range features.SpectralFeatures.SpectralFlatness {
		s.GreaterOrEqual(flatness, 0.0, "Spectral flatness should be non-negative at frame %d", i)
		s.LessOrEqual(flatness, 1.0, "Spectral flatness should be <= 1.0 at frame %d", i)
	}

	// Energy features should always be present
	s.NotNil(features.EnergyFeatures)
	if features.EnergyFeatures != nil {
		s.GreaterOrEqual(features.EnergyFeatures.EnergyVariance, 0.0)
		s.GreaterOrEqual(features.EnergyFeatures.LoudnessRange, 0.0)
	}

	// If MFCC is enabled and present, validate dimensions
	if features.MFCC != nil {
		s.Len(features.MFCC, s.testSpectrogram.TimeFrames)
		if len(features.MFCC) > 0 {
			expectedCoeffs := s.testConfig.MFCCCoefficients
			if expectedCoeffs == 0 {
				expectedCoeffs = 13
			}
			s.Len(features.MFCC[0], expectedCoeffs)
		}
	}

	// If temporal features are present, validate
	if features.TemporalFeatures != nil {
		s.GreaterOrEqual(features.TemporalFeatures.DynamicRange, 0.0)
		s.GreaterOrEqual(features.TemporalFeatures.SilenceRatio, 0.0)
		s.LessOrEqual(features.TemporalFeatures.SilenceRatio, 1.0)
		s.GreaterOrEqual(features.TemporalFeatures.PeakAmplitude, 0.0)
		s.GreaterOrEqual(features.TemporalFeatures.AverageAmplitude, 0.0)
	}

	/* featuresJSON, err := json.Marshal(features.ChromaFeatures)
	if err != nil {
		s.Errorf(err, "Failed to marshal ChromaFeatures")
	}
	log.Printf("Extracted Chroma Features: %v", string(featuresJSON)) */
}

// Benchmark tests for performance
func (s *FeatureExtractorTestSuite) TestPerformance() {
	extractor, err := s.factory.CreateExtractor(config.ContentMusic, s.testConfig)
	s.Require().NoError(err)

	// Measure extraction time
	start := time.Now()
	features, err := extractor.ExtractFeatures(s.testSpectrogram, s.testPCM, s.testSampleRate)
	duration := time.Since(start)

	s.Require().NoError(err)
	s.Require().NotNil(features)

	// Should complete within reasonable time (adjust threshold as needed)
	s.Less(duration, 5*time.Second, "Feature extraction took too long: %v", duration)
}

// Run the test suite
func TestFeatureExtractorSuite(t *testing.T) {
	suite.Run(t, new(FeatureExtractorTestSuite))
}

// Additional unit tests for specific functionality
func TestFeatureExtractorFactory(t *testing.T) {
	factory := NewFeatureExtractorFactory()
	assert.NotNil(t, factory)

	testConfig := config.FeatureConfig{
		EnableMFCC:   true,
		EnableChroma: true,
	}

	// Test each content type
	contentTypes := []config.ContentType{
		config.ContentMusic,
		config.ContentNews,
		config.ContentTalk,
		config.ContentSports,
		config.ContentMixed,
		config.ContentUnknown,
	}

	for _, contentType := range contentTypes {
		t.Run(string(contentType), func(t *testing.T) {
			extractor, err := factory.CreateExtractor(contentType, testConfig)
			assert.NoError(t, err)
			assert.NotNil(t, extractor)
			assert.NotEmpty(t, extractor.GetName())
			assert.Equal(t, contentType, extractor.GetContentType())
		})
	}
}

func TestFeatureWeightsValidation(t *testing.T) {
	factory := NewFeatureExtractorFactory()
	testConfig := config.FeatureConfig{}

	extractors := []config.ContentType{
		config.ContentMusic,
		config.ContentNews,
		config.ContentSports,
		config.ContentMixed,
		config.ContentUnknown,
	}

	for _, contentType := range extractors {
		t.Run(string(contentType), func(t *testing.T) {
			extractor, err := factory.CreateExtractor(contentType, testConfig)
			require.NoError(t, err)

			weights := extractor.GetFeatureWeights()
			assert.NotEmpty(t, weights)

			// Validate weights sum approximately to 1.0
			sum := 0.0
			for _, weight := range weights {
				assert.GreaterOrEqual(t, weight, 0.0)
				sum += weight
			}
			assert.InDelta(t, 1.0, sum, 0.01, "Weights should sum to approximately 1.0")
		})
	}
}
