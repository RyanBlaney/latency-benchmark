package extractors

import (
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// TODO: make distinction between talk and news

// SpeechFeatureExtractor extracts features optimized for speech/news content
type SpeechFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
}

// NewSpeechFeatureExtractor creates a speech-specific feature extractor
func NewSpeechFeatureExtractor(config *config.FeatureConfig) *SpeechFeatureExtractor {
	return &SpeechFeatureExtractor{
		config: config,
		logger: logging.WithFields(logging.Fields{
			"component": "speech_feature_extractor",
		}),
	}
}

func (s *SpeechFeatureExtractor) GetName() string {
	return "SpeechFeatureExtractor"
}

func (s *SpeechFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentNews
}

func (s *SpeechFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if s.config.SimilarityWeights != nil {
		return s.config.SimilarityWeights
	}

	// Default weights for speech content
	return map[string]float64{
		"mfcc":     0.50,
		"speech":   0.25,
		"spectral": 0.15,
		"temporal": 0.10,
	}
}

func (s *SpeechFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "ExtractFeatures",
		"frames":   spectrogram.TimeFrames,
	})

	logger.Debug("Extracting speech-specific features")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	if s.config.EnableSpeechFeatures {
		speechFeatures, err := s.extractSpeechFeatures(spectrogram, pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract speech features")
		} else {
			features.SpeechFeatures = speechFeatures
		}
	}

	// Extract basic spectral features (focused on speech range)
	spectralFeatures, err := s.extractSpeechSpectralFeatures(spectrogram)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")
	} else {
		features.SpectralFeatures = spectralFeatures
	}

	// Extract temporal features (pauses, speech rate)
	if s.config.EnableTemporalFeatures {
		temporalFeatures, err := s.extractSpeechTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "speech"
	features.ExtractionMetadata["mfcc_coefficients"] = s.config.MFCCCoefficients
	features.ExtractionMetadata["speech_features_enabled"] = s.config.EnableSpeechFeatures

	logger.Info("Speech feature extraction completed")
	return features, nil
}

func (s *SpeechFeatureExtractor) extractMFCC(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	return make([][]float64, spectrogram.TimeFrames), nil
}

func (s *SpeechFeatureExtractor) extractSpeechFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*SpeechFeatures, error) {
	return &SpeechFeatures{}, nil
}

func (s *SpeechFeatureExtractor) extractSpeechSpectralFeatures(spectrogram *analyzers.SpectrogramResult) (*SpectralFeatures, error) {
	return &SpectralFeatures{}, nil
}

func (s *SpeechFeatureExtractor) extractSpeechTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	return &TemporalFeatures{}, nil
}
