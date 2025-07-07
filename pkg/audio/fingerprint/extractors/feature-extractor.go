package extractors

import (
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// FeatureExtractor defines the interface for content-specific feature extraction
type FeatureExtractor interface {
	ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error)
	GetFeatureWeights() map[string]float64
	GetName() string
	GetContentType() config.ContentType
}

// FeatureExtractorFactory creates feature extractors based on content type
type FeatureExtractorFactory struct {
	logger logging.Logger
}

// NewFeatureExtractorFactory creates a new feature extractor factory
func NewFeatureExtractorFactory() *FeatureExtractorFactory {
	return &FeatureExtractorFactory{
		logger: logging.WithFields(logging.Fields{
			"component": "feature_extractor_factory",
		}),
	}
}

// CreateExtractor creates a feature extractor for the specified content type
func (f *FeatureExtractorFactory) CreateExtractor(contentType config.ContentType, featureConfig config.FeatureConfig) (FeatureExtractor, error) {
	logger := f.logger.WithFields(logging.Fields{
		"function":     "CreateExtractor",
		"content_type": contentType,
	})

	switch contentType {
	case config.ContentMusic:
		logger.Debug("Creating music feature extractor")
		return NewMusicFeatureExtractor(&featureConfig), nil

	case config.ContentNews:
		logger.Debug("Creating news feature extractor")
		return NewSpeechFeatureExtractor(&featureConfig, true), nil

	case config.ContentTalk:
		logger.Debug("Creating talk feature extractor")
		return NewSpeechFeatureExtractor(&featureConfig, false), nil

	case config.ContentSports:
		logger.Debug("Creating sports feature extractor")
		return NewSportsFeatureExtractor(&featureConfig), nil

	case config.ContentMixed:
		logger.Debug("Creating mixed content feature extractor")
		return NewMixedFeatureExtractor(&featureConfig), nil

	default:
		logger.Debug("Creating general feature extractor for unknown content")
		return NewGeneralFeatureExtractor(&featureConfig), nil
	}
}
