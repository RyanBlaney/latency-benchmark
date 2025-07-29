package config

type ContentAwareConfig struct {
	EnableContentDetection bool                           `json:"enable_content_detection"`
	DefaultContentType     ContentType                    `json:"default_content_type"`
	ContentConfigs         map[ContentType]*FeatureConfig `json:"content_configs"`
	AutoDetectThreshold    float64                        `json:"auto_detect_threshold"`
	FallbackStrategy       string                         `json:"fallback_strategy"` // "conservative", "aggressive", "adaptive"
}

type FeatureConfig struct {
	// Spectral Analysis
	SampleRate int        `json:"sample_rate"`
	WindowSize int        `json:"window_size"`
	HopSize    int        `json:"hop_size"`
	FreqRange  [2]float64 `json:"freq_range"` // [min, max] Hz

	// Feature Selection
	EnableChroma           bool `json:"enable_chroma"`
	EnableMFCC             bool `json:"enable_mfcc"`
	EnableSpectralContrast bool `json:"enable_spectral_contrast"`
	EnableTemporalFeatures bool `json:"enable_temporal_features"`
	EnableSpeechFeatures   bool `json:"enable_speech_features"`
	EnableHarmonicFeatures bool `json:"enable_harmonic_features"`

	// Content-specific parameters
	MFCCCoefficients int `json:"mfcc_coefficients"`
	ChromaBins       int `json:"chroma_bins"`
	ContrastBands    int `json:"contrast_bands"`

	// Matching parameters
	SimilarityWeights map[string]float64 `json:"similarity_weights"`
	MatchThreshold    float64            `json:"match_threshold"`
}

type ContentType string

const (
	ContentMusic   ContentType = "music"
	ContentNews    ContentType = "news"
	ContentSports  ContentType = "sports"
	ContentTalk    ContentType = "talk"
	ContentMixed   ContentType = "mixed"
	ContentUnknown ContentType = "unknown"
)

// ComparisonConfig configures fingerprint comparison (simplified for users)
type ComparisonConfig struct {
	SimilarityThreshold float64 `json:"similarity_threshold"` // 0.0-1.0

	// Method selection
	Method string `json:"method"` // "auto", "precise", "fast"

	// Optional advanced settings
	EnableDetailedMetrics bool `json:"enable_detailed_metrics,omitempty"`
	MaxCandidates         int  `json:"max_candidates,omitempty"`

	// Skips in-depth feature comparison if quantized fingerprints don't match
	EnableContentFilter bool `json:"enable_content_filter"`
}

type AlignmentConfig struct {
	// Core parameters
	MaxLagSeconds float64 `json:"max_lag_seconds"`
	MinConfidence float64 `json:"min_confidence"`
	StepSize      int     `json:"step_size"`

	// Method preferences
	PreferredMethod string `json:"preferred_method"` // "hybrid", "dtw", "correlation"
	FallbackMethod  string `json:"fallback_method"`

	// Quality thresholds
	MinSimilarity float64 `json:"min_similarity"`
	MinQuality    float64 `json:"min_quality"`

	// Algorithm settings
	DTWBandRadius     int     `json:"dtw_band_radius"`
	CorrNormalize     bool    `json:"corr_normalize"`
	ConsistencyTrials int     `json:"consistency_trials"`
	NoiseThreshold    float64 `json:"noise_threshold"`
}

func DefaultAlignmentConfig() AlignmentConfig {
	return AlignmentConfig{
		MaxLagSeconds:     30.0,
		MinConfidence:     0.6,
		StepSize:          1,
		PreferredMethod:   "hybrid",
		FallbackMethod:    "correlation",
		MinSimilarity:     0.3,
		MinQuality:        0.4,
		DTWBandRadius:     50,
		CorrNormalize:     true,
		ConsistencyTrials: 5,
		NoiseThreshold:    0.1,
	}
}

// DefaultComparisonConfig returns sensible defaults for comparison
func DefaultComparisonConfig() *ComparisonConfig {
	return &ComparisonConfig{
		SimilarityThreshold:   0.75,   // 75% similarity required
		Method:                "auto", // Let system choose best method
		MaxCandidates:         50,
		EnableDetailedMetrics: false, // Keep it simple by default
		EnableContentFilter:   false, // Prioritizes accuracy
	}
}

// GetContentOptimizedComparisonConfig returns optimized comparison config for content type
func GetContentOptimizedComparisonConfig(contentType ContentType) *ComparisonConfig {
	config := DefaultComparisonConfig()

	switch contentType {
	case ContentMusic:
		config.SimilarityThreshold = 0.80 // Higher threshold for music
		config.Method = "precise"

	case ContentNews, ContentTalk:
		config.SimilarityThreshold = 0.70 // Lower threshold for speech
		config.EnableContentFilter = false
		config.Method = "precise"

	case ContentSports:
		config.SimilarityThreshold = 0.75
		config.Method = "auto"

	case ContentMixed:
		config.SimilarityThreshold = 0.72
		config.Method = "auto"
		config.EnableDetailedMetrics = true // More analysis for mixed content
	}

	return config
}

// AlignmentConfigForContent generates an `AlignmentConfig` based on the specified
// `ContentType`. The configuration determines parameters such as minimum confidence
// required for valid alignment and the preferred method for alignment.
func AlignmentConfigForContent(contentType ContentType) *AlignmentConfig {
	config := DefaultAlignmentConfig()

	switch contentType {
	case ContentNews, ContentTalk:
		config.MinConfidence = 0.5
		config.PreferredMethod = "dtw"

	case ContentMusic:
		config.MinConfidence = 0.7
		config.PreferredMethod = "hybrid"

	case ContentSports:
		config.MinConfidence = 0.4

	case ContentMixed:
		config.MinConfidence = 0.5
		config.PreferredMethod = "hybrid"
	}

	return &config
}

// ComparisonConfigForContent generates a `ComparisonConfig` based on the specified
// `ContentType`. The configuration sets parameters for comparing audio,
// including similarity thresholds and comparison methods.
func ComparisonConfigForContent(contentType ContentType) ComparisonConfig {
	switch contentType {
	case ContentMusic:
		return ComparisonConfig{
			SimilarityThreshold: 0.80,
			Method:              "precise",
		}
	case ContentNews, ContentTalk:
		return ComparisonConfig{
			SimilarityThreshold: 0.70,
			Method:              "precise",
		}
	case ContentSports:
		return ComparisonConfig{
			SimilarityThreshold: 0.75,
			Method:              "auto",
		}
	default:
		return ComparisonConfig{
			SimilarityThreshold: 0.75,
			Method:              "auto",
		}
	}
}
