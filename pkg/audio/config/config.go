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
	// Basic settings
	SimilarityThreshold float64 `json:"similarity_threshold"` // 0.0-1.0, main threshold users care about
	Method              string  `json:"method"`               // "auto", "precise", "fast"

	// Advanced settings (optional)
	EnableContentFilter   bool `json:"enable_content_filter"`   // Filter by content type
	MaxCandidates         int  `json:"max_candidates"`          // Limit results
	EnableDetailedMetrics bool `json:"enable_detailed_metrics"` // Calculate extra metrics
}

// DefaultComparisonConfig returns sensible defaults for comparison
func DefaultComparisonConfig() *ComparisonConfig {
	return &ComparisonConfig{
		SimilarityThreshold:   0.75,   // 75% similarity required
		Method:                "auto", // Let system choose best method
		EnableContentFilter:   true,
		MaxCandidates:         50,
		EnableDetailedMetrics: false, // Keep it simple by default
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
		config.Method = "fast"

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
