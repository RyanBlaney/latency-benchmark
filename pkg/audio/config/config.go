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
