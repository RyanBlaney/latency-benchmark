package fingerprint

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/transcode"
)

// AudioFingerprint represents a complete audio fingerprint
type AudioFingerprint struct {
	ID               string                        `json:"id"`
	StreamURL        string                        `json:"stream_url"`
	StreamType       common.StreamType             `json:"stream_type"`
	ContentType      config.ContentType            `json:"content_type"`
	Timestamp        time.Time                     `json:"timestamp"`
	Duration         time.Duration                 `json:"duration"`
	SampleRate       int                           `json:"sample_rate"`
	HopSize          int                           `json:"hop_size"` // for alignment
	Channels         int                           `json:"channels"`
	Features         *extractors.ExtractedFeatures `json:"features"`
	CompactHash      string                        `json:"compact_hash"` // shortened and quantized
	DetailedHash     string                        `json:"detailed_hash"`
	PerceptualHashes map[string]string             `json:"perceptual_hash"`
	Metadata         map[string]any                `json:"metadata,omitempty"`
}

// FingerprintConfig holds configuration for fingerprint generation
type FingerprintConfig struct {
	WindowSize           int                              `json:"window_size"`
	HopSize              int                              `json:"hop_size"`
	EnableContentDetect  bool                             `json:"enable_content_detect"`
	HashResolution       HashResolution                   `json:"hash_resolution"`
	FeatureConfig        *config.FeatureConfig            `json:"feature_config"`
	ContentConfig        *config.ContentAwareConfig       `json:"content_config"`
	Quantization         int                              `json:"quantization"` // How many decimals to quantize features (only affects compact hash)
	PerceptualHashTypes  []PerceptualHashType             `json:"perceptual_hash_types"`
	UsePerceptualHashing bool                             `json:"use_perceptual_hashing"`
	PerceptualHashParams *extractors.PerceptualHashParams `json:"perceptual_hash_params"`
}

// HashResolution defines the resolution of the hash
type HashResolution string

const (
	HashLow    HashResolution = "low"    // 64-bit hash
	HashMedium HashResolution = "medium" // 128-bit hash
	HashHigh   HashResolution = "high"   // 256-bit hash
)

// PerceptualHashType defines different types of perceptual hashes
type PerceptualHashType = extractors.PerceptualHashType

const (
	HashSpectral = extractors.PerceptualSpectral
	HashTemporal = extractors.PerceptualTemporal
	HashMFCC     = extractors.PerceptualMFCC
	HashChroma   = extractors.PerceptualChroma
	HashCombined = extractors.PerceptualCombined
)

// FingerprintGenerator generates audio fingerprints
type FingerprintGenerator struct {
	config           *FingerprintConfig
	extractorFactory *extractors.FeatureExtractorFactory
	ContentDetector  *ContentDetector
	spectralAnalyzer *analyzers.SpectralAnalyzer
	perceptualHasher *extractors.PerceptualHasher
	logger           logging.Logger
}

// NewFingerprintGenerator creates a new fingerprint generator with configuration
func NewFingerprintGenerator(config *FingerprintConfig) *FingerprintGenerator {
	if config == nil {
		config = DefaultFingerprintConfig()
	}

	logger := logging.WithFields(logging.Fields{
		"component": "fingerprint_generator",
	})

	var perceptualHasher *extractors.PerceptualHasher
	if config.UsePerceptualHashing && config.PerceptualHashParams != nil {
		perceptualHasher = extractors.NewPerceptualHasherWithParams(*config.PerceptualHashParams)
	} else if config.UsePerceptualHashing {
		perceptualHasher = extractors.NewPerceptualHasher()
	}

	return &FingerprintGenerator{
		config:           config,
		extractorFactory: extractors.NewFeatureExtractorFactory(),
		spectralAnalyzer: analyzers.NewSpectralAnalyzer(44100), // Will be updated with actual sample rate
		ContentDetector:  NewContentDetector(config.ContentConfig),
		perceptualHasher: perceptualHasher,
		logger:           logger,
	}
}

// DefaultFingerprintConfig return default fingerprint configuration
func DefaultFingerprintConfig() *FingerprintConfig {
	return &FingerprintConfig{
		WindowSize:          2048,
		HopSize:             512,
		EnableContentDetect: true,
		HashResolution:      HashMedium,
		PerceptualHashTypes: []PerceptualHashType{
			HashSpectral,
			HashTemporal,
			HashMFCC,
			HashCombined,
		},
		FeatureConfig: &config.FeatureConfig{
			EnableMFCC:             true,
			EnableChroma:           true,
			EnableSpectralContrast: true,
			EnableHarmonicFeatures: false, // for performance
			EnableSpeechFeatures:   false, // enabled for speech content
			EnableTemporalFeatures: true,
			MFCCCoefficients:       13,
			ChromaBins:             12,
			SimilarityWeights: map[string]float64{
				"mfcc":     0.40,
				"spectral": 0.25,
				"chroma":   0.20,
				"temporal": 0.15,
			},
		},
		ContentConfig: &config.ContentAwareConfig{
			EnableContentDetection: true,
			DefaultContentType:     config.ContentUnknown,
			AutoDetectThreshold:    2.0,
		},
		Quantization:         3,
		UsePerceptualHashing: true, // Most accurate method
		PerceptualHashParams: &extractors.PerceptualHashParams{
			HashType:             extractors.PerceptualCombined, // Will be changed
			BinSize:              1.0,
			MaxCoefficients:      8,
			UseCoarseQuant:       true,
			HashLength:           16,
			SpectralBins:         100.0,
			TemporalBins:         5.0,
			MFCCBins:             0.5,
			BrightnessThresholds: [2]float64{1000, 2500},
			RolloffThresholds:    [2]float64{3000, 7000},
			DynamicsThresholds:   [2]float64{20, 40},
			SilenceThresholds:    [2]float64{0.1, 0.3},
		},
	}
}

func ContentOptimizedFingerprintConfig(contentType config.ContentType) *FingerprintConfig {
	cfg := DefaultFingerprintConfig()

	// Use content-optimized perceptual hash parameters
	cfg.PerceptualHashParams = extractors.ContentOptimizedPerceptualHashParams(contentType)

	// Adjust other settings based on content type
	switch contentType {
	case config.ContentNews, config.ContentTalk:
		cfg.PerceptualHashTypes = []PerceptualHashType{
			HashSpectral,
			HashMFCC,
			HashCombined,
		}

	case config.ContentMusic:
		cfg.PerceptualHashTypes = []PerceptualHashType{
			HashSpectral,
			HashMFCC,
			HashChroma,
			HashCombined,
		}

	case config.ContentSports:
		cfg.PerceptualHashTypes = []PerceptualHashType{
			HashTemporal,
			HashSpectral,
			HashCombined,
		}

	default:
		// Use all hash types for unknown content
	}

	return cfg
}

// GenerateFingerprint generates a complete audio fingerprint from audio data
func (fg *FingerprintGenerator) GenerateFingerprint(audioData *transcode.AudioData) (*AudioFingerprint, error) {
	if audioData == nil {
		return nil, fmt.Errorf("audio data cannot be nil")
	}

	logger := fg.logger.WithFields(logging.Fields{
		"function":    "GenerateFingerprint",
		"sample_rate": audioData.SampleRate,
		"channels":    audioData.Channels,
		"samples":     len(audioData.PCM),
	})

	logger.Info("Starting fingerprint generation")

	// Update spectral analyzer with correct sample rate
	fg.spectralAnalyzer = analyzers.NewSpectralAnalyzer(audioData.SampleRate)

	// Detect content type if enabled
	contentType := config.ContentUnknown
	if fg.config.EnableContentDetect {
		contentType = fg.ContentDetector.DetectContentType(audioData)
		logger.Info("Content type detected", logging.Fields{
			"content_type": contentType,
		})
	}

	// Update feature config based on content type
	adaptedConfig := fg.adaptConfigForContent(contentType)

	windowSize := fg.config.FeatureConfig.WindowSize
	hopSize := fg.config.FeatureConfig.HopSize
	windowType := analyzers.WindowHann // Good default for audio analysis

	logger.Info("STFT Configuration", logging.Fields{
		"window_size": windowSize,
		"hop_size":    hopSize,
		"pcm_length":  len(audioData.PCM),
	})

	spectrogram, err := fg.spectralAnalyzer.ComputeSTFTWithWindow(
		audioData.PCM,
		windowSize,
		hopSize,
		windowType)
	if err != nil {
		logger.Error(err, "Failed to compute STFT")
	}

	logger.Info("STFT Debug Info", logging.Fields{
		"pcm_length":     len(audioData.PCM),
		"window_size":    windowSize,
		"hop_size":       hopSize,
		"stft_frames":    spectrogram.TimeFrames,
		"stft_freq_bins": spectrogram.FreqBins,
	})

	// Extract features using appropriate extractor
	extractor, err := fg.extractorFactory.CreateExtractor(contentType, *adaptedConfig)
	if err != nil {
		logger.Error(err, "Failed to create feature extractor")
		return nil, err
	}

	features, err := extractor.ExtractFeatures(spectrogram, audioData.PCM, audioData.SampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract features")
		return nil, err
	}

	// Generate fingerprint
	fingerprint := &AudioFingerprint{
		ID:               generateID(audioData),
		StreamURL:        audioData.Metadata.URL,
		StreamType:       common.StreamType(audioData.Metadata.Type),
		ContentType:      contentType,
		Timestamp:        time.Now(),
		Duration:         calculateDuration(audioData),
		SampleRate:       audioData.SampleRate,
		HopSize:          fg.config.FeatureConfig.HopSize,
		Channels:         audioData.Channels,
		Features:         features,
		PerceptualHashes: make(map[string]string),
		Metadata:         make(map[string]any),
	}

	// Generate hashes
	if err := fg.generateHashes(fingerprint); err != nil {
		logger.Error(err, "Failed to generate hashes")
		return nil, err
	}

	// Add metadata
	addMetadata(fingerprint, audioData, extractor, fg.config)

	logger.Info("Fingerprint generation completed", logging.Fields{
		"fingerprint_id": fingerprint.ID,
		"compact_hash":   fingerprint.CompactHash[:16] + "...", // show first 16 characters
		"content_type":   fingerprint.ContentType,
	})

	return fingerprint, nil
}

// adaptConfigForContent adapts feature configuration based on content type
func (fg *FingerprintGenerator) adaptConfigForContent(contentType config.ContentType) *config.FeatureConfig {
	adaptedConfig := *fg.config.FeatureConfig // Copy

	// TODO: get rid of magic numbers with a possible config

	switch contentType {
	case config.ContentMusic:
		adaptedConfig.EnableHarmonicFeatures = true
		adaptedConfig.EnableChroma = true
		adaptedConfig.EnableSpeechFeatures = false
		adaptedConfig.SimilarityWeights = map[string]float64{
			"mfcc":     0.35,
			"chroma":   0.30,
			"harmonic": 0.20,
			"spectral": 0.15,
		}

	case config.ContentNews, config.ContentTalk:
		adaptedConfig.EnableSpeechFeatures = true
		adaptedConfig.EnableHarmonicFeatures = false
		adaptedConfig.EnableChroma = false
		adaptedConfig.SimilarityWeights = map[string]float64{
			"mfcc":     0.50,
			"speech":   0.25,
			"spectral": 0.15,
			"temporal": 0.10,
		}

	case config.ContentSports:
		adaptedConfig.EnableTemporalFeatures = true
		adaptedConfig.EnableSpeechFeatures = false
		adaptedConfig.SimilarityWeights = map[string]float64{
			"mfcc":     0.30,
			"spectral": 0.25,
			"temporal": 0.25,
			"energy":   0.20,
		}

	case config.ContentMixed:
		// Enable all features for mixed content
		adaptedConfig.EnableHarmonicFeatures = true
		adaptedConfig.EnableSpeechFeatures = true
		adaptedConfig.EnableChroma = true
		adaptedConfig.SimilarityWeights = map[string]float64{
			"mfcc":     0.30,
			"spectral": 0.20,
			"temporal": 0.20,
			"chroma":   0.15,
			"speech":   0.15,
		}
	}

	return &adaptedConfig
}

// generateHashes generates various types of hashes for the fingerprint
func (fg *FingerprintGenerator) generateHashes(fingerprint *AudioFingerprint) error {
	// Generate detailed hash from all features
	detailedData, err := json.Marshal(fingerprint.Features)
	if err != nil {
		fg.logger.Error(err, "Failed to marshal features for detailed hash")
		return err
	}

	detailedHasher := sha256.New()
	detailedHasher.Write(detailedData)
	fingerprint.DetailedHash = hex.EncodeToString(detailedHasher.Sum(nil))

	if fg.config.UsePerceptualHashing && fg.perceptualHasher != nil {
		// Use perceptual hashing for robustness
		result, err := fg.perceptualHasher.GenerateHash(fingerprint.Features)
		if err != nil {
			fg.logger.Warn("Failed to generate perceptual hash, falling back to traditional method", logging.Fields{
				"error": err.Error(),
			})
			// Fall back to traditional method
			fg.generateCompactHashFallback(fingerprint)
		} else {
			fingerprint.CompactHash = result.Hash
			fg.logger.Info("Generated perceptual compact hash", logging.Fields{
				"hash":          result.Hash,
				"hash_type":     result.HashType,
				"feature_count": result.FeatureCount,
				"bin_count":     result.BinCount,
				"robustness":    result.Robustness,
			})
		}
	} else {
		// Use traditional cryptographic hashing
		fg.generateCompactHashFallback(fingerprint)
	}

	// Generate perceptual hashes for different feature types
	for _, hashType := range fg.config.PerceptualHashTypes {
		if fg.perceptualHasher != nil {
			// Create a hasher for this specific type
			params := *fg.config.PerceptualHashParams
			params.HashType = extractors.PerceptualHashType(hashType)
			typeHasher := extractors.NewPerceptualHasherWithParams(params)

			result, err := typeHasher.GenerateHash(fingerprint.Features)
			if err != nil {
				fg.logger.Warn("Failed to generate perceptual hash", logging.Fields{
					"hash_type": hashType,
					"error":     err.Error(),
				})
				continue
			}
			fingerprint.PerceptualHashes[string(hashType)] = result.Hash

			fg.logger.Debug("Generated perceptual hash type", logging.Fields{
				"hash_type":     hashType,
				"hash":          result.Hash,
				"feature_count": result.FeatureCount,
				"robustness":    result.Robustness,
			})
		} else {
			// Fall back to traditional method
			hash, err := fg.generatePerceptualHashFallback(fingerprint.Features, hashType)
			if err != nil {
				fg.logger.Warn("Failed to generate fallback perceptual hash", logging.Fields{
					"hash_type": hashType,
					"error":     err.Error(),
				})
				continue
			}
			fingerprint.PerceptualHashes[string(hashType)] = hash
		}
	}

	return nil
}

func (fg *FingerprintGenerator) generateCompactHashFallback(fingerprint *AudioFingerprint) {
	compactData := fg.extractCompactFeatures(fingerprint.Features)
	compactHasher := sha256.New()
	compactHasher.Write(compactData)
	fullCompactHash := hex.EncodeToString(compactHasher.Sum(nil))

	// Truncate based on resolution
	switch fg.config.HashResolution {
	case HashLow:
		fingerprint.CompactHash = fullCompactHash[:16] // 64-bit
	case HashMedium:
		fingerprint.CompactHash = fullCompactHash[:32] // 128-bit
	case HashHigh:
		fingerprint.CompactHash = fullCompactHash // 256-bit
	default:
		fingerprint.CompactHash = fullCompactHash[:32] // Default: medium
	}

	fg.logger.Debug("Generated traditional compact hash", logging.Fields{
		"hash": fingerprint.CompactHash,
	})
}

func (fg *FingerprintGenerator) generatePerceptualHashFallback(features *extractors.ExtractedFeatures, hashType PerceptualHashType) (string, error) {
	var hashData []byte

	switch hashType {
	case HashSpectral:
		hashData = fg.generateSpectralHash(features)
	case HashTemporal:
		hashData = fg.generateTemporalHash(features)
	case HashMFCC:
		hashData = fg.generateMFCCHash(features)
	case HashChroma:
		hashData = fg.generateChromaHash(features)
	case HashCombined:
		hashData = fg.generateCombinedHash(features)
	default:
		return "", fmt.Errorf("unsupported hash type: %s", hashType)
	}

	if len(hashData) == 0 {
		return "", fmt.Errorf("no data available for hash type: %s", hashType)
	}

	hasher := sha256.New()
	hasher.Write(hashData)
	hash := hex.EncodeToString(hasher.Sum(nil))[:32] // 128-bit hash

	fg.logger.Debug("Generated fallback perceptual hash", logging.Fields{
		"hash_type": hashType,
		"hash":      hash,
		"method":    "traditional_cryptographic",
	})

	return hash, nil
}

// extractCompactFeatures extracts the most important features compact hashing
func (fg *FingerprintGenerator) extractCompactFeatures(features *extractors.ExtractedFeatures) []byte {
	compactFeatures := make(map[string]any)

	// MFCC (most important for audio similarity)
	if len(features.MFCC) > 0 {
		// Use mean and std of first few MFCC coefficients
		compactFeatures["mfcc_mean"] = quantizeSlice(calculateMFCCStats(features.MFCC, "mean"), fg.config.Quantization)
		compactFeatures["mfcc_std"] = quantizeSlice(calculateMFCCStats(features.MFCC, "std"), fg.config.Quantization)
	}

	// Spectral features summary
	if features.SpectralFeatures != nil {
		compactFeatures["spectral_centroid_mean"] = quantizeFloat(calculateMean(features.SpectralFeatures.SpectralCentroid), fg.config.Quantization)
		compactFeatures["spectral_rolloff_mean"] = quantizeFloat(calculateMean(features.SpectralFeatures.SpectralRolloff), fg.config.Quantization)
		compactFeatures["spectral_flatness_mean"] = quantizeFloat(calculateMean(features.SpectralFeatures.SpectralFlatness), fg.config.Quantization)
	}

	// Chroma features (for harmonic content)
	if len(features.ChromaFeatures) > 0 {
		compactFeatures["chroma_mean"] = quantizeSlice(calculateChromaStats(features.ChromaFeatures, "mean"), fg.config.Quantization)
	}

	// Temporal features summary
	if features.TemporalFeatures != nil {
		compactFeatures["dynamic_range"] = quantizeFloat(features.TemporalFeatures.DynamicRange, fg.config.Quantization)
		compactFeatures["silence_ratio"] = quantizeFloat(features.TemporalFeatures.SilenceRatio, fg.config.Quantization)
	}

	// Convert to JSON for consistent hashing
	data, _ := json.Marshal(compactFeatures)
	return data
}

// generateSpectralHash creates hash from spectral features
func (fg *FingerprintGenerator) generateSpectralHash(features *extractors.ExtractedFeatures) []byte {
	if features.SpectralFeatures == nil {
		return []byte{}
	}

	spectralData := map[string]any{
		"centroid_mean":  calculateMean(features.SpectralFeatures.SpectralCentroid),
		"rolloff_mean":   calculateMean(features.SpectralFeatures.SpectralRolloff),
		"flatness_mean":  calculateMean(features.SpectralFeatures.SpectralFlatness),
		"bandwidth_mean": calculateMean(features.SpectralFeatures.SpectralBandwidth),
		"flux_mean":      calculateMean(features.SpectralFeatures.SpectralFlux),
	}

	data, err := json.Marshal(spectralData)
	if err != nil {
		fg.logger.Warn("Failed to marshal spectral hash", logging.Fields{
			"error": err.Error(),
		})
		return []byte{}
	}
	return data
}

// generateTemporalHash creates hash from temporal features
func (fg *FingerprintGenerator) generateTemporalHash(features *extractors.ExtractedFeatures) []byte {
	if features.TemporalFeatures == nil {
		return []byte{}
	}

	temporalData := map[string]any{
		"dynamic_range":   features.TemporalFeatures.DynamicRange,
		"silence_ratio":   features.TemporalFeatures.SilenceRatio,
		"onet_density":    features.TemporalFeatures.OnsetDensity,
		"rms_energy_mean": calculateMean(features.TemporalFeatures.RMSEnergy),
	}

	data, err := json.Marshal(temporalData)
	if err != nil {
		fg.logger.Warn("Failed to marshal temporal hash", logging.Fields{
			"error": err.Error(),
		})
		return []byte{}
	}
	return data
}

// generateMFCCHash creates hash from MFCC features
func (fg *FingerprintGenerator) generateMFCCHash(features *extractors.ExtractedFeatures) []byte {
	if len(features.MFCC) == 0 {
		return []byte{}
	}

	mfccData := map[string]any{
		"mfcc_mean":  calculateMFCCStats(features.MFCC, "mean"),
		"mfcc_std":   calculateMFCCStats(features.MFCC, "std"),
		"mfcc_delta": calculateMFCCStats(features.MFCC, "delta"),
	}

	data, err := json.Marshal(mfccData)
	if err != nil {
		fg.logger.Warn("Failed to marshal MFCC hash", logging.Fields{
			"error": err.Error(),
		})
		return []byte{}
	}
	return data
}

// generateChromaHash creates hash from chroma features
func (fg *FingerprintGenerator) generateChromaHash(features *extractors.ExtractedFeatures) []byte {
	if len(features.ChromaFeatures) == 0 {
		return []byte{}
	}

	chromaData := map[string]any{
		"chroma_mean": calculateChromaStats(features.ChromaFeatures, "mean"),
		"chroma_std":  calculateChromaStats(features.ChromaFeatures, "std"),
	}

	data, err := json.Marshal(chromaData)
	if err != nil {
		fg.logger.Warn("Failed to marshal chroma hash", logging.Fields{
			"error": err.Error(),
		})
		return []byte{}
	}
	return data
}

// generateCombinedHash creates a hash from all available features
func (fg *FingerprintGenerator) generateCombinedHash(features *extractors.ExtractedFeatures) []byte {
	combinedData := make(map[string]any)

	// Add spectral features
	if features.SpectralFeatures != nil {
		combinedData["spectral_centroid"] = calculateMean(features.SpectralFeatures.SpectralCentroid)
		combinedData["spectral_rolloff"] = calculateMean(features.SpectralFeatures.SpectralRolloff)
	}

	// Add MFCC features
	if len(features.MFCC) > 0 {
		combinedData["mfcc_mean"] = calculateMFCCStats(features.MFCC, "mean")
	}

	// Add temporal features
	if features.TemporalFeatures != nil {
		combinedData["dynamic_range"] = features.TemporalFeatures.DynamicRange
		combinedData["silence_ratio"] = features.TemporalFeatures.SilenceRatio
	}

	// Add chroma features
	if len(features.ChromaFeatures) > 0 {
		combinedData["chroma_mean"] = calculateChromaStats(features.ChromaFeatures, "mean")
	}

	data, err := json.Marshal(combinedData)
	if err != nil {
		fg.logger.Warn("Failed to marshal combined hash", logging.Fields{
			"error": err.Error(),
		})
		return []byte{}
	}
	return data
}
