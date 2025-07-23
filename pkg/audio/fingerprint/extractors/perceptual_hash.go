package extractors

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// PerceptualHashType defines different types of perceptual hashes
type PerceptualHashType string

const (
	PerceptualSpectral PerceptualHashType = "spectral"
	PerceptualTemporal PerceptualHashType = "temporal"
	PerceptualMFCC     PerceptualHashType = "mfcc"
	PerceptualChroma   PerceptualHashType = "chroma"
	PerceptualCombined PerceptualHashType = "combined"
)

// PerceptualHashParams configures perceptual hash generation
type PerceptualHashParams struct {
	HashType        PerceptualHashType `json:"hash_type"`
	BinSize         float64            `json:"bin_size"`         // Size of quantization bins
	MaxCoefficients int                `json:"max_coefficients"` // Max MFCC coefficients to use
	UseCoarseQuant  bool               `json:"use_coarse_quant"` // Use very coarse quantization
	HashLength      int                `json:"hash_length"`      // Length of output hash

	// Robustness settings
	SpectralBins float64 `json:"spectral_bins"` // Hz per bin for spectral features
	TemporalBins float64 `json:"temporal_bins"` // dB per bin for temporal features
	MFCCBins     float64 `json:"mfcc_bins"`     // Units per bin for MFCC

	// Classification thresholds
	BrightnessThresholds [2]float64 `json:"brightness_thresholds"` // [dark/medium, medium/bright]
	RolloffThresholds    [2]float64 `json:"rolloff_thresholds"`    // [low/medium, medium/high]
	DynamicsThresholds   [2]float64 `json:"dynamics_thresholds"`   // [compressed/normal, normal/dynamic]
	SilenceThresholds    [2]float64 `json:"silence_thresholds"`    // [continuous/some, some/many]
}

func ContentOptimizedPerceptualHashParams(contentType config.ContentType) *PerceptualHashParams {
	params := DefaultPerceptualHashParams()

	switch contentType {
	case config.ContentNews, config.ContentTalk:
		// For speech content, focus on MFCC and spectral features
		params.HashType = PerceptualCombined
		params.MaxCoefficients = 6  // Fewer MFCC coefficients for robustness
		params.MFCCBins = 1.0       // Larger bins for more robustness
		params.SpectralBins = 150.0 // Larger spectral bins

	case config.ContentMusic:
		// For music, include chroma and more detailed spectral analysis
		params.HashType = PerceptualCombined
		params.MaxCoefficients = 8
		params.MFCCBins = 0.5
		params.SpectralBins = 100.0

	case config.ContentSports:
		// For sports, focus on temporal and energy characteristics
		params.HashType = PerceptualTemporal
		params.TemporalBins = 3.0  // Smaller bins for energy characteristics
		params.MaxCoefficients = 4 // Fewer MFCC coefficients

	default:
		// Use defaults for mixed/unknown content
	}

	return &params
}

// PerceptualHashResult contains the generated hash and metadata
type PerceptualHashResult struct {
	Hash         string             `json:"hash"`
	HashType     PerceptualHashType `json:"hash_type"`
	BinCount     int                `json:"bin_count"`     // Number of bins used
	FeatureCount int                `json:"feature_count"` // Number of features hashed
	Metadata     map[string]any     `json:"metadata"`      // Additional info about the hash
	Robustness   float64            `json:"robustness"`    // Estimated robustness level (0-1)
}

// PerceptualHasher generates perceptual hashes from audio features
type PerceptualHasher struct {
	params PerceptualHashParams
	logger logging.Logger
}

// NewPerceptualHasher creates a new perceptual hasher
func NewPerceptualHasher() *PerceptualHasher {
	return &PerceptualHasher{
		params: DefaultPerceptualHashParams(),
		logger: logging.WithFields(logging.Fields{
			"component": "perceptual_hasher",
		}),
	}
}

// NewPerceptualHasherWithParams creates a hasher with custom parameters
func NewPerceptualHasherWithParams(params PerceptualHashParams) *PerceptualHasher {
	return &PerceptualHasher{
		params: params,
		logger: logging.WithFields(logging.Fields{
			"component": "perceptual_hasher",
		}),
	}
}

// DefaultPerceptualHashParams returns sensible defaults for perceptual hashing
func DefaultPerceptualHashParams() PerceptualHashParams {
	return PerceptualHashParams{
		HashType:        PerceptualCombined,
		BinSize:         1.0,
		MaxCoefficients: 8, // First 8 MFCC coefficients are most stable
		UseCoarseQuant:  true,
		HashLength:      16, // 16 character hash for good balance

		SpectralBins: 100.0, // 100 Hz bins for spectral features
		TemporalBins: 5.0,   // 5 dB bins for temporal features
		MFCCBins:     0.5,   // 0.5 unit bins for MFCC

		BrightnessThresholds: [2]float64{1000, 2500}, // Hz thresholds
		RolloffThresholds:    [2]float64{3000, 7000}, // Hz thresholds
		DynamicsThresholds:   [2]float64{20, 40},     // dB thresholds
		SilenceThresholds:    [2]float64{0.1, 0.3},   // Ratio thresholds
	}
}

// GenerateHash generates a perceptual hash from extracted features
func (ph *PerceptualHasher) GenerateHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	if features == nil {
		return nil, fmt.Errorf("features cannot be nil")
	}

	ph.logger.Debug("Generating perceptual hash", logging.Fields{
		"hash_type": ph.params.HashType,
	})

	switch ph.params.HashType {
	case PerceptualSpectral:
		return ph.generateSpectralHash(features)
	case PerceptualTemporal:
		return ph.generateTemporalHash(features)
	case PerceptualMFCC:
		return ph.generateMFCCHash(features)
	case PerceptualChroma:
		return ph.generateChromaHash(features)
	case PerceptualCombined:
		return ph.generateCombinedHash(features)
	default:
		return nil, fmt.Errorf("unsupported hash type: %s", ph.params.HashType)
	}
}

// generateSpectralHash creates a perceptual hash from spectral features
func (ph *PerceptualHasher) generateSpectralHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	if features.SpectralFeatures == nil {
		return nil, fmt.Errorf("no spectral features available")
	}

	spectralData := make(map[string]any)
	featureCount := 0

	// Spectral centroid (brightness)
	if len(features.SpectralFeatures.SpectralCentroid) > 0 {
		centroidMean := ph.calculateMean(features.SpectralFeatures.SpectralCentroid)
		spectralData["brightness"] = ph.classifyBrightness(centroidMean)
		spectralData["centroid_bin"] = ph.binValue(centroidMean, ph.params.SpectralBins)
		featureCount++

		ph.logger.Debug("Spectral centroid analysis", logging.Fields{
			"mean":       centroidMean,
			"brightness": spectralData["brightness"],
			"bin":        spectralData["centroid_bin"],
		})
	}

	// Spectral rolloff (high frequency content)
	if len(features.SpectralFeatures.SpectralRolloff) > 0 {
		rolloffMean := ph.calculateMean(features.SpectralFeatures.SpectralRolloff)
		spectralData["rolloff_class"] = ph.classifyRolloff(rolloffMean)
		spectralData["rolloff_bin"] = ph.binValue(rolloffMean, ph.params.SpectralBins*2) // Larger bins for rolloff
		featureCount++
	}

	// Spectral flatness (tonal vs noise)
	if len(features.SpectralFeatures.SpectralFlatness) > 0 {
		flatnessMean := ph.calculateMean(features.SpectralFeatures.SpectralFlatness)
		spectralData["tonality"] = ph.classifyTonality(flatnessMean)
		featureCount++
	}

	// Spectral bandwidth (spectral spread)
	if len(features.SpectralFeatures.SpectralBandwidth) > 0 {
		bandwidthMean := ph.calculateMean(features.SpectralFeatures.SpectralBandwidth)
		spectralData["bandwidth_bin"] = ph.binValue(bandwidthMean, ph.params.SpectralBins)
		featureCount++
	}

	hash, binCount := ph.createHash(spectralData)

	result := &PerceptualHashResult{
		Hash:         hash,
		HashType:     PerceptualSpectral,
		BinCount:     binCount,
		FeatureCount: featureCount,
		Metadata:     spectralData,
		Robustness:   ph.estimateRobustness(featureCount, binCount),
	}

	ph.logger.Debug("Generated spectral hash", logging.Fields{
		"hash":          hash,
		"feature_count": featureCount,
		"robustness":    result.Robustness,
	})

	return result, nil
}

// generateMFCCHash creates a perceptual hash from MFCC features
func (ph *PerceptualHasher) generateMFCCHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	if len(features.MFCC) == 0 {
		return nil, fmt.Errorf("no MFCC features available")
	}

	mfccData := make(map[string]any)
	featureCount := 0

	// Use only the most stable coefficients
	numCoeffs := min(ph.params.MaxCoefficients, len(features.MFCC[0]))

	ph.logger.Debug("Processing MFCC coefficients", logging.Fields{
		"total_coeffs": len(features.MFCC[0]),
		"using_coeffs": numCoeffs,
		"frames":       len(features.MFCC),
	})

	for c := range numCoeffs {
		// Extract coefficient values across time
		values := make([]float64, len(features.MFCC))
		for t := range len(features.MFCC) {
			if c < len(features.MFCC[t]) {
				values[t] = features.MFCC[t][c]
			}
		}

		if len(values) > 0 {
			mean := ph.calculateMean(values)
			stddev := ph.calculateStdDev(values)

			// Bin the mean and standard deviation
			mfccData[fmt.Sprintf("c%d_mean_bin", c)] = ph.binValue(mean, ph.params.MFCCBins)
			mfccData[fmt.Sprintf("c%d_std_bin", c)] = ph.binValue(stddev, ph.params.MFCCBins)
			featureCount += 2

			if c < 3 { // Log first few coefficients for debugging
				ph.logger.Debug("MFCC coefficient analysis", logging.Fields{
					"coeff":    c,
					"mean":     mean,
					"stddev":   stddev,
					"mean_bin": mfccData[fmt.Sprintf("c%d_mean_bin", c)],
					"std_bin":  mfccData[fmt.Sprintf("c%d_std_bin", c)],
				})
			}
		}
	}

	// Special handling for C0 (energy-related)
	if len(features.MFCC) > 0 && len(features.MFCC[0]) > 0 {
		c0Values := make([]float64, len(features.MFCC))
		for t := range len(features.MFCC) {
			c0Values[t] = features.MFCC[t][0]
		}
		c0Mean := ph.calculateMean(c0Values)
		mfccData["energy_class"] = ph.classifyEnergy(c0Mean)

		ph.logger.Debug("MFCC C0 (energy) analysis", logging.Fields{
			"c0_mean":      c0Mean,
			"energy_class": mfccData["energy_class"],
		})
	}

	hash, binCount := ph.createHash(mfccData)

	result := &PerceptualHashResult{
		Hash:         hash,
		HashType:     PerceptualMFCC,
		BinCount:     binCount,
		FeatureCount: featureCount,
		Metadata:     mfccData,
		Robustness:   ph.estimateRobustness(featureCount, binCount),
	}

	ph.logger.Debug("Generated MFCC hash", logging.Fields{
		"hash":          hash,
		"feature_count": featureCount,
		"robustness":    result.Robustness,
	})

	return result, nil
}

// generateTemporalHash creates a perceptual hash from temporal features
func (ph *PerceptualHasher) generateTemporalHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	if features.TemporalFeatures == nil {
		return nil, fmt.Errorf("no temporal features available")
	}

	temporalData := make(map[string]any)
	featureCount := 0

	// Dynamic range
	if features.TemporalFeatures.DynamicRange > 0 {
		temporalData["dynamics_class"] = ph.classifyDynamics(features.TemporalFeatures.DynamicRange)
		temporalData["dynamics_bin"] = ph.binValue(features.TemporalFeatures.DynamicRange, ph.params.TemporalBins)
		featureCount++

		ph.logger.Debug("Dynamic range analysis", logging.Fields{
			"dynamic_range":  features.TemporalFeatures.DynamicRange,
			"dynamics_class": temporalData["dynamics_class"],
		})
	}

	// Silence ratio
	temporalData["silence_class"] = ph.classifySilence(features.TemporalFeatures.SilenceRatio)
	temporalData["silence_bin"] = ph.binValue(features.TemporalFeatures.SilenceRatio*100, 5.0) // 5% bins
	featureCount++

	// RMS energy pattern (if available)
	if len(features.TemporalFeatures.RMSEnergy) > 0 {
		energyMean := ph.calculateMean(features.TemporalFeatures.RMSEnergy)
		energyStd := ph.calculateStdDev(features.TemporalFeatures.RMSEnergy)

		temporalData["energy_level"] = ph.classifyEnergyLevel(energyMean)
		temporalData["energy_var"] = ph.classifyEnergyVariation(energyStd)
		featureCount += 2
	}

	// Peak amplitude
	if features.TemporalFeatures.PeakAmplitude > 0 {
		temporalData["peak_class"] = ph.classifyPeakLevel(features.TemporalFeatures.PeakAmplitude)
		featureCount++
	}

	hash, binCount := ph.createHash(temporalData)

	result := &PerceptualHashResult{
		Hash:         hash,
		HashType:     PerceptualTemporal,
		BinCount:     binCount,
		FeatureCount: featureCount,
		Metadata:     temporalData,
		Robustness:   ph.estimateRobustness(featureCount, binCount),
	}

	ph.logger.Debug("Generated temporal hash", logging.Fields{
		"hash":          hash,
		"feature_count": featureCount,
		"robustness":    result.Robustness,
	})

	return result, nil
}

// generateChromaHash creates a perceptual hash from chroma features
func (ph *PerceptualHasher) generateChromaHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	if len(features.ChromaFeatures) == 0 {
		return nil, fmt.Errorf("no chroma features available")
	}

	chromaData := make(map[string]any)

	// Calculate mean chroma vector
	chromaMeans := ph.calculateMeanChromaVector(features.ChromaFeatures)
	if len(chromaMeans) == 0 {
		return nil, fmt.Errorf("failed to calculate chroma means")
	}

	// Find dominant pitch classes (most prominent notes)
	dominantClasses := ph.findDominantPitchClasses(chromaMeans, 3) // Top 3
	chromaData["dominant_classes"] = dominantClasses

	// Calculate chroma energy distribution
	totalEnergy := 0.0
	for _, energy := range chromaMeans {
		totalEnergy += energy
	}

	if totalEnergy > 0 {
		// Classify the harmonic content
		chromaData["harmonic_complexity"] = ph.classifyHarmonicComplexity(chromaMeans)
		chromaData["tonal_strength"] = ph.classifyTonalStrength(totalEnergy)
	}

	hash, binCount := ph.createHash(chromaData)

	result := &PerceptualHashResult{
		Hash:         hash,
		HashType:     PerceptualChroma,
		BinCount:     binCount,
		FeatureCount: len(chromaData),
		Metadata:     chromaData,
		Robustness:   ph.estimateRobustness(len(chromaData), binCount),
	}

	ph.logger.Debug("Generated chroma hash", logging.Fields{
		"hash":          hash,
		"feature_count": len(chromaData),
		"robustness":    result.Robustness,
	})

	return result, nil
}

// generateCombinedHash creates the most robust hash combining multiple feature types
func (ph *PerceptualHasher) generateCombinedHash(features *ExtractedFeatures) (*PerceptualHashResult, error) {
	combinedData := make(map[string]any)
	featureCount := 0

	ph.logger.Debug("Generating combined perceptual hash")

	// Add spectral characteristics (if available)
	if features.SpectralFeatures != nil {
		if len(features.SpectralFeatures.SpectralCentroid) > 0 {
			centroidMean := ph.calculateMean(features.SpectralFeatures.SpectralCentroid)
			combinedData["brightness"] = ph.classifyBrightness(centroidMean)
			featureCount++
		}

		if len(features.SpectralFeatures.SpectralRolloff) > 0 {
			rolloffMean := ph.calculateMean(features.SpectralFeatures.SpectralRolloff)
			combinedData["rolloff_class"] = ph.classifyRolloff(rolloffMean)
			featureCount++
		}
	}

	// Add MFCC characteristics (if available)
	if len(features.MFCC) > 0 && len(features.MFCC[0]) > 0 {
		// Use only first few coefficients for robustness
		for c := 0; c < min(4, len(features.MFCC[0])); c++ {
			values := make([]float64, len(features.MFCC))
			for t := range len(features.MFCC) {
				if c < len(features.MFCC[t]) {
					values[t] = features.MFCC[t][c]
				}
			}
			mean := ph.calculateMean(values)
			combinedData[fmt.Sprintf("mfcc_c%d", c)] = ph.binValue(mean, ph.params.MFCCBins*2) // Larger bins for robustness
		}
		featureCount += min(4, len(features.MFCC[0]))
	}

	// Add temporal characteristics (if available)
	if features.TemporalFeatures != nil {
		if features.TemporalFeatures.DynamicRange > 0 {
			combinedData["dynamics"] = ph.classifyDynamics(features.TemporalFeatures.DynamicRange)
			featureCount++
		}

		combinedData["silence"] = ph.classifySilence(features.TemporalFeatures.SilenceRatio)
		featureCount++
	}

	// Add chroma characteristics (if available)
	if len(features.ChromaFeatures) > 0 {
		chromaMeans := ph.calculateMeanChromaVector(features.ChromaFeatures)
		if len(chromaMeans) > 0 {
			combinedData["harmonic"] = ph.classifyHarmonicComplexity(chromaMeans)
			featureCount++
		}
	}

	hash, binCount := ph.createHash(combinedData)

	result := &PerceptualHashResult{
		Hash:         hash,
		HashType:     PerceptualCombined,
		BinCount:     binCount,
		FeatureCount: featureCount,
		Metadata:     combinedData,
		Robustness:   ph.estimateRobustness(featureCount, binCount),
	}

	ph.logger.Info("Generated combined perceptual hash", logging.Fields{
		"hash":          hash,
		"feature_count": featureCount,
		"robustness":    result.Robustness,
		"metadata":      combinedData,
	})

	return result, nil
}

// Classification helper functions
func (ph *PerceptualHasher) binValue(value, binSize float64) int {
	if binSize <= 0 {
		return int(math.Round(value))
	}
	return int(math.Floor(value / binSize))
}

func (ph *PerceptualHasher) classifyBrightness(centroidMean float64) string {
	if centroidMean < ph.params.BrightnessThresholds[0] {
		return "dark"
	} else if centroidMean < ph.params.BrightnessThresholds[1] {
		return "medium"
	} else {
		return "bright"
	}
}

func (ph *PerceptualHasher) classifyRolloff(rolloffMean float64) string {
	if rolloffMean < ph.params.RolloffThresholds[0] {
		return "low"
	} else if rolloffMean < ph.params.RolloffThresholds[1] {
		return "medium"
	} else {
		return "high"
	}
}

func (ph *PerceptualHasher) classifyTonality(flatnessMean float64) string {
	if flatnessMean < 0.1 {
		return "tonal"
	} else if flatnessMean < 0.5 {
		return "mixed"
	} else {
		return "noise"
	}
}

func (ph *PerceptualHasher) classifyDynamics(dynamicRange float64) string {
	if dynamicRange < ph.params.DynamicsThresholds[0] {
		return "compressed"
	} else if dynamicRange < ph.params.DynamicsThresholds[1] {
		return "normal"
	} else {
		return "dynamic"
	}
}

func (ph *PerceptualHasher) classifySilence(silenceRatio float64) string {
	if silenceRatio < ph.params.SilenceThresholds[0] {
		return "continuous"
	} else if silenceRatio < ph.params.SilenceThresholds[1] {
		return "some_pauses"
	} else {
		return "many_pauses"
	}
}

func (ph *PerceptualHasher) classifyEnergy(c0Mean float64) string {
	if c0Mean < -10 {
		return "quiet"
	} else if c0Mean < 10 {
		return "medium"
	} else {
		return "loud"
	}
}

func (ph *PerceptualHasher) classifyEnergyLevel(energyMean float64) string {
	if energyMean < 0.01 {
		return "low"
	} else if energyMean < 0.1 {
		return "medium"
	} else {
		return "high"
	}
}

func (ph *PerceptualHasher) classifyEnergyVariation(energyStd float64) string {
	if energyStd < 0.01 {
		return "stable"
	} else if energyStd < 0.05 {
		return "moderate"
	} else {
		return "variable"
	}
}

func (ph *PerceptualHasher) classifyPeakLevel(peakAmplitude float64) string {
	if peakAmplitude < 0.1 {
		return "low"
	} else if peakAmplitude < 0.7 {
		return "medium"
	} else {
		return "high"
	}
}

func (ph *PerceptualHasher) classifyHarmonicComplexity(chromaMeans []float64) string {
	nonZeroCount := 0
	for _, mean := range chromaMeans {
		if mean > 0.1 { // Threshold for significant energy
			nonZeroCount++
		}
	}

	if nonZeroCount < 3 {
		return "simple"
	} else if nonZeroCount < 6 {
		return "moderate"
	} else {
		return "complex"
	}
}

func (ph *PerceptualHasher) classifyTonalStrength(totalEnergy float64) string {
	if totalEnergy < 1.0 {
		return "weak"
	} else if totalEnergy < 5.0 {
		return "moderate"
	} else {
		return "strong"
	}
}

// Calculation helper functions
func (ph *PerceptualHasher) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (ph *PerceptualHasher) calculateStdDev(values []float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}

	mean := ph.calculateMean(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	return math.Sqrt(sumSquares / float64(len(values)-1))
}

func (ph *PerceptualHasher) calculateMeanChromaVector(chroma [][]float64) []float64 {
	if len(chroma) == 0 || len(chroma[0]) == 0 {
		return nil
	}

	numBins := len(chroma[0])
	means := make([]float64, numBins)

	for b := range numBins {
		sum := 0.0
		count := 0
		for t := range len(chroma) {
			if b < len(chroma[t]) {
				sum += chroma[t][b]
				count++
			}
		}
		if count > 0 {
			means[b] = sum / float64(count)
		}
	}

	return means
}

func (ph *PerceptualHasher) findDominantPitchClasses(chromaMeans []float64, topN int) []int {
	type pitchClass struct {
		index  int
		energy float64
	}

	classes := make([]pitchClass, len(chromaMeans))
	for i, energy := range chromaMeans {
		classes[i] = pitchClass{index: i, energy: energy}
	}

	// Sort by energy (descending)
	for i := 0; i < len(classes)-1; i++ {
		for j := i + 1; j < len(classes); j++ {
			if classes[j].energy > classes[i].energy {
				classes[i], classes[j] = classes[j], classes[i]
			}
		}
	}

	// Return top N indices
	result := make([]int, min(topN, len(classes)))
	for i := range len(result) {
		result[i] = classes[i].index
	}

	return result
}

func (ph *PerceptualHasher) createHash(data map[string]any) (string, int) {
	// Convert to JSON for consistent ordering
	jsonData, _ := json.Marshal(data)

	// Create hash
	hasher := sha256.New()
	hasher.Write(jsonData)
	fullHash := hex.EncodeToString(hasher.Sum(nil))

	// Truncate to desired length
	hashLength := min(ph.params.HashLength, len(fullHash))
	return fullHash[:hashLength], len(data)
}

func (ph *PerceptualHasher) estimateRobustness(featureCount, binCount int) float64 {
	// More features and fewer bins generally means more robustness
	if featureCount == 0 {
		return 0.0
	}

	// Simple heuristic: more features = more robust, fewer bins per feature = more robust
	avgBinsPerFeature := float64(binCount) / float64(featureCount)
	robustness := math.Min(1.0, float64(featureCount)/10.0) * (1.0 / (1.0 + avgBinsPerFeature/10.0))

	return math.Max(0.0, math.Min(1.0, robustness))
}

// ComparePerceptualHashes compares two perceptual hashes for similarity
func ComparePerceptualHashes(hash1, hash2 string) float64 {
	if hash1 == hash2 {
		return 1.0
	}

	if len(hash1) != len(hash2) {
		return 0.0
	}

	// Simple character-based comparison
	matches := 0
	for i := 0; i < len(hash1); i++ {
		if hash1[i] == hash2[i] {
			matches++
		}
	}

	return float64(matches) / float64(len(hash1))
}
