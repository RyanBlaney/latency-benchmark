package fingerprint

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

type similarityMethod string

const (
	methodWeighted  similarityMethod = "weighted"
	methodEuclidean similarityMethod = "euclidean"
	methodCosine    similarityMethod = "cosine"
	methodPearson   similarityMethod = "pearson"
	methodAdaptive  similarityMethod = "adaptive"
)

// SimilarityResult holds the result of fingerprint comparison
type SimilarityResult struct {
	OverallSimilarity     float64                   `json:"overall_similarity"` // 0.0-1.0
	HashSimilarity        float64                   `json:"hash_similarity"`    // Hash-based similarity
	FeatureSimilarity     float64                   `json:"feature_similarity"` // Feature-based similarity
	ContentTypeMatch      bool                      `json:"content_type_match"`
	PerceptualHashMatches map[string]float64        `json:"perceptual_hash_matches"` // Per hash type similarity
	FeatureDistances      map[string]float64        `json:"feature_distances"`       // Per feature distances
	QualityMetrics        *ComparisonQualityMetrics `json:"qualtiy_metrics"`         // Comparison quality
	ProcessingTime        time.Duration             `json:"processing_time"`
	Confidence            float64                   `json:"confidence"`
	Metadata              map[string]any            `json:"metadata"`
	AlignmentApplied      bool                      `json:"alignment_applied"`
	TemporalOffset        float64                   `json:"temporal_offset"`
}

// ComparisonQualityMetrics holds quality metrics for the comparison
type ComparisonQualityMetrics struct {
	DataAvailability  float64 `json:"data_availability"`
	FeatureCoverage   float64 `json:"feature_coverage"`
	TemporalAlignment float64 `json:"temporal_alignment"`
	NoiseLevel        float64 `json:"noise_level"`
	DynamicRangeMatch float64 `json:"dynamic_range_match"`
	SpectralCoherence float64 `json:"spectral_coherence"`
}

// Match represents a fingerprint match with score and metadata
type Match struct {
	Fingerprint *AudioFingerprint `json:"fingerprint"`
	Similarity  *SimilarityResult `json:"similarity"`
	Rank        int               `json:"rank"`
	MatchType   string            `json:"match_type"`   // 'exact', 'similar', 'weak'
	MatchRegion *TimeRegion       `json:"match_region"` // Time region of match
}

// TimeRegion represents a time region in audio
type TimeRegion struct {
	StartTime time.Duration `json:"start_time"`
	EndTime   time.Duration `json:"end_time"`
	Duration  time.Duration `json:"duration"`
	Offset    time.Duration `json:"offset"`
}

// FingerprintComparator handles fingerprint comparison operations
type FingerprintComparator struct {
	config         *config.ComparisonConfig
	internalMethod similarityMethod
	hashWeight     float64
	featureWeight  float64
	logger         logging.Logger
}

// NewFingerprintComparator creates a new fingerprint comparator
func NewFingerprintComparator(cfg *config.ComparisonConfig) *FingerprintComparator {
	if cfg == nil {
		cfg = config.DefaultComparisonConfig()
	}

	// Convert user-friendly method to internal method
	var internalMethod similarityMethod
	var hashWeight, featureWeight float64

	switch cfg.Method {
	case "fast":
		internalMethod = methodCosine
		hashWeight = 0.5
		featureWeight = 0.5

	case "precise":
		internalMethod = methodPearson
		hashWeight = 0.2
		featureWeight = 0.8

	case "auto":
		internalMethod = methodAdaptive
		hashWeight = 0.3
		featureWeight = 0.7

	default:
		internalMethod = methodAdaptive
		hashWeight = 0.3
		featureWeight = 0.7
	}

	return &FingerprintComparator{
		config:         cfg,
		internalMethod: internalMethod,
		hashWeight:     hashWeight,
		featureWeight:  featureWeight,
		logger: logging.WithFields(logging.Fields{
			"component": "fingerprint_comparator",
		}),
	}
}

// DefaultComparisonConfig returns a default comparison config
func DefaultComparisonConfig() *config.ComparisonConfig {
	return config.DefaultComparisonConfig()
}

// ContentOptimizedComparisonConfig returns comparison config optimized for the content type
func ContentOptimizedComparisonConfig(contentType config.ContentType) *config.ComparisonConfig {
	return config.GetContentOptimizedComparisonConfig(contentType)
}

// Compare compares two fingerprints and returns the similarity result
// TODO: ways to optimize:
// - Check compact hash similarity
// - Use perceptual hashes FIRST for similarity detection (possibly skip feature comp)
// - Use feature comparison only for PRECISE similarity scoring
func (fc *FingerprintComparator) Compare(fp1, fp2 *AudioFingerprint) (*SimilarityResult, error) {
	if fp1 == nil || fp2 == nil {
		return nil, fmt.Errorf("fingerprints cannot be nil")
	}

	startTime := time.Now()

	logger := fc.logger.WithFields(logging.Fields{
		"function": "Compare",
		"method":   fc.config.Method,
		"fp1_id":   fp1.ID,
		"fp2_id":   fp2.ID,
	})

	logger.Debug("Startin fingerprint comparison")

	// Initialize result
	result := &SimilarityResult{
		PerceptualHashMatches: make(map[string]float64),
		FeatureDistances:      make(map[string]float64),
		Metadata:              make(map[string]any),
	}

	result.ContentTypeMatch = fp1.ContentType == fp2.ContentType

	// Apply content filtering
	if fc.config.EnableContentFilter && !result.ContentTypeMatch {
		logger.Debug("Content types don't match, applying penalty")
		result.OverallSimilarity = 0.0
		result.Confidence = 0.1
		result.ProcessingTime = time.Since(startTime)
		return result, nil
	}

	// Calculate hash similarity
	hashSimilarity, err := fc.calculateHashSimilarity(fp1, fp2)
	if err != nil {
		logger.Error(err, "Failed to calculate hash similarity")
		hashSimilarity = 0.0
	}
	result.HashSimilarity = hashSimilarity

	// Early exit if hash filtering is enabled and similarity is too low
	if fc.config.SimilarityThreshold > 0.5 && hashSimilarity < fc.config.SimilarityThreshold*0.5 {
		logger.Debug("Hash similarity too low, early exit")
		result.OverallSimilarity = hashSimilarity
		result.FeatureSimilarity = 0.0
		result.Confidence = 0.3
		result.ProcessingTime = time.Since(startTime)
		return result, nil
	}

	// Calculate feature similarity
	featureSimilarity, err := fc.calculateFeatureSimilarity(fp1, fp2, result)
	if err != nil {
		logger.Error(err, "Failed to calculate feature similarity")
		featureSimilarity = 0.0
	}
	result.FeatureSimilarity = featureSimilarity

	result.OverallSimilarity = fc.calculateOverallSimilarity(result)

	if fc.config.EnableDetailedMetrics {
		result.QualityMetrics = fc.calculateQualityMetrics(fp1, fp2, result)
	}

	result.Confidence = fc.calculateConfidence(result)

	result.ProcessingTime = time.Since(startTime)

	logger.Info("Fingerprint comparison completed", logging.Fields{
		"overall_similarity": result.OverallSimilarity,
		"hash_similarity":    result.HashSimilarity,
		"feature_similarity": result.FeatureSimilarity,
		"content_type_match": result.ContentTypeMatch,
		"confidence":         result.Confidence,
		"processing_time":    result.ProcessingTime,
	})

	return result, nil
}

func (fc *FingerprintComparator) CompareWithAlignment(fp1, fp2 *AudioFingerprint, alignment *extractors.AlignmentFeatures, alignmentConf *config.AlignmentConfig) (*SimilarityResult, error) {
	if fp1 == nil || fp2 == nil {
		return nil, fmt.Errorf("fingerprints cannot be nil")
	}

	startTime := time.Now()

	logger := fc.logger.WithFields(logging.Fields{
		"function":      "CompareWithAlignment",
		"method":        fc.config.Method,
		"fp1_id":        fp1.ID,
		"fp2_id":        fp2.ID,
		"has_alignment": alignment != nil,
	})

	if alignment != nil {
		logger = logger.WithFields(logging.Fields{
			"temporal_offset":      alignment.TemporalOffset,
			"alignment_confidence": alignment.OffsetConfidence,
		})
	}

	logger.Debug("Starting fingerprint comparison with alignment")

	result := &SimilarityResult{
		PerceptualHashMatches: make(map[string]float64),
		FeatureDistances:      make(map[string]float64),
		Metadata:              make(map[string]any),
		AlignmentApplied:      alignment != nil,
	}

	if alignment != nil {
		result.TemporalOffset = alignment.TemporalOffset
	}

	result.ContentTypeMatch = fp1.ContentType == fp2.ContentType

	// Apply content filtering
	if fc.config.EnableContentFilter && !result.ContentTypeMatch {
		logger.Debug("Content types don't match, applying penalty")
		result.OverallSimilarity = 0.0
		result.Confidence = 0.1
		result.ProcessingTime = time.Since(startTime)
		return result, nil
	}

	hashSimilarity, err := fc.calculateHashSimilarity(fp1, fp2)
	if err != nil {
		logger.Error(err, "Failed to calculate hash similarity")
		hashSimilarity = 0.0
	}
	result.HashSimilarity = hashSimilarity

	// Early exit if hash filtering is enabled and similarity is too low
	if fc.config.SimilarityThreshold > 0.5 && hashSimilarity < fc.config.SimilarityThreshold*0.5 {
		logger.Debug("Hash similarity too low, early exit")
		result.OverallSimilarity = hashSimilarity
		result.FeatureSimilarity = 0.0
		result.Confidence = 0.3
		result.ProcessingTime = time.Since(startTime)
		return result, nil
	}

	// Calculate feature similarity with optional alignment
	featureSimilarity, err := fc.calculateFeatureSimilarityWithAlignment(fp1, fp2, result, alignment)
	if err != nil {
		logger.Error(err, "Failed to calculate feature similarity")
		featureSimilarity = 0.0
	}
	result.FeatureSimilarity = featureSimilarity

	result.OverallSimilarity = fc.calculateOverallSimilarity(result)

	if fc.config.EnableDetailedMetrics {
		result.QualityMetrics = fc.calculateQualityMetrics(fp1, fp2, result)
	}

	result.Confidence = fc.calculateConfidence(result)
	result.ProcessingTime = time.Since(startTime)

	logger.Info("Fingerprint comparison completed", logging.Fields{
		"overall_similarity":  result.OverallSimilarity,
		"hash_similarity":     result.HashSimilarity,
		"feature_similarity":  result.FeatureSimilarity,
		"content_type_match":  result.ContentTypeMatch,
		"confidence":          result.Confidence,
		"alignment_applied":   result.AlignmentApplied,
		"processing_time":     result.ProcessingTime,
		"mfcc_similarity":     result.FeatureDistances["mfcc"],
		"spectral_similarity": result.FeatureDistances["spectral"],
	})

	return result, nil
}

func (fc *FingerprintComparator) calculateFeatureSimilarityWithAlignment(fp1, fp2 *AudioFingerprint, result *SimilarityResult, alignment *extractors.AlignmentFeatures) (float64, error) {
	if fp1.Features == nil || fp2.Features == nil {
		err := fmt.Errorf("features cannot be nil")
		fc.logger.Error(err, "Features cannot be nil", logging.Fields{
			"function": "calculateFeatureSimilarityWithAlignment",
		})
		return 0.0, err
	}

	features1 := fp1.Features
	features2 := fp2.Features

	var similarities []float64
	var weights []float64

	// Get feature weights
	featureWeights := fc.getEffectiveWeights(fp1)

	// Compare MFCC features with optional alignment
	if len(features1.MFCC) > 0 && len(features2.MFCC) > 0 {
		var mfccSim, distance float64

		sampleRate := min(fp1.SampleRate, fp2.SampleRate)
		hopSize := min(fp1.HopSize, fp2.HopSize)

		if alignment != nil {
			// Use alignment-aware MFCC comparison
			mfccSim, distance = fc.compareMFCCWithAlignment(
				features1.MFCC, features2.MFCC,
				fp1.ContentType, fp2.ContentType,
				alignment.TemporalOffset, hopSize, sampleRate)
		} else {
			// Use standard MFCC comparison
			mfccSim, distance = fc.compareMFCC(features1.MFCC, features2.MFCC, fp1.ContentType, fp2.ContentType)
		}

		similarities = append(similarities, mfccSim)
		weights = append(weights, featureWeights["mfcc"])
		result.FeatureDistances["mfcc"] = distance
	}

	// Compare spectral features
	if features1.SpectralFeatures != nil && features2.SpectralFeatures != nil {
		spectralSim, distance := fc.compareSpectralFeatures(features1.SpectralFeatures, features2.SpectralFeatures)
		similarities = append(similarities, spectralSim)
		weights = append(weights, featureWeights["spectral"])
		result.FeatureDistances["spectral"] = distance
	}

	// Compare chroma features
	if len(features1.ChromaFeatures) > 0 && len(features2.ChromaFeatures) > 0 {
		chromaSim, distance := fc.compareChromaFeatures(features1.ChromaFeatures, features2.ChromaFeatures)
		similarities = append(similarities, chromaSim)
		weights = append(weights, featureWeights["chroma"])
		result.FeatureDistances["chroma"] = distance
	}

	// Compare temporal features
	if features1.TemporalFeatures != nil && features2.TemporalFeatures != nil {
		temporalSim, distance := fc.compareTemporalFeatures(features1.TemporalFeatures, features2.TemporalFeatures)
		similarities = append(similarities, temporalSim)
		weights = append(weights, featureWeights["temporal"])
		result.FeatureDistances["temporal"] = distance
	}

	// Compare speech features
	if features1.SpeechFeatures != nil && features2.SpeechFeatures != nil {
		speechSim, distance := fc.compareSpeechFeatures(features1.SpeechFeatures, features2.SpeechFeatures)
		similarities = append(similarities, speechSim)
		weights = append(weights, featureWeights["speech"])
		result.FeatureDistances["speech"] = distance
	}

	// Compare harmonic features
	if features1.HarmonicFeatures != nil && features2.HarmonicFeatures != nil {
		harmonicSim, distance := fc.compareHarmonicFeatures(features1.HarmonicFeatures, features2.HarmonicFeatures)
		similarities = append(similarities, harmonicSim)
		weights = append(weights, featureWeights["harmonic"])
		result.FeatureDistances["harmonic"] = distance
	}

	if len(similarities) == 0 {
		err := fmt.Errorf("no comparable features found")
		fc.logger.Error(err, "No comparable features found", logging.Fields{
			"function": "calculateFeatureSimilarityWithAlignment",
		})
		return 0.0, err
	}

	return fc.calculateWeightedMean(similarities, weights), nil
}

func (fc *FingerprintComparator) compareMFCCWithAlignment(mfcc1, mfcc2 [][]float64, contentType1, contentType2 config.ContentType, temporalOffset float64, hopSize int, sampleRate int) (float64, float64) {
	if len(mfcc1) == 0 || len(mfcc2) == 0 {
		return 0.0, 1.0
	}

	// Apply temporal alignment by shifting MFCC frames
	alignedMFCC1, alignedMFCC2 := fc.applyTemporalAlignment(mfcc1, mfcc2, temporalOffset, hopSize, sampleRate)

	if len(alignedMFCC1) == 0 || len(alignedMFCC2) == 0 {
		return 0.0, 1.0
	}

	// Now proceed with normal MFCC comparison on aligned features
	return fc.compareMFCC(alignedMFCC1, alignedMFCC2, contentType1, contentType2)
}

func (fc *FingerprintComparator) applyTemporalAlignment(mfcc1, mfcc2 [][]float64, temporalOffsetSeconds float64, hopSize int, sampleRate int) ([][]float64, [][]float64) {
	// Convert temporal offset to frame offset
	frameOffset := int(math.Round(temporalOffsetSeconds * float64(sampleRate) / float64(hopSize)))

	fc.logger.Debug("Applying temporal alignment", logging.Fields{
		"temporal_offset_sec": temporalOffsetSeconds,
		"frame_offset":        frameOffset,
		"mfcc1_frames":        len(mfcc1),
		"mfcc2_frames":        len(mfcc2),
		"hop_size":            hopSize,
		"sample_rate":         sampleRate,
	})

	var alignedMFCC1, alignedMFCC2 [][]float64

	if frameOffset > 0 {
		// mfcc2 is ahead, so shift mfcc1 forward (skip early frames of mfcc1)
		if frameOffset < len(mfcc1) {
			alignedMFCC1 = mfcc1[frameOffset:]
		} else {
			return nil, nil // Offset too large
		}
		alignedMFCC2 = mfcc2
	} else if frameOffset < 0 {
		// mfcc1 is ahead, so shift mfcc2 forward (skip early frames of mfcc2)
		frameOffset = -frameOffset
		if frameOffset < len(mfcc2) {
			alignedMFCC2 = mfcc2[frameOffset:]
		} else {
			return nil, nil // Offset too large
		}
		alignedMFCC1 = mfcc1
	} else {
		// No offset needed
		alignedMFCC1 = mfcc1
		alignedMFCC2 = mfcc2
	}

	// Trim to same length
	minLen := min(len(alignedMFCC1), len(alignedMFCC2))
	if minLen == 0 {
		return nil, nil
	}

	alignedMFCC1 = alignedMFCC1[:minLen]
	alignedMFCC2 = alignedMFCC2[:minLen]

	fc.logger.Debug("Alignment applied", logging.Fields{
		"aligned_mfcc1_frames": len(alignedMFCC1),
		"aligned_mfcc2_frames": len(alignedMFCC2),
		"frames_trimmed":       frameOffset,
	})

	return alignedMFCC1, alignedMFCC2
}

// FindBestMatches find the best matches from a list candidates
func (fc *FingerprintComparator) FindBestMatches(query *AudioFingerprint, candidates []*AudioFingerprint) ([]*Match, error) {
	if query == nil {
		return nil, fmt.Errorf("query fingerprint cannot be nil")
	}

	logger := fc.logger.WithFields(logging.Fields{
		"function":  "FindBestMatches",
		"query_id":  query.ID,
		"candiates": len(candidates),
	})

	logger.Info("Finding best matches")

	var matches []*Match

	// Compare with each candidate
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}

		if query.ID == candidate.ID {
			continue
		}

		similarity, err := fc.Compare(query, candidate)
		if err != nil {
			logger.Warn("Failed to compare with candidate", logging.Fields{
				"candidate_id": candidate.ID,
				"error":        err.Error(),
			})
			continue
		}

		// Only include matches above threshold
		if similarity.OverallSimilarity >= fc.config.SimilarityThreshold {
			match := &Match{
				Fingerprint: candidate,
				Similarity:  similarity,
				MatchType:   fc.classifyMatch(similarity),
			}
			matches = append(matches, match)
		}
	}

	// Sort by similarity (descending)
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Similarity.OverallSimilarity > matches[j].Similarity.OverallSimilarity
	})

	// Assign ranks and limit results
	maxResults := fc.config.MaxCandidates
	if len(matches) > maxResults {
		matches = matches[:maxResults]
	}

	for i, match := range matches {
		match.Rank = i + 1
	}

	logger.Info("Best matches found", logging.Fields{
		"total_matches": len(matches),
		"threshold":     fc.config.SimilarityThreshold,
	})

	return matches, nil
}

// calculateHashSimilarity calculates simlarity based on hashes
func (fc *FingerprintComparator) calculateHashSimilarity(fp1, fp2 *AudioFingerprint) (float64, error) {
	similarities := make([]float64, 0)

	// Compare compact hashes
	if fp1.CompactHash != "" && fp2.CompactHash != "" {
		var compactSim float64

		// Check if these are perceptual hashes (shorter length indicates perceptual)
		if len(fp1.CompactHash) <= 16 && len(fp2.CompactHash) <= 16 {
			// Use perceptual hash comparison
			compactSim = extractors.ComparePerceptualHashes(fp1.CompactHash, fp2.CompactHash)
			fc.logger.Info("Perceptual compact hash comparison", logging.Fields{
				"hash1":      fp1.CompactHash,
				"hash2":      fp2.CompactHash,
				"similarity": compactSim,
			})
		} else {
			// Use traditional exact hash comparison
			compactSim = fc.compareHashes(fp1.CompactHash, fp2.CompactHash)
			fc.logger.Debug("Traditional hash comparison", logging.Fields{
				"hash1_prefix": fp1.CompactHash[:min(8, len(fp1.CompactHash))],
				"hash2_prefix": fp2.CompactHash[:min(8, len(fp2.CompactHash))],
				"similarity":   compactSim,
			})
		}

		similarities = append(similarities, compactSim)
	}

	// Compare perceptual hashes with perceptual comparison
	for hashType, hash1 := range fp1.PerceptualHashes {
		if hash2, exists := fp2.PerceptualHashes[hashType]; exists {
			hashSim := extractors.ComparePerceptualHashes(hash1, hash2)
			similarities = append(similarities, hashSim)

			fc.logger.Info("Perceptual hash type comparison", logging.Fields{
				"type":       hashType,
				"hash1":      hash1,
				"hash2":      hash2,
				"similarity": hashSim,
			})
		}
	}

	if len(similarities) == 0 {
		return 0.0, nil
	}

	// Weight more recent perceptual hashes higher than compact hash
	if len(similarities) > 1 {
		// Give less weight to the first hash (compact) if we have perceptual hashes
		weights := make([]float64, len(similarities))
		weights[0] = 0.2 // Compact hash gets less weight
		for i := 1; i < len(weights); i++ {
			weights[i] = 0.8 / float64(len(weights)-1) // Distribute remaining weight among perceptual hashes
		}

		weightedSim := fc.calculateWeightedMean(similarities, weights)
		fc.logger.Info("Weighted hash similarity", logging.Fields{
			"individual_sims": similarities,
			"weights":         weights,
			"final_sim":       weightedSim,
		})

		return weightedSim, nil
	}

	return similarities[0], nil
}

// compareHashes compares two hash strings using Hamming distance
func (fc *FingerprintComparator) compareHashes(hash1, hash2 string) float64 {
	if hash1 == "" || hash2 == "" {
		return 0.0
	}

	if hash1 == hash2 {
		return 1.0
	}

	// Calculate Hamming distance for hex sxtrings
	minLen := min(len(hash1), len(hash2))

	if minLen == 0 {
		return 0.0
	}

	matches := 0
	for i := range minLen {
		if hash1[i] == hash2[i] {
			matches++
		}
	}

	return float64(matches) / float64(minLen)
}

// calculateFeatureSimilarity calculates similarity based on extracted features
func (fc *FingerprintComparator) calculateFeatureSimilarity(fp1, fp2 *AudioFingerprint, result *SimilarityResult) (float64, error) {
	if fp1.Features == nil || fp2.Features == nil {
		err := fmt.Errorf("features cannot be nil")
		fc.logger.Error(err, "Features cannot be nil", logging.Fields{
			"function": "calculateFeatureSimilarity",
		})
		return 0.0, err
	}

	features1 := fp1.Features
	features2 := fp2.Features

	var similarities []float64
	var weights []float64

	// Get feature weights
	featureWeights := fc.getEffectiveWeights(fp1)

	// Compare MFCC features
	if len(features1.MFCC) > 0 && len(features2.MFCC) > 0 {
		mfccSim, distance := fc.compareMFCC(features1.MFCC, features2.MFCC, fp1.ContentType, fp2.ContentType)
		similarities = append(similarities, mfccSim)
		weights = append(weights, featureWeights["mfcc"])
		result.FeatureDistances["mfcc"] = distance
	}

	// Compare spectral features
	if features1.SpectralFeatures != nil && features2.SpectralFeatures != nil {
		spectralSim, distance := fc.compareSpectralFeatures(features1.SpectralFeatures, features2.SpectralFeatures)
		similarities = append(similarities, spectralSim)
		weights = append(weights, featureWeights["spectral"])
		result.FeatureDistances["spectral"] = distance
	}

	// Compare chroma features
	if len(features1.ChromaFeatures) > 0 && len(features2.ChromaFeatures) > 0 {
		chromaSim, distance := fc.compareChromaFeatures(features1.ChromaFeatures, features2.ChromaFeatures)
		similarities = append(similarities, chromaSim)
		weights = append(weights, featureWeights["chroma"])
		result.FeatureDistances["chroma"] = distance
	}

	// Compare temporal features
	if features1.TemporalFeatures != nil && features2.TemporalFeatures != nil {
		temporalSim, distance := fc.compareTemporalFeatures(features1.TemporalFeatures, features2.TemporalFeatures)
		similarities = append(similarities, temporalSim)
		weights = append(weights, featureWeights["temporal"])
		result.FeatureDistances["temporal"] = distance
	}

	// Compare speech features
	if features1.SpeechFeatures != nil && features2.SpeechFeatures != nil {
		speechSim, distance := fc.compareSpeechFeatures(features1.SpeechFeatures, features2.SpeechFeatures)
		similarities = append(similarities, speechSim)
		weights = append(weights, featureWeights["speech"])
		result.FeatureDistances["speech"] = distance
	}

	// Compare harmonic features
	if features1.HarmonicFeatures != nil && features2.HarmonicFeatures != nil {
		harmonicSim, distance := fc.compareHarmonicFeatures(features1.HarmonicFeatures, features2.HarmonicFeatures)
		similarities = append(similarities, harmonicSim)
		weights = append(weights, featureWeights["harmonic"])
		result.FeatureDistances["harmonic"] = distance
	}

	if len(similarities) == 0 {
		err := fmt.Errorf("no comparable features found")
		fc.logger.Error(err, "No comparable features found", logging.Fields{
			"function": "calculateFeatureSimilarity",
		})
		return 0.0, err
	}

	return fc.calculateWeightedMean(similarities, weights), nil
}

// CompareMFCC compares MFCC features using GoNum statistical functions
func (fc *FingerprintComparator) compareMFCC(mfcc1, mfcc2 [][]float64, contentType1, contentType2 config.ContentType) (float64, float64) {
	if len(mfcc1) == 0 || len(mfcc2) == 0 {
		return 0.0, 1.0
	}

	// Extract statistical features using GoNum
	stats1 := fc.extractMFCCStatistics(mfcc1)
	stats2 := fc.extractMFCCStatistics(mfcc2)

	var statsSimilarity float64
	// Method 1: Cosine Similarity
	if len(stats1) > 0 && len(stats2) > 0 {
		statsSimilarity = fc.cosineSimilarity(stats1, stats2)
	}

	// Method 2: Sequence-based comparison
	sequenceSimilarity := fc.compareMFCCSequences(mfcc1, mfcc2)

	// Method3: DTW-based comparison for temporal alignment
	dtwSimilarity := fc.compareMFCCWithDTW(mfcc1, mfcc2)

	// Combine the three methods with weights by content type
	var contentType config.ContentType
	if contentType1 != contentType2 {
		contentType = config.ContentMixed
	}
	var combinedSimilarity float64
	switch contentType {
	case config.ContentMusic:
		// Music: DTW is crucial for tempo variations, sequence for melody structure
		// Stats less reliable due to dynamic range compression in different encodings
		combinedSimilarity = 0.15*statsSimilarity + 0.35*sequenceSimilarity + 0.50*dtwSimilarity

	case config.ContentTalk, config.ContentNews:
		// Speech: Stats capture spectral envelope (formants), sequence for prosody
		// DTW handles speaking rate differences between encodings
		combinedSimilarity = 0.40*statsSimilarity + 0.35*sequenceSimilarity + 0.25*dtwSimilarity

	case config.ContentSports:
		// Sports: High variance (crowd noise, commentary, music)
		// DTW most robust for temporal events, stats for ambient characteristics
		combinedSimilarity = 0.25*statsSimilarity + 0.25*sequenceSimilarity + 0.50*dtwSimilarity

	case config.ContentMixed:
		// Mixed/Cross-encoding: Conservative weighting, emphasize robust methods
		// DTW most resilient to encoding artifacts and quality differences
		combinedSimilarity = 0.20*statsSimilarity + 0.30*sequenceSimilarity + 0.50*dtwSimilarity

	default:
		// Unknown content: Balanced approach with slight DTW preference
		// Handles MP3 vs HLS compression differences best
		combinedSimilarity = 0.30*statsSimilarity + 0.30*sequenceSimilarity + 0.40*dtwSimilarity
	}
	return combinedSimilarity, 1.0 - combinedSimilarity
}

// compareMFCCSequences compares MFCC using sequence analysis from your algorithms
func (fc *FingerprintComparator) compareMFCCSequences(mfcc1, mfcc2 [][]float64) float64 {
	if len(mfcc1) == 0 || len(mfcc2) == 0 {
		return 0.0
	}

	// Focus on first 8-10 coefficients (most perceptually relevant)
	numCoeffs := min(len(mfcc1[0]), len(mfcc2[0]), 10)
	if numCoeffs == 0 {
		return 0.0
	}

	var similarities []float64

	// Compare each coefficient sequence using cross-correlation from your algorithms
	for coeff := range numCoeffs {
		seq1 := fc.extractCoefficientSequence(mfcc1, coeff)
		seq2 := fc.extractCoefficientSequence(mfcc2, coeff)

		if len(seq1) == 0 || len(seq2) == 0 {
			continue
		}

		// Use your CrossCorrelation algorithm for sequence comparison
		maxLag := min(len(seq1), len(seq2)) / 4 // Reasonable lag limit
		crossCorr := stats.NewCrossCorrelation(maxLag)

		result, err := crossCorr.Compute(seq1, seq2)
		if err == nil && result != nil {
			// Use peak correlation as similarity measure
			similarity := math.Max(0.0, result.PeakCorrelation)
			similarities = append(similarities, similarity)
		}
	}

	if len(similarities) == 0 {
		return 0.0
	}

	// Use GoNum for weighted mean (weight higher coefficients more)
	weights := make([]float64, len(similarities))
	for i := range weights {
		// Weight decreases for higher-order coefficients
		weights[i] = math.Exp(-float64(i) * 0.2)
	}

	return stat.Mean(similarities, weights)
}

// compareMFCCWithDTW uses DTW for alignment-aware MFCC comparison
func (fc *FingerprintComparator) compareMFCCWithDTW(mfcc1, mfcc2 [][]float64) float64 {
	if len(mfcc1) == 0 || len(mfcc2) == 0 {
		return 0.0
	}

	// Prepare MFCC vectors for DTW (use first 8 coefficients per frame)
	vectors1 := fc.prepareMFCCVectorsForDTW(mfcc1)
	vectors2 := fc.prepareMFCCVectorsForDTW(mfcc2)

	if len(vectors1) == 0 || len(vectors2) == 0 {
		return 0.0
	}

	// Create DTW analyzer with reasonable constraints
	constraintBand := max(len(vectors1), len(vectors2)) / 4 // 25% constraint band
	dtwAnalyzer := stats.NewDTWAlignmentWithParams(
		constraintBand,
		"symmetric", // Good for speech
		stats.EuclideanDistance,
	)

	// Perform DTW alignment on vector sequences
	result, err := dtwAnalyzer.Align(vectors1, vectors2)
	if err != nil {
		return 0.0
	}

	// Convert DTW distance to similarity
	// Normalize by sequence lengths and vector dimensions
	maxLen := float64(max(len(vectors1), len(vectors2)))
	vectorDim := float64(len(vectors1[0]))
	normalizedDistance := result.Distance / (maxLen * vectorDim)

	// Convert to similarity (0-1 range)
	similarity := math.Exp(-normalizedDistance)
	return math.Max(0.0, math.Min(1.0, similarity))
}

func (fc *FingerprintComparator) prepareMFCCVectorsForDTW(mfcc [][]float64) [][]float64 {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return nil
	}

	numCoeffs := min(len(mfcc[0]), 8) // Use first 8 coefficients
	vectors := make([][]float64, len(mfcc))

	for frame := range len(mfcc) {
		vector := make([]float64, numCoeffs)
		for coeff := 0; coeff < numCoeffs && coeff < len(mfcc[frame]); coeff++ {
			vector[coeff] = mfcc[frame][coeff]
		}
		vectors[frame] = vector
	}

	return vectors
}

// Helper function to extract a single coefficient sequence across all frames
func (fc *FingerprintComparator) extractCoefficientSequence(mfcc [][]float64, coeffIndex int) []float64 {
	if coeffIndex >= len(mfcc[0]) {
		return nil
	}

	sequence := make([]float64, len(mfcc))
	for frame := range len(mfcc) {
		if coeffIndex < len(mfcc[frame]) {
			sequence[frame] = mfcc[frame][coeffIndex]
		}
	}
	return sequence
}

// compareSpectralFeatures compares spectral features
func (fc *FingerprintComparator) compareSpectralFeatures(spec1, spec2 *extractors.SpectralFeatures) (float64, float64) {
	similarities := make([]float64, 0)

	// Compare each spectral feature using statistical measures
	if len(spec1.SpectralCentroid) > 0 && len(spec2.SpectralCentroid) > 0 {
		sim := fc.compareSequenceStats(spec1.SpectralCentroid, spec2.SpectralCentroid)
		similarities = append(similarities, sim)
	}

	if len(spec1.SpectralRolloff) > 0 && len(spec2.SpectralRolloff) > 0 {
		sim := fc.compareSequenceStats(spec1.SpectralRolloff, spec2.SpectralRolloff)
		similarities = append(similarities, sim)
	}

	if len(spec1.SpectralFlux) > 0 && len(spec2.SpectralFlux) > 0 {
		sim := fc.compareSequenceStats(spec1.SpectralFlux, spec2.SpectralFlux)
		similarities = append(similarities, sim)
	}

	if len(similarities) == 0 {
		return 0.0, 1.0
	}

	similarity := stat.Mean(similarities, nil)
	return similarity, 1.0 - similarity
}

func (fc *FingerprintComparator) compareChromaFeatures(chroma1, chroma2 [][]float64) (float64, float64) {
	if len(chroma1) == 0 || len(chroma2) == 0 {
		return 0.0, 1.0
	}

	mean1 := fc.calculateMeanChromaVector(chroma1)
	mean2 := fc.calculateMeanChromaVector(chroma2)

	if len(mean1) == 0 || len(mean2) == 0 {
		return 0.0, 1.0
	}

	// Use cosine similarity for chroma comparison
	similarity := fc.cosineSimilarity(mean1, mean2)
	return similarity, 1.0 - similarity
}

func (fc *FingerprintComparator) compareTemporalFeatures(temp1, temp2 *extractors.TemporalFeatures) (float64, float64) {
	similarities := make([]float64, 0)

	// Compare scalar features
	if temp1.DynamicRange > 0 && temp2.DynamicRange > 0 {
		drSim := fc.compareScalarFeatures(temp1.DynamicRange, temp2.DynamicRange)
		similarities = append(similarities, drSim)
	}

	silenceSim := fc.compareScalarFeatures(temp1.SilenceRatio, temp2.SilenceRatio)
	similarities = append(similarities, silenceSim)

	if temp1.OnsetDensity > 0 && temp2.OnsetDensity > 0 {
		onsetSim := fc.compareScalarFeatures(temp1.OnsetDensity, temp2.OnsetDensity)
		similarities = append(similarities, onsetSim)
	}

	// Compare RMS energy sequences if available
	if len(temp1.RMSEnergy) > 0 && len(temp2.RMSEnergy) > 0 {
		rmsSim := fc.compareSequenceStats(temp1.RMSEnergy, temp2.RMSEnergy)
		similarities = append(similarities, rmsSim)
	}

	if len(similarities) == 0 {
		return 0.0, 1.0
	}

	similarity := stat.Mean(similarities, nil)
	return similarity, 1.0 - similarity
}

func (fc *FingerprintComparator) compareSpeechFeatures(speech1, speech2 *extractors.SpeechFeatures) (float64, float64) {
	similarities := make([]float64, 0)

	// Compare scalar speech features
	if speech1.SpeechRate > 0 && speech2.SpeechRate > 0 {
		rateSim := fc.compareScalarFeatures(speech1.SpeechRate, speech2.SpeechRate)
		similarities = append(similarities, rateSim)
	}

	if speech1.VocalTractLength > 0 && speech2.VocalTractLength > 0 {
		vtlSim := fc.compareScalarFeatures(speech1.VocalTractLength, speech2.VocalTractLength)
		similarities = append(similarities, vtlSim)
	}

	// Compare voicing probability sequence if available
	if len(speech1.VoicingProbability) > 0 && len(speech2.VoicingProbability) > 0 {
		voicingSim := fc.compareSequenceStats(speech1.VoicingProbability, speech2.VoicingProbability)
		similarities = append(similarities, voicingSim)
	}

	if len(similarities) == 0 {
		return 0.0, 1.0
	}

	similarity := stat.Mean(similarities, nil)
	return similarity, 1.0 - similarity
}

func (fc *FingerprintComparator) compareHarmonicFeatures(harm1, harm2 *extractors.HarmonicFeatures) (float64, float64) {
	similarities := make([]float64, 0)

	// Compare harmonic ratio sequence
	if len(harm1.HarmonicRatio) > 0 && len(harm2.HarmonicRatio) > 0 {
		harmonicSim := fc.compareSequenceStats(harm1.HarmonicRatio, harm2.HarmonicRatio)
		similarities = append(similarities, harmonicSim)
	}

	// Compare pitch estimates sequence
	if len(harm1.PitchEstimate) > 0 && len(harm2.PitchEstimate) > 0 {
		pitchSim := fc.compareSequenceStats(harm1.PitchEstimate, harm2.PitchEstimate)
		similarities = append(similarities, pitchSim)
	}

	if len(similarities) == 0 {
		return 0.0, 1.0
	}

	similarity := stat.Mean(similarities, nil)
	return similarity, 1.0 - similarity
}

// Helper functions

func (fc *FingerprintComparator) extractMFCCStatistics(mfcc [][]float64) []float64 {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return nil
	}

	numCoeffs := len(mfcc[0])
	stats := make([]float64, numCoeffs*2)

	for c := range numCoeffs {
		// Extract coefficient values across time
		values := make([]float64, len(mfcc))
		for t := range len(mfcc) {
			if c < len(mfcc[t]) {
				values[t] = mfcc[t][c]
			}
		}

		mean := stat.Mean(values, nil)
		variance := stat.Variance(values, nil)
		std := math.Sqrt(variance)

		stats[c] = mean
		stats[c+numCoeffs] = std
	}

	return stats
}

func (fc *FingerprintComparator) calculateMeanChromaVector(chroma [][]float64) []float64 {
	if len(chroma) == 0 || len(chroma[0]) == 0 {
		return nil
	}

	numBins := len(chroma[0])
	means := make([]float64, numBins)

	for b := range numBins {
		// Extract values for this chroma bin across time
		values := make([]float64, 0, len(chroma))
		for t := range len(chroma) {
			if b < len(chroma[t]) {
				values = append(values, chroma[t][b])
			}
		}

		if len(values) > 0 {
			means[b] = stat.Mean(values, nil)
		}
	}

	return means
}

func (fc *FingerprintComparator) compareSequenceStats(seq1, seq2 []float64) float64 {
	if len(seq1) == 0 || len(seq2) == 0 {
		return 0.0
	}

	mean1 := stat.Mean(seq1, nil)
	mean2 := stat.Mean(seq2, nil)
	std1 := math.Sqrt(stat.Variance(seq1, nil))
	std2 := math.Sqrt(stat.Variance(seq2, nil))

	// Create feature vectors from statistics
	features1 := []float64{mean1, std1}
	features2 := []float64{mean2, std2}

	return fc.cosineSimilarity(features1, features2)
}

func (fc *FingerprintComparator) compareScalarFeatures(val1, val2 float64) float64 {
	if val1 == 0 && val2 == 0 {
		return 1.0
	}

	maxVal := math.Max(math.Abs(val1), math.Abs(val2))
	if maxVal == 0 {
		return 1.0
	}

	diff := math.Abs(val1 - val2)
	return math.Max(0.0, 1.0-diff/maxVal)
}

func (fc *FingerprintComparator) cosineSimilarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) || len(vec1) == 0 {
		return 0.0
	}

	// Calculate dot product and norms
	dotProduct := floats.Dot(vec1, vec2)
	norm1 := floats.Norm(vec1, 2)
	norm2 := floats.Norm(vec2, 2)

	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	return dotProduct / (norm1 * norm2)
}

func (fc *FingerprintComparator) calculateWeightedMean(values, weights []float64) float64 {
	if len(values) != len(weights) || len(values) == 0 {
		return 0.0
	}

	// Use GoNum's weighted mean calculation
	return stat.Mean(values, weights)
}

// calculateOverallSimilarity combines hash and feature similarities
func (fc *FingerprintComparator) calculateOverallSimilarity(result *SimilarityResult) float64 {
	switch fc.internalMethod {
	case methodAdaptive:
		return fc.calculateAdaptiveSimilarity(result)
	default:
		// Default weighted combination
		return fc.hashWeight*result.HashSimilarity + fc.featureWeight*result.FeatureSimilarity
	}
}

// calculateAdaptiveSimilarity calculates similarity adaptively based on available data
func (fc *FingerprintComparator) calculateAdaptiveSimilarity(result *SimilarityResult) float64 {
	// Adapt weights based on data availability and quality
	hashWeight := fc.hashWeight
	featureWeight := fc.featureWeight

	// If hash similarity is very high, increase its weight
	if result.HashSimilarity > 0.9 {
		hashWeight += 0.2
		featureWeight -= 0.2
	}

	// If feature similarity is very reliable, increase its weight
	if result.FeatureSimilarity > 0.8 && len(result.FeatureDistances) >= 3 {
		featureWeight += 0.1
		hashWeight -= 0.1
	}

	// Ensure weights are valid
	hashWeight = math.Max(0.1, math.Min(0.9, hashWeight))
	featureWeight = 1.0 - hashWeight

	return hashWeight*result.HashSimilarity + featureWeight*result.FeatureSimilarity
}

// calculateQualityMetrics calculates quality metrics for the comparison
func (fc *FingerprintComparator) calculateQualityMetrics(fp1, fp2 *AudioFingerprint, result *SimilarityResult) *ComparisonQualityMetrics {
	metrics := &ComparisonQualityMetrics{}

	// Calculate data availability
	availableFeatures := 0
	totalFeatures := 6 // Total possible feature types

	if fp1.Features.MFCC != nil && fp2.Features.MFCC != nil {
		availableFeatures++
	}
	if fp1.Features.SpectralFeatures != nil && fp2.Features.SpectralFeatures != nil {
		availableFeatures++
	}
	if fp1.Features.ChromaFeatures != nil && fp2.Features.ChromaFeatures != nil {
		availableFeatures++
	}
	if fp1.Features.TemporalFeatures != nil && fp2.Features.TemporalFeatures != nil {
		availableFeatures++
	}
	if fp1.Features.SpeechFeatures != nil && fp2.Features.SpeechFeatures != nil {
		availableFeatures++
	}
	if fp1.Features.HarmonicFeatures != nil && fp2.Features.HarmonicFeatures != nil {
		availableFeatures++
	}

	metrics.DataAvailability = float64(availableFeatures) / float64(totalFeatures)
	metrics.FeatureCoverage = float64(len(result.FeatureDistances)) / float64(totalFeatures)

	// Calculate temporal alignment
	durationDiff := math.Abs(fp1.Duration.Seconds() - fp2.Duration.Seconds())
	maxDuration := math.Max(fp1.Duration.Seconds(), fp2.Duration.Seconds())
	if maxDuration > 0 {
		metrics.TemporalAlignment = 1.0 - math.Min(1.0, durationDiff/maxDuration)
	} else {
		metrics.TemporalAlignment = 1.0
	}

	// Estimate noise level and other metrics
	metrics.NoiseLevel = fc.estimateNoiseLevel(result)
	metrics.DynamicRangeMatch = fc.calculateDynamicRangeMatch(fp1, fp2)
	metrics.SpectralCoherence = fc.calculateSpectralCoherence(fp1, fp2)

	return metrics
}

// estimateNoiseLevel estimates noise level from feature inconsistencies
func (fc *FingerprintComparator) estimateNoiseLevel(result *SimilarityResult) float64 {
	if len(result.FeatureDistances) == 0 {
		return 0.5 // Unknown noise level
	}

	// Calculate variance in feature similarities using GoNum
	similarities := make([]float64, 0, len(result.FeatureDistances))
	for _, distance := range result.FeatureDistances {
		similarities = append(similarities, 1.0-distance) // Convert distance to similarity
	}

	if len(similarities) <= 1 {
		return 0.0
	}

	variance := stat.Variance(similarities, nil)

	// High variance suggests more noise
	return math.Min(1.0, math.Sqrt(variance))
}

// calculateDynamicRangeMatch compares dynamic ranges
func (fc *FingerprintComparator) calculateDynamicRangeMatch(fp1, fp2 *AudioFingerprint) float64 {
	if fp1.Features.TemporalFeatures == nil || fp2.Features.TemporalFeatures == nil {
		return 0.5 // Unknown
	}

	dr1 := fp1.Features.TemporalFeatures.DynamicRange
	dr2 := fp2.Features.TemporalFeatures.DynamicRange

	if dr1 <= 0 || dr2 <= 0 {
		return 0.5
	}

	return fc.compareScalarFeatures(dr1, dr2)
}

// calculateSpectralCoherence calculates spectral coherence using GoNum correlation
func (fc *FingerprintComparator) calculateSpectralCoherence(fp1, fp2 *AudioFingerprint) float64 {
	if fp1.Features.SpectralFeatures == nil || fp2.Features.SpectralFeatures == nil {
		return 0.5
	}

	spec1 := fp1.Features.SpectralFeatures
	spec2 := fp2.Features.SpectralFeatures

	coherences := make([]float64, 0)

	// Compare spectral centroids using GoNum correlation
	if len(spec1.SpectralCentroid) > 0 && len(spec2.SpectralCentroid) > 0 {
		correlation := stat.Correlation(spec1.SpectralCentroid, spec2.SpectralCentroid, nil)
		if !math.IsNaN(correlation) {
			coherences = append(coherences, math.Abs(correlation))
		}
	}

	// Compare spectral rolloffs
	if len(spec1.SpectralRolloff) > 0 && len(spec2.SpectralRolloff) > 0 {
		correlation := stat.Correlation(spec1.SpectralRolloff, spec2.SpectralRolloff, nil)
		if !math.IsNaN(correlation) {
			coherences = append(coherences, math.Abs(correlation))
		}
	}

	if len(coherences) == 0 {
		return 0.5
	}

	return stat.Mean(coherences, nil)
}

// calculateConfidence calculates confidence in the comparison result
func (fc *FingerprintComparator) calculateConfidence(result *SimilarityResult) float64 {
	confidence := 0.5 // Base confidence

	// High similarity increases confidence
	if result.OverallSimilarity > 0.8 {
		confidence += 0.3
	} else if result.OverallSimilarity > 0.6 {
		confidence += 0.2
	}

	// Content type match increases confidence
	if result.ContentTypeMatch {
		confidence += 0.1
	}

	// More feature types increase confidence
	featureCount := float64(len(result.FeatureDistances))
	confidence += featureCount * 0.05

	// Quality metrics affect confidence
	if result.QualityMetrics != nil {
		confidence += result.QualityMetrics.DataAvailability * 0.1
		confidence -= result.QualityMetrics.NoiseLevel * 0.1
	}

	// Consistent hash and feature similarities increase confidence
	hashFeatureDiff := math.Abs(result.HashSimilarity - result.FeatureSimilarity)
	if hashFeatureDiff < 0.2 {
		confidence += 0.1
	}

	return math.Max(0.0, math.Min(1.0, confidence))
}

// classifyMatch classifies the type of match based on similarity
func (fc *FingerprintComparator) classifyMatch(similarity *SimilarityResult) string {
	if similarity.OverallSimilarity >= 0.95 {
		return "exact"
	} else if similarity.OverallSimilarity >= 0.85 {
		return "very_similar"
	} else if similarity.OverallSimilarity >= 0.75 {
		return "similar"
	} else if similarity.OverallSimilarity >= 0.6 {
		return "somewhat_similar"
	} else {
		return "weak"
	}
}

// getEffectiveWeights gets effective feature weights for comparison
func (fc *FingerprintComparator) getEffectiveWeights(fp *AudioFingerprint) map[string]float64 {
	// First try to get weights from fingerprint metadata
	if weights, exists := fp.Metadata["feature_weights"].(map[string]float64); exists && weights != nil {
		return weights
	}

	switch fp.ContentType {
	case config.ContentNews, config.ContentTalk:
		return map[string]float64{
			"mfcc":     0.50, // Increased for speech content
			"spectral": 0.25,
			"temporal": 0.15,
			"speech":   0.10,
			"chroma":   0.05, // Reduced for speech
			"harmonic": 0.05, // Reduced for speech
			"energy":   0.10,
		}
	case config.ContentMusic:
		return map[string]float64{
			"mfcc":     0.30,
			"chroma":   0.25, // Important for music
			"spectral": 0.20,
			"harmonic": 0.15, // Important for music
			"temporal": 0.10,
			"speech":   0.05,
			"energy":   0.10,
		}
	case config.ContentSports:
		return map[string]float64{
			"energy":   0.30, // Important for crowd dynamics
			"temporal": 0.25, // Important for excitement patterns
			"mfcc":     0.25,
			"spectral": 0.20,
			"speech":   0.10,
			"chroma":   0.05,
			"harmonic": 0.05,
		}
	default:
		// Default balanced weights
		return map[string]float64{
			"mfcc":     0.35,
			"spectral": 0.25,
			"temporal": 0.20,
			"energy":   0.15,
			"chroma":   0.10,
			"speech":   0.10,
			"harmonic": 0.10,
		}
	}
}

// BatchCompare compares one fingerprint against multiple candidates efficiently
func (fc *FingerprintComparator) BatchCompare(query *AudioFingerprint, candidates []*AudioFingerprint) ([]*SimilarityResult, error) {
	if query == nil {
		return nil, fmt.Errorf("query fingerprint cannot be nil")
	}

	logger := fc.logger.WithFields(logging.Fields{
		"function":   "BatchCompare",
		"query_id":   query.ID,
		"candidates": len(candidates),
	})

	logger.Info("Starting batch comparison")

	results := make([]*SimilarityResult, 0, len(candidates))

	for i, candidate := range candidates {
		if candidate == nil {
			continue
		}

		// Skip self-comparison
		if query.ID == candidate.ID {
			continue
		}

		result, err := fc.Compare(query, candidate)
		if err != nil {
			logger.Warn("Failed to compare with candidate", logging.Fields{
				"candidate_index": i,
				"candidate_id":    candidate.ID,
				"error":           err.Error(),
			})
			continue
		}

		results = append(results, result)
	}

	logger.Info("Batch comparison completed", logging.Fields{
		"successful_comparisons": len(results),
		"total_candidates":       len(candidates),
	})

	return results, nil
}

// GetSimilarityStatistics calculates statistics for a set of similarity results using GoNum
func GetSimilarityStatistics(results []*SimilarityResult) map[string]float64 {
	if len(results) == 0 {
		return map[string]float64{}
	}

	similarities := make([]float64, len(results))
	hashSims := make([]float64, len(results))
	featureSims := make([]float64, len(results))
	confidences := make([]float64, len(results))

	for i, result := range results {
		similarities[i] = result.OverallSimilarity
		hashSims[i] = result.HashSimilarity
		featureSims[i] = result.FeatureSimilarity
		confidences[i] = result.Confidence
	}

	// Use GoNum for statistical calculations
	calculateStats := func(values []float64) map[string]float64 {
		// Sort for percentiles
		sorted := make([]float64, len(values))
		copy(sorted, values)
		sort.Float64s(sorted)

		mean := stat.Mean(values, nil)
		variance := stat.Variance(values, nil)

		return map[string]float64{
			"mean":   mean,
			"min":    sorted[0],
			"max":    sorted[len(sorted)-1],
			"median": stat.Quantile(0.5, stat.Empirical, sorted, nil),
			"std":    math.Sqrt(variance),
		}
	}

	overallStats := calculateStats(similarities)
	hashStats := calculateStats(hashSims)
	featureStats := calculateStats(featureSims)
	confidenceStats := calculateStats(confidences)

	return map[string]float64{
		"overall_mean":      overallStats["mean"],
		"overall_min":       overallStats["min"],
		"overall_max":       overallStats["max"],
		"overall_median":    overallStats["median"],
		"overall_std":       overallStats["std"],
		"hash_mean":         hashStats["mean"],
		"feature_mean":      featureStats["mean"],
		"confidence_mean":   confidenceStats["mean"],
		"total_comparisons": float64(len(results)),
	}
}

// ValidateConfig validates the comparison configuration
func (fc *FingerprintComparator) ValidateConfig() error {
	if fc.config.SimilarityThreshold < 0 || fc.config.SimilarityThreshold > 1 {
		return fmt.Errorf("similarity threshold must be between 0 and 1: %f", fc.config.SimilarityThreshold)
	}

	if fc.config.MaxCandidates <= 0 {
		return fmt.Errorf("max candidates must be positive: %d", fc.config.MaxCandidates)
	}

	validMethods := map[string]bool{"auto": true, "fast": true, "precise": true}
	if !validMethods[fc.config.Method] {
		return fmt.Errorf("invalid method: %s (must be 'auto', 'fast', or 'precise')", fc.config.Method)
	}

	return nil
}
