package extractors

import (
	"fmt"
	"math"
	"strconv"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/temporal"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// AlignmentExtractor performs audio alignment and synchronization analysis
// WHY: Critical for fingerprint matching, determining temporal relationships,
// and measuring similarity between audio segments of different lengths
type AlignmentExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger

	// Alignment algorithms from algorithms package
	alignmentAnalyzer *stats.AlignmentAnalyzer
	dtwAnalyzer       *stats.DTWAlignment
	crossCorr         *stats.CrossCorrelation
	energy            *temporal.Energy

	// Alignment parameters
	maxLagSamples    int
	maxLagSeconds    float64
	stepSize         int
	confidenceThresh float64
}

// AlignmentFeatures contains alignment and synchronization results
type AlignmentFeatures struct {
	// Primary alignment results
	BestAlignment *AlignmentResult `json:"best_alignment"` // Best overall alignment
	DTWAlignment  *AlignmentResult `json:"dtw_alignment"`  // DTW-based alignment
	CorrAlignment *AlignmentResult `json:"corr_alignment"` // Cross-correlation alignment

	// Offset and timing information
	TemporalOffset   float64 `json:"temporal_offset_seconds"` // Best offset in seconds
	OffsetConfidence float64 `json:"offset_confidence"`       // Confidence in offset estimate
	TimeStretch      float64 `json:"time_stretch"`            // Estimated time stretch factor

	// Similarity metrics
	OverallSimilarity float64            `json:"overall_similarity"` // Combined similarity score
	FeatureSimilarity map[string]float64 `json:"feature_similarity"` // Per-feature similarity

	// Quality and consistency
	AlignmentQuality float64               `json:"alignment_quality"` // Quality of alignment
	Consistency      *stats.AlignmentStats `json:"consistency"`       // Consistency analysis

	// Analysis metadata
	Method          string  `json:"method"`           // Best alignment method
	ProcessingTime  float64 `json:"processing_time"`  // Processing time (ms)
	QueryLength     float64 `json:"query_length"`     // Query length in seconds
	ReferenceLength float64 `json:"reference_length"` // Reference length in seconds
}

// AlignmentResult wraps the stats package result with additional context
type AlignmentResult struct {
	*stats.AlignmentResult
	FeatureType string `json:"feature_type"`        // Type of features used
	Success     bool   `json:"success"`             // Whether alignment succeeded
	ErrorMsg    string `json:"error_msg,omitempty"` // Error message if failed
}

// NewAlignmentExtractor creates a new alignment feature extractor
func NewAlignmentExtractor(featureConf *config.FeatureConfig, alignmentConf *config.AlignmentConfig) *AlignmentExtractor {
	logger := logging.WithFields(logging.Fields{
		"component": "alignment_extractor",
	})

	maxLagSeconds := alignmentConf.MaxLagSeconds
	maxLagSamples := int(maxLagSeconds * float64(featureConf.SampleRate))

	maxLagFrames := maxLagSamples / featureConf.HopSize

	return &AlignmentExtractor{
		config: featureConf,
		logger: logger,

		// Initialize alignment algorithms
		alignmentAnalyzer: stats.NewAlignmentAnalyzer(stats.AlignmentHybrid, maxLagFrames, featureConf.SampleRate, featureConf.HopSize, featureConf.WindowSize, alignmentConf.MinConfidence),
		dtwAnalyzer:       stats.NewDTWAlignment(),
		crossCorr:         stats.NewCrossCorrelation(maxLagFrames),
		energy:            temporal.NewEnergy(featureConf.WindowSize, featureConf.HopSize, featureConf.SampleRate),

		// Set parameters
		maxLagSamples:    maxLagSamples,
		maxLagSeconds:    maxLagSeconds,
		stepSize:         alignmentConf.StepSize,
		confidenceThresh: alignmentConf.MinConfidence,
	}
}

// NewAlignmentExtractorWithMaxLag creates a new alignment feature extractor based on the max lag (in seconds)
func NewAlignmentExtractorWithMaxLag(featureConf *config.FeatureConfig, alignmentConf *config.AlignmentConfig, maxLagSeconds float64) *AlignmentExtractor {
	logger := logging.WithFields(logging.Fields{
		"component": "alignment_extractor",
	})

	maxLagSamples := int(maxLagSeconds * float64(featureConf.SampleRate))

	hopSize := featureConf.HopSize
	maxLagFrames := maxLagSamples / hopSize

	logger.Debug("Alignment extractor configuration", logging.Fields{
		"maxLagSeconds": maxLagSeconds,
		"maxLagSamples": maxLagSamples,
		"maxLagFrames":  maxLagFrames,
		"hopSize":       hopSize,
	})

	return &AlignmentExtractor{
		config: featureConf,
		logger: logger,

		alignmentAnalyzer: stats.NewAlignmentAnalyzer(
			stats.AlignmentHybrid,
			maxLagFrames,
			featureConf.SampleRate,
			featureConf.HopSize,
			featureConf.WindowSize,
			alignmentConf.MinConfidence),
		dtwAnalyzer: stats.NewDTWAlignment(),
		crossCorr:   stats.NewCrossCorrelation(maxLagFrames),
		energy:      temporal.NewEnergy(featureConf.WindowSize, featureConf.HopSize, featureConf.SampleRate),

		maxLagSamples:    maxLagSamples,
		maxLagSeconds:    maxLagSeconds,
		stepSize:         alignmentConf.StepSize,
		confidenceThresh: alignmentConf.MinConfidence,
	}
}

// ExtractAlignmentFeatures performs comprehensive alignment analysis between two feature sets
func (ae *AlignmentExtractor) ExtractAlignmentFeatures(
	queryFeatures, referenceFeatures *ExtractedFeatures,
	queryPCM, referencePCM []float64,
	sampleRate int,
) (*AlignmentFeatures, error) {

	if queryFeatures == nil || referenceFeatures == nil {
		return nil, fmt.Errorf("feature sets cannot be nil")
	}

	logger := ae.logger.WithFields(logging.Fields{
		"function":      "ExtractAlignmentFeatures",
		"query_pcm_len": len(queryPCM),
		"ref_pcm_len":   len(referencePCM),
		"sample_rate":   sampleRate,
	})

	logger.Debug("Starting alignment feature extraction")

	result := &AlignmentFeatures{
		FeatureSimilarity: make(map[string]float64),
		QueryLength:       float64(len(queryPCM)) / float64(sampleRate),
		ReferenceLength:   float64(len(referencePCM)) / float64(sampleRate),
	}

	// Step 1: Try alignment with multiple feature types
	alignments := ae.performMultiFeatureAlignment(queryFeatures, referenceFeatures, sampleRate)

	// Step 2: Select best alignment
	bestAlignment := ae.selectBestAlignment(alignments)
	if bestAlignment != nil {
		result.BestAlignment = bestAlignment
		result.TemporalOffset = bestAlignment.OffsetSeconds
		result.OffsetConfidence = bestAlignment.Confidence
		result.OverallSimilarity = bestAlignment.Similarity
		result.AlignmentQuality = bestAlignment.AlignmentQuality
		result.Method = bestAlignment.FeatureType
	}

	// Step 3: Store individual alignment results
	for featureType, alignment := range alignments {
		switch featureType {
		case "dtw_mfcc":
			if alignment.AlignmentResult != nil && alignment.DTWResult != nil {
				result.DTWAlignment = alignment
			}
		case "corr_energy":
			if alignment.AlignmentResult != nil && alignment.CrossCorrResult != nil {
				result.CorrAlignment = alignment
			}
		}

		// Store feature similarities
		if alignment.Success {
			result.FeatureSimilarity[featureType] = alignment.Similarity
		}
	}

	// Step 4: Estimate time stretch factor
	result.TimeStretch = ae.estimateTimeStretch(result.BestAlignment, result.QueryLength, result.ReferenceLength)

	// Step 5: Analyze alignment consistency
	// TODO: fix the bottleneck but ensure consistency
	/* if result.BestAlignment != nil && result.BestAlignment.Success {
		consistency, err := ae.analyzeConsistency(queryFeatures, referenceFeatures, sampleRate)
		if err != nil {
			logger.Warn("Failed to analyze alignment consistency", logging.Fields{"error": err})
		} else {
			result.Consistency = consistency
		}
	} */

	logger.Debug("Alignment feature extraction completed", logging.Fields{
		"best_method":             result.Method,
		"temporal_offset_seconds": result.TemporalOffset,
		"similarity":              result.OverallSimilarity,
		"quality":                 result.AlignmentQuality,
	})

	return result, nil
}

// TruncateToAlignmentPCM takes in the PCM audio and the `AlignmentFeatures` and purges all audio that doesn't align.
// This is an essential step before fingerprinting.
func (ae *AlignmentExtractor) TruncateToAlignmentPCM(pcm1, pcm2 []float64, sampleRate int, alignment *AlignmentFeatures) ([]float64, []float64, error) {
	offsetSeconds := alignment.TemporalOffset
	sampleRateFloat := float64(sampleRate)

	// Convert to samples with proper rounding
	offsetSamples := int(math.Round(math.Abs(offsetSeconds) * sampleRateFloat))

	logger := ae.logger.WithFields(logging.Fields{
		"function":    "ExtractAlignmentFeatures",
		"pcm1_len":    len(pcm1),
		"pcm2_len":    len(pcm2),
		"sample_rate": sampleRate,
	})

	var start1, start2, commonLength int

	if offsetSeconds > 0 {
		// Stream 2 is ahead: skip beginning of stream 2, keep beginning of stream 1
		start1 = 0
		start2 = offsetSamples

		if start2 >= len(pcm2) {
			return nil, nil, fmt.Errorf("offset too large: need to skip %d samples but pcm2 only has %d", start2, len(pcm2))
		}

		// Calculate how much audio remains after skipping
		remaining1 := len(pcm1) - start1 // All of pcm1
		remaining2 := len(pcm2) - start2 // PCM2 after skipping
		commonLength = min(remaining1, remaining2)

	} else if offsetSeconds < 0 {
		// Stream 1 is ahead: skip beginning of stream 1, keep beginning of stream 2
		start1 = offsetSamples
		start2 = 0

		if start1 >= len(pcm1) {
			return nil, nil, fmt.Errorf("offset too large: need to skip %d samples but pcm1 only has %d", start1, len(pcm1))
		}

		remaining1 := len(pcm1) - start1 // PCM1 after skipping
		remaining2 := len(pcm2) - start2 // All of pcm2
		commonLength = min(remaining1, remaining2)

	} else {
		// No offset
		start1, start2 = 0, 0
		commonLength = min(len(pcm1), len(pcm2))
	}

	if commonLength <= 0 {
		return nil, nil, fmt.Errorf("no overlapping audio after alignment")
	}

	// Add some padding to ensure we get the best aligned portion
	// Skip a bit more at the beginning and end to avoid edge effects
	paddingSamples := int(0.5 * sampleRateFloat) // 0.5 second padding
	if commonLength > 2*paddingSamples {
		start1 += paddingSamples
		start2 += paddingSamples
		commonLength -= 2 * paddingSamples
	}

	logger.Debug("After alignment",
		logging.Fields{
			"start1": start1,
			"start2": start2,
			"common_length_" + strconv.Itoa(commonLength): float64(commonLength) / sampleRateFloat,
		})

	// Return aligned PCM segments
	alignedPCM1 := pcm1[start1 : start1+commonLength]
	alignedPCM2 := pcm2[start2 : start2+commonLength]

	return alignedPCM1, alignedPCM2, nil
}

// performMultiFeatureAlignment tries alignment with different feature types
func (ae *AlignmentExtractor) performMultiFeatureAlignment(
	queryFeatures, referenceFeatures *ExtractedFeatures,
	sampleRate int,
) map[string]*AlignmentResult {

	alignments := make(map[string]*AlignmentResult)

	ae.logger.Debug("Available features for alignment", logging.Fields{
		"query_mfcc_available":     queryFeatures.MFCC != nil,
		"ref_mfcc_available":       referenceFeatures.MFCC != nil,
		"query_energy_available":   queryFeatures.EnergyFeatures != nil,
		"ref_energy_available":     referenceFeatures.EnergyFeatures != nil,
		"query_spectral_available": queryFeatures.SpectralFeatures != nil,
		"ref_spectral_available":   referenceFeatures.SpectralFeatures != nil,
	})

	// 1. MFCC-based DTW alignment (best for speech content)
	// if queryFeatures.MFCC != nil && referenceFeatures.MFCC != nil {
	// alignment := ae.alignWithFeatures("dtw_mfcc", queryFeatures.MFCC, referenceFeatures.MFCC, sampleRate, stats.AlignmentDTW)
	// alignments["dtw_mfcc"] = alignment
	// }

	// 2. Energy-based cross-correlation (fast, good for similar content)
	if queryFeatures.EnergyFeatures != nil && referenceFeatures.EnergyFeatures != nil &&
		len(queryFeatures.EnergyFeatures.ShortTimeEnergy) > 0 && len(referenceFeatures.EnergyFeatures.ShortTimeEnergy) > 0 {

		// Convert energy to 2D features
		queryEnergy := ae.convertEnergyTo2D(queryFeatures.EnergyFeatures.ShortTimeEnergy)
		refEnergy := ae.convertEnergyTo2D(referenceFeatures.EnergyFeatures.ShortTimeEnergy)

		alignment := ae.alignWithFeatures("corr_energy", queryEnergy, refEnergy, sampleRate, stats.AlignmentCrossCorrelation)
		alignments["corr_energy"] = alignment
	}

	// 3. Spectral centroid alignment (good for timbral changes)
	// if queryFeatures.SpectralFeatures != nil && referenceFeatures.SpectralFeatures != nil &&
	// len(queryFeatures.SpectralFeatures.SpectralCentroid) > 0 && len(referenceFeatures.SpectralFeatures.SpectralCentroid) > 0 {
	//
	// queryCentroid := ae.convertSpectralTo2D(queryFeatures.SpectralFeatures.SpectralCentroid)
	// refCentroid := ae.convertSpectralTo2D(referenceFeatures.SpectralFeatures.SpectralCentroid)
	//
	// alignment := ae.alignWithFeatures("dtw_centroid", queryCentroid, refCentroid, sampleRate, stats.AlignmentDTW)
	// alignments["dtw_centroid"] = alignment
	// }

	// 4. Chroma alignment (good for harmonic content)
	if queryFeatures.ChromaFeatures != nil && referenceFeatures.ChromaFeatures != nil &&
		len(queryFeatures.ChromaFeatures) > 0 && len(referenceFeatures.ChromaFeatures) > 0 {

		alignment := ae.alignWithFeatures("dtw_chroma", queryFeatures.ChromaFeatures, referenceFeatures.ChromaFeatures, sampleRate, stats.AlignmentDTW)
		alignments["dtw_chroma"] = alignment
	}

	return alignments
}

// alignWithFeatures performs alignment using specified features and method
func (ae *AlignmentExtractor) alignWithFeatures(
	featureType string,
	queryFeatures, referenceFeatures [][]float64,
	sampleRate int,
	method stats.AlignmentMethod,
) *AlignmentResult {

	logger := ae.logger.WithFields(logging.Fields{
		"feature_type": featureType,
		"method":       method,
		"query_frames": len(queryFeatures),
		"ref_frames":   len(referenceFeatures),
	})

	// Clamp maxLagFrames to actual data bounds
	minFrames := min(len(queryFeatures), len(referenceFeatures))
	maxLagFrames := ae.maxLagSamples / ae.config.HopSize
	maxLagFrames = min(maxLagFrames, minFrames-1)

	logger.Debug("Creating alignment analyzer", logging.Fields{
		"maxLagSamples": ae.maxLagSamples,
		"maxLagFrames":  maxLagFrames,
		"hopSize":       ae.config.HopSize,
	})

	// Create analyzer with FRAME-based lag
	analyzer := stats.NewAlignmentAnalyzer(method, maxLagFrames, sampleRate, ae.config.HopSize, ae.config.WindowSize, ae.confidenceThresh)

	logger.Debug("Starting alignment computation")

	// Perform alignment
	result, err := analyzer.AlignFeatures(queryFeatures, referenceFeatures, sampleRate)
	if err != nil {
		logger.Warn("Alignment failed", logging.Fields{"error": err})
		return &AlignmentResult{
			FeatureType: featureType,
			Success:     false,
			ErrorMsg:    err.Error(),
		}
	}

	logger.Debug("Alignment succeeded", logging.Fields{
		"offset_seconds": result.OffsetSeconds,
		"confidence":     result.Confidence,
		"similarity":     result.Similarity,
	})

	return &AlignmentResult{
		AlignmentResult: result,
		FeatureType:     featureType,
		Success:         true,
	}
}

// selectBestAlignment chooses the best alignment from multiple attempts
func (ae *AlignmentExtractor) selectBestAlignment(alignments map[string]*AlignmentResult) *AlignmentResult {
	var bestAlignment *AlignmentResult
	bestScore := 0.0

	// Define priority weights for different feature types
	weights := map[string]float64{
		// "dtw_mfcc":     1.0, // Highest priority for speech content
		"corr_energy": 1.0, // Good general-purpose alignment (was 0.8, now the only active)
		"dtw_chroma":  0.7, // Good for harmonic content
		// "dtw_centroid": 0.6, // Lower priority but still useful
	}

	for featureType, alignment := range alignments {
		if !alignment.Success || alignment.AlignmentResult == nil {
			continue
		}

		// Calculate weighted score
		weight := weights[featureType]
		if weight == 0 {
			weight = 0.5 // Default weight for unknown feature types
		}

		// Combine confidence, similarity, and quality
		score := weight * (0.4*alignment.Confidence + 0.4*alignment.Similarity + 0.2*alignment.AlignmentQuality)

		if score > bestScore {
			bestScore = score
			bestAlignment = alignment
		}
	}

	return bestAlignment
}

// estimateTimeStretch estimates the time stretch factor between sequences
func (ae *AlignmentExtractor) estimateTimeStretch(alignment *AlignmentResult, queryLen, refLen float64) float64 {
	if alignment == nil || !alignment.Success || queryLen <= 0 || refLen <= 0 {
		return 1.0 // No stretch
	}

	// Simple ratio-based stretch estimation
	lengthRatio := queryLen / refLen

	// If we have DTW path, use it for more accurate estimation
	if alignment.DTWResult != nil && len(alignment.DTWResult.Path) > 0 {
		// Calculate average slope of DTW path
		path := alignment.DTWResult.Path
		if len(path) > 1 {
			startPoint := path[0]
			endPoint := path[len(path)-1]

			querySpan := float64(endPoint.QueryIndex - startPoint.QueryIndex + 1)
			refSpan := float64(endPoint.RefIndex - startPoint.RefIndex + 1)

			if refSpan > 0 {
				pathRatio := querySpan / refSpan
				// Weighted combination of length ratio and path ratio
				return 0.7*pathRatio + 0.3*lengthRatio
			}
		}
	}

	return lengthRatio
}

// Helper methods for feature conversion

func (ae *AlignmentExtractor) convertEnergyTo2D(energy []float64) [][]float64 {
	result := make([][]float64, len(energy))
	for i, val := range energy {
		result[i] = []float64{val}
	}
	return result
}

// AlignAudioFiles provides a high-level interface for aligning two audio files
func (ae *AlignmentExtractor) AlignAudioFiles(
	queryPCM, referencePCM []float64,
	sampleRate int,
	extractorFactory *FeatureExtractorFactory,
	contentType config.ContentType,
) (*AlignmentFeatures, error) {

	logger := ae.logger.WithFields(logging.Fields{
		"function":     "AlignAudioFiles",
		"content_type": contentType,
	})

	// Create appropriate feature extractor
	// TODO: unused extractor
	//extractor, err := extractorFactory.CreateExtractor(contentType, *ae.config)
	//if err != nil {
	//	return nil, fmt.Errorf("failed to create feature extractor: %w", err)
	//}

	// TODO:
	// Extract features from both audio files
	// Note: This is simplified - in practice you'd need spectrograms
	// For now, we'll use energy-based alignment as a fallback

	// Extract energy features directly
	queryEnergy := ae.energy.ComputeShortTimeEnergy(queryPCM)
	refEnergy := ae.energy.ComputeShortTimeEnergy(referencePCM)

	// Convert to 2D features
	queryFeats := ae.convertEnergyTo2D(queryEnergy)
	refFeats := ae.convertEnergyTo2D(refEnergy)

	// Perform alignment
	result, err := ae.alignmentAnalyzer.AlignFeatures(queryFeats, refFeats, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("alignment failed: %w", err)
	}

	// Create alignment features result
	alignmentFeatures := &AlignmentFeatures{
		BestAlignment: &AlignmentResult{
			AlignmentResult: result,
			FeatureType:     "energy",
			Success:         true,
		},
		TemporalOffset:    result.OffsetSeconds,
		OffsetConfidence:  result.Confidence,
		OverallSimilarity: result.Similarity,
		AlignmentQuality:  result.AlignmentQuality,
		Method:            "energy_correlation",
		QueryLength:       float64(len(queryPCM)) / float64(sampleRate),
		ReferenceLength:   float64(len(referencePCM)) / float64(sampleRate),
		FeatureSimilarity: map[string]float64{
			"energy": result.Similarity,
		},
	}

	logger.Debug("Audio file alignment completed", logging.Fields{
		"offset_seconds": alignmentFeatures.TemporalOffset,
		"similarity":     alignmentFeatures.OverallSimilarity,
		"confidence":     alignmentFeatures.OffsetConfidence,
	})

	return alignmentFeatures, nil
}

// GetAlignmentSummary provides a human-readable summary of alignment results
func (ae *AlignmentExtractor) GetAlignmentSummary(features *AlignmentFeatures) map[string]any {
	summary := make(map[string]any)

	if features == nil {
		summary["status"] = "failed"
		return summary
	}

	summary["status"] = "success"
	summary["method"] = features.Method
	summary["offset_seconds"] = features.TemporalOffset
	summary["similarity_percent"] = features.OverallSimilarity * 100
	summary["confidence_percent"] = features.OffsetConfidence * 100
	summary["quality_percent"] = features.AlignmentQuality * 100

	// Classify alignment quality
	if features.OffsetConfidence > 0.8 {
		summary["quality_description"] = "excellent"
	} else if features.OffsetConfidence > 0.6 {
		summary["quality_description"] = "good"
	} else if features.OffsetConfidence > 0.4 {
		summary["quality_description"] = "fair"
	} else {
		summary["quality_description"] = "poor"
	}

	// Time stretch information
	summary["time_stretch_factor"] = features.TimeStretch
	if math.Abs(features.TimeStretch-1.0) > 0.05 {
		summary["time_stretch_detected"] = true
	} else {
		summary["time_stretch_detected"] = false
	}

	return summary
}
