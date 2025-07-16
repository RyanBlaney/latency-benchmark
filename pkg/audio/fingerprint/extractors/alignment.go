package extractors

import (
	"fmt"
	"math"

	"github.com/goccmack/godsp"
)

// LightweightAlignment performs fast temporal alignment using go-dsp cross-correlation
type LightweightAlignment struct {
	MaxOffsetSeconds float64 // Maximum expected offset
	FeatureRateHz    float64 // Feature extraction rate (features per second)
}

// NewLightweightAlignment creates a new lightweight alignment analyzer
func NewLightweightAlignment(maxOffsetSeconds, featureRateHz float64) *LightweightAlignment {
	return &LightweightAlignment{
		MaxOffsetSeconds: maxOffsetSeconds,
		FeatureRateHz:    featureRateHz,
	}
}

// AlignmentResult contains the alignment results
type AlignmentResult struct {
	OffsetSeconds float64 // Time offset in seconds (stream2 relative to stream1)
	Confidence    float64 // Correlation coefficient (0-1)
	IsValid       bool    // Whether alignment is reliable
}

// FindAlignment finds temporal offset between two feature sets using go-dsp
func (la *LightweightAlignment) FindAlignment(features1, features2 *ExtractedFeatures) (*AlignmentResult, error) {
	// Extract the best signals for alignment (prioritize energy for speech)
	signal1 := la.extractBestSignal(features1)
	signal2 := la.extractBestSignal(features2)

	if len(signal1) == 0 || len(signal2) == 0 {
		return nil, fmt.Errorf("no suitable signals for alignment")
	}

	fmt.Printf("DEBUG Alignment: signal lengths: sig1=%d, sig2=%d, max_offset=%.1fs\n",
		len(signal1), len(signal2), la.MaxOffsetSeconds)

	// Use manual cross-correlation (more reliable than go-dsp for alignment)
	offset, confidence := la.crossCorrelateManual(signal1, signal2)

	fmt.Printf("DEBUG Alignment: result: offset=%.3fs, confidence=%.4f\n", offset, confidence)

	// Determine if alignment is valid
	isValid := confidence > 0.6

	return &AlignmentResult{
		OffsetSeconds: offset,
		Confidence:    confidence,
		IsValid:       isValid,
	}, nil
}

// crossCorrelateManual performs manual cross-correlation for better control
func (la *LightweightAlignment) crossCorrelateManual(signal1, signal2 []float64) (float64, float64) {
	// Calculate max delay in samples
	maxDelaySamples := int(la.MaxOffsetSeconds * la.FeatureRateHz)

	fmt.Printf("DEBUG CrossCorr: manual implementation with maxDelay=%d samples (%.1fs)\n",
		maxDelaySamples, la.MaxOffsetSeconds)

	// Normalize signals to improve correlation
	norm1 := la.normalizeSignal(signal1)
	norm2 := la.normalizeSignal(signal2)

	// Ensure we don't search beyond signal boundaries
	minLen := len(norm1)
	if len(norm2) < minLen {
		minLen = len(norm2)
	}

	// Limit search range to prevent out-of-bounds
	maxSearchDelay := minInt(maxDelaySamples, minLen-1)

	fmt.Printf("DEBUG CrossCorr: searching range: -%d to +%d samples\n", maxSearchDelay, maxSearchDelay)

	maxCorr := -1.0
	bestOffset := 0

	// Search both positive and negative offsets
	for delay := -maxSearchDelay; delay <= maxSearchDelay; delay++ {
		corr := la.calculateCorrelation(norm1, norm2, delay)

		if corr > maxCorr {
			maxCorr = corr
			bestOffset = delay
		}
	}

	// Convert to seconds
	offsetSeconds := float64(bestOffset) / la.FeatureRateHz

	fmt.Printf("DEBUG CrossCorr: best offset: %d samples (%.3fs), correlation: %.4f\n",
		bestOffset, offsetSeconds, maxCorr)

	return offsetSeconds, maxCorr
}

// calculateCorrelation calculates normalized correlation at a specific delay
func (la *LightweightAlignment) calculateCorrelation(signal1, signal2 []float64, delay int) float64 {
	// Determine overlap region
	start1, end1 := 0, len(signal1)
	start2, end2 := 0, len(signal2)

	if delay >= 0 {
		// signal2 is delayed relative to signal1
		start2 = delay
		end1 = minInt(end1, len(signal2)-delay)
	} else {
		// signal1 is delayed relative to signal2
		start1 = -delay
		end2 = minInt(end2, len(signal1)+delay)
	}

	// Calculate correlation for overlap region
	overlapLen := minInt(end1-start1, end2-start2)
	if overlapLen <= 0 {
		return 0.0
	}

	var sum1, sum2, sum12, sum1sq, sum2sq float64
	count := 0

	for i := 0; i < overlapLen; i++ {
		idx1 := start1 + i
		idx2 := start2 + i

		if idx1 >= 0 && idx1 < len(signal1) && idx2 >= 0 && idx2 < len(signal2) {
			val1 := signal1[idx1]
			val2 := signal2[idx2]

			sum1 += val1
			sum2 += val2
			sum12 += val1 * val2
			sum1sq += val1 * val1
			sum2sq += val2 * val2
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	// Calculate Pearson correlation coefficient
	n := float64(count)
	numerator := n*sum12 - sum1*sum2
	denominator1 := n*sum1sq - sum1*sum1
	denominator2 := n*sum2sq - sum2*sum2

	denominator := math.Sqrt(denominator1 * denominator2)
	if denominator < 1e-10 {
		return 0.0
	}

	correlation := numerator / denominator

	// Ensure correlation is in valid range [-1, 1]
	if correlation > 1.0 {
		correlation = 1.0
	} else if correlation < -1.0 {
		correlation = -1.0
	}

	return correlation
}

// crossCorrelateWithGoDSP performs cross-correlation using the go-dsp library (backup method)
func (la *LightweightAlignment) crossCorrelateWithGoDSP(signal1, signal2 []float64) (float64, float64) {
	// Calculate max delay in samples
	maxDelaySamples := int(la.MaxOffsetSeconds * la.FeatureRateHz)

	fmt.Printf("DEBUG CrossCorr: using go-dsp with maxDelay=%d samples (%.1fs)\n",
		maxDelaySamples, la.MaxOffsetSeconds)

	// Normalize signals to improve correlation
	norm1 := la.normalizeSignal(signal1)
	norm2 := la.normalizeSignal(signal2)

	// Use go-dsp cross-correlation
	correlation := godsp.Xcorr(norm1, norm2, maxDelaySamples)

	// Find the peak correlation
	maxCorr := -1.0
	bestOffset := 0

	for i, corr := range correlation {
		if corr > maxCorr {
			maxCorr = corr
			bestOffset = i - maxDelaySamples // Center around zero
		}
	}

	// Convert to seconds
	offsetSeconds := float64(bestOffset) / la.FeatureRateHz

	fmt.Printf("DEBUG CrossCorr: best offset: %d samples (%.3fs), correlation: %.4f\n",
		bestOffset, offsetSeconds, maxCorr)

	return offsetSeconds, maxCorr
}

// extractBestSignal extracts the most reliable signal for alignment
func (la *LightweightAlignment) extractBestSignal(features *ExtractedFeatures) []float64 {
	// Priority 1: Energy features (best for speech alignment)
	if features.EnergyFeatures != nil && len(features.EnergyFeatures.ShortTimeEnergy) > 0 {
		fmt.Printf("DEBUG: Using energy features for alignment (length: %d)\n", len(features.EnergyFeatures.ShortTimeEnergy))
		return features.EnergyFeatures.ShortTimeEnergy
	}

	// Priority 2: Temporal RMS energy
	if features.TemporalFeatures != nil && len(features.TemporalFeatures.RMSEnergy) > 0 {
		fmt.Printf("DEBUG: Using temporal RMS energy for alignment (length: %d)\n", len(features.TemporalFeatures.RMSEnergy))
		return features.TemporalFeatures.RMSEnergy
	}

	// Priority 3: Spectral centroid (fallback)
	if features.SpectralFeatures != nil && len(features.SpectralFeatures.SpectralCentroid) > 0 {
		fmt.Printf("DEBUG: Using spectral centroid for alignment (length: %d)\n", len(features.SpectralFeatures.SpectralCentroid))
		return features.SpectralFeatures.SpectralCentroid
	}

	fmt.Printf("DEBUG: No suitable features found for alignment\n")
	return []float64{}
}

// normalizeSignal normalizes a signal to zero mean and unit variance
func (la *LightweightAlignment) normalizeSignal(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Calculate mean
	mean := 0.0
	for _, val := range signal {
		mean += val
	}
	mean /= float64(len(signal))

	// Calculate standard deviation
	variance := 0.0
	for _, val := range signal {
		diff := val - mean
		variance += diff * diff
	}
	variance /= float64(len(signal))
	stdDev := math.Sqrt(variance)

	// Handle constant signals
	if stdDev < 1e-10 {
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - mean
		}
		return normalized
	}

	// Normalize to zero mean, unit variance
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - mean) / stdDev
	}

	return normalized
}

// AlignStreams is a convenience function for aligning two feature sets
func AlignStreams(features1, features2 *ExtractedFeatures, maxOffsetSeconds, featureRateHz float64) (*AlignmentResult, error) {
	aligner := NewLightweightAlignment(maxOffsetSeconds, featureRateHz)
	return aligner.FindAlignment(features1, features2)
}

// AlignStreamsWithEnergyOnly performs energy-only alignment (fastest, best for speech)
func AlignStreamsWithEnergyOnly(features1, features2 *ExtractedFeatures, maxOffsetSeconds, featureRateHz float64) (*AlignmentResult, error) {
	aligner := NewLightweightAlignment(maxOffsetSeconds, featureRateHz)

	// Extract energy signals directly
	var signal1, signal2 []float64

	if features1.EnergyFeatures != nil && len(features1.EnergyFeatures.ShortTimeEnergy) > 0 {
		signal1 = features1.EnergyFeatures.ShortTimeEnergy
	} else {
		return nil, fmt.Errorf("no energy features in stream 1")
	}

	if features2.EnergyFeatures != nil && len(features2.EnergyFeatures.ShortTimeEnergy) > 0 {
		signal2 = features2.EnergyFeatures.ShortTimeEnergy
	} else {
		return nil, fmt.Errorf("no energy features in stream 2")
	}

	offset, confidence := aligner.crossCorrelateManual(signal1, signal2)

	return &AlignmentResult{
		OffsetSeconds: offset,
		Confidence:    confidence,
		IsValid:       confidence > 0.5, // Lower threshold for energy-only
	}, nil
}
