package extractors

import (
	"fmt"
	"math"
)

// LightweightAlignment performs fast temporal alignment using a
// cross-correlation on existing features
type LightweightAlignment struct {
	MaxOffsetSeconds float64 // Maximum expected offset
	SampleRateHz     float64 // Feature extraction rate (features per second)
}

// NewLightweightAlignment creates a new lightweight alignment analyzer
func NewLightweightAlignment(maxOffsetSeconds, featureRateHz float64) *LightweightAlignment {
	return &LightweightAlignment{
		MaxOffsetSeconds: maxOffsetSeconds,
		SampleRateHz:     featureRateHz,
	}
}

// AlignmentResult contains the alignment results
type AlignmentResult struct {
	OffsetSeconds float64 // Time offset in seconds (stream2 relative to stream1)
	Confidence    float64 // Correlation coefficient (0-1)
	IsValid       bool    // Whether alignment is reliable
}

// FindAlignmment finds temporal offset between two feature sets
func (la *LightweightAlignment) FindAlignment(features1, features2 *ExtractedFeatures) (*AlignmentResult, error) {
	// Use the most distinctive features for alignment
	signals1 := la.extractSignals(features1)
	signals2 := la.extractSignals(features2)

	if len(signals1) == 0 || len(signals2) == 0 {
		return nil, fmt.Errorf("no suitable signals for alignment")
	}

	// Perform cross-correlation on primary signal (spectral centroid)
	bestOffset, bestCorr := la.crossCorrelate(signals1[0], signals2[0])

	// Validate with secondary signals if available
	if len(signals1) > 1 && len(signals2) > 1 {
		offset2, corr2 := la.crossCorrelate(signals1[1], signals2[1])

		// Check if offsets are consistent (within 0.5 seconds)
		if math.Abs(bestOffset-offset2) > 0.5 {
			// Use the one with higher correlation
			if corr2 > bestCorr {
				bestOffset = offset2
				bestCorr = corr2
			}
		} else {
			// Average the offsets, keep the better correlation
			bestOffset = (bestOffset + offset2) / 2
			if corr2 > bestCorr {
				bestCorr = corr2
			}
		}
	}

	return &AlignmentResult{
		OffsetSeconds: bestOffset,
		Confidence:    bestCorr,
		IsValid:       bestCorr > 0.6, // TODO: Config for correlation threshold
	}, nil
}

// extractSignals extracts the best signals for alignment from features
func (la *LightweightAlignment) extractSignals(features *ExtractedFeatures) [][]float64 {
	var signals [][]float64

	// Primary: Spectral Centroid (most distinctive)
	if features.SpectralFeatures != nil && len(features.SpectralFeatures.SpectralCentroid) > 0 {
		signals = append(signals, features.SpectralFeatures.SpectralCentroid)
	}

	// Secondary: Energy (for validation)
	if features.EnergyFeatures != nil && len(features.EnergyFeatures.ShortTimeEnergy) > 0 {
		signals = append(signals, features.EnergyFeatures.ShortTimeEnergy)
	} else if features.TemporalFeatures != nil && len(features.TemporalFeatures.RMSEnergy) > 0 {
		signals = append(signals, features.TemporalFeatures.RMSEnergy)
	}

	// Tertiary: Zero Crossing Rate (texture changes)
	if features.SpectralFeatures != nil && len(features.SpectralFeatures.ZeroCrossingRate) > 0 {
		signals = append(signals, features.SpectralFeatures.ZeroCrossingRate)
	}

	return signals
}

// crossCorrelate performs normalized cross-correlation between two signals
func (la *LightweightAlignment) crossCorrelate(signal1, signal2 []float64) (float64, float64) {
	// Normalize signals (zero mean, unit variance)
	norm1 := la.normalizeSignal(signal1)
	norm2 := la.normalizeSignal(signal2)

	maxOffset := int(la.MaxOffsetSeconds * la.SampleRateHz)

	bestCorr := -1.0
	bestOffset := 0

	// Try different offsets
	for offset := -maxOffset; offset <= maxOffset; offset++ {
		corr := la.computeCorrelation(norm1, norm2, offset)
		if corr > bestCorr {
			bestCorr = corr
			bestOffset = offset
		}

	}

	// Convert offset to seconds
	offsetSeconds := float64(bestOffset) / la.SampleRateHz

	return offsetSeconds, bestCorr

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

	if stdDev == 0 {
		return signal // Avoid division by zero
	}

	// Normalize
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - mean) / stdDev
	}

	return normalized
}

// computeCorrelation computes correlation between two signals at a given offset
func (la *LightweightAlignment) computeCorrelation(signal1, signal2 []float64, offset int) float64 {
	n1, n2 := len(signal1), len(signal2)

	// Determine overlap region
	var start1, end1, start2, end2 int

	if offset >= 0 {
		// signal2 is shifted right (starts later)
		start1 = offset
		end1 = n1
		start2 = 0
		end2 = n2 - offset
	} else {
		// signal2 is shifted left (starts earlier)
		start1 = 0
		end1 = n1 + offset
		start2 = -offset
		end2 = n2
	}

	// Ensure bounds are valid
	if start1 >= n1 || start2 >= n2 || end1 <= start1 || end2 <= start2 {
		return 0.0
	}

	if end1 > n1 {
		end1 = n1
	}
	if end2 > n2 {
		end2 = n2
	}

	// Compute correlation for overlap region
	overlapLen := min(end1-start1, end2-start2)
	if overlapLen <= 0 {
		return 0.0
	}

	correlation := 0.0
	for i := range overlapLen {
		correlation += signal1[start1+i] * signal2[start2+i]
	}

	return correlation / float64(overlapLen)
}

// QuickAlign performs ultra-fast alignment using only energy features
func (la *LightweightAlignment) QuickAlign(
	features1, features2 *ExtractedFeatures,
) (*AlignmentResult, error) {

	// Use only energy - fastest alignment for music curation
	var signal1, signal2 []float64

	if features1.EnergyFeatures != nil && len(features1.EnergyFeatures.ShortTimeEnergy) > 0 {
		signal1 = features1.EnergyFeatures.ShortTimeEnergy
	} else if features1.TemporalFeatures != nil && len(features1.TemporalFeatures.RMSEnergy) > 0 {
		signal1 = features1.TemporalFeatures.RMSEnergy
	} else {
		return nil, fmt.Errorf("no energy features available in stream 1")
	}

	if features2.EnergyFeatures != nil && len(features2.EnergyFeatures.ShortTimeEnergy) > 0 {
		signal2 = features2.EnergyFeatures.ShortTimeEnergy
	} else if features2.TemporalFeatures != nil && len(features2.TemporalFeatures.RMSEnergy) > 0 {
		signal2 = features2.TemporalFeatures.RMSEnergy
	} else {
		return nil, fmt.Errorf("no energy features available in stream 2")
	}

	offset, corr := la.crossCorrelate(signal1, signal2)

	return &AlignmentResult{
		OffsetSeconds: offset,
		Confidence:    corr,
		IsValid:       corr > 0.5, // Lower threshold for quick alignment
	}, nil
}

// AlignStreams is a convenience function that aligns two streams and returns synchronized segments
func AlignStreams(features1, features2 *ExtractedFeatures, maxOffsetSec, featureRateHz float64) (*AlignmentResult, error) {
	aligner := NewLightweightAlignment(maxOffsetSec, featureRateHz)
	return aligner.FindAlignment(features1, features2)
}

// QuickAlignStreams is a convenience function for ultra-fast energy-based alignment
func QuickAlignStreams(features1, features2 *ExtractedFeatures, maxOffsetSec, featureRateHz float64) (*AlignmentResult, error) {
	aligner := NewLightweightAlignment(maxOffsetSec, featureRateHz)
	return aligner.QuickAlign(features1, features2)
}
