package common

import (
	"math"
)

// NormalizationType defines normalization method
type NormalizationType int

const (
	ZScore NormalizationType = iota
	MinMax
	Energy
	Peak
	RMSNorm
	Quantile
	Robust
)

// Normalizer provides various signal normalization methods
type Normalizer struct {
	method NormalizationType
}

// NewNormalizer creates a new normalizer
func NewNormalizer(method NormalizationType) *Normalizer {
	return &Normalizer{
		method: method,
	}
}

// Normalize normalizes signal using the specified method
func (n *Normalizer) Normalize(signal []float64) []float64 {
	switch n.method {
	case ZScore:
		return n.zScoreNormalize(signal)
	case MinMax:
		return n.minMaxNormalize(signal)
	case Energy:
		return n.energyNormalize(signal)
	case Peak:
		return n.peakNormalize(signal)
	case RMSNorm:
		return n.rmsNormalize(signal)
	case Quantile:
		return n.quantileNormalize(signal, 0.05, 0.95)
	case Robust:
		return n.robustNormalize(signal)
	default:
		return n.zScoreNormalize(signal)
	}
}

// zScoreNormalize normalizes to zero mean and unit variance
func (n *Normalizer) zScoreNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	mean := Mean(signal)
	std := StandardDeviation(signal)

	if std < 1e-10 {
		// Handle constant signal
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - mean
		}
		return normalized
	}

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - mean) / std
	}

	return normalized
}

// minMaxNormalize normalizes to [0, 1] range
func (n *Normalizer) minMaxNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	min := signal[0]
	max := signal[0]

	for _, val := range signal {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	if math.Abs(max-min) < 1e-10 {
		// Handle constant signal
		normalized := make([]float64, len(signal))
		return normalized // All zeros
	}

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - min) / (max - min)
	}

	return normalized
}

// energyNormalize normalizes by total energy (L2 norm)
func (n *Normalizer) energyNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	energy := 0.0
	for _, val := range signal {
		energy += val * val
	}

	if energy < 1e-10 {
		return signal // Return unchanged if no energy
	}

	energyNorm := math.Sqrt(energy)
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = val / energyNorm
	}

	return normalized
}

// peakNormalize normalizes by peak absolute value
func (n *Normalizer) peakNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	peak := 0.0
	for _, val := range signal {
		abs := math.Abs(val)
		if abs > peak {
			peak = abs
		}
	}

	if peak < 1e-10 {
		return signal // Return unchanged if no peak
	}

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = val / peak
	}

	return normalized
}

// rmsNormalize normalizes by RMS value
func (n *Normalizer) rmsNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	rms := RMS(signal)

	if rms < 1e-10 {
		return signal // Return unchanged if no RMS
	}

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = val / rms
	}

	return normalized
}

// quantileNormalize normalizes using percentile range
func (n *Normalizer) quantileNormalize(signal []float64, lowQuantile, highQuantile float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	lowVal := Percentile(signal, lowQuantile)
	highVal := Percentile(signal, highQuantile)

	if math.Abs(highVal-lowVal) < 1e-10 {
		// Handle constant range
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - lowVal
		}
		return normalized
	}

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		// Clamp and normalize
		clampedVal := Clamp(val, lowVal, highVal)
		normalized[i] = (clampedVal - lowVal) / (highVal - lowVal)
	}

	return normalized
}

// robustNormalize uses median and MAD (Median Absolute Deviation)
func (n *Normalizer) robustNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	median := Percentile(signal, 0.5)

	// Calculate MAD (Median Absolute Deviation)
	deviations := make([]float64, len(signal))
	for i, val := range signal {
		deviations[i] = math.Abs(val - median)
	}

	mad := Percentile(deviations, 0.5)

	if mad < 1e-10 {
		// Handle constant signal
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - median
		}
		return normalized
	}

	// Scale factor for normal distribution consistency
	scaleFactor := 1.4826 * mad

	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - median) / scaleFactor
	}

	return normalized
}

// AdaptiveNormalize chooses normalization method based on signal characteristics
func (n *Normalizer) AdaptiveNormalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Analyze signal characteristics
	mean := Mean(signal)
	std := StandardDeviation(signal)

	// Calculate peak and RMS
	peak := 0.0
	rms := RMS(signal)
	for _, val := range signal {
		abs := math.Abs(val)
		if abs > peak {
			peak = abs
		}
	}

	crestFactor := 0.0
	if rms > 1e-10 {
		crestFactor = peak / rms
	}

	// Decision logic
	if crestFactor > 10.0 {
		// High crest factor indicates spiky signal - use robust normalization
		return n.robustNormalize(signal)
	} else if std < 1e-6 {
		// Nearly constant signal - use simple centering
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - mean
		}
		return normalized
	} else if math.Abs(mean) > 3*std {
		// Signal has large DC offset - use Z-score
		return n.zScoreNormalize(signal)
	} else {
		// Normal signal - use energy normalization
		return n.energyNormalize(signal)
	}
}

// NormalizeInPlace normalizes signal in-place
func (n *Normalizer) NormalizeInPlace(signal []float64) {
	normalized := n.Normalize(signal)
	copy(signal, normalized)
}

// NormalizeToTarget normalizes signal to specific target range
func (n *Normalizer) NormalizeToTarget(signal []float64, targetMin, targetMax float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// First normalize to [0, 1]
	normalized := n.minMaxNormalize(signal)

	// Scale to target range
	targetRange := targetMax - targetMin
	result := make([]float64, len(normalized))
	for i, val := range normalized {
		result[i] = targetMin + val*targetRange
	}

	return result
}

// NormalizeDB normalizes signal to specific dB level
func (n *Normalizer) NormalizeDB(signal []float64, targetDB float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Calculate current RMS level
	currentRMS := RMS(signal)
	if currentRMS < 1e-10 {
		return signal // Cannot normalize silent signal
	}

	// Convert target dB to linear scale
	targetLinear := math.Pow(10.0, targetDB/20.0)

	// Calculate scaling factor
	scaleFactor := targetLinear / currentRMS

	// Apply scaling
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = val * scaleFactor
	}

	return normalized
}

// NormalizeLUFS normalizes signal to specific LUFS level (simplified)
func (n *Normalizer) NormalizeLUFS(signal []float64, targetLUFS float64, sampleRate int) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Simplified LUFS calculation (real implementation would need proper K-weighting)
	// This is a basic approximation using RMS energy
	windowSize := int(0.4 * float64(sampleRate)) // 400ms windows
	if windowSize > len(signal) {
		windowSize = len(signal)
	}

	// Calculate loudness in overlapping windows
	hopSize := windowSize / 4
	numWindows := (len(signal)-windowSize)/hopSize + 1

	if numWindows <= 0 {
		return n.NormalizeDB(signal, targetLUFS) // Fallback to dB normalization
	}

	loudnessSum := 0.0
	validWindows := 0

	for i := 0; i < numWindows; i++ {
		startIdx := i * hopSize
		endIdx := startIdx + windowSize

		if endIdx > len(signal) {
			endIdx = len(signal)
		}

		// Calculate RMS for this window
		windowRMS := 0.0
		windowSamples := endIdx - startIdx
		for j := startIdx; j < endIdx; j++ {
			windowRMS += signal[j] * signal[j]
		}
		windowRMS = math.Sqrt(windowRMS / float64(windowSamples))

		if windowRMS > 1e-10 {
			// Convert to loudness units (simplified)
			loudness := -0.691 + 10.0*math.Log10(windowRMS*windowRMS)
			loudnessSum += math.Pow(10.0, loudness/10.0)
			validWindows++
		}
	}

	if validWindows == 0 {
		return signal // Cannot normalize silent signal
	}

	// Calculate integrated loudness
	integratedLoudness := -0.691 + 10.0*math.Log10(loudnessSum/float64(validWindows))

	// Calculate gain needed to reach target
	gainDB := targetLUFS - integratedLoudness
	gainLinear := math.Pow(10.0, gainDB/20.0)

	// Apply gain
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = val * gainLinear
	}

	return normalized
}

// FrameNormalize normalizes each frame independently
func (n *Normalizer) FrameNormalize(signal []float64, frameSize, hopSize int) []float64 {
	if len(signal) < frameSize {
		return n.Normalize(signal)
	}

	normalized := make([]float64, len(signal))
	copy(normalized, signal) // Start with original signal

	numFrames := (len(signal)-frameSize)/hopSize + 1

	for i := 0; i < numFrames; i++ {
		startIdx := i * hopSize
		endIdx := startIdx + frameSize

		if endIdx > len(signal) {
			endIdx = len(signal)
		}

		// Extract frame
		frame := signal[startIdx:endIdx]

		// Normalize frame
		normalizedFrame := n.Normalize(frame)

		// Apply normalized frame back (with overlap handling)
		for j, val := range normalizedFrame {
			idx := startIdx + j
			if idx < len(normalized) {
				if i == 0 || startIdx+j < startIdx+hopSize {
					// First frame or non-overlapping region
					normalized[idx] = val
				} else {
					// Overlapping region - blend with existing value
					weight := float64(j) / float64(len(normalizedFrame))
					normalized[idx] = (1.0-weight)*normalized[idx] + weight*val
				}
			}
		}
	}

	return normalized
}

// GetNormalizationStats returns statistics about the normalization applied
func (n *Normalizer) GetNormalizationStats(original, normalized []float64) map[string]float64 {
	stats := make(map[string]float64)

	if len(original) != len(normalized) || len(original) == 0 {
		return stats
	}

	// Original signal stats
	origMean := Mean(original)
	origStd := StandardDeviation(original)
	origRMS := RMS(original)
	origPeak := 0.0
	for _, val := range original {
		abs := math.Abs(val)
		if abs > origPeak {
			origPeak = abs
		}
	}

	// Normalized signal stats
	normMean := Mean(normalized)
	normStd := StandardDeviation(normalized)
	normRMS := RMS(normalized)
	normPeak := 0.0
	for _, val := range normalized {
		abs := math.Abs(val)
		if abs > normPeak {
			normPeak = abs
		}
	}

	// Calculate gains
	stats["original_mean"] = origMean
	stats["original_std"] = origStd
	stats["original_rms"] = origRMS
	stats["original_peak"] = origPeak

	stats["normalized_mean"] = normMean
	stats["normalized_std"] = normStd
	stats["normalized_rms"] = normRMS
	stats["normalized_peak"] = normPeak

	if origRMS > 1e-10 {
		stats["rms_gain_db"] = 20.0 * math.Log10(normRMS/origRMS)
	}
	if origPeak > 1e-10 {
		stats["peak_gain_db"] = 20.0 * math.Log10(normPeak/origPeak)
	}

	return stats
}
