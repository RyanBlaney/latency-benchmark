package temporal

import (
	"math"
	"sort"
)

// DynamicRange analyzes amplitude dynamics and statistics
type DynamicRange struct {
	envelopeExtractor *Envelope
}

// NewDynamicRange creates a new dynamic range analyzer
func NewDynamicRange() *DynamicRange {
	return &DynamicRange{
		envelopeExtractor: NewEnvelope(),
	}
}

// ComputeRange calculates dynamic range in dB between percentiles
func (dr *DynamicRange) ComputeRange(signal []float64, lowPercentile, highPercentile float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate RMS values for analysis
	frameSize := 1024
	hopSize := 512
	rmsValues := dr.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(rmsValues) == 0 {
		return 0.0
	}

	return dr.calculatePercentileRange(rmsValues, lowPercentile, highPercentile)
}

// ComputePeakRange calculates peak-based dynamic range
func (dr *DynamicRange) ComputePeakRange(signal []float64, lowPercentile, highPercentile float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate peak values for analysis
	frameSize := 1024
	hopSize := 512
	peakValues := dr.envelopeExtractor.ComputePeak(signal, frameSize, hopSize)

	if len(peakValues) == 0 {
		return 0.0
	}

	return dr.calculatePercentileRange(peakValues, lowPercentile, highPercentile)
}

// calculatePercentileRange calculates range between percentiles in dB
func (dr *DynamicRange) calculatePercentileRange(values []float64, lowPercentile, highPercentile float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Sort values for percentile calculation
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	// Calculate percentile indices
	lowIdx := int(lowPercentile * float64(len(sorted)-1))
	highIdx := int(highPercentile * float64(len(sorted)-1))

	lowValue := sorted[lowIdx]
	highValue := sorted[highIdx]

	// Avoid log(0)
	if lowValue <= 0.0 {
		lowValue = 1e-10
	}
	if highValue <= 0.0 {
		return 0.0
	}

	// Return range in dB
	return 20.0 * math.Log10(highValue/lowValue)
}

// ComputeCrestFactor calculates crest factor (peak-to-RMS ratio)
func (dr *DynamicRange) ComputeCrestFactor(signal []float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Find peak amplitude
	peak := 0.0
	sumSquares := 0.0

	for _, sample := range signal {
		abs := math.Abs(sample)
		if abs > peak {
			peak = abs
		}
		sumSquares += sample * sample
	}

	// Calculate RMS
	rms := math.Sqrt(sumSquares / float64(len(signal)))

	if rms == 0.0 {
		return 0.0
	}

	return peak / rms
}

// ComputeFrameCrestFactors calculates crest factor for each frame
func (dr *DynamicRange) ComputeFrameCrestFactors(signal []float64, frameSize, hopSize int) []float64 {
	if len(signal) < frameSize || frameSize <= 0 || hopSize <= 0 {
		return []float64{}
	}

	numFrames := (len(signal)-frameSize)/hopSize + 1
	crestFactors := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * hopSize
		endIdx := startIdx + frameSize

		if endIdx > len(signal) {
			break
		}

		frame := signal[startIdx:endIdx]
		crestFactors[i] = dr.ComputeCrestFactor(frame)
	}

	return crestFactors
}

// ComputeLoudnessRange calculates loudness range (EBU R128 style)
func (dr *DynamicRange) ComputeLoudnessRange(signal []float64, sampleRate int) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate momentary loudness values (400ms windows)
	windowSize := int(0.4 * float64(sampleRate)) // 400ms
	hopSize := windowSize / 4                    // 25% overlap

	loudnessValues := dr.envelopeExtractor.ComputeRMS(signal, windowSize, hopSize)

	if len(loudnessValues) == 0 {
		return 0.0
	}

	// Convert to loudness units (simplified)
	for i := range loudnessValues {
		if loudnessValues[i] > 0 {
			loudnessValues[i] = -0.691 + 10.0*math.Log10(loudnessValues[i]*loudnessValues[i])
		} else {
			loudnessValues[i] = -70.0 // Silence threshold
		}
	}

	// Calculate loudness range (10th to 95th percentile)
	return dr.calculatePercentileRange(loudnessValues, 0.10, 0.95)
}

// ComputeStatistics calculates comprehensive dynamic range statistics
func (dr *DynamicRange) ComputeStatistics(signal []float64) map[string]float64 {
	stats := make(map[string]float64)

	if len(signal) == 0 {
		return stats
	}

	// Basic amplitude statistics
	peak := 0.0
	sumSquares := 0.0
	sumAbs := 0.0

	for _, sample := range signal {
		abs := math.Abs(sample)
		if abs > peak {
			peak = abs
		}
		sumSquares += sample * sample
		sumAbs += abs
	}

	rms := math.Sqrt(sumSquares / float64(len(signal)))
	mean := sumAbs / float64(len(signal))

	stats["peak_amplitude"] = peak
	stats["rms_amplitude"] = rms
	stats["mean_amplitude"] = mean
	stats["crest_factor"] = dr.ComputeCrestFactor(signal)

	// Dynamic range measurements
	stats["dynamic_range_10_90"] = dr.ComputeRange(signal, 0.10, 0.90)
	stats["dynamic_range_5_95"] = dr.ComputeRange(signal, 0.05, 0.95)
	stats["peak_range_10_90"] = dr.ComputePeakRange(signal, 0.10, 0.90)

	return stats
}
