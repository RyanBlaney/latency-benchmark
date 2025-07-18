package spectral

import (
	"math"
)

// SpectralFlatness computes spectral flatness (Wiener entropy)
// Critical for distinguishing speech from music - speech has lower flatness
type SpectralFlatness struct {
	minThreshold float64 // Minimum value to avoid log(0)
}

// NewSpectralFlatness creates a new spectral flatness calculator
func NewSpectralFlatness() *SpectralFlatness {
	return &SpectralFlatness{
		minThreshold: 1e-10, // Avoid log(0) issues
	}
}

// NewSpectralFlatnessWithThreshold creates calculator with custom threshold
func NewSpectralFlatnessWithThreshold(threshold float64) *SpectralFlatness {
	return &SpectralFlatness{
		minThreshold: threshold,
	}
}

// Compute calculates spectral flatness for a single magnitude spectrum
// Returns ratio of geometric mean to arithmetic mean (0-1 range)
// Lower values (0.0-0.3) indicate tonal content (speech, music)
// Higher values (0.7-1.0) indicate noise-like content
func (sf *SpectralFlatness) Compute(magnitudeSpectrum []float64) float64 {
	if len(magnitudeSpectrum) == 0 {
		return 0.0
	}

	// Calculate geometric mean (using log domain for numerical stability)
	logSum := 0.0
	validCount := 0

	for _, magnitude := range magnitudeSpectrum {
		if magnitude > sf.minThreshold {
			logSum += math.Log(magnitude)
			validCount++
		}
	}

	if validCount == 0 {
		return 0.0
	}

	geometricMean := math.Exp(logSum / float64(validCount))

	// Calculate arithmetic mean
	arithmeticMean := 0.0
	for _, magnitude := range magnitudeSpectrum {
		arithmeticMean += magnitude
	}
	arithmeticMean /= float64(len(magnitudeSpectrum))

	if arithmeticMean <= sf.minThreshold {
		return 0.0
	}

	// Spectral flatness = geometric mean / arithmetic mean
	flatness := geometricMean / arithmeticMean

	// Ensure result is in valid range [0, 1]
	if flatness > 1.0 {
		flatness = 1.0
	}

	return flatness
}

// ComputeFrames processes multiple frames efficiently
func (sf *SpectralFlatness) ComputeFrames(spectrogram [][]float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	flatness := make([]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		flatness[t] = sf.Compute(magnitudeSpectrum)
	}

	return flatness
}

// ComputeInDB calculates spectral flatness in decibels
// Useful for setting thresholds:
// - Speech typically: -15 to -5 dB
// - Music typically: -20 to -8 dB
// - Noise typically: -3 to 0 dB
func (sf *SpectralFlatness) ComputeInDB(magnitudeSpectrum []float64) float64 {
	flatness := sf.Compute(magnitudeSpectrum)

	if flatness <= sf.minThreshold {
		return -100.0 // Very low flatness
	}

	return 10.0 * math.Log10(flatness)
}

// ComputeFramesInDB processes multiple frames and returns dB values
func (sf *SpectralFlatness) ComputeFramesInDB(spectrogram [][]float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	flatnessDB := make([]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		flatnessDB[t] = sf.ComputeInDB(magnitudeSpectrum)
	}

	return flatnessDB
}

// ComputeBandLimited calculates spectral flatness for a frequency band
// Useful for analyzing specific frequency ranges (e.g., speech formants)
func (sf *SpectralFlatness) ComputeBandLimited(magnitudeSpectrum []float64, startBin, endBin int) float64 {
	if startBin < 0 || endBin >= len(magnitudeSpectrum) || startBin >= endBin {
		return 0.0
	}

	// Extract the frequency band
	band := magnitudeSpectrum[startBin : endBin+1]
	return sf.Compute(band)
}

// ComputeSpeechBands calculates flatness for speech-relevant frequency bands
// Returns flatness for: [low: 0-1kHz, mid: 1-4kHz, high: 4-8kHz]
func (sf *SpectralFlatness) ComputeSpeechBands(magnitudeSpectrum []float64, sampleRate int) (float64, float64, float64) {
	if len(magnitudeSpectrum) == 0 {
		return 0.0, 0.0, 0.0
	}

	nyquist := float64(sampleRate) / 2.0
	freqPerBin := nyquist / float64(len(magnitudeSpectrum)-1)

	// Calculate bin indices for speech bands
	lowBand := int(1000.0 / freqPerBin)  // 0-1kHz
	midBand := int(4000.0 / freqPerBin)  // 1-4kHz
	highBand := int(8000.0 / freqPerBin) // 4-8kHz

	// Clamp to valid ranges
	lowBand = min(lowBand, len(magnitudeSpectrum)-1)
	midBand = min(midBand, len(magnitudeSpectrum)-1)
	highBand = min(highBand, len(magnitudeSpectrum)-1)

	lowFlatness := sf.ComputeBandLimited(magnitudeSpectrum, 0, lowBand)
	midFlatness := sf.ComputeBandLimited(magnitudeSpectrum, lowBand, midBand)
	highFlatness := sf.ComputeBandLimited(magnitudeSpectrum, midBand, highBand)

	return lowFlatness, midFlatness, highFlatness
}

// IsContentTonal determines if content is tonal based on flatness threshold
// Returns true for speech/music, false for noise
func (sf *SpectralFlatness) IsContentTonal(flatness float64, threshold float64) bool {
	return flatness < threshold
}

// EstimateContentType provides basic content classification based on flatness
func (sf *SpectralFlatness) EstimateContentType(flatnessValues []float64) string {
	if len(flatnessValues) == 0 {
		return "unknown"
	}

	// Calculate mean flatness
	meanFlatness := 0.0
	for _, flatness := range flatnessValues {
		meanFlatness += flatness
	}
	meanFlatness /= float64(len(flatnessValues))

	// Calculate variance for stability measure
	variance := 0.0
	for _, flatness := range flatnessValues {
		diff := flatness - meanFlatness
		variance += diff * diff
	}
	variance /= float64(len(flatnessValues))

	// Content classification heuristics
	if meanFlatness < 0.2 && variance < 0.01 {
		return "music" // Low flatness, stable
	} else if meanFlatness < 0.35 && variance > 0.01 {
		return "speech" // Low-medium flatness, variable
	} else if meanFlatness > 0.6 {
		return "noise" // High flatness
	} else {
		return "mixed" // Medium flatness
	}
}
