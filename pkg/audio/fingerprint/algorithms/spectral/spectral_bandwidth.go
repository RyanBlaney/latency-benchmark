package spectral

import (
	"math"
)

// SpectralBandwidth computes spectral bandwidth around centroid
// Extracted from your existing working implementation
type SpectralBandwidth struct {
	sampleRate  int
	freqBins    []float64 // Pre-calculated frequency bins
	initialized bool
}

// NewSpectralBandwidth creates a new spectral bandwidth calculator
func NewSpectralBandwidth(sampleRate int) *SpectralBandwidth {
	return &SpectralBandwidth{
		sampleRate: sampleRate,
	}
}

// Compute calculates spectral bandwidth for a single spectrum given its centroid
// This is your existing working implementation
func (sb *SpectralBandwidth) Compute(spectrum []float64, centroid float64) float64 {
	if len(spectrum) == 0 {
		return 0.0
	}

	// Initialize frequency bins if needed
	if !sb.initialized || len(sb.freqBins) != len(spectrum) {
		sb.initializeFreqBins(len(spectrum))
	}

	// Your existing implementation from calculateSpectralBandwidth
	numerator := 0.0
	denominator := 0.0

	for i := range len(spectrum) {
		diff := sb.freqBins[i] - centroid
		numerator += diff * diff * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	return math.Sqrt(numerator / denominator)
}

// ComputeFrames processes multiple frames with their corresponding centroids
func (sb *SpectralBandwidth) ComputeFrames(spectrogram [][]float64, centroids []float64) []float64 {
	if len(spectrogram) == 0 || len(centroids) != len(spectrogram) {
		return []float64{}
	}

	bandwidths := make([]float64, len(spectrogram))

	for t, spectrum := range spectrogram {
		bandwidths[t] = sb.Compute(spectrum, centroids[t])
	}

	return bandwidths
}

// initializeFreqBins pre-calculates frequency bins (matches your existing GetFrequencyBins)
func (sb *SpectralBandwidth) initializeFreqBins(numBins int) {
	sb.freqBins = make([]float64, numBins)
	for i := range numBins {
		sb.freqBins[i] = float64(i) * float64(sb.sampleRate) / float64((numBins-1)*2)
	}
	sb.initialized = true
}
