package spectral

import (
	"math"
)

// SpectralSlope computes spectral slope via linear regression
// Extracted from your existing calculateSpectralSlope implementation
type SpectralSlope struct {
	sampleRate  int
	freqBins    []float64 // Pre-calculated frequency bins
	initialized bool
}

// NewSpectralSlope creates a new spectral slope calculator
func NewSpectralSlope(sampleRate int) *SpectralSlope {
	return &SpectralSlope{
		sampleRate: sampleRate,
	}
}

// Compute calculates spectral slope for a single magnitude spectrum
// This is your existing working implementation
func (ss *SpectralSlope) Compute(spectrum []float64) float64 {
	if len(spectrum) < 2 {
		return 0
	}

	// Initialize frequency bins if needed
	if !ss.initialized || len(ss.freqBins) != len(spectrum) {
		ss.initializeFreqBins(len(spectrum))
	}

	// Convert to log domain for linear regression
	n := 0
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i := range len(spectrum) {
		if spectrum[i] > 1e-10 && ss.freqBins[i] > 0 {
			x := math.Log10(ss.freqBins[i])
			y := math.Log10(spectrum[i])

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
			n++
		}
	}

	if n < 2 {
		return 0
	}

	// Linear regression slope
	denominator := float64(n)*sumXX - sumX*sumX
	if denominator == 0 {
		return 0
	}

	slope := (float64(n)*sumXY - sumX*sumY) / denominator
	return slope
}

// ComputeFrames processes multiple frames efficiently
func (ss *SpectralSlope) ComputeFrames(spectrogram [][]float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	slopes := make([]float64, len(spectrogram))

	for t, spectrum := range spectrogram {
		slopes[t] = ss.Compute(spectrum)
	}

	return slopes
}

// initializeFreqBins pre-calculates frequency bins (matches your existing GetFrequencyBins)
func (ss *SpectralSlope) initializeFreqBins(numBins int) {
	ss.freqBins = make([]float64, numBins)
	for i := range numBins {
		ss.freqBins[i] = float64(i) * float64(ss.sampleRate) / float64((numBins-1)*2)
	}
	ss.initialized = true
}
