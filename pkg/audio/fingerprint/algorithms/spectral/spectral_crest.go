package spectral

import (
	"math"
)

// SpectralCrest computes spectral crest factor (peak-to-RMS ratio)
// Extracted from your existing calculateSpectralCrest implementation
type SpectralCrest struct {
	// No state needed - TODO: highlight the why
}

// NewSpectralCrest creates a new spectral crest calculator
func NewSpectralCrest() *SpectralCrest {
	return &SpectralCrest{}
}

// Compute calculates spectral crest factor for a single magnitude spectrum
// This is your existing working implementation
func (sc *SpectralCrest) Compute(spectrum []float64) float64 {
	if len(spectrum) == 0 {
		return 0
	}

	maxVal := 0.0
	sumSquares := 0.0

	for _, mag := range spectrum {
		if mag > maxVal {
			maxVal = mag
		}
		sumSquares += mag * mag
	}

	rms := math.Sqrt(sumSquares / float64(len(spectrum)))

	if rms == 0 {
		return 0
	}

	return maxVal / rms
}

// ComputeFrames processes multiple frames efficiently
func (sc *SpectralCrest) ComputeFrames(spectrogram [][]float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	crests := make([]float64, len(spectrogram))

	for t, spectrum := range spectrogram {
		crests[t] = sc.Compute(spectrum)
	}

	return crests
}
