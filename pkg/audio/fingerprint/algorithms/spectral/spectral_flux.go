package spectral

import (
	"math"
)

// SpectralFlux computes spectral flux (measure of spectral change)
// Extracted from your existing ComputeSpectralFlux implementation
type SpectralFlux struct {
	// No state needed - TODO: why
}

// NewSpectralFlux creates a new spectral flux calculator
func NewSpectralFlux() *SpectralFlux {
	return &SpectralFlux{}
}

// Compute calculates spectral flux for a spectrogram
// This is your existing working implementation
func (sf *SpectralFlux) Compute(spectrogram [][]float64) []float64 {
	if len(spectrogram) < 2 {
		return []float64{}
	}

	flux := make([]float64, len(spectrogram)-1)

	for t := 1; t < len(spectrogram); t++ {
		sum := 0.0
		for f := 0; f < len(spectrogram[t]); f++ {
			diff := spectrogram[t][f] - spectrogram[t-1][f]
			if diff > 0 { // Only positive changes (energy increases)
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

// ComputeAllChanges calculates spectral flux including both positive and negative changes
func (sf *SpectralFlux) ComputeAllChanges(spectrogram [][]float64) []float64 {
	if len(spectrogram) < 2 {
		return []float64{}
	}

	flux := make([]float64, len(spectrogram)-1)

	for t := 1; t < len(spectrogram); t++ {
		sum := 0.0
		for f := 0; f < len(spectrogram[t]); f++ {
			diff := spectrogram[t][f] - spectrogram[t-1][f]
			sum += diff * diff // All changes, positive and negative
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}
