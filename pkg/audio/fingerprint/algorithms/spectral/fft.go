package spectral

import (
	"github.com/mjibson/go-dsp/fft"
)

// FFT provides Fast Fourier Transform functionality
// Extracted from your existing SpectralAnalyzer
type FFT struct {
	// No state needed for now
}

// NewFFT creates a new FFT calculator
func NewFFT() *FFT {
	return &FFT{}
}

// Compute computes Fast Fourier Transform using mjibson/go-dsp
// Takes []float64 input and returns []complex128 output
// This is your existing working implementation
func (f *FFT) Compute(x []float64) []complex128 {
	if len(x) == 0 {
		return []complex128{}
	}

	// mjibson/go-dsp handles all sizes efficiently, including non-power-of-2
	return fft.FFTReal(x)
}

// ComputeInverse computes inverse FFT
func (f *FFT) ComputeInverse(x []complex128) []complex128 {
	if len(x) == 0 {
		return []complex128{}
	}

	return fft.IFFT(x)
}

// ComputeInverseReal computes inverse FFT and returns real part only
func (f *FFT) ComputeInverseReal(x []complex128) []float64 {
	if len(x) == 0 {
		return []float64{}
	}

	result := fft.IFFT(x)
	realResult := make([]float64, len(result))

	for i, val := range result {
		realResult[i] = real(val)
	}

	return realResult
}
