package windowing

import (
	"fmt"
	"math"
)

// BlackmanHarris represents a Blackman-Harris window function
// Extracted from your existing generateBlackmanHarris implementation
type BlackmanHarris struct {
	size         int
	symmetric    bool
	coefficients []float64
}

// NewBlackmanHarris creates a new Blackman-Harris window
func NewBlackmanHarris(size int, symmetric bool) *BlackmanHarris {
	bh := &BlackmanHarris{
		size:      size,
		symmetric: symmetric,
	}
	bh.generate()
	return bh
}

// generate creates Blackman-Harris window coefficients
// This is your existing working implementation
func (bh *BlackmanHarris) generate() {
	bh.coefficients = make([]float64, bh.size)

	denominator := float64(bh.size)
	if bh.symmetric {
		denominator = float64(bh.size - 1)
	}

	a0, a1, a2, a3 := 0.35875, 0.48829, 0.14128, 0.01168

	for i := range bh.size {
		arg := 2 * math.Pi * float64(i) / denominator
		bh.coefficients[i] = a0 - a1*math.Cos(arg) + a2*math.Cos(2*arg) - a3*math.Cos(3*arg)
	}
}

// Apply applies the window to a signal (creates new array)
func (bh *BlackmanHarris) Apply(signal []float64) []float64 {
	if len(signal) != bh.size {
		return nil
	}

	windowed := make([]float64, bh.size)
	for i := 0; i < bh.size; i++ {
		windowed[i] = signal[i] * bh.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (bh *BlackmanHarris) ApplyInPlace(signal []float64) error {
	if len(signal) != bh.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), bh.size)
	}

	for i := 0; i < bh.size; i++ {
		signal[i] *= bh.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (bh *BlackmanHarris) GetCoefficients() []float64 {
	coeffs := make([]float64, len(bh.coefficients))
	copy(coeffs, bh.coefficients)
	return coeffs
}

// GetSize returns the window size
func (bh *BlackmanHarris) GetSize() int {
	return bh.size
}

// GetType returns the window type
func (bh *BlackmanHarris) GetType() string {
	return "blackman_harris"
}
