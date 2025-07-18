package windowing

import (
	"fmt"
	"math"
)

// Hamming represents a Hamming window function
// Extracted from your existing generateHamming implementation
type Hamming struct {
	size         int
	symmetric    bool
	coefficients []float64
}

// NewHamming creates a new Hamming window
func NewHamming(size int, symmetric bool) *Hamming {
	h := &Hamming{
		size:      size,
		symmetric: symmetric,
	}
	h.generate()
	return h
}

// generate creates Hamming window coefficients
// This is your existing working implementation
func (h *Hamming) generate() {
	h.coefficients = make([]float64, h.size)

	denominator := float64(h.size)
	if h.symmetric {
		denominator = float64(h.size - 1)
	}

	for i := range h.size {
		h.coefficients[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/denominator)
	}
}

// Apply applies the window to a signal (creates new array)
func (h *Hamming) Apply(signal []float64) []float64 {
	if len(signal) != h.size {
		return nil
	}

	windowed := make([]float64, h.size)
	for i := 0; i < h.size; i++ {
		windowed[i] = signal[i] * h.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (h *Hamming) ApplyInPlace(signal []float64) error {
	if len(signal) != h.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), h.size)
	}

	for i := 0; i < h.size; i++ {
		signal[i] *= h.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (h *Hamming) GetCoefficients() []float64 {
	coeffs := make([]float64, len(h.coefficients))
	copy(coeffs, h.coefficients)
	return coeffs
}

// GetSize returns the window size
func (h *Hamming) GetSize() int {
	return h.size
}

// GetType returns the window type
func (h *Hamming) GetType() string {
	return "hamming"
}
