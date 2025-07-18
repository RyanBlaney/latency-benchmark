package windowing

import (
	"fmt"
)

// Rectangular represents a rectangular (boxcar) window function
// Extracted from your existing generateRectangular implementation
type Rectangular struct {
	size         int
	coefficients []float64
}

// NewRectangular creates a new rectangular window
func NewRectangular(size int) *Rectangular {
	r := &Rectangular{
		size: size,
	}
	r.generate()
	return r
}

// generate creates rectangular window coefficients
// This is your existing working implementation
func (r *Rectangular) generate() {
	r.coefficients = make([]float64, r.size)
	for i := range r.coefficients {
		r.coefficients[i] = 1.0
	}
}

// Apply applies the window to a signal (creates new array)
func (r *Rectangular) Apply(signal []float64) []float64 {
	if len(signal) != r.size {
		return nil
	}

	// For rectangular window, just return a copy
	windowed := make([]float64, r.size)
	copy(windowed, signal)
	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (r *Rectangular) ApplyInPlace(signal []float64) error {
	if len(signal) != r.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), r.size)
	}

	// For rectangular window, signal remains unchanged
	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (r *Rectangular) GetCoefficients() []float64 {
	coeffs := make([]float64, len(r.coefficients))
	copy(coeffs, r.coefficients)
	return coeffs
}

// GetSize returns the window size
func (r *Rectangular) GetSize() int {
	return r.size
}

// GetType returns the window type
func (r *Rectangular) GetType() string {
	return "rectangular"
}
