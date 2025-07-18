package windowing

import (
	"fmt"
)

// Bartlett represents a Bartlett (triangular) window function
// Extracted from your existing generateBartlett implementation
type Bartlett struct {
	size         int
	symmetric    bool
	coefficients []float64
}

// NewBartlett creates a new Bartlett window
func NewBartlett(size int, symmetric bool) *Bartlett {
	b := &Bartlett{
		size:      size,
		symmetric: symmetric,
	}
	b.generate()
	return b
}

// generate creates Bartlett window coefficients
// This is your existing working implementation
func (b *Bartlett) generate() {
	b.coefficients = make([]float64, b.size)

	for i := range b.size {
		if i <= b.size/2 {
			b.coefficients[i] = 2.0 * float64(i) / float64(b.size-1)
		} else {
			b.coefficients[i] = 2.0 - 2.0*float64(i)/float64(b.size-1)
		}
	}
}

// Apply applies the window to a signal (creates new array)
func (b *Bartlett) Apply(signal []float64) []float64 {
	if len(signal) != b.size {
		return nil
	}

	windowed := make([]float64, b.size)
	for i := 0; i < b.size; i++ {
		windowed[i] = signal[i] * b.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (b *Bartlett) ApplyInPlace(signal []float64) error {
	if len(signal) != b.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), b.size)
	}

	for i := 0; i < b.size; i++ {
		signal[i] *= b.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (b *Bartlett) GetCoefficients() []float64 {
	coeffs := make([]float64, len(b.coefficients))
	copy(coeffs, b.coefficients)
	return coeffs
}

// GetSize returns the window size
func (b *Bartlett) GetSize() int {
	return b.size
}

// GetType returns the window type
func (b *Bartlett) GetType() string {
	return "bartlett"
}
