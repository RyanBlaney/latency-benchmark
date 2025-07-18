package windowing

import (
	"fmt"
	"math"
)

// Blackman represents a Blackman window function
// Extracted from your existing generateBlackman implementation
type Blackman struct {
	size         int
	symmetric    bool
	coefficients []float64
}

// NewBlackman creates a new Blackman window
func NewBlackman(size int, symmetric bool) *Blackman {
	b := &Blackman{
		size:      size,
		symmetric: symmetric,
	}
	b.generate()
	return b
}

// generate creates Blackman window coefficients
// This is your existing working implementation
func (b *Blackman) generate() {
	b.coefficients = make([]float64, b.size)

	denominator := float64(b.size)
	if b.symmetric {
		denominator = float64(b.size - 1)
	}

	a0, a1, a2 := 0.42, 0.5, 0.08

	for i := range b.size {
		arg := 2 * math.Pi * float64(i) / denominator
		b.coefficients[i] = a0 - a1*math.Cos(arg) + a2*math.Cos(2*arg)
	}
}

// Apply applies the window to a signal (creates new array)
func (b *Blackman) Apply(signal []float64) []float64 {
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
func (b *Blackman) ApplyInPlace(signal []float64) error {
	if len(signal) != b.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), b.size)
	}

	for i := 0; i < b.size; i++ {
		signal[i] *= b.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (b *Blackman) GetCoefficients() []float64 {
	coeffs := make([]float64, len(b.coefficients))
	copy(coeffs, b.coefficients)
	return coeffs
}

// GetSize returns the window size
func (b *Blackman) GetSize() int {
	return b.size
}

// GetType returns the window type
func (b *Blackman) GetType() string {
	return "blackman"
}
