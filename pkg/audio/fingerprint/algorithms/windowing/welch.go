package windowing

import (
	"fmt"
)

// Welch represents a Welch window function
// Extracted from your existing generateWelch implementation
type Welch struct {
	size         int
	coefficients []float64
}

// NewWelch creates a new Welch window
func NewWelch(size int) *Welch {
	w := &Welch{
		size: size,
	}
	w.generate()
	return w
}

// generate creates Welch window coefficients
// This is your existing working implementation
func (w *Welch) generate() {
	w.coefficients = make([]float64, w.size)

	for i := range w.size {
		arg := (float64(i) - float64(w.size-1)/2.0) / (float64(w.size-1) / 2.0)
		w.coefficients[i] = 1.0 - arg*arg
	}
}

// Apply applies the window to a signal (creates new array)
func (w *Welch) Apply(signal []float64) []float64 {
	if len(signal) != w.size {
		return nil
	}

	windowed := make([]float64, w.size)
	for i := 0; i < w.size; i++ {
		windowed[i] = signal[i] * w.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (w *Welch) ApplyInPlace(signal []float64) error {
	if len(signal) != w.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), w.size)
	}

	for i := 0; i < w.size; i++ {
		signal[i] *= w.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (w *Welch) GetCoefficients() []float64 {
	coeffs := make([]float64, len(w.coefficients))
	copy(coeffs, w.coefficients)
	return coeffs
}

// GetSize returns the window size
func (w *Welch) GetSize() int {
	return w.size
}

// GetType returns the window type
func (w *Welch) GetType() string {
	return "welch"
}
