package windowing

import (
	"fmt"
	"math"
)

// Tukey represents a Tukey window function
// Extracted from your existing generateTukey implementation
type Tukey struct {
	size         int
	alpha        float64
	symmetric    bool
	coefficients []float64
}

// NewTukey creates a new Tukey window
func NewTukey(size int, alpha float64, symmetric bool) *Tukey {
	t := &Tukey{
		size:      size,
		alpha:     alpha,
		symmetric: symmetric,
	}
	t.generate()
	return t
}

// generate creates Tukey window coefficients
// This is your existing working implementation
func (t *Tukey) generate() {
	t.coefficients = make([]float64, t.size)

	// Tukey window is rectangular in the middle with cosine tapers on the sides
	taperLength := int(t.alpha * float64(t.size) / 2.0)

	for i := range t.size {
		if i < taperLength {
			// Rising cosine taper
			arg := math.Pi * float64(i) / float64(taperLength)
			t.coefficients[i] = 0.5 * (1 + math.Cos(arg-math.Pi))
		} else if i >= t.size-taperLength {
			// Falling cosine taper
			arg := math.Pi * float64(i-(t.size-taperLength)) / float64(taperLength)
			t.coefficients[i] = 0.5 * (1 + math.Cos(arg))
		} else {
			// Rectangular middle section
			t.coefficients[i] = 1.0
		}
	}
}

// Apply applies the window to a signal (creates new array)
func (t *Tukey) Apply(signal []float64) []float64 {
	if len(signal) != t.size {
		return nil
	}

	windowed := make([]float64, t.size)
	for i := 0; i < t.size; i++ {
		windowed[i] = signal[i] * t.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (t *Tukey) ApplyInPlace(signal []float64) error {
	if len(signal) != t.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), t.size)
	}

	for i := 0; i < t.size; i++ {
		signal[i] *= t.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (t *Tukey) GetCoefficients() []float64 {
	coeffs := make([]float64, len(t.coefficients))
	copy(coeffs, t.coefficients)
	return coeffs
}

// GetSize returns the window size
func (t *Tukey) GetSize() int {
	return t.size
}

// GetType returns the window type
func (t *Tukey) GetType() string {
	return "tukey"
}

// GetAlpha returns the Tukey alpha parameter
func (t *Tukey) GetAlpha() float64 {
	return t.alpha
}
