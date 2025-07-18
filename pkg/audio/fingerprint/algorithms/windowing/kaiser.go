package windowing

import (
	"fmt"
	"math"
)

// Kaiser represents a Kaiser window function
// Extracted from your existing generateKaiser implementation
type Kaiser struct {
	size         int
	beta         float64
	symmetric    bool
	coefficients []float64
}

// NewKaiser creates a new Kaiser window
func NewKaiser(size int, beta float64, symmetric bool) *Kaiser {
	k := &Kaiser{
		size:      size,
		beta:      beta,
		symmetric: symmetric,
	}
	k.generate()
	return k
}

// generate creates Kaiser window coefficients
// This is your existing working implementation
func (k *Kaiser) generate() {
	k.coefficients = make([]float64, k.size)

	denominator := float64(k.size)
	if k.symmetric {
		denominator = float64(k.size - 1)
	}

	// Calculate I0(beta) for normalization
	i0Beta := k.besselI0(k.beta)

	for i := range k.size {
		arg := 2.0*float64(i)/denominator - 1.0
		k.coefficients[i] = k.besselI0(k.beta*math.Sqrt(1-arg*arg)) / i0Beta
	}
}

// besselI0 computes the zero-order modified Bessel function of the first kind
// This is your existing working implementation
func (k *Kaiser) besselI0(x float64) float64 {
	// Series expansion approximation
	sum := 1.0
	term := 1.0

	for i := 1; i < 50; i++ {
		term *= (x / (2.0 * float64(i))) * (x / (2.0 * float64(i)))
		sum += term

		// Check for convergence
		if term < 1e-12 {
			break
		}
	}

	return sum
}

// Apply applies the window to a signal (creates new array)
func (k *Kaiser) Apply(signal []float64) []float64 {
	if len(signal) != k.size {
		return nil
	}

	windowed := make([]float64, k.size)
	for i := 0; i < k.size; i++ {
		windowed[i] = signal[i] * k.coefficients[i]
	}

	return windowed
}

// ApplyInPlace applies the window to a signal in-place
func (k *Kaiser) ApplyInPlace(signal []float64) error {
	if len(signal) != k.size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), k.size)
	}

	for i := 0; i < k.size; i++ {
		signal[i] *= k.coefficients[i]
	}

	return nil
}

// GetCoefficients returns a copy of the window coefficients
func (k *Kaiser) GetCoefficients() []float64 {
	coeffs := make([]float64, len(k.coefficients))
	copy(coeffs, k.coefficients)
	return coeffs
}

// GetSize returns the window size
func (k *Kaiser) GetSize() int {
	return k.size
}

// GetType returns the window type
func (k *Kaiser) GetType() string {
	return "kaiser"
}

// GetBeta returns the Kaiser beta parameter
func (k *Kaiser) GetBeta() float64 {
	return k.beta
}
