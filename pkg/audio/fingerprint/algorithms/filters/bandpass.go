package filters

import (
	"fmt"
	"math"
)

// BandpassFilter implements a digital bandpass filter using biquad topology.
//
// This implementation uses the cookbook formulas from Robert Bristow-Johnson's
// "Cookbook formulae for audio EQ biquad filter coefficients"
// Reference: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
type BandpassFilter struct {
	sampleRate int
	centerFreq float64 // Center frequency in Hz
	bandwidth  float64 // Bandwidth in Hz
	qFactor    float64 // Quality factor (centerFreq/bandwidth)

	// Biquad coefficients
	b0, b1, b2 float64 // Numerator coefficients
	a0, a1, a2 float64 // Denominator coefficients

	// State variables for direct form II implementation
	x1, x2 float64 // Input delay line
	y1, y2 float64 // Output delay line

	initialized bool
}

// FilterType represents different bandpass filter design methods
type FilterType int

const (
	// Butterworth design - maximally flat passband
	Butterworth FilterType = iota
	// Chebyshev Type I - ripple in passband, sharp transition
	ChebyshevI
	// Bessel - maximally flat group delay (linear phase)
	Bessel
)

// NewBandpassFilter creates a new bandpass filter with specified parameters.
//
// Parameters:
//   - sampleRate: Sample rate in Hz
//   - centerFreq: Center frequency in Hz
//   - bandwidth: Bandwidth in Hz
//
// The Q factor is automatically calculated as centerFreq/bandwidth.
// Higher Q values create narrower, more selective filters.
func NewBandpassFilter(sampleRate int, centerFreq, bandwidth float64) *BandpassFilter {
	bf := &BandpassFilter{
		sampleRate: sampleRate,
		centerFreq: centerFreq,
		bandwidth:  bandwidth,
		qFactor:    centerFreq / bandwidth,
	}

	bf.computeCoefficients()
	return bf
}

// NewBandpassFilterWithQ creates a bandpass filter with explicit Q factor.
//
// Parameters:
//   - sampleRate: Sample rate in Hz
//   - centerFreq: Center frequency in Hz
//   - qFactor: Quality factor (higher = narrower filter)
func NewBandpassFilterWithQ(sampleRate int, centerFreq, qFactor float64) *BandpassFilter {
	bf := &BandpassFilter{
		sampleRate: sampleRate,
		centerFreq: centerFreq,
		qFactor:    qFactor,
		bandwidth:  centerFreq / qFactor,
	}

	bf.computeCoefficients()
	return bf
}

// computeCoefficients calculates the biquad coefficients using the cookbook formula.
func (bf *BandpassFilter) computeCoefficients() {
	// Normalize frequency: w0 = 2*pi*f0/Fs
	w0 := 2.0 * math.Pi * bf.centerFreq / float64(bf.sampleRate)

	// Prevent numerical issues at Nyquist
	if w0 >= math.Pi {
		w0 = math.Pi * 0.99
	}

	cosW0 := math.Cos(w0)
	sinW0 := math.Sin(w0)

	// Alpha parameter: alpha = sin(w0)/(2*Q)
	alpha := sinW0 / (2.0 * bf.qFactor)

	// Bandpass coefficients (cookbook formula)
	bf.b0 = alpha        // sin(w0)/2
	bf.b1 = 0.0          // 0
	bf.b2 = -alpha       // -sin(w0)/2
	bf.a0 = 1.0 + alpha  // 1 + alpha
	bf.a1 = -2.0 * cosW0 // -2*cos(w0)
	bf.a2 = 1.0 - alpha  // 1 - alpha

	// Normalize by a0 for direct form II implementation
	bf.b0 /= bf.a0
	bf.b1 /= bf.a0
	bf.b2 /= bf.a0
	bf.a1 /= bf.a0
	bf.a2 /= bf.a0
	bf.a0 = 1.0

	bf.initialized = true
}

// Process applies the bandpass filter to a single sample.
// Uses Direct Form II biquad implementation for numerical stability.
//
// The difference equation is:
// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
func (bf *BandpassFilter) Process(input float64) float64 {
	if !bf.initialized {
		bf.computeCoefficients()
	}

	// Direct Form II implementation
	// w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
	w := input - bf.a1*bf.x1 - bf.a2*bf.x2

	// y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
	output := bf.b0*w + bf.b1*bf.x1 + bf.b2*bf.x2

	// Update delay line
	bf.x2 = bf.x1
	bf.x1 = w

	return output
}

// ProcessBuffer applies the bandpass filter to an entire buffer of samples.
func (bf *BandpassFilter) ProcessBuffer(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, sample := range input {
		output[i] = bf.Process(sample)
	}
	return output
}

// Reset clears the filter's internal state (delay line).
// Call this when processing discontinuous audio segments.
func (bf *BandpassFilter) Reset() {
	bf.x1, bf.x2 = 0.0, 0.0
	bf.y1, bf.y2 = 0.0, 0.0
}

// SetParameters updates the filter parameters and recomputes coefficients.
func (bf *BandpassFilter) SetParameters(centerFreq, bandwidth float64) error {
	if centerFreq <= 0 || centerFreq >= float64(bf.sampleRate)/2 {
		return fmt.Errorf("center frequency must be between 0 and Nyquist frequency (%d Hz)", bf.sampleRate/2)
	}

	if bandwidth <= 0 {
		return fmt.Errorf("bandwidth must be positive")
	}

	bf.centerFreq = centerFreq
	bf.bandwidth = bandwidth
	bf.qFactor = centerFreq / bandwidth
	bf.computeCoefficients()

	return nil
}

// GetFrequencyResponse computes the magnitude and phase response at given frequency.
// Returns magnitude (linear scale) and phase (radians).
//
// The frequency response is computed as:
// H(e^jw) = (b0 + b1*e^-jw + b2*e^-j2w) / (a0 + a1*e^-jw + a2*e^-j2w)
func (bf *BandpassFilter) GetFrequencyResponse(frequency float64) (magnitude, phase float64) {
	w := 2.0 * math.Pi * frequency / float64(bf.sampleRate)

	// Compute complex exponentials
	cosW := math.Cos(w)
	sinW := math.Sin(w)
	cos2W := math.Cos(2 * w)
	sin2W := math.Sin(2 * w)

	// Numerator: b0 + b1*e^-jw + b2*e^-j2w
	numReal := bf.b0 + bf.b1*cosW + bf.b2*cos2W
	numImag := -bf.b1*sinW - bf.b2*sin2W

	// Denominator: a0 + a1*e^-jw + a2*e^-j2w
	denReal := bf.a0 + bf.a1*cosW + bf.a2*cos2W
	denImag := -bf.a1*sinW - bf.a2*sin2W

	// H(e^jw) = numerator / denominator
	denMagSq := denReal*denReal + denImag*denImag

	hReal := (numReal*denReal + numImag*denImag) / denMagSq
	hImag := (numImag*denReal - numReal*denImag) / denMagSq

	magnitude = math.Sqrt(hReal*hReal + hImag*hImag)
	phase = math.Atan2(hImag, hReal)

	return magnitude, phase
}

// GetParameters returns the current filter parameters.
func (bf *BandpassFilter) GetParameters() (centerFreq, bandwidth, qFactor float64) {
	return bf.centerFreq, bf.bandwidth, bf.qFactor
}

// GetCoefficients returns the current biquad coefficients.
// Useful for debugging or implementing the filter elsewhere.
func (bf *BandpassFilter) GetCoefficients() (b0, b1, b2, a0, a1, a2 float64) {
	return bf.b0, bf.b1, bf.b2, bf.a0, bf.a1, bf.a2
}
