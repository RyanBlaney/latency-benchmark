package filters

import (
	"math"
)

// DCRemoval implements a DC blocking filter (high-pass filter) to remove
// the DC component (0 Hz) from audio signals.
//
// References:
//   - Julius O. Smith III, "Introduction to Digital Filters with Audio Applications"
//     https://ccrma.stanford.edu/~jos/filters/DC_Blocker.html
//   - Udo Zölzer, "Digital Audio Signal Processing", 2nd Edition, Chapter 5
//
// This filter has several advantages:
// 1. Very low computational cost (3 operations per sample)
// 2. No ripple in passband
// 3. Configurable cutoff frequency
// 4. Excellent DC blocking performance
type DCRemoval struct {
	poleLocation float64 // R parameter (0 < R < 1)
	cutoffFreq   float64 // -3dB cutoff frequency in Hz
	sampleRate   int     // Sample rate in Hz

	// State variables
	x1 float64 // Previous input sample x[n-1]
	y1 float64 // Previous output sample y[n-1]

	initialized bool
}

// NewDCRemoval creates a new DC removal filter with default settings.
// Uses a pole location of 0.995, which gives a cutoff frequency of
// approximately 8 Hz at 44.1 kHz sample rate.
func NewDCRemoval() *DCRemoval {
	return &DCRemoval{
		poleLocation: 0.995, // Standard value for audio applications
		initialized:  true,
	}
}

// NewDCRemovalWithCutoff creates a DC removal filter with specified cutoff frequency.
//
// Parameters:
//   - sampleRate: Sample rate in Hz
//   - cutoffFreq: Desired -3dB cutoff frequency in Hz
//
// The pole location R is calculated as:
// R = 1 - 2*pi*fc/fs
// Where fc is the cutoff frequency and fs is the sample rate.
func NewDCRemovalWithCutoff(sampleRate int, cutoffFreq float64) *DCRemoval {
	dc := &DCRemoval{
		sampleRate: sampleRate,
		cutoffFreq: cutoffFreq,
	}

	dc.computePoleLocation()
	return dc
}

// NewDCRemovalWithPole creates a DC removal filter with explicit pole location.
//
// Parameters:
//   - poleLocation: R parameter (0 < R < 1)
//     Closer to 1 = lower cutoff frequency (more DC blocking)
//     Closer to 0 = higher cutoff frequency (less DC blocking)
//
// Common values:
//   - 0.99: Aggressive DC blocking (cutoff ≈ 35 Hz at 44.1 kHz)
//   - 0.995: Standard DC blocking (cutoff ≈ 8 Hz at 44.1 kHz)
//   - 0.999: Conservative DC blocking (cutoff ≈ 3.5 Hz at 44.1 kHz)
func NewDCRemovalWithPole(poleLocation float64) *DCRemoval {
	return &DCRemoval{
		poleLocation: poleLocation,
		initialized:  true,
	}
}

// computePoleLocation calculates the pole location from the desired cutoff frequency.
// Uses the approximation: R ≈ 1 - 2*pi*fc/fs
// This is valid for small cutoff frequencies (fc << fs/2).
func (dc *DCRemoval) computePoleLocation() {
	if dc.sampleRate > 0 && dc.cutoffFreq > 0 {
		// R = 1 - 2*pi*fc/fs (small angle approximation)
		dc.poleLocation = 1.0 - (2.0 * math.Pi * dc.cutoffFreq / float64(dc.sampleRate))

		// Clamp to valid range
		if dc.poleLocation >= 1.0 {
			dc.poleLocation = 0.999
		} else if dc.poleLocation <= 0.0 {
			dc.poleLocation = 0.001
		}

		dc.initialized = true
	}
}

// Process applies DC removal to a single sample.
// Implements the difference equation:
// y[n] = x[n] - x[n-1] + R * y[n-1]
func (dc *DCRemoval) Process(input float64) float64 {
	if !dc.initialized {
		dc.poleLocation = 0.995
		dc.initialized = true
	}

	// Apply the difference equation
	output := input - dc.x1 + dc.poleLocation*dc.y1

	// Update state
	dc.x1 = input
	dc.y1 = output

	return output
}

// ProcessBuffer applies DC removal to an entire buffer of samples.
func (dc *DCRemoval) ProcessBuffer(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, sample := range input {
		output[i] = dc.Process(sample)
	}
	return output
}

// Reset clears the filter's internal state.
// Call this when processing discontinuous audio segments.
func (dc *DCRemoval) Reset() {
	dc.x1 = 0.0
	dc.y1 = 0.0
}

// SetCutoffFrequency updates the cutoff frequency and recomputes the pole location.
func (dc *DCRemoval) SetCutoffFrequency(sampleRate int, cutoffFreq float64) {
	dc.sampleRate = sampleRate
	dc.cutoffFreq = cutoffFreq
	dc.computePoleLocation()
}

// SetPoleLocation directly sets the pole location parameter.
func (dc *DCRemoval) SetPoleLocation(poleLocation float64) {
	if poleLocation > 0 && poleLocation < 1 {
		dc.poleLocation = poleLocation
		dc.initialized = true
	}
}

// GetCutoffFrequency calculates the approximate -3dB cutoff frequency.
// Uses the inverse of the design formula: fc ≈ (1-R)*fs/(2*pi)
func (dc *DCRemoval) GetCutoffFrequency(sampleRate int) float64 {
	if sampleRate <= 0 {
		return 0.0
	}

	return (1.0 - dc.poleLocation) * float64(sampleRate) / (2.0 * math.Pi)
}

// GetPoleLocation returns the current pole location parameter.
func (dc *DCRemoval) GetPoleLocation() float64 {
	return dc.poleLocation
}

// GetFrequencyResponse computes the magnitude and phase response at given frequency.
// Returns magnitude (linear scale) and phase (radians).
//
// The frequency response is:
// H(e^jw) = (1 - e^-jw) / (1 - R*e^-jw)
func (dc *DCRemoval) GetFrequencyResponse(frequency float64, sampleRate int) (magnitude, phase float64) {
	w := 2.0 * math.Pi * frequency / float64(sampleRate)

	cosW := math.Cos(w)
	sinW := math.Sin(w)

	// Numerator: 1 - e^-jw = 1 - cos(w) + j*sin(w)
	numReal := 1.0 - cosW
	numImag := sinW

	// Denominator: 1 - R*e^-jw = 1 - R*cos(w) + j*R*sin(w)
	denReal := 1.0 - dc.poleLocation*cosW
	denImag := dc.poleLocation * sinW

	// H(e^jw) = numerator / denominator
	denMagSq := denReal*denReal + denImag*denImag

	hReal := (numReal*denReal + numImag*denImag) / denMagSq
	hImag := (numImag*denReal - numReal*denImag) / denMagSq

	magnitude = math.Sqrt(hReal*hReal + hImag*hImag)
	phase = math.Atan2(hImag, hReal)

	return magnitude, phase
}

// GetGroupDelay computes the group delay at given frequency.
// Group delay = -d(phase)/dw, useful for analyzing phase linearity.
//
// For the DC blocker, group delay is approximately:
// τ(w) ≈ R*sin(w) / (1 - R*cos(w))
func (dc *DCRemoval) GetGroupDelay(frequency float64, sampleRate int) float64 {
	w := 2.0 * math.Pi * frequency / float64(sampleRate)

	cosW := math.Cos(w)
	sinW := math.Sin(w)

	// Group delay approximation for DC blocker
	numerator := dc.poleLocation * sinW
	denominator := (1.0 - dc.poleLocation*cosW) * (1.0 - dc.poleLocation*cosW)

	if denominator != 0 {
		return numerator / denominator
	}

	return 0.0
}
