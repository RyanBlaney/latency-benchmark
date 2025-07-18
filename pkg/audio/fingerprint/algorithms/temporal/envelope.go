package temporal

import (
	"math"
)

// Envelope provides amplitude envelope extraction
type Envelope struct {
	// No state needed - stateless calculation
}

// NewEnvelope creates a new envelope extractor
func NewEnvelope() *Envelope {
	return &Envelope{}
}

// ComputeRMS computes RMS envelope with given frame and hop sizes
func (e *Envelope) ComputeRMS(signal []float64, frameSize, hopSize int) []float64 {
	if len(signal) < frameSize || frameSize <= 0 || hopSize <= 0 {
		return []float64{}
	}

	numFrames := (len(signal)-frameSize)/hopSize + 1
	envelope := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * hopSize
		endIdx := startIdx + frameSize

		if endIdx > len(signal) {
			break
		}

		// Calculate RMS for this frame
		sumSquares := 0.0
		for j := startIdx; j < endIdx; j++ {
			sumSquares += signal[j] * signal[j]
		}
		envelope[i] = math.Sqrt(sumSquares / float64(frameSize))
	}

	return envelope
}

// ComputePeak computes peak envelope (maximum absolute value per frame)
func (e *Envelope) ComputePeak(signal []float64, frameSize, hopSize int) []float64 {
	if len(signal) < frameSize || frameSize <= 0 || hopSize <= 0 {
		return []float64{}
	}

	numFrames := (len(signal)-frameSize)/hopSize + 1
	envelope := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * hopSize
		endIdx := startIdx + frameSize

		if endIdx > len(signal) {
			break
		}

		// Find peak in this frame
		peak := 0.0
		for j := startIdx; j < endIdx; j++ {
			abs := math.Abs(signal[j])
			if abs > peak {
				peak = abs
			}
		}
		envelope[i] = peak
	}

	return envelope
}

// ComputeHilbert computes envelope using Hilbert transform approximation
func (e *Envelope) ComputeHilbert(signal []float64) []float64 {
	if len(signal) == 0 {
		return []float64{}
	}

	envelope := make([]float64, len(signal))

	// Simple Hilbert transform approximation using finite differences
	for i := range signal {
		real := signal[i]

		// Approximate imaginary part using derivative
		var imag float64
		if i == 0 {
			imag = signal[1] - signal[0]
		} else if i == len(signal)-1 {
			imag = signal[i] - signal[i-1]
		} else {
			imag = (signal[i+1] - signal[i-1]) / 2.0
		}

		// Envelope is magnitude of complex signal
		envelope[i] = math.Sqrt(real*real + imag*imag)
	}

	return envelope
}

// ComputeSmoothed computes smoothed envelope using moving average
func (e *Envelope) ComputeSmoothed(envelope []float64, windowSize int) []float64 {
	if len(envelope) == 0 || windowSize <= 0 {
		return envelope
	}

	if windowSize > len(envelope) {
		windowSize = len(envelope)
	}

	smoothed := make([]float64, len(envelope))
	halfWindow := windowSize / 2

	for i := range envelope {
		sum := 0.0
		count := 0

		// Average over window
		for j := i - halfWindow; j <= i+halfWindow; j++ {
			if j >= 0 && j < len(envelope) {
				sum += envelope[j]
				count++
			}
		}

		if count > 0 {
			smoothed[i] = sum / float64(count)
		}
	}

	return smoothed
}
