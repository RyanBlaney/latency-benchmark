package temporal

import (
	"math"
)

// AttackDecay analyzes transient characteristics of audio signals
type AttackDecay struct {
	envelopeExtractor *Envelope
}

// NewAttackDecay creates a new attack/decay analyzer
func NewAttackDecay() *AttackDecay {
	return &AttackDecay{
		envelopeExtractor: NewEnvelope(),
	}
}

// ComputeAttackTime calculates attack time for audio segments
// Returns time from start to peak amplitude
func (ad *AttackDecay) ComputeAttackTime(signal []float64, sampleRate int, threshold float64) []float64 {
	if len(signal) == 0 {
		return []float64{}
	}

	// Use envelope for analysis
	frameSize := 512
	hopSize := 256
	envelope := ad.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(envelope) == 0 {
		return []float64{}
	}

	// Find peaks in envelope
	peaks := ad.findPeaks(envelope, threshold)
	attackTimes := make([]float64, len(peaks))

	for i, peakIdx := range peaks {
		// Find start of attack (where envelope rises above noise floor)
		startIdx := ad.findAttackStart(envelope, peakIdx, threshold*0.1)

		// Calculate attack time in seconds
		attackFrames := peakIdx - startIdx
		attackTimes[i] = float64(attackFrames*hopSize) / float64(sampleRate)
	}

	return attackTimes
}

// ComputeDecayTime calculates decay time for audio segments
// Returns time from peak to sustain level
func (ad *AttackDecay) ComputeDecayTime(signal []float64, sampleRate int, threshold float64) []float64 {
	if len(signal) == 0 {
		return []float64{}
	}

	// Use envelope for analysis
	frameSize := 512
	hopSize := 256
	envelope := ad.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(envelope) == 0 {
		return []float64{}
	}

	// Find peaks in envelope
	peaks := ad.findPeaks(envelope, threshold)
	decayTimes := make([]float64, len(peaks))

	for i, peakIdx := range peaks {
		// Find end of decay (where envelope reaches sustain level)
		sustainLevel := envelope[peakIdx] * 0.1 // 10% of peak
		endIdx := ad.findDecayEnd(envelope, peakIdx, sustainLevel)

		// Calculate decay time in seconds
		decayFrames := endIdx - peakIdx
		decayTimes[i] = float64(decayFrames*hopSize) / float64(sampleRate)
	}

	return decayTimes
}

// findPeaks finds peaks in envelope above threshold
func (ad *AttackDecay) findPeaks(envelope []float64, threshold float64) []int {
	if len(envelope) < 3 {
		return []int{}
	}

	var peaks []int

	for i := 1; i < len(envelope)-1; i++ {
		// Check if it's a local maximum above threshold
		if envelope[i] > envelope[i-1] &&
			envelope[i] > envelope[i+1] &&
			envelope[i] >= threshold {
			peaks = append(peaks, i)
		}
	}

	return peaks
}

// findAttackStart finds the start of attack before a peak
func (ad *AttackDecay) findAttackStart(envelope []float64, peakIdx int, noiseFloor float64) int {
	for i := peakIdx; i >= 0; i-- {
		if envelope[i] <= noiseFloor {
			return i
		}
	}
	return 0
}

// findDecayEnd finds the end of decay after a peak
func (ad *AttackDecay) findDecayEnd(envelope []float64, peakIdx int, sustainLevel float64) int {
	for i := peakIdx; i < len(envelope); i++ {
		if envelope[i] <= sustainLevel {
			return i
		}
	}
	return len(envelope) - 1
}

// ComputeTransientRatio calculates ratio of transient to steady-state energy
func (ad *AttackDecay) ComputeTransientRatio(signal []float64, sampleRate int) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Use envelope for analysis
	frameSize := 512
	hopSize := 256
	envelope := ad.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(envelope) == 0 {
		return 0.0
	}

	// Calculate derivative to find transient regions
	derivative := make([]float64, len(envelope)-1)
	for i := range len(derivative) {
		derivative[i] = math.Abs(envelope[i+1] - envelope[i])
	}

	// Sum transient energy (high derivative)
	transientEnergy := 0.0
	steadyEnergy := 0.0
	threshold := ad.calculateDerivativeThreshold(derivative)

	for i, deriv := range derivative {
		energy := envelope[i] * envelope[i]
		if deriv > threshold {
			transientEnergy += energy
		} else {
			steadyEnergy += energy
		}
	}

	if steadyEnergy == 0 {
		return 1.0 // All transient
	}

	return transientEnergy / (transientEnergy + steadyEnergy)
}

// calculateDerivativeThreshold calculates adaptive threshold for derivative
func (ad *AttackDecay) calculateDerivativeThreshold(derivative []float64) float64 {
	if len(derivative) == 0 {
		return 0.0
	}

	// Calculate mean and standard deviation
	mean := 0.0
	for _, deriv := range derivative {
		mean += deriv
	}
	mean /= float64(len(derivative))

	variance := 0.0
	for _, deriv := range derivative {
		diff := deriv - mean
		variance += diff * diff
	}
	variance /= float64(len(derivative))
	stdDev := math.Sqrt(variance)

	// Threshold at mean + 2 standard deviations
	return mean + 2.0*stdDev
}
