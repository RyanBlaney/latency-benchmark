package temporal

import (
	"math"
)

// TempoEstimation provides basic tempo estimation for future music analysis
type TempoEstimation struct {
	onsetDetector     *OnsetDetection
	envelopeExtractor *Envelope
}

// NewTempoEstimation creates a new tempo estimator
func NewTempoEstimation() *TempoEstimation {
	return &TempoEstimation{
		onsetDetector:     NewOnsetDetection(),
		envelopeExtractor: NewEnvelope(),
	}
}

// EstimateTempo estimates tempo in BPM using onset detection
func (te *TempoEstimation) EstimateTempo(signal []float64, sampleRate int) (float64, error) {
	if len(signal) == 0 {
		return 0.0, nil
	}

	// Detect onsets
	onsets, err := te.onsetDetector.DetectOnsetsComplex(signal, sampleRate)
	if err != nil {
		return 0.0, err
	}

	if len(onsets) < 2 {
		return 0.0, nil // Need at least 2 onsets
	}

	// Calculate inter-onset intervals
	intervals := make([]float64, len(onsets)-1)
	for i := range len(intervals) {
		intervalSamples := onsets[i+1] - onsets[i]
		intervals[i] = float64(intervalSamples) / float64(sampleRate)
	}

	// Find most common interval using autocorrelation
	tempo := te.findTempoFromIntervals(intervals)

	return tempo, nil
}

// EstimateTempoAutocorrelation estimates tempo using autocorrelation of energy
func (te *TempoEstimation) EstimateTempoAutocorrelation(signal []float64, sampleRate int) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate energy envelope for beat tracking
	frameSize := int(0.1 * float64(sampleRate)) // 100ms frames for beat analysis
	hopSize := frameSize / 4                    // 25% overlap

	envelope := te.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(envelope) < 10 {
		return 0.0
	}

	// Apply autocorrelation to find periodic patterns
	maxLag := len(envelope) / 2
	autocorr := te.calculateAutocorrelation(envelope, maxLag)

	// Find peaks in autocorrelation corresponding to beat periods
	tempo := te.findTempoFromAutocorrelation(autocorr, hopSize, sampleRate)

	return tempo
}

// findTempoFromIntervals finds tempo from inter-onset intervals
func (te *TempoEstimation) findTempoFromIntervals(intervals []float64) float64 {
	if len(intervals) == 0 {
		return 0.0
	}

	// Calculate histogram of intervals (quantized to common beat intervals)
	tempoRange := []float64{60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200}
	tempoCounts := make([]int, len(tempoRange))

	for _, interval := range intervals {
		if interval > 0.2 && interval < 2.0 { // Valid beat interval range (30-300 BPM)
			tempo := 60.0 / interval

			// Find closest tempo bin
			bestIdx := 0
			bestDiff := math.Abs(tempo - tempoRange[0])
			for i, refTempo := range tempoRange {
				diff := math.Abs(tempo - refTempo)
				if diff < bestDiff {
					bestDiff = diff
					bestIdx = i
				}
			}

			if bestDiff < 10.0 { // Within 10 BPM tolerance
				tempoCounts[bestIdx]++
			}
		}
	}

	// Find most frequent tempo
	maxCount := 0
	bestTempo := 120.0 // Default
	for i, count := range tempoCounts {
		if count > maxCount {
			maxCount = count
			bestTempo = tempoRange[i]
		}
	}

	return bestTempo
}

// calculateAutocorrelation calculates autocorrelation function
func (te *TempoEstimation) calculateAutocorrelation(signal []float64, maxLag int) []float64 {
	if maxLag > len(signal) {
		maxLag = len(signal)
	}

	autocorr := make([]float64, maxLag)

	for lag := 0; lag < maxLag; lag++ {
		sum := 0.0
		count := 0

		for i := 0; i < len(signal)-lag; i++ {
			sum += signal[i] * signal[i+lag]
			count++
		}

		if count > 0 {
			autocorr[lag] = sum / float64(count)
		}
	}

	// Normalize
	if len(autocorr) > 0 && autocorr[0] > 0 {
		for i := range autocorr {
			autocorr[i] /= autocorr[0]
		}
	}

	return autocorr
}

// findTempoFromAutocorrelation finds tempo from autocorrelation peaks
func (te *TempoEstimation) findTempoFromAutocorrelation(autocorr []float64, hopSize int, sampleRate int) float64 {
	if len(autocorr) < 10 {
		return 0.0
	}

	// Look for peaks in autocorrelation corresponding to beat periods
	// Convert lag to time period
	timePerFrame := float64(hopSize) / float64(sampleRate)

	// Search in reasonable tempo range (60-180 BPM)
	minPeriodSec := 60.0 / 180.0 // 180 BPM
	maxPeriodSec := 1.0          // (60.0 / 60.0) for 60 BPM

	minLag := int(minPeriodSec / timePerFrame)
	maxLag := int(maxPeriodSec / timePerFrame)

	if minLag < 1 {
		minLag = 1
	}
	if maxLag >= len(autocorr) {
		maxLag = len(autocorr) - 1
	}

	// Find highest peak in tempo range
	maxVal := 0.0
	bestLag := 0

	for lag := minLag; lag <= maxLag; lag++ {
		// Check if it's a local maximum
		if lag > 0 && lag < len(autocorr)-1 {
			if autocorr[lag] > autocorr[lag-1] &&
				autocorr[lag] > autocorr[lag+1] &&
				autocorr[lag] > maxVal {
				maxVal = autocorr[lag]
				bestLag = lag
			}
		}
	}

	if bestLag == 0 {
		return 120.0 // Default tempo
	}

	// Convert lag back to tempo
	period := float64(bestLag) * timePerFrame
	tempo := 60.0 / period

	return tempo
}

// EstimateTempoRange estimates tempo and provides confidence range
func (te *TempoEstimation) EstimateTempoRange(signal []float64, sampleRate int) (float64, float64, float64) {
	// Get tempo from both methods
	onsetTempo, _ := te.EstimateTempo(signal, sampleRate)
	autocorrTempo := te.EstimateTempoAutocorrelation(signal, sampleRate)

	// Average the results
	avgTempo := (onsetTempo + autocorrTempo) / 2.0

	// Calculate confidence based on agreement
	diff := math.Abs(onsetTempo - autocorrTempo)
	confidence := math.Max(0.0, 1.0-diff/50.0) // Confidence decreases with disagreement

	return avgTempo, confidence, diff
}

// ClassifyTempoCategory classifies tempo into broad categories
func (te *TempoEstimation) ClassifyTempoCategory(tempo float64) string {
	if tempo < 60 {
		return "very_slow"
	} else if tempo < 90 {
		return "slow"
	} else if tempo < 120 {
		return "moderate"
	} else if tempo < 150 {
		return "fast"
	} else {
		return "very_fast"
	}
}
