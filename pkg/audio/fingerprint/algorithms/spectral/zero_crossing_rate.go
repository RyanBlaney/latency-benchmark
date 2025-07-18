package spectral

import (
	"math"

	"gonum.org/v1/gonum/stat"
)

// ZeroCrossingRate calculates zero crossing rate for voice activity detection
// High ZCR indicates fricatives/unvoiced speech, low ZCR indicates voiced speech
type ZeroCrossingRate struct {
	sampleRate int
	frameSize  int
	hopSize    int
}

// NewZeroCrossingRate creates a new zero crossing rate calculator
func NewZeroCrossingRate(sampleRate int) *ZeroCrossingRate {
	return &ZeroCrossingRate{
		sampleRate: sampleRate,
		frameSize:  1024, // Default frame size
		hopSize:    512,  // Default hop size (50% overlap)
	}
}

// NewZeroCrossingRateWithParams creates calculator with custom parameters
func NewZeroCrossingRateWithParams(sampleRate, frameSize, hopSize int) *ZeroCrossingRate {
	return &ZeroCrossingRate{
		sampleRate: sampleRate,
		frameSize:  frameSize,
		hopSize:    hopSize,
	}
}

// Compute calculates ZCR for a single frame
// Returns rate as crossings per second
func (zcr *ZeroCrossingRate) Compute(frame []float64) float64 {
	if len(frame) < 2 {
		return 0.0
	}

	crossings := 0
	for i := 1; i < len(frame); i++ {
		// Check for sign change (zero crossing)
		if (frame[i-1] >= 0 && frame[i] < 0) || (frame[i-1] < 0 && frame[i] >= 0) {
			crossings++
		}
	}

	// Convert to crossings per second
	frameDuration := float64(len(frame)) / float64(zcr.sampleRate)
	return float64(crossings) / frameDuration
}

// ComputeNormalized calculates normalized ZCR (0-1 range)
// Useful for content-agnostic comparison
func (zcr *ZeroCrossingRate) ComputeNormalized(frame []float64) float64 {
	if len(frame) < 2 {
		return 0.0
	}

	crossings := 0
	for i := 1; i < len(frame); i++ {
		if (frame[i-1] >= 0 && frame[i] < 0) || (frame[i-1] < 0 && frame[i] >= 0) {
			crossings++
		}
	}

	// Normalize by maximum possible crossings (alternating signal)
	maxCrossings := len(frame) - 1
	if maxCrossings == 0 {
		return 0.0
	}

	return float64(crossings) / float64(maxCrossings)
}

// ComputeFrames calculates ZCR for overlapping frames of a signal
func (zcr *ZeroCrossingRate) ComputeFrames(signal []float64) []float64 {
	if len(signal) < zcr.frameSize {
		return []float64{}
	}

	numFrames := (len(signal)-zcr.frameSize)/zcr.hopSize + 1
	zcrValues := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * zcr.hopSize
		endIdx := startIdx + zcr.frameSize

		if endIdx > len(signal) {
			break
		}

		frame := signal[startIdx:endIdx]
		zcrValues[i] = zcr.Compute(frame)
	}

	return zcrValues
}

// ComputeFramesNormalized calculates normalized ZCR for overlapping frames
func (zcr *ZeroCrossingRate) ComputeFramesNormalized(signal []float64) []float64 {
	if len(signal) < zcr.frameSize {
		return []float64{}
	}

	numFrames := (len(signal)-zcr.frameSize)/zcr.hopSize + 1
	zcrValues := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * zcr.hopSize
		endIdx := startIdx + zcr.frameSize

		if endIdx > len(signal) {
			break
		}

		frame := signal[startIdx:endIdx]
		zcrValues[i] = zcr.ComputeNormalized(frame)
	}

	return zcrValues
}

// ComputeWithThreshold calculates ZCR with amplitude threshold
// Only counts crossings above a minimum amplitude to reduce noise sensitivity
func (zcr *ZeroCrossingRate) ComputeWithThreshold(frame []float64, threshold float64) float64 {
	if len(frame) < 2 {
		return 0.0
	}

	crossings := 0
	for i := 1; i < len(frame); i++ {
		// Only count crossing if both samples are above threshold
		if math.Abs(frame[i-1]) > threshold && math.Abs(frame[i]) > threshold {
			if (frame[i-1] >= 0 && frame[i] < 0) || (frame[i-1] < 0 && frame[i] >= 0) {
				crossings++
			}
		}
	}

	frameDuration := float64(len(frame)) / float64(zcr.sampleRate)
	return float64(crossings) / frameDuration
}

// DetectVoiceActivity uses ZCR for basic voice activity detection
// Returns true if frame likely contains speech
func (zcr *ZeroCrossingRate) DetectVoiceActivity(frame []float64, energyThreshold, zcrLowThreshold, zcrHighThreshold float64) bool {
	// Calculate frame energy
	energy := 0.0
	for _, sample := range frame {
		energy += sample * sample
	}
	energy /= float64(len(frame))

	// Low energy indicates silence
	if energy < energyThreshold {
		return false
	}

	// Calculate ZCR
	zcrRate := zcr.ComputeNormalized(frame)

	// Voice activity based on ZCR range
	// Very low ZCR: likely voiced speech
	// Medium ZCR: likely voiced speech
	// Very high ZCR: likely unvoiced speech/fricatives
	// Extremely high ZCR: likely noise
	return zcrRate >= zcrLowThreshold && zcrRate <= zcrHighThreshold
}

// DetectSpeechSegments performs voice activity detection on entire signal
// Returns start/end indices of speech segments
func (zcr *ZeroCrossingRate) DetectSpeechSegments(signal []float64, energyThreshold, zcrLowThreshold, zcrHighThreshold float64, minSegmentLength int) [][]int {
	zcrValues := zcr.ComputeFramesNormalized(signal)
	if len(zcrValues) == 0 {
		return [][]int{}
	}

	// Calculate frame energies
	energies := make([]float64, len(zcrValues))
	for i := range len(zcrValues) {
		startIdx := i * zcr.hopSize
		endIdx := startIdx + zcr.frameSize
		endIdx = min(endIdx, len(signal))

		frame := signal[startIdx:endIdx]
		energy := 0.0
		for _, sample := range frame {
			energy += sample * sample
		}
		energies[i] = energy / float64(len(frame))
	}

	// Voice activity detection
	var segments [][]int
	currentStart := -1

	for i, zcrValue := range zcrValues {
		isVoice := energies[i] >= energyThreshold &&
			zcrValue >= zcrLowThreshold &&
			zcrValue <= zcrHighThreshold

		if isVoice && currentStart == -1 {
			// Start of speech segment
			currentStart = i * zcr.hopSize
		} else if !isVoice && currentStart != -1 {
			// End of speech segment
			segmentLength := (i * zcr.hopSize) - currentStart
			if segmentLength >= minSegmentLength {
				segments = append(segments, []int{currentStart, i * zcr.hopSize})
			}
			currentStart = -1
		}
	}

	// Handle segment that extends to end of signal
	if currentStart != -1 {
		segmentLength := len(signal) - currentStart
		if segmentLength >= minSegmentLength {
			segments = append(segments, []int{currentStart, len(signal)})
		}
	}

	return segments
}

// ClassifyFrameType classifies frame based on ZCR characteristics
func (zcr *ZeroCrossingRate) ClassifyFrameType(frame []float64, energy float64) string {
	if energy < 0.001 { // Low energy threshold
		return "silence"
	}

	zcrRate := zcr.ComputeNormalized(frame)

	if zcrRate < 0.1 {
		return "voiced" // Low ZCR = voiced speech, music
	} else if zcrRate < 0.4 {
		return "mixed" // Medium ZCR = mixed voiced/unvoiced
	} else if zcrRate < 0.7 {
		return "unvoiced" // High ZCR = fricatives, unvoiced speech
	} else {
		return "noise" // Very high ZCR = noise
	}
}

// GetOptimalThresholds returns recommended thresholds for speech detection
func (zcr *ZeroCrossingRate) GetOptimalThresholds() (energyThreshold, zcrLowThreshold, zcrHighThreshold float64) {
	return 0.001, // Energy threshold
		0.02, // ZCR low threshold (2% of max crossings)
		0.6 // ZCR high threshold (60% of max crossings)
}

// ComputeStatistics calculates ZCR statistics for content analysis using gonum
func (zcr *ZeroCrossingRate) ComputeStatistics(zcrValues []float64) (mean, variance, min, max float64) {
	if len(zcrValues) == 0 {
		return 0, 0, 0, 0
	}

	// Use gonum for robust statistical calculations
	mean = stat.Mean(zcrValues, nil)
	variance = stat.Variance(zcrValues, nil)

	// Find min and max
	min = zcrValues[0]
	max = zcrValues[0]
	for _, value := range zcrValues {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	return mean, variance, min, max
}
