package temporal

import (
	"math"
)

// SilenceDetection provides voice activity detection and silence analysis
type SilenceDetection struct {
	envelopeExtractor *Envelope
}

// NewSilenceDetection creates a new silence detector
func NewSilenceDetection() *SilenceDetection {
	return &SilenceDetection{
		envelopeExtractor: NewEnvelope(),
	}
}

// DetectSilence detects silent segments in audio signal
func (sd *SilenceDetection) DetectSilence(signal []float64, sampleRate int, energyThreshold float64, minSilenceDuration float64) [][]int {
	if len(signal) == 0 {
		return [][]int{}
	}

	// Calculate frame-based energy
	frameSize := int(0.025 * float64(sampleRate)) // 25ms frames
	hopSize := frameSize / 2                      // 50% overlap

	energies := sd.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(energies) == 0 {
		return [][]int{}
	}

	// Convert minimum silence duration to frames
	minSilenceFrames := int(minSilenceDuration * float64(sampleRate) / float64(hopSize))

	// Find silent frames
	silentFrames := make([]bool, len(energies))
	for i, energy := range energies {
		silentFrames[i] = energy < energyThreshold
	}

	// Group consecutive silent frames into segments
	var silenceSegments [][]int
	currentStart := -1

	for i, isSilent := range silentFrames {
		if isSilent && currentStart == -1 {
			// Start of silence segment
			currentStart = i
		} else if !isSilent && currentStart != -1 {
			// End of silence segment
			segmentLength := i - currentStart
			if segmentLength >= minSilenceFrames {
				startSample := currentStart * hopSize
				endSample := i * hopSize
				silenceSegments = append(silenceSegments, []int{startSample, endSample})
			}
			currentStart = -1
		}
	}

	// Handle segment that extends to end
	if currentStart != -1 {
		segmentLength := len(silentFrames) - currentStart
		if segmentLength >= minSilenceFrames {
			startSample := currentStart * hopSize
			endSample := len(signal)
			silenceSegments = append(silenceSegments, []int{startSample, endSample})
		}
	}

	return silenceSegments
}

// DetectVoiceActivity detects voice activity using energy and ZCR
func (sd *SilenceDetection) DetectVoiceActivity(signal []float64, sampleRate int, energyThreshold, zcrLowThreshold, zcrHighThreshold float64) [][]int {
	if len(signal) == 0 {
		return [][]int{}
	}

	frameSize := int(0.025 * float64(sampleRate)) // 25ms frames
	hopSize := frameSize / 2                      // 50% overlap

	// Calculate energy
	energies := sd.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	// Calculate zero crossing rate
	zcrValues := sd.calculateZCR(signal, frameSize, hopSize)

	if len(energies) != len(zcrValues) {
		return [][]int{}
	}

	// Detect voice activity
	voiceFrames := make([]bool, len(energies))
	for i := range energies {
		isVoice := energies[i] >= energyThreshold &&
			zcrValues[i] >= zcrLowThreshold &&
			zcrValues[i] <= zcrHighThreshold
		voiceFrames[i] = isVoice
	}

	// Group consecutive voice frames into segments
	var voiceSegments [][]int
	currentStart := -1
	minVoiceFrames := int(0.1 * float64(sampleRate) / float64(hopSize)) // 100ms minimum

	for i, isVoice := range voiceFrames {
		if isVoice && currentStart == -1 {
			currentStart = i
		} else if !isVoice && currentStart != -1 {
			segmentLength := i - currentStart
			if segmentLength >= minVoiceFrames {
				startSample := currentStart * hopSize
				endSample := i * hopSize
				voiceSegments = append(voiceSegments, []int{startSample, endSample})
			}
			currentStart = -1
		}
	}

	// Handle segment that extends to end
	if currentStart != -1 {
		segmentLength := len(voiceFrames) - currentStart
		if segmentLength >= minVoiceFrames {
			startSample := currentStart * hopSize
			endSample := len(signal)
			voiceSegments = append(voiceSegments, []int{startSample, endSample})
		}
	}

	return voiceSegments
}

// calculateZCR calculates zero crossing rate for frames
func (sd *SilenceDetection) calculateZCR(signal []float64, frameSize, hopSize int) []float64 {
	if len(signal) < frameSize {
		return []float64{}
	}

	numFrames := (len(signal)-frameSize)/hopSize + 1
	zcrValues := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * hopSize
		endIdx := startIdx + frameSize

		if endIdx > len(signal) {
			break
		}

		// Count zero crossings in frame
		crossings := 0
		for j := startIdx + 1; j < endIdx; j++ {
			if (signal[j-1] >= 0 && signal[j] < 0) || (signal[j-1] < 0 && signal[j] >= 0) {
				crossings++
			}
		}

		// Normalize by maximum possible crossings
		maxCrossings := frameSize - 1
		zcrValues[i] = float64(crossings) / float64(maxCrossings)
	}

	return zcrValues
}

// ComputeSilenceRatio calculates the ratio of silent frames
func (sd *SilenceDetection) ComputeSilenceRatio(signal []float64, sampleRate int, energyThreshold float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	frameSize := int(0.025 * float64(sampleRate)) // 25ms frames
	hopSize := frameSize / 2                      // 50% overlap

	energies := sd.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(energies) == 0 {
		return 0.0
	}

	silentFrames := 0
	for _, energy := range energies {
		if energy < energyThreshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

// AdaptiveThreshold calculates adaptive energy threshold based on signal statistics
func (sd *SilenceDetection) AdaptiveThreshold(signal []float64, sampleRate int) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	frameSize := int(0.025 * float64(sampleRate)) // 25ms frames
	hopSize := frameSize / 2                      // 50% overlap

	energies := sd.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(energies) == 0 {
		return 0.0
	}

	// Calculate statistics
	mean := 0.0
	for _, energy := range energies {
		mean += energy
	}
	mean /= float64(len(energies))

	variance := 0.0
	for _, energy := range energies {
		diff := energy - mean
		variance += diff * diff
	}
	variance /= float64(len(energies))
	stdDev := math.Sqrt(variance)

	// Adaptive threshold: mean - 2 * standard deviation
	threshold := mean - 2.0*stdDev
	if threshold < 0 {
		threshold = mean * 0.1 // Fallback to 10% of mean
	}

	return threshold
}

// GetOptimalThresholds returns recommended thresholds for voice activity detection
func (sd *SilenceDetection) GetOptimalThresholds() (energyThreshold, zcrLowThreshold, zcrHighThreshold float64) {
	return 0.001, // Energy threshold
		0.02, // ZCR low threshold (2% of max crossings)
		0.6 // ZCR high threshold (60% of max crossings)
}
