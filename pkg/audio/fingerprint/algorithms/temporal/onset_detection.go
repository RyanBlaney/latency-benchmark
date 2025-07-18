package temporal

import (
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// OnsetDetection detects note/event onsets in audio signals
type OnsetDetection struct {
	spectralFlux      *spectral.SpectralFlux
	envelopeExtractor *Envelope
	stft              *spectral.STFT
}

// NewOnsetDetection creates a new onset detector
func NewOnsetDetection() *OnsetDetection {
	return &OnsetDetection{
		spectralFlux:      spectral.NewSpectralFlux(),
		envelopeExtractor: NewEnvelope(),
		stft:              spectral.NewSTFT(),
	}
}

// DetectOnsets detects onsets using spectral flux
func (od *OnsetDetection) DetectOnsets(signal []float64, sampleRate int, threshold float64, minInterval float64) ([]int, error) {
	if len(signal) == 0 {
		return []int{}, nil
	}

	// Compute STFT
	windowSize := 1024
	hopSize := 512
	stftResult, err := od.stft.ComputeWithWindow(signal, windowSize, hopSize, sampleRate, nil)
	if err != nil {
		return nil, err
	}

	// Calculate spectral flux using the spectral algorithm
	flux := od.spectralFlux.Compute(stftResult.Magnitude)

	if len(flux) == 0 {
		return []int{}, nil
	}

	// Find peaks in spectral flux
	onsetFrames := od.findFluxPeaks(flux, threshold, minInterval, hopSize, sampleRate)

	// Convert frame indices to sample indices
	onsetSamples := make([]int, len(onsetFrames))
	for i, frameIdx := range onsetFrames {
		onsetSamples[i] = frameIdx * hopSize
	}

	return onsetSamples, nil
}

// DetectOnsetsEnergy detects onsets using energy-based method
func (od *OnsetDetection) DetectOnsetsEnergy(signal []float64, sampleRate int, threshold float64, minInterval float64) []int {
	if len(signal) == 0 {
		return []int{}
	}

	// Calculate energy envelope
	frameSize := 512
	hopSize := 256
	envelope := od.envelopeExtractor.ComputeRMS(signal, frameSize, hopSize)

	if len(envelope) == 0 {
		return []int{}
	}

	// Calculate energy derivative for onset detection
	energyDiff := make([]float64, len(envelope)-1)
	for i := range len(energyDiff) {
		diff := envelope[i+1] - envelope[i]
		if diff > 0 {
			energyDiff[i] = diff
		} else {
			energyDiff[i] = 0 // Only positive changes
		}
	}

	// Find peaks in energy difference
	onsetFrames := od.findFluxPeaks(energyDiff, threshold, minInterval, hopSize, sampleRate)

	// Convert frame indices to sample indices
	onsetSamples := make([]int, len(onsetFrames))
	for i, frameIdx := range onsetFrames {
		onsetSamples[i] = frameIdx * hopSize
	}

	return onsetSamples
}

// findFluxPeaks finds peaks in flux/energy difference signals
func (od *OnsetDetection) findFluxPeaks(flux []float64, threshold float64, minInterval float64, hopSize int, sampleRate int) []int {
	if len(flux) < 3 {
		return []int{}
	}

	// Convert minimum interval to frames
	minIntervalFrames := int(minInterval * float64(sampleRate) / float64(hopSize))

	var peaks []int
	lastPeakFrame := -minIntervalFrames // Allow first peak

	for i := 1; i < len(flux)-1; i++ {
		// Check if it's a local maximum above threshold
		if flux[i] > flux[i-1] &&
			flux[i] > flux[i+1] &&
			flux[i] >= threshold &&
			i-lastPeakFrame >= minIntervalFrames {
			peaks = append(peaks, i)
			lastPeakFrame = i
		}
	}

	return peaks
}

// DetectOnsetsComplex detects onsets using multiple methods and combines results
func (od *OnsetDetection) DetectOnsetsComplex(signal []float64, sampleRate int) ([]int, error) {
	if len(signal) == 0 {
		return []int{}, nil
	}

	// Parameters
	fluxThreshold := 0.3
	energyThreshold := 0.1
	minInterval := 0.05 // 50ms minimum interval

	// Detect using spectral flux
	fluxOnsets, err := od.DetectOnsets(signal, sampleRate, fluxThreshold, minInterval)
	if err != nil {
		return nil, err
	}

	// Detect using energy
	energyOnsets := od.DetectOnsetsEnergy(signal, sampleRate, energyThreshold, minInterval)

	// Combine and deduplicate onsets
	combinedOnsets := od.combineOnsets(fluxOnsets, energyOnsets, int(minInterval*float64(sampleRate)))

	return combinedOnsets, nil
}

// combineOnsets combines onset lists and removes duplicates within tolerance
func (od *OnsetDetection) combineOnsets(onsets1, onsets2 []int, tolerance int) []int {
	// Combine all onsets
	allOnsets := append(onsets1, onsets2...)

	if len(allOnsets) == 0 {
		return []int{}
	}

	// Sort onsets
	for i := 0; i < len(allOnsets)-1; i++ {
		for j := i + 1; j < len(allOnsets); j++ {
			if allOnsets[j] < allOnsets[i] {
				allOnsets[i], allOnsets[j] = allOnsets[j], allOnsets[i]
			}
		}
	}

	// Remove duplicates within tolerance
	var uniqueOnsets []int
	for _, onset := range allOnsets {
		isDuplicate := false
		for _, existing := range uniqueOnsets {
			if abs(onset-existing) <= tolerance {
				isDuplicate = true
				break
			}
		}
		if !isDuplicate {
			uniqueOnsets = append(uniqueOnsets, onset)
		}
	}

	return uniqueOnsets
}

// ComputeOnsetDensity calculates onset density (onsets per second)
func (od *OnsetDetection) ComputeOnsetDensity(signal []float64, sampleRate int) (float64, error) {
	onsets, err := od.DetectOnsetsComplex(signal, sampleRate)
	if err != nil {
		return 0.0, err
	}

	duration := float64(len(signal)) / float64(sampleRate)
	if duration == 0 {
		return 0.0, nil
	}

	return float64(len(onsets)) / duration, nil
}

// AdaptiveThreshold calculates adaptive threshold based on flux statistics
func (od *OnsetDetection) AdaptiveThreshold(flux []float64) float64 {
	if len(flux) == 0 {
		return 0.0
	}

	// Calculate mean and standard deviation
	mean := 0.0
	for _, val := range flux {
		mean += val
	}
	mean /= float64(len(flux))

	variance := 0.0
	for _, val := range flux {
		diff := val - mean
		variance += diff * diff
	}
	variance /= float64(len(flux))
	stdDev := math.Sqrt(variance)

	// Adaptive threshold: mean + 2 * standard deviation
	return mean + 2.0*stdDev
}

// abs returns absolute value of integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
