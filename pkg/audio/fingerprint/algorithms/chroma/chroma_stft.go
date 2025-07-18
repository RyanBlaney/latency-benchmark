package chroma

import (
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// ChromaSTFT computes chromagram using Short-Time Fourier Transform
//
// DIFFERENCE FROM spectral/stft.go:
// - spectral/stft.go: Generic STFT for any spectral analysis
// - chroma/chroma_stft.go: Specialized for pitch class analysis
//   - Maps frequencies to 12 semitone bins (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
//   - Octave-folded representation (all C notes map to same bin)
//   - Logarithmic frequency mapping for musical perception
//   - Tuning frequency adjustable (default A4=440Hz)
type ChromaSTFT struct {
	sampleRate int
	stft       *spectral.STFT
	tuningFreq float64 // A4 frequency (default 440 Hz)
	chromaBins int     // Number of chroma bins (always 12)
	minFreq    float64 // Minimum frequency to consider
	maxFreq    float64 // Maximum frequency to consider
}

// NewChromaSTFT creates a new STFT-based chromagram calculator
func NewChromaSTFT(sampleRate int, tuningFreq float64) *ChromaSTFT {
	return &ChromaSTFT{
		sampleRate: sampleRate,
		stft:       spectral.NewSTFT(),
		tuningFreq: tuningFreq,
		chromaBins: 12,
		minFreq:    80.0,   // Approximate E2
		maxFreq:    8000.0, // High enough for harmonics
	}
}

// NewChromaSTFTDefault creates chromagram with standard A4=440Hz tuning
func NewChromaSTFTDefault(sampleRate int) *ChromaSTFT {
	return NewChromaSTFT(sampleRate, 440.0)
}

// ComputeChroma computes chromagram from audio signal
func (cs *ChromaSTFT) ComputeChroma(signal []float64, windowSize, hopSize int, window spectral.Window) ([][]float64, error) {
	if len(signal) == 0 {
		return nil, nil
	}

	// Compute STFT using the generic spectral STFT
	stftResult, err := cs.stft.ComputeWithWindow(signal, windowSize, hopSize, cs.sampleRate, window)
	if err != nil {
		return nil, err
	}

	// Convert magnitude spectrogram to chromagram
	chromagram := cs.convertSTFTToChroma(stftResult)

	return chromagram, nil
}

// convertSTFTToChroma converts STFT magnitude spectrogram to chromagram
func (cs *ChromaSTFT) convertSTFTToChroma(stftResult *spectral.STFTResult) [][]float64 {
	chromagram := make([][]float64, stftResult.TimeFrames)

	// Pre-calculate frequency to chroma bin mapping
	chromaMapping := cs.calculateChromaMapping(stftResult.FreqBins, stftResult.FreqResolution)

	for t := 0; t < stftResult.TimeFrames; t++ {
		chromagram[t] = make([]float64, cs.chromaBins)

		// Map magnitude spectrum to chroma bins
		for f := 0; f < stftResult.FreqBins; f++ {
			magnitude := stftResult.Magnitude[t][f]
			chromaBin := chromaMapping[f]

			if chromaBin >= 0 && chromaBin < cs.chromaBins {
				// Use magnitude squared for energy
				chromagram[t][chromaBin] += magnitude * magnitude
			}
		}

		// Normalize chroma vector
		cs.normalizeChromaFrame(chromagram[t])
	}

	return chromagram
}

// calculateChromaMapping maps FFT bins to chroma bins
func (cs *ChromaSTFT) calculateChromaMapping(freqBins int, freqResolution float64) []int {
	mapping := make([]int, freqBins)

	for f := range freqBins {
		frequency := float64(f) * freqResolution

		if frequency < cs.minFreq || frequency > cs.maxFreq {
			mapping[f] = -1 // Outside valid range
			continue
		}

		// Convert frequency to MIDI note number
		midiNote := cs.frequencyToMIDI(frequency)

		// Map to chroma bin (0-11)
		chromaBin := int(math.Round(midiNote)) % 12
		mapping[f] = chromaBin
	}

	return mapping
}

// frequencyToMIDI converts frequency to MIDI note number
func (cs *ChromaSTFT) frequencyToMIDI(frequency float64) float64 {
	if frequency <= 0 {
		return 0
	}

	// MIDI note number: 69 + 12 * log2(f/440)
	// A4 (440 Hz) = MIDI note 69
	return 69.0 + 12.0*math.Log2(frequency/cs.tuningFreq)
}

// normalizeChromaFrame normalizes a single chroma frame
func (cs *ChromaSTFT) normalizeChromaFrame(chromaFrame []float64) {
	// Calculate total energy
	totalEnergy := 0.0
	for _, energy := range chromaFrame {
		totalEnergy += energy
	}

	// Normalize to unit sum
	if totalEnergy > 1e-10 {
		for i := range chromaFrame {
			chromaFrame[i] /= totalEnergy
		}
	}
}

// ComputeChromaWithCustomRange computes chromagram with custom frequency range
func (cs *ChromaSTFT) ComputeChromaWithCustomRange(signal []float64, windowSize, hopSize int, window spectral.Window, minFreq, maxFreq float64) ([][]float64, error) {
	// Temporarily set custom range
	origMinFreq := cs.minFreq
	origMaxFreq := cs.maxFreq
	cs.minFreq = minFreq
	cs.maxFreq = maxFreq

	// Compute chromagram
	chromagram, err := cs.ComputeChroma(signal, windowSize, hopSize, window)

	// Restore original range
	cs.minFreq = origMinFreq
	cs.maxFreq = origMaxFreq

	return chromagram, err
}

// GetChromaLabels returns the chroma bin labels
func (cs *ChromaSTFT) GetChromaLabels() []string {
	return []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
}

// GetDominantChroma finds the dominant chroma for each time frame
func (cs *ChromaSTFT) GetDominantChroma(chromagram [][]float64) []int {
	if len(chromagram) == 0 {
		return []int{}
	}

	dominantChroma := make([]int, len(chromagram))

	for t, chromaFrame := range chromagram {
		maxEnergy := 0.0
		maxBin := 0

		for bin, energy := range chromaFrame {
			if energy > maxEnergy {
				maxEnergy = energy
				maxBin = bin
			}
		}

		dominantChroma[t] = maxBin
	}

	return dominantChroma
}

// ComputeChromaStatistics calculates statistics for chromagram
func (cs *ChromaSTFT) ComputeChromaStatistics(chromagram [][]float64) map[string][]float64 {
	if len(chromagram) == 0 {
		return map[string][]float64{}
	}

	stats := make(map[string][]float64)

	// Mean chroma across time
	meanChroma := make([]float64, cs.chromaBins)
	for t := range chromagram {
		for bin := range chromagram[t] {
			meanChroma[bin] += chromagram[t][bin]
		}
	}
	for bin := range meanChroma {
		meanChroma[bin] /= float64(len(chromagram))
	}
	stats["mean"] = meanChroma

	// Variance chroma across time
	varChroma := make([]float64, cs.chromaBins)
	for t := range chromagram {
		for bin := range chromagram[t] {
			diff := chromagram[t][bin] - meanChroma[bin]
			varChroma[bin] += diff * diff
		}
	}
	for bin := range varChroma {
		varChroma[bin] /= float64(len(chromagram))
	}
	stats["variance"] = varChroma

	return stats
}

// ComputeChromaEnergy calculates total energy per chroma bin
func (cs *ChromaSTFT) ComputeChromaEnergy(chromagram [][]float64) []float64 {
	if len(chromagram) == 0 {
		return make([]float64, cs.chromaBins)
	}

	energy := make([]float64, cs.chromaBins)

	for t := range chromagram {
		for bin := range chromagram[t] {
			energy[bin] += chromagram[t][bin]
		}
	}

	return energy
}

// EstimateKey estimates musical key from chromagram
func (cs *ChromaSTFT) EstimateKey(chromagram [][]float64) (string, string) {
	// Calculate mean chroma profile
	meanChroma := cs.ComputeChromaStatistics(chromagram)["mean"]
	if len(meanChroma) == 0 {
		return "C", "major"
	}

	// Key profiles for major and minor scales (simplified)
	majorProfile := []float64{1.0, 0.2, 0.6, 0.2, 0.8, 0.6, 0.2, 1.0, 0.2, 0.6, 0.2, 0.4}
	minorProfile := []float64{1.0, 0.2, 0.4, 0.6, 0.2, 0.8, 0.2, 0.6, 0.8, 0.2, 0.4, 0.2}

	chromaLabels := cs.GetChromaLabels()
	bestKey := "C"
	bestMode := "major"
	bestCorr := -1.0

	// Test all 12 keys for both major and minor
	for root := range 12 {
		// Test major
		majorCorr := cs.calculateProfileCorrelation(meanChroma, majorProfile, root)
		if majorCorr > bestCorr {
			bestCorr = majorCorr
			bestKey = chromaLabels[root]
			bestMode = "major"
		}

		// Test minor
		minorCorr := cs.calculateProfileCorrelation(meanChroma, minorProfile, root)
		if minorCorr > bestCorr {
			bestCorr = minorCorr
			bestKey = chromaLabels[root]
			bestMode = "minor"
		}
	}

	return bestKey, bestMode
}

// calculateProfileCorrelation calculates correlation between chroma and key profile
func (cs *ChromaSTFT) calculateProfileCorrelation(chroma, profile []float64, rootOffset int) float64 {
	if len(chroma) != len(profile) {
		return 0.0
	}

	// Shift profile to match root
	shiftedProfile := make([]float64, len(profile))
	for i := range profile {
		shiftedProfile[i] = profile[(i+rootOffset)%len(profile)]
	}

	// Calculate Pearson correlation
	return cs.pearsonCorrelation(chroma, shiftedProfile)
}

// pearsonCorrelation calculates Pearson correlation coefficient
func (cs *ChromaSTFT) pearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	n := float64(len(x))
	sumX, sumY, sumXY, sumXX, sumYY := 0.0, 0.0, 0.0, 0.0, 0.0

	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumXX += x[i] * x[i]
		sumYY += y[i] * y[i]
	}

	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumXX - sumX*sumX) * (n*sumYY - sumY*sumY))

	if denominator < 1e-10 {
		return 0.0
	}

	return numerator / denominator
}

// SetTuning updates the tuning frequency (A4)
func (cs *ChromaSTFT) SetTuning(tuningFreq float64) {
	cs.tuningFreq = tuningFreq
}

// GetTuning returns the current tuning frequency
func (cs *ChromaSTFT) GetTuning() float64 {
	return cs.tuningFreq
}
