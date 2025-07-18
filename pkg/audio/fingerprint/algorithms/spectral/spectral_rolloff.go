package spectral

// SpectralRolloff computes spectral rolloff frequency
// Extracted from your existing working implementation
type SpectralRolloff struct {
	sampleRate  int
	freqBins    []float64 // Pre-calculated frequency bins
	initialized bool
}

// NewSpectralRolloff creates a new spectral rolloff calculator
func NewSpectralRolloff(sampleRate int) *SpectralRolloff {
	return &SpectralRolloff{
		sampleRate: sampleRate,
	}
}

// Compute calculates spectral rolloff for a single magnitude spectrum
// threshold: typically 0.85 for 85th percentile
func (sr *SpectralRolloff) Compute(spectrum []float64, threshold float64) float64 {
	if len(spectrum) == 0 {
		return 0.0
	}

	// Initialize frequency bins if needed
	if !sr.initialized || len(sr.freqBins) != len(spectrum) {
		sr.initializeFreqBins(len(spectrum))
	}

	// This is your existing working implementation
	totalEnergy := 0.0
	for _, mag := range spectrum {
		totalEnergy += mag * mag
	}

	if totalEnergy == 0 {
		return 0
	}

	targetEnergy := threshold * totalEnergy
	cumulativeEnergy := 0.0

	for i := range len(spectrum) {
		cumulativeEnergy += spectrum[i] * spectrum[i]
		if cumulativeEnergy >= targetEnergy {
			if i < len(sr.freqBins) {
				return sr.freqBins[i]
			}
			break
		}
	}

	if len(sr.freqBins) > 0 {
		return sr.freqBins[len(sr.freqBins)-1]
	}
	return 0
}

// ComputeFrames processes multiple frames efficiently
func (sr *SpectralRolloff) ComputeFrames(spectrogram [][]float64, threshold float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	rolloffs := make([]float64, len(spectrogram))

	for t, spectrum := range spectrogram {
		rolloffs[t] = sr.Compute(spectrum, threshold)
	}

	return rolloffs
}

// initializeFreqBins pre-calculates frequency bins (matches your existing GetFrequencyBins)
func (sr *SpectralRolloff) initializeFreqBins(numBins int) {
	sr.freqBins = make([]float64, numBins)
	for i := range numBins {
		sr.freqBins[i] = float64(i) * float64(sr.sampleRate) / float64((numBins-1)*2)
	}
	sr.initialized = true
}
