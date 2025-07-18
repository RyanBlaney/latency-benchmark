package spectral

// SpectralCentroid computes the spectral centroid (center of mass) of a spectrum
// Extracted from your existing working implementation
type SpectralCentroid struct {
	sampleRate  int
	freqBins    []float64 // Pre-calculated frequency bins for efficiency
	initialized bool
}

// NewSpectralCentroid creates a new spectral centroid calculator
func NewSpectralCentroid(sampleRate int) *SpectralCentroid {
	return &SpectralCentroid{
		sampleRate: sampleRate,
	}
}

// Compute calculates spectral centroid for a single magnitude spectrum
// This is your existing working implementation
func (sc *SpectralCentroid) Compute(spectrum []float64) float64 {
	if len(spectrum) == 0 {
		return 0.0
	}

	// Initialize frequency bins if needed (matches your GetFrequencyBins)
	if !sc.initialized || len(sc.freqBins) != len(spectrum) {
		sc.initializeFreqBins(len(spectrum))
	}

	// Your existing implementation from calculateSpectralCentroid
	numerator := 0.0
	denominator := 0.0

	for i := range len(spectrum) {
		numerator += sc.freqBins[i] * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// ComputeFrames processes multiple frames efficiently
func (sc *SpectralCentroid) ComputeFrames(spectrogram [][]float64) []float64 {
	if len(spectrogram) == 0 {
		return []float64{}
	}

	centroids := make([]float64, len(spectrogram))

	for t, spectrum := range spectrogram {
		centroids[t] = sc.Compute(spectrum)
	}

	return centroids
}

// initializeFreqBins pre-calculates frequency bins (matches your existing GetFrequencyBins)
func (sc *SpectralCentroid) initializeFreqBins(numBins int) {
	sc.freqBins = make([]float64, numBins)
	for i := range numBins {
		sc.freqBins[i] = float64(i) * float64(sc.sampleRate) / float64((numBins-1)*2)
	}
	sc.initialized = true
}

// GetFrequencyBins returns the frequency bins used for calculation
func (sc *SpectralCentroid) GetFrequencyBins() []float64 {
	if !sc.initialized {
		return nil
	}

	// Return copy to prevent modification
	bins := make([]float64, len(sc.freqBins))
	copy(bins, sc.freqBins)
	return bins
}
