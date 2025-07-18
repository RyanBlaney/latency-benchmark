package spectral

import (
	"math"
)

// SpectralContrast computes spectral contrast features
// Measures the difference between peaks and valleys in spectrum
type SpectralContrast struct {
	sampleRate  int
	numBands    int
	freqBins    []float64
	bandEdges   []int
	initialized bool
}

// NewSpectralContrast creates a new spectral contrast calculator
func NewSpectralContrast(sampleRate int, numBands int) *SpectralContrast {
	return &SpectralContrast{
		sampleRate: sampleRate,
		numBands:   numBands,
	}
}

// Compute calculates spectral contrast for a single magnitude spectrum
func (sc *SpectralContrast) Compute(magnitudeSpectrum []float64) []float64 {
	if len(magnitudeSpectrum) == 0 {
		return []float64{}
	}

	// Initialize frequency bands if needed
	if !sc.initialized || len(sc.freqBins) != len(magnitudeSpectrum) {
		sc.initializeBands(len(magnitudeSpectrum))
	}

	contrast := make([]float64, sc.numBands)

	for band := 0; band < sc.numBands; band++ {
		startBin := sc.bandEdges[band]
		endBin := sc.bandEdges[band+1]
		endBin = min(endBin, len(magnitudeSpectrum))

		if startBin >= endBin {
			contrast[band] = 0.0
			continue
		}

		// Extract band spectrum
		bandSpectrum := magnitudeSpectrum[startBin:endBin]
		contrast[band] = sc.calculateBandContrast(bandSpectrum)
	}

	return contrast
}

// ComputeFrames processes multiple frames efficiently
func (sc *SpectralContrast) ComputeFrames(spectrogram [][]float64) [][]float64 {
	if len(spectrogram) == 0 {
		return [][]float64{}
	}

	contrasts := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		contrasts[t] = sc.Compute(magnitudeSpectrum)
	}

	return contrasts
}

// calculateBandContrast calculates contrast for a frequency band
func (sc *SpectralContrast) calculateBandContrast(bandSpectrum []float64) float64 {
	if len(bandSpectrum) == 0 {
		return 0.0
	}

	// Convert to power spectrum
	powerSpectrum := make([]float64, len(bandSpectrum))
	for i, mag := range bandSpectrum {
		powerSpectrum[i] = mag * mag
	}

	// Sort to find peaks and valleys
	sortedPower := make([]float64, len(powerSpectrum))
	copy(sortedPower, powerSpectrum)

	// Simple insertion sort
	for i := 1; i < len(sortedPower); i++ {
		key := sortedPower[i]
		j := i - 1
		for j >= 0 && sortedPower[j] > key {
			sortedPower[j+1] = sortedPower[j]
			j--
		}
		sortedPower[j+1] = key
	}

	// Calculate valley and peak energies
	valleyRatio := 0.2 // Bottom 20% for valleys
	peakRatio := 0.2   // Top 20% for peaks

	valleyCount := int(valleyRatio * float64(len(sortedPower)))
	peakCount := int(peakRatio * float64(len(sortedPower)))

	if valleyCount == 0 {
		valleyCount = 1
	}
	if peakCount == 0 {
		peakCount = 1
	}

	// Average valley energy (bottom percentile)
	valleyEnergy := 0.0
	for i := 0; i < valleyCount; i++ {
		valleyEnergy += sortedPower[i]
	}
	valleyEnergy /= float64(valleyCount)

	// Average peak energy (top percentile)
	peakEnergy := 0.0
	startIdx := len(sortedPower) - peakCount
	for i := startIdx; i < len(sortedPower); i++ {
		peakEnergy += sortedPower[i]
	}
	peakEnergy /= float64(peakCount)

	// Calculate contrast in dB
	if valleyEnergy <= 0 {
		valleyEnergy = 1e-10 // Avoid log(0)
	}
	if peakEnergy <= 0 {
		return 0.0
	}

	contrast := 10.0 * math.Log10(peakEnergy/valleyEnergy)
	return contrast
}

// initializeBands creates frequency band boundaries
func (sc *SpectralContrast) initializeBands(numBins int) {
	sc.freqBins = make([]float64, numBins)
	sc.bandEdges = make([]int, sc.numBands+1)

	// Calculate frequency bins
	nyquist := float64(sc.sampleRate) / 2.0
	for i := range numBins {
		sc.freqBins[i] = float64(i) * nyquist / float64(numBins-1)
	}

	// Create logarithmically spaced frequency bands
	minFreq := 200.0   // Start at 200 Hz
	maxFreq := nyquist // End at Nyquist

	if maxFreq <= minFreq {
		maxFreq = minFreq * 2
	}

	logMinFreq := math.Log10(minFreq)
	logMaxFreq := math.Log10(maxFreq)
	logStep := (logMaxFreq - logMinFreq) / float64(sc.numBands)

	// Convert frequency boundaries to bin indices
	for i := 0; i <= sc.numBands; i++ {
		logFreq := logMinFreq + float64(i)*logStep
		freq := math.Pow(10.0, logFreq)

		// Find closest bin
		binIdx := int(freq * float64(numBins-1) / nyquist)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		if binIdx < 0 {
			binIdx = 0
		}

		sc.bandEdges[i] = binIdx
	}

	// Ensure monotonic increasing band edges
	for i := 1; i <= sc.numBands; i++ {
		if sc.bandEdges[i] <= sc.bandEdges[i-1] {
			sc.bandEdges[i] = sc.bandEdges[i-1] + 1
		}
	}

	sc.initialized = true
}

// ComputeWithCustomBands calculates spectral contrast with custom frequency bands
func (sc *SpectralContrast) ComputeWithCustomBands(magnitudeSpectrum []float64, bandFreqs []float64) []float64 {
	if len(magnitudeSpectrum) == 0 || len(bandFreqs) < 2 {
		return []float64{}
	}

	// Initialize frequency bins if needed
	if !sc.initialized || len(sc.freqBins) != len(magnitudeSpectrum) {
		sc.freqBins = make([]float64, len(magnitudeSpectrum))
		nyquist := float64(sc.sampleRate) / 2.0
		for i := range len(magnitudeSpectrum) {
			sc.freqBins[i] = float64(i) * nyquist / float64(len(magnitudeSpectrum)-1)
		}
	}

	numBands := len(bandFreqs) - 1
	contrast := make([]float64, numBands)

	for band := range numBands {
		startFreq := bandFreqs[band]
		endFreq := bandFreqs[band+1]

		// Convert frequencies to bin indices
		startBin := sc.freqToBin(startFreq, len(magnitudeSpectrum))
		endBin := sc.freqToBin(endFreq, len(magnitudeSpectrum))
		endBin = min(endBin, len(magnitudeSpectrum))

		if startBin >= endBin {
			contrast[band] = 0.0
			continue
		}

		// Extract band spectrum
		bandSpectrum := magnitudeSpectrum[startBin:endBin]
		contrast[band] = sc.calculateBandContrast(bandSpectrum)
	}

	return contrast
}

// freqToBin converts frequency to bin index
func (sc *SpectralContrast) freqToBin(freq float64, numBins int) int {
	nyquist := float64(sc.sampleRate) / 2.0
	binIdx := int(freq * float64(numBins-1) / nyquist)
	binIdx = max(binIdx, 0)

	if binIdx >= numBins {
		binIdx = numBins - 1
	}

	return binIdx
}

// GetBandFrequencies returns the frequency boundaries of the bands
func (sc *SpectralContrast) GetBandFrequencies() []float64 {
	if !sc.initialized {
		return nil
	}

	freqs := make([]float64, len(sc.bandEdges))
	for i, binIdx := range sc.bandEdges {
		if binIdx < len(sc.freqBins) {
			freqs[i] = sc.freqBins[binIdx]
		}
	}

	return freqs
}

// ComputeMean calculates mean spectral contrast across frames
func (sc *SpectralContrast) ComputeMean(contrasts [][]float64) []float64 {
	if len(contrasts) == 0 {
		return []float64{}
	}

	numBands := len(contrasts[0])
	meanContrast := make([]float64, numBands)

	for band := range numBands {
		sum := 0.0
		count := 0

		for t := range contrasts {
			if band < len(contrasts[t]) {
				sum += contrasts[t][band]
				count++
			}
		}

		if count > 0 {
			meanContrast[band] = sum / float64(count)
		}
	}

	return meanContrast
}

// ComputeVariance calculates variance of spectral contrast across frames
func (sc *SpectralContrast) ComputeVariance(contrasts [][]float64) []float64 {
	if len(contrasts) == 0 {
		return []float64{}
	}

	meanContrast := sc.ComputeMean(contrasts)
	numBands := len(meanContrast)
	variance := make([]float64, numBands)

	for band := range numBands {
		sumSquaredDiff := 0.0
		count := 0

		for t := range contrasts {
			if band < len(contrasts[t]) {
				diff := contrasts[t][band] - meanContrast[band]
				sumSquaredDiff += diff * diff
				count++
			}
		}

		if count > 1 {
			variance[band] = sumSquaredDiff / float64(count-1)
		}
	}

	return variance
}
