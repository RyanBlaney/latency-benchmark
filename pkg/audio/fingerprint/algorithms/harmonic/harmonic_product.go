package harmonic

import (
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// HarmonicProduct implements Harmonic Product Spectrum for F0 estimation
type HarmonicProduct struct {
	sampleRate    int
	numHarmonics  int
	minF0         float64
	maxF0         float64
	fft           *spectral.FFT
	powerSpectrum *spectral.PowerSpectrum
}

// NewHarmonicProduct creates a new harmonic product spectrum analyzer
func NewHarmonicProduct(sampleRate int, numHarmonics int, minF0, maxF0 float64) *HarmonicProduct {
	return &HarmonicProduct{
		sampleRate:    sampleRate,
		numHarmonics:  numHarmonics,
		minF0:         minF0,
		maxF0:         maxF0,
		fft:           spectral.NewFFT(),
		powerSpectrum: spectral.NewPowerSpectrum(),
	}
}

// ComputeHPS computes Harmonic Product Spectrum from magnitude spectrum
func (hp *HarmonicProduct) ComputeHPS(magnitudeSpectrum []float64) []float64 {
	if len(magnitudeSpectrum) == 0 {
		return []float64{}
	}

	// Convert to power spectrum for better peak detection
	powerSpec := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		powerSpec[i] = mag * mag
	}

	// Initialize HPS with first harmonic (fundamental)
	hps := make([]float64, len(powerSpec))
	copy(hps, powerSpec)

	// Multiply with downsampled versions (harmonics)
	for harmonic := 2; harmonic <= hp.numHarmonics; harmonic++ {
		downsampled := hp.downsampleSpectrum(powerSpec, harmonic)

		// Multiply element-wise
		for i := 0; i < len(hps) && i < len(downsampled); i++ {
			hps[i] *= downsampled[i]
		}
	}

	return hps
}

// EstimateF0 estimates fundamental frequency using HPS
func (hp *HarmonicProduct) EstimateF0(signal []float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Apply window and compute FFT
	windowedSignal := hp.applyWindow(signal)
	fftResult := hp.fft.Compute(windowedSignal)

	// Convert to magnitude spectrum (positive frequencies only)
	freqBins := len(fftResult)/2 + 1
	magnitudeSpec := make([]float64, freqBins)
	for i := range freqBins {
		magnitudeSpec[i] = math.Sqrt(real(fftResult[i])*real(fftResult[i]) +
			imag(fftResult[i])*imag(fftResult[i]))
	}

	// Compute HPS
	hps := hp.ComputeHPS(magnitudeSpec)

	// Find peak in valid F0 range
	f0BinIdx := hp.findF0Peak(hps, len(signal))

	if f0BinIdx == 0 {
		return 0.0
	}

	// Convert bin index to frequency
	freqResolution := float64(hp.sampleRate) / float64(len(signal))
	return float64(f0BinIdx) * freqResolution
}

// downsampleSpectrum downsamples spectrum by factor for harmonic analysis
func (hp *HarmonicProduct) downsampleSpectrum(spectrum []float64, factor int) []float64 {
	if factor <= 1 {
		// Return copy of original
		downsampled := make([]float64, len(spectrum))
		copy(downsampled, spectrum)
		return downsampled
	}

	// Downsample by taking every factor-th sample
	downsampledLen := len(spectrum) / factor
	if downsampledLen == 0 {
		return []float64{}
	}

	downsampled := make([]float64, len(spectrum))

	// Fill with zeros first
	for i := range downsampled {
		downsampled[i] = 0.0
	}

	// Copy downsampled values to corresponding positions
	for i := range downsampledLen {
		sourceIdx := i * factor
		if sourceIdx < len(spectrum) {
			downsampled[i] = spectrum[sourceIdx]
		}
	}

	return downsampled
}

// findF0Peak finds the fundamental frequency peak in HPS
func (hp *HarmonicProduct) findF0Peak(hps []float64, signalLength int) int {
	freqResolution := float64(hp.sampleRate) / float64(signalLength)

	// Convert F0 range to bin indices
	minBin := int(hp.minF0 / freqResolution)
	maxBin := int(hp.maxF0 / freqResolution)

	if minBin < 1 {
		minBin = 1
	}
	if maxBin >= len(hps) {
		maxBin = len(hps) - 1
	}

	bestBin := 0
	bestValue := 0.0

	// Find maximum in valid range
	for bin := minBin; bin <= maxBin; bin++ {
		if hps[bin] > bestValue {
			bestValue = hps[bin]
			bestBin = bin
		}
	}

	// Verify it's a local maximum
	if bestBin > 0 && bestBin < len(hps)-1 {
		if hps[bestBin] > hps[bestBin-1] && hps[bestBin] > hps[bestBin+1] {
			return bestBin
		}
	}

	return bestBin
}

// ComputeHPSWithInterpolation computes HPS with interpolated harmonics
func (hp *HarmonicProduct) ComputeHPSWithInterpolation(magnitudeSpectrum []float64) []float64 {
	if len(magnitudeSpectrum) == 0 {
		return []float64{}
	}

	// Convert to power spectrum
	powerSpec := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		powerSpec[i] = mag * mag
	}

	// Initialize HPS
	hps := make([]float64, len(powerSpec))
	copy(hps, powerSpec)

	// Apply harmonic multiplication with interpolation
	for harmonic := 2; harmonic <= hp.numHarmonics; harmonic++ {
		// For each bin, find the corresponding harmonic frequency
		for bin := range len(hps) {
			harmonicBin := float64(bin) / float64(harmonic)

			// Interpolate value at fractional bin index
			interpolatedValue := hp.interpolateSpectrum(powerSpec, harmonicBin)
			hps[bin] *= interpolatedValue
		}
	}

	return hps
}

// interpolateSpectrum performs linear interpolation in spectrum
func (hp *HarmonicProduct) interpolateSpectrum(spectrum []float64, index float64) float64 {
	if index < 0 || index >= float64(len(spectrum)-1) {
		return 0.0
	}

	leftBin := int(index)
	rightBin := leftBin + 1
	frac := index - float64(leftBin)

	if rightBin >= len(spectrum) {
		return spectrum[leftBin]
	}

	// Linear interpolation
	return spectrum[leftBin] + frac*(spectrum[rightBin]-spectrum[leftBin])
}

// applyWindow applies Hann window to signal
func (hp *HarmonicProduct) applyWindow(signal []float64) []float64 {
	windowed := make([]float64, len(signal))

	for i, sample := range signal {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(len(signal)-1)))
		windowed[i] = sample * window
	}

	return windowed
}

// ComputeHarmonicStrength calculates strength of harmonic components
func (hp *HarmonicProduct) ComputeHarmonicStrength(magnitudeSpectrum []float64, f0Freq float64, signalLength int) []float64 {
	if len(magnitudeSpectrum) == 0 || f0Freq <= 0 {
		return []float64{}
	}

	freqResolution := float64(hp.sampleRate) / float64(signalLength)
	strengths := make([]float64, hp.numHarmonics)

	for harmonic := 1; harmonic <= hp.numHarmonics; harmonic++ {
		harmonicFreq := f0Freq * float64(harmonic)
		harmonicBin := harmonicFreq / freqResolution

		if harmonicBin >= float64(len(magnitudeSpectrum)) {
			strengths[harmonic-1] = 0.0
			continue
		}

		// Get magnitude at harmonic frequency (with interpolation)
		magnitude := hp.interpolateSpectrum(magnitudeSpectrum, harmonicBin)
		strengths[harmonic-1] = magnitude
	}

	return strengths
}

// ComputeHarmonicity calculates harmonicity measure
func (hp *HarmonicProduct) ComputeHarmonicity(magnitudeSpectrum []float64, f0Freq float64, signalLength int) float64 {
	strengths := hp.ComputeHarmonicStrength(magnitudeSpectrum, f0Freq, signalLength)

	if len(strengths) == 0 {
		return 0.0
	}

	// Calculate ratio of harmonic energy to total energy
	harmonicEnergy := 0.0
	for _, strength := range strengths {
		harmonicEnergy += strength * strength
	}

	totalEnergy := 0.0
	for _, mag := range magnitudeSpectrum {
		totalEnergy += mag * mag
	}

	if totalEnergy == 0 {
		return 0.0
	}

	return harmonicEnergy / totalEnergy
}

// EstimateF0WithConfidence estimates F0 and returns confidence measure
func (hp *HarmonicProduct) EstimateF0WithConfidence(signal []float64) (float64, float64) {
	f0 := hp.EstimateF0(signal)

	if f0 == 0 {
		return 0.0, 0.0
	}

	// Compute magnitude spectrum for confidence calculation
	windowedSignal := hp.applyWindow(signal)
	fftResult := hp.fft.Compute(windowedSignal)

	freqBins := len(fftResult)/2 + 1
	magnitudeSpec := make([]float64, freqBins)
	for i := range freqBins {
		magnitudeSpec[i] = math.Sqrt(real(fftResult[i])*real(fftResult[i]) +
			imag(fftResult[i])*imag(fftResult[i]))
	}

	// Use harmonicity as confidence measure
	confidence := hp.ComputeHarmonicity(magnitudeSpec, f0, len(signal))

	return f0, confidence
}

// GetOptimalNumHarmonics returns recommended number of harmonics for given F0 range
func (hp *HarmonicProduct) GetOptimalNumHarmonics() int {
	// Calculate how many harmonics fit in the frequency range
	nyquist := float64(hp.sampleRate) / 2.0
	maxHarmonics := int(nyquist / hp.minF0)

	// Typical range is 3-7 harmonics for speech analysis
	if maxHarmonics > 7 {
		return 5 // Good balance of accuracy and computation
	} else if maxHarmonics > 3 {
		return maxHarmonics - 1
	} else {
		return 2 // Minimum useful number
	}
}
