package analyzers

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
)

// SpectralAnalyzer provides core FFT and spectral analysis functionality
type SpectralAnalyzer struct {
	windowGenerator *WindowGenerator
	sampleRate      int
	logger          logging.Logger
}

// SpectrogramResult holds the result of STFT analysis
type SpectrogramResult struct {
	Magnitude      [][]float64    `json:"magnitude"`       // Time x Frequency magnitude matrix
	Phase          [][]float64    `json:"phase"`           // Time x Frequency phase matrix
	Complex        [][]complex128 `json:"-"`               // Raw complex spectrogram (not serialized)
	TimeFrames     int            `json:"time_frames"`     // Number of time frames
	FreqBins       int            `json:"freq_bins"`       // Number of frequency bins
	SampleRate     int            `json:"sample_rate"`     // Sample rate
	WindowSize     int            `json:"window_size"`     // FFT window size
	HopSize        int            `json:"hop_size"`        // Hop size between frames
	FreqResolution float64        `json:"freq_resolution"` // Frequency resolution (Hz/bin)
	TimeResolution float64        `json:"time_resolution"` // Time resolution (seconds/frame)
}

// FrequencyDomainFeatures holds basic frequency domain characteristics
type FrequencyDomainFeatures struct {
	SpectralCentroid  float64 `json:"spectral_centroid"`
	SpectralRolloff   float64 `json:"spectral_rolloff"`
	SpectralBandwidth float64 `json:"spectral_bandwidth"`
	SpectralFlatness  float64 `json:"spectral_flatness"`
	SpectralCrest     float64 `json:"spectral_crest"`
	SpectralSlope     float64 `json:"spectral_slope"`
	SpectralKurtosis  float64 `json:"spectral_kurtosis"`
	SpectralSkewness  float64 `json:"spectral_skewness"`
	Energy            float64 `json:"energy"`
	ZeroCrossingRate  float64 `json:"zero_crossing_rate"`
}

// NewSpectralAnalyzer creates a new spectral analyzer
func NewSpectralAnalyzer(sampleRate int) *SpectralAnalyzer {
	return &SpectralAnalyzer{
		windowGenerator: NewWindowGenerator(),
		sampleRate:      sampleRate,
		logger: logging.WithFields(logging.Fields{
			"component":   "spectral_analyzer",
			"sample_rate": sampleRate,
		}),
	}
}

// ComputeFFT computes FFT and returns a SpectrogramResult with both complex and magnitude/phase data
// This is a single-frame "spectrogram" - useful for simple FFT analysis
func (sa *SpectralAnalyzer) ComputeFFT(signal []float64) (*SpectrogramResult, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal")
	}

	logger := sa.logger.WithFields(logging.Fields{
		"function":      "ComputeFFT",
		"signal_length": len(signal),
	})

	logger.Debug("Computing FFT")

	// Compute FFT
	fftResult := sa.FFT(signal)

	// Only keep positive frequencies (including DC and Nyquist)
	freqBins := len(fftResult)/2 + 1
	freqBins = min(len(fftResult), freqBins)

	// Create single-frame result matrices
	magnitude := make([][]float64, 1)
	phase := make([][]float64, 1)
	complexSpectrum := make([][]complex128, 1)

	magnitude[0] = make([]float64, freqBins)
	phase[0] = make([]float64, freqBins)
	complexSpectrum[0] = make([]complex128, freqBins)

	// Extract magnitude and phase for positive frequencies
	for i := 0; i < freqBins; i++ {
		complexSpectrum[0][i] = fftResult[i]
		magnitude[0][i] = cmplx.Abs(fftResult[i])
		phase[0][i] = cmplx.Phase(fftResult[i])
	}

	result := &SpectrogramResult{
		Magnitude:      magnitude,
		Phase:          phase,
		Complex:        complexSpectrum,
		TimeFrames:     1, // Single frame
		FreqBins:       freqBins,
		SampleRate:     sa.sampleRate,
		WindowSize:     len(signal),                                   // Original signal length
		HopSize:        len(signal),                                   // No overlap for single frame
		FreqResolution: float64(sa.sampleRate) / float64(len(signal)), // Frequency resolution
		TimeResolution: float64(len(signal)) / float64(sa.sampleRate), // Duration of the signal
	}

	logger.Debug("FFT computation completed", logging.Fields{
		"freq_bins":       result.FreqBins,
		"freq_resolution": result.FreqResolution,
		"signal_duration": result.TimeResolution,
	})

	return result, nil
}

// FFT computes Fast Fourier Transform using mjibson/go-dsp
// Takes []float64 input and returns []complex128 output - perfect for your fingerprinting library!
func (sa *SpectralAnalyzer) FFT(x []float64) []complex128 {
	if len(x) == 0 {
		return []complex128{}
	}

	// mjibson/go-dsp handles all sizes efficiently, including non-power-of-2
	return fft.FFTReal(x)
}

// ComputePowerSpectrum computes power spectral density
func (sa *SpectralAnalyzer) ComputePowerSpectrum(spectrogram *SpectrogramResult) [][]float64 {
	power := make([][]float64, spectrogram.TimeFrames)

	for t := 0; t < spectrogram.TimeFrames; t++ {
		power[t] = make([]float64, spectrogram.FreqBins)
		for f := 0; f < spectrogram.FreqBins; f++ {
			mag := spectrogram.Magnitude[t][f]
			power[t][f] = mag * mag
		}
	}

	return power
}

// ComputeLogPowerSpectrum computes log power spectrum in dB
func (sa *SpectralAnalyzer) ComputeLogPowerSpectrum(spectrogram *SpectrogramResult, floorDB float64) [][]float64 {
	logPower := make([][]float64, spectrogram.TimeFrames)
	floor := math.Pow(10, floorDB/10.0)

	for t := 0; t < spectrogram.TimeFrames; t++ {
		logPower[t] = make([]float64, spectrogram.FreqBins)
		for f := 0; f < spectrogram.FreqBins; f++ {
			mag := spectrogram.Magnitude[t][f]
			power := mag * mag
			if power < floor {
				power = floor
			}
			logPower[t][f] = 10 * math.Log10(power)
		}
	}

	return logPower
}

// ExtractFrameFeatures extracts frequency domain features from a single spectrum frame
func (sa *SpectralAnalyzer) ExtractFrameFeatures(magnitudeSpectrum []float64) *FrequencyDomainFeatures {
	features := &FrequencyDomainFeatures{}

	if len(magnitudeSpectrum) == 0 {
		return features
	}

	// Generate frequency bins
	freqs := sa.GetFrequencyBins(len(magnitudeSpectrum))

	// Spectral Centroid (center of mass)
	features.SpectralCentroid = sa.calculateSpectralCentroid(magnitudeSpectrum, freqs)

	// Spectral Rolloff (85th percentile frequency)
	features.SpectralRolloff = sa.calculateSpectralRolloff(magnitudeSpectrum, freqs, 0.85)

	// Spectral Bandwidth (second moment around centroid)
	features.SpectralBandwidth = sa.calculateSpectralBandwidth(magnitudeSpectrum, freqs, features.SpectralCentroid)

	// Spectral Flatness (geometric mean / arithmetic mean)
	features.SpectralFlatness = sa.calculateSpectralFlatness(magnitudeSpectrum)

	// Spectral Crest (peak / RMS ratio)
	features.SpectralCrest = sa.calculateSpectralCrest(magnitudeSpectrum)

	// Spectral Slope (linear regression slope)
	features.SpectralSlope = sa.calculateSpectralSlope(magnitudeSpectrum, freqs)

	// Higher order moments
	features.SpectralKurtosis = sa.calculateSpectralKurtosis(magnitudeSpectrum, freqs, features.SpectralCentroid)
	features.SpectralSkewness = sa.calculateSpectralSkewness(magnitudeSpectrum, freqs, features.SpectralCentroid)

	// Energy
	features.Energy = sa.calculateEnergy(magnitudeSpectrum)

	return features
}

// GetFrequencyBins returns frequency values for each FFT bin
func (sa *SpectralAnalyzer) GetFrequencyBins(numBins int) []float64 {
	freqs := make([]float64, numBins)
	for i := range numBins {
		freqs[i] = float64(i) * float64(sa.sampleRate) / float64((numBins-1)*2)
	}
	return freqs
}

// calculateSpectralCentroid computes spectral centroid
func (sa *SpectralAnalyzer) calculateSpectralCentroid(spectrum []float64, freqs []float64) float64 {
	if len(spectrum) != len(freqs) {
		return 0
	}

	numerator := 0.0
	denominator := 0.0

	for i := range len(spectrum) {
		numerator += freqs[i] * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// calculateSpectralRolloff computes spectral rolloff frequency
func (sa *SpectralAnalyzer) calculateSpectralRolloff(spectrum []float64, freqs []float64, threshold float64) float64 {
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
			if i < len(freqs) {
				return freqs[i]
			}
			break
		}
	}

	if len(freqs) > 0 {
		return freqs[len(freqs)-1]
	}
	return 0
}

// calculateSpectralBandwidth computes spectral bandwidth
func (sa *SpectralAnalyzer) calculateSpectralBandwidth(spectrum []float64, freqs []float64, centroid float64) float64 {
	if len(spectrum) != len(freqs) {
		return 0
	}

	numerator := 0.0
	denominator := 0.0

	for i := range len(spectrum) {
		diff := freqs[i] - centroid
		numerator += diff * diff * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	return math.Sqrt(numerator / denominator)
}

// calculateSpectralFlatness computes spectral flatness (Wiener entropy)
func (sa *SpectralAnalyzer) calculateSpectralFlatness(spectrum []float64) float64 {
	if len(spectrum) == 0 {
		return 0
	}

	// Geometric mean
	logSum := 0.0
	count := 0

	for _, mag := range spectrum {
		if mag > 1e-10 { // Avoid log(0)
			logSum += math.Log(mag)
			count++
		}
	}

	if count == 0 {
		return 0
	}

	geometricMean := math.Exp(logSum / float64(count))

	// Arithmetic mean
	arithmeticMean := 0.0
	for _, mag := range spectrum {
		arithmeticMean += mag
	}
	arithmeticMean /= float64(len(spectrum))

	if arithmeticMean == 0 {
		return 0
	}

	return geometricMean / arithmeticMean
}

// calculateSpectralCrest computes spectral crest factor
func (sa *SpectralAnalyzer) calculateSpectralCrest(spectrum []float64) float64 {
	if len(spectrum) == 0 {
		return 0
	}

	maxVal := 0.0
	sumSquares := 0.0

	for _, mag := range spectrum {
		if mag > maxVal {
			maxVal = mag
		}
		sumSquares += mag * mag
	}

	rms := math.Sqrt(sumSquares / float64(len(spectrum)))

	if rms == 0 {
		return 0
	}

	return maxVal / rms
}

// calculateSpectralSlope computes spectral slope via linear regression
func (sa *SpectralAnalyzer) calculateSpectralSlope(spectrum []float64, freqs []float64) float64 {
	if len(spectrum) != len(freqs) || len(spectrum) < 2 {
		return 0
	}

	// Convert to log domain for linear regression
	n := 0
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i := range len(spectrum) {
		if spectrum[i] > 1e-10 && freqs[i] > 0 {
			x := math.Log10(freqs[i])
			y := math.Log10(spectrum[i])

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
			n++
		}
	}

	if n < 2 {
		return 0
	}

	// Linear regression slope
	denominator := float64(n)*sumXX - sumX*sumX
	if denominator == 0 {
		return 0
	}

	slope := (float64(n)*sumXY - sumX*sumY) / denominator
	return slope
}

// calculateSpectralKurtosis computes spectral kurtosis
func (sa *SpectralAnalyzer) calculateSpectralKurtosis(spectrum []float64, freqs []float64, centroid float64) float64 {
	if len(spectrum) != len(freqs) || len(spectrum) < 2 {
		return 0
	}

	// Calculate fourth moment
	numerator := 0.0
	denominator := 0.0
	variance := 0.0

	// First calculate variance
	for i := range len(spectrum) {
		diff := freqs[i] - centroid
		variance += diff * diff * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	variance /= denominator
	if variance == 0 {
		return 0
	}

	// Calculate fourth moment
	for i := range len(spectrum) {
		diff := freqs[i] - centroid
		numerator += math.Pow(diff, 4) * spectrum[i]
	}

	kurtosis := (numerator / denominator) / (variance * variance)
	return kurtosis - 3.0 // Excess kurtosis (subtract 3 for normal distribution)
}

// calculateSpectralSkewness computes spectral skewness
func (sa *SpectralAnalyzer) calculateSpectralSkewness(spectrum []float64, freqs []float64, centroid float64) float64 {
	if len(spectrum) != len(freqs) || len(spectrum) < 2 {
		return 0
	}

	// Calculate third moment and standard deviation
	numerator := 0.0
	denominator := 0.0
	variance := 0.0

	// Calculate variance first
	for i := range len(spectrum) {
		diff := freqs[i] - centroid
		variance += diff * diff * spectrum[i]
		denominator += spectrum[i]
	}

	if denominator == 0 {
		return 0
	}

	variance /= denominator
	if variance == 0 {
		return 0
	}

	stdDev := math.Sqrt(variance)

	// Calculate third moment
	for i := range len(spectrum) {
		diff := freqs[i] - centroid
		numerator += math.Pow(diff, 3) * spectrum[i]
	}

	skewness := (numerator / denominator) / math.Pow(stdDev, 3)
	return skewness
}

// calculateEnergy computes total energy
func (sa *SpectralAnalyzer) calculateEnergy(spectrum []float64) float64 {
	energy := 0.0
	for _, mag := range spectrum {
		energy += mag * mag
	}
	return energy
}

// GetSpectrogramSlice extracts a frequency slice across all time frames
func (sa *SpectralAnalyzer) GetSpectrogramSlice(spectrogram *SpectrogramResult, freqBin int) []float64 {
	if freqBin < 0 || freqBin >= spectrogram.FreqBins {
		return nil
	}

	slice := make([]float64, spectrogram.TimeFrames)
	for t := 0; t < spectrogram.TimeFrames; t++ {
		slice[t] = spectrogram.Magnitude[t][freqBin]
	}

	return slice
}

// ComputeSpectralFlux computes spectral flux (measure of spectral change)
func (sa *SpectralAnalyzer) ComputeSpectralFlux(spectrogram *SpectrogramResult) []float64 {
	if spectrogram.TimeFrames < 2 {
		return nil
	}

	flux := make([]float64, spectrogram.TimeFrames-1)

	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		for f := 0; f < spectrogram.FreqBins; f++ {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			if diff > 0 { // Only positive changes (energy increases)
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

