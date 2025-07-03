package analyzers

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
)

// SpectralAnalyzer provides core FFT and spectral analysis functionality
type SpectralAnalyzer struct {
	windowGenerator *WindowGenerator
	sampleRate      int
	logger          logging.Logger
}

// STFTConfig holds configuration for Short-Time Fourier Transform
type STFTConfig struct {
	WindowSize   int           `json:"window_size"`
	HopSize      int           `json:"hop_size"`
	WindowType   WindowType    `json:"window_type"`
	WindowConfig *WindowConfig `json:"window_config,omitempty"`
	OverlapRatio float64       `json:"overlap_ratio"` // Alternative to HopSize
	ZeroPadding  int           `json:"zero_padding"`  // Additional zero padding
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

// DefaultSTFTConfig returns default STFT configuration
func DefaultSTFTConfig() *STFTConfig {
	return &STFTConfig{
		WindowSize:   2048,
		HopSize:      512, // 75% overlap
		WindowType:   WindowHann,
		OverlapRatio: 0.75,
		ZeroPadding:  0,
	}
}

// ContentOptimizedSTFTConfig returns STFT config optimized for content type
func ContentOptimizedSTFTConfig(contentType config.ContentType) *STFTConfig {
	stftConfig := DefaultSTFTConfig()

	switch contentType {
	case config.ContentMusic:
		stftConfig.WindowSize = 2048
		stftConfig.HopSize = 512
		stftConfig.WindowType = WindowBlackman // Better for harmonic content

	case config.ContentNews, config.ContentTalk:
		stftConfig.WindowSize = 1024 // Shorter window for speech
		stftConfig.HopSize = 256
		stftConfig.WindowType = WindowHamming // Standard for speech

	case config.ContentSports:
		stftConfig.WindowSize = 1024 // Shorter for dynamic content
		stftConfig.HopSize = 256
		stftConfig.WindowType = WindowHann // General purpose

	case config.ContentMixed, config.ContentUnknown:
		// Use defaults - balanced approach
		stftConfig.WindowType = WindowHann
	}

	return stftConfig
}

// STFT computes Short-Time Fourier Transform
func (sa *SpectralAnalyzer) STFT(signal []float64, config *STFTConfig) (*SpectrogramResult, error) {
	if config == nil {
		config = DefaultSTFTConfig()
	}

	logger := sa.logger.WithFields(logging.Fields{
		"function":      "STFT",
		"signal_length": len(signal),
		"window_size":   config.WindowSize,
		"hop_size":      config.HopSize,
		"window_type":   config.WindowType,
	})

	logger.Debug("Starting STFT computation")

	if len(signal) < config.WindowSize {
		return nil, fmt.Errorf("signal length (%d) is shorter than window size (%d)",
			len(signal), config.WindowSize)
	}

	// Generate window function
	windowConfig := &WindowConfig{
		Type:      config.WindowType,
		Size:      config.WindowSize,
		Normalize: true,
		Symmetric: true,
	}
	if config.WindowConfig != nil {
		windowConfig = config.WindowConfig
	}

	window, err := sa.windowGenerator.Generate(windowConfig)
	if err != nil {
		logger.Error(err, "Failed to generate window function")
		return nil, fmt.Errorf("failed to generate window: %w", err)
	}

	// Calculate number of frames
	hopSize := config.HopSize
	if hopSize == 0 {
		hopSize = int(float64(config.WindowSize) * (1.0 - config.OverlapRatio))
	}

	numFrames := (len(signal)-config.WindowSize)/hopSize + 1
	if numFrames <= 0 {
		return nil, fmt.Errorf("insufficient signal length for given hop size")
	}

	logger.Debug("STFT parameters calculated", logging.Fields{
		"num_frames":         numFrames,
		"effective_hop_size": hopSize,
		"overlap_ratio":      1.0 - float64(hopSize)/float64(config.WindowSize),
	})

	// Prepare FFT size (with zero padding)
	fftSize := config.WindowSize + config.ZeroPadding
	if fftSize&(fftSize-1) != 0 {
		// Round up to next power of 2 for efficient FFT
		nextPow2 := 1
		for nextPow2 < fftSize {
			nextPow2 <<= 1
		}
		fftSize = nextPow2
	}

	freqBins := fftSize/2 + 1 // Only positive frequencies

	// Allocate result matrices
	magnitude := make([][]float64, numFrames)
	phase := make([][]float64, numFrames)
	complex_spectrogram := make([][]complex128, numFrames)

	// Process each frame
	for frame := range numFrames {
		start := frame * hopSize
		end := start + config.WindowSize

		if end > len(signal) {
			break
		}

		// Extract and window signal frame
		frameSignal := signal[start:end]
		windowedFrame, err := window.Apply(frameSignal)
		if err != nil {
			return nil, fmt.Errorf("failed to apply window to frame %d: %w", frame, err)
		}

		// Zero-pad if necessary
		if fftSize > config.WindowSize {
			paddedFrame := make([]float64, fftSize)
			copy(paddedFrame, windowedFrame)
			windowedFrame = paddedFrame
		}

		// Convert to complex for FFT
		complexFrame := make([]complex128, len(windowedFrame))
		for i, val := range windowedFrame {
			complexFrame[i] = complex(val, 0)
		}

		// Compute FFT
		fft := sa.FFT(complexFrame)

		// Store results (only positive frequencies)
		magnitude[frame] = make([]float64, freqBins)
		phase[frame] = make([]float64, freqBins)
		complex_spectrogram[frame] = make([]complex128, freqBins)

		for i := range freqBins {
			complex_spectrogram[frame][i] = fft[i]
			magnitude[frame][i] = cmplx.Abs(fft[i])
			phase[frame][i] = cmplx.Phase(fft[i])
		}
	}

	result := &SpectrogramResult{
		Magnitude:      magnitude,
		Phase:          phase,
		Complex:        complex_spectrogram,
		TimeFrames:     numFrames,
		FreqBins:       freqBins,
		SampleRate:     sa.sampleRate,
		WindowSize:     config.WindowSize,
		HopSize:        hopSize,
		FreqResolution: float64(sa.sampleRate) / float64(fftSize),
		TimeResolution: float64(hopSize) / float64(sa.sampleRate),
	}

	logger.Info("STFT computation completed", logging.Fields{
		"time_frames":     result.TimeFrames,
		"freq_bins":       result.FreqBins,
		"freq_resolution": result.FreqResolution,
		"time_resolution": result.TimeResolution,
	})

	return result, nil
}

// FFT computes Fast Fourier Transform using Cooley-Tukey algorithm
func (sa *SpectralAnalyzer) FFT(x []complex128) []complex128 {
	N := len(x)

	// Base case
	if N <= 1 {
		return x
	}

	// Ensure N is power of 2 by zero-padding
	if N&(N-1) != 0 {
		nextPow2 := 1
		for nextPow2 < N {
			nextPow2 <<= 1
		}
		padded := make([]complex128, nextPow2)
		copy(padded, x)
		x = padded
		N = nextPow2
	}

	return sa.fftRecursive(x)
}

// fftRecursive performs recursive FFT implementation
func (sa *SpectralAnalyzer) fftRecursive(x []complex128) []complex128 {
	N := len(x)

	if N <= 1 {
		return x
	}

	// Divide
	even := make([]complex128, N/2)
	odd := make([]complex128, N/2)

	for i := 0; i < N/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Conquer
	evenFFT := sa.fftRecursive(even)
	oddFFT := sa.fftRecursive(odd)

	// Combine
	result := make([]complex128, N)

	for k := 0; k < N/2; k++ {
		t := cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(N))) * oddFFT[k]
		result[k] = evenFFT[k] + t
		result[k+N/2] = evenFFT[k] - t
	}

	return result
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
