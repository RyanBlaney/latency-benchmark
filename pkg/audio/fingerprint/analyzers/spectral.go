package analyzers

import (
	"fmt"
	"math"
	"math/cmplx"
	"runtime"
	"sync"

	"github.com/mjibson/go-dsp/fft"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
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
		// My linter doesn't like math.Pow(diff, 3) here
		numerator += (diff * diff * diff) * spectrum[i]
	}

	skewness := (numerator / denominator) / (stdDev * stdDev * stdDev)
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

// getOptimalWorkerCount determines the optimal number of workers based on workload
func (sa *SpectralAnalyzer) getOptimalWorkerCount(numFrames int) int {
	// Base number on available CPUs
	numCPU := runtime.NumCPU()

	// For small workloads, don't over-parallelize
	if numFrames < 100 {
		return min(numCPU/2, numFrames)
	}

	// For medium workloads, use most CPUs
	if numFrames < 1000 {
		return min(numCPU, 8) // Cap at 8 for medium loads
	}

	// For large workloads, use all available CPUs
	return numCPU
}

// ComputeSTFTBatch processes multiple signals in parallel (useful for batch fingerprinting)
func (sa *SpectralAnalyzer) ComputeSTFTBatch(signals [][]float64, windowSize int, hopSize int, windowType WindowType) ([]*SpectrogramResult, error) {
	if len(signals) == 0 {
		return nil, fmt.Errorf("no signals provided")
	}

	results := make([]*SpectrogramResult, len(signals))
	errors := make([]error, len(signals))

	// Use goroutines for batch processing
	var wg sync.WaitGroup
	numWorkers := min(runtime.NumCPU(), len(signals))

	// Channel for signal processing jobs
	type signalJob struct {
		index  int
		signal []float64
	}

	jobs := make(chan signalJob, len(signals))

	// Launch workers
	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				result, err := sa.ComputeSTFTWithWindow(job.signal, windowSize, hopSize, windowType)
				results[job.index] = result
				errors[job.index] = err
			}
		}()
	}

	// Send jobs
	go func() {
		defer close(jobs)
		for i, signal := range signals {
			jobs <- signalJob{index: i, signal: signal}
		}
	}()

	wg.Wait()

	// Check for any errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("error processing signal %d: %w", i, err)
		}
	}

	return results, nil
}

// ComputeSTFTStreaming processes audio in streaming fashion with overlap-add
// Useful for real-time audio fingerprinting
func (sa *SpectralAnalyzer) ComputeSTFTStreaming(windowSize int, hopSize int, windowType WindowType) (*STFTStreamer, error) {
	windowConfig := &WindowConfig{
		Type:      windowType,
		Size:      windowSize,
		Normalize: true,
		Symmetric: true,
	}

	window, err := sa.windowGenerator.Generate(windowConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate window: %w", err)
	}

	return &STFTStreamer{
		analyzer:   sa,
		window:     window,
		windowSize: windowSize,
		hopSize:    hopSize,
		buffer:     make([]float64, 0, windowSize*2),
		freqBins:   windowSize/2 + 1,
	}, nil
}

// STFTStreamer handles streaming STFT computation
type STFTStreamer struct {
	analyzer   *SpectralAnalyzer
	window     *Window
	windowSize int
	hopSize    int
	buffer     []float64
	freqBins   int
}

// ProcessChunk processes a new chunk of audio data
func (s *STFTStreamer) ProcessChunk(chunk []float64) ([]*SpectrogramFrame, error) {
	if len(chunk) == 0 {
		return nil, nil
	}

	// Add new data to buffer
	s.buffer = append(s.buffer, chunk...)

	var frames []*SpectrogramFrame

	// Process as many complete frames as possible
	for len(s.buffer) >= s.windowSize {
		// Extract frame
		frameData := make([]float64, s.windowSize)
		copy(frameData, s.buffer[:s.windowSize])

		// Apply window
		err := s.window.ApplyInPlace(frameData)
		if err != nil {
			return nil, err
		}

		// Compute FFT
		fftResult := s.analyzer.FFT(frameData)

		// Create frame result
		frame := &SpectrogramFrame{
			Magnitude: make([]float64, s.freqBins),
			Phase:     make([]float64, s.freqBins),
			Complex:   make([]complex128, s.freqBins),
		}

		// Extract positive frequencies
		for i := 0; i < s.freqBins; i++ {
			frame.Complex[i] = fftResult[i]
			frame.Magnitude[i] = cmplx.Abs(fftResult[i])
			frame.Phase[i] = cmplx.Phase(fftResult[i])
		}

		frames = append(frames, frame)

		// Advance buffer by hop size
		if s.hopSize >= len(s.buffer) {
			s.buffer = s.buffer[:0]
		} else {
			copy(s.buffer, s.buffer[s.hopSize:])
			s.buffer = s.buffer[:len(s.buffer)-s.hopSize]
		}
	}

	return frames, nil
}

// SpectrogramFrame represents a single frame of spectrogram data
type SpectrogramFrame struct {
	Magnitude []float64    `json:"magnitude"`
	Phase     []float64    `json:"phase"`
	Complex   []complex128 `json:"-"`
}

// ComputeSTFTWithWindow computes STFT with parallel processing and custom window type
// This is the main optimized implementation with goroutines
func (sa *SpectralAnalyzer) ComputeSTFTWithWindow(signal []float64, windowSize int, hopSize int, windowType WindowType) (*SpectrogramResult, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal")
	}

	if windowSize <= 0 {
		return nil, fmt.Errorf("window size must be positive")
	}

	if hopSize <= 0 {
		return nil, fmt.Errorf("hop size must be positive")
	}

	logger := sa.logger.WithFields(logging.Fields{
		"function":      "ComputeSTFTWithWindow",
		"signal_length": len(signal),
		"window_size":   windowSize,
		"hop_size":      hopSize,
		"window_type":   windowType,
	})

	logger.Debug("Computing STFT with parallel processing")

	// Calculate number of frames
	numFrames := (len(signal)-windowSize)/hopSize + 1
	if numFrames <= 0 {
		return nil, fmt.Errorf("signal too short for given window size and hop size")
	}

	// Generate window function using your WindowGenerator
	windowConfig := &WindowConfig{
		Type:      windowType,
		Size:      windowSize,
		Normalize: true,
		Symmetric: true,
	}

	window, err := sa.windowGenerator.Generate(windowConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate window: %w", err)
	}

	// Calculate frequency bins (positive frequencies only)
	freqBins := windowSize/2 + 1

	// Initialize result matrices
	magnitude := make([][]float64, numFrames)
	phase := make([][]float64, numFrames)
	complexSpectrum := make([][]complex128, numFrames)

	// Pre-allocate all arrays
	for i := range numFrames {
		magnitude[i] = make([]float64, freqBins)
		phase[i] = make([]float64, freqBins)
		complexSpectrum[i] = make([]complex128, freqBins)
	}

	// Determine optimal number of workers based on system and workload
	numWorkers := sa.getOptimalWorkerCount(numFrames)

	// Channel for frame processing jobs
	type frameJob struct {
		frameIdx int
		startIdx int
		endIdx   int
	}

	jobs := make(chan frameJob, numFrames)

	// Launch worker goroutines
	var wg sync.WaitGroup

	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Reuse frame buffer for this worker
			frameBuffer := make([]float64, windowSize)

			for job := range jobs {
				// Safety check
				if job.endIdx > len(signal) {
					continue
				}

				// Extract and window the frame
				copy(frameBuffer, signal[job.startIdx:job.endIdx])

				// Apply window function in-place
				err := window.ApplyInPlace(frameBuffer)
				if err != nil {
					logger.Error(err, "Failed to apply window", logging.Fields{"frame": job.frameIdx})
					continue
				}

				// Compute FFT
				fftResult := sa.FFT(frameBuffer)

				// Extract positive frequencies and compute magnitude/phase
				for i := range freqBins {
					complexSpectrum[job.frameIdx][i] = fftResult[i]
					magnitude[job.frameIdx][i] = cmplx.Abs(fftResult[i])
					phase[job.frameIdx][i] = cmplx.Phase(fftResult[i])
				}
			}
		}()
	}

	// Send jobs to workers
	go func() {
		defer close(jobs)
		for frameIdx := range numFrames {
			startIdx := frameIdx * hopSize
			endIdx := startIdx + windowSize

			if endIdx <= len(signal) {
				jobs <- frameJob{
					frameIdx: frameIdx,
					startIdx: startIdx,
					endIdx:   endIdx,
				}
			}
		}
	}()

	// Wait for all workers to complete
	wg.Wait()

	result := &SpectrogramResult{
		Magnitude:      magnitude,
		Phase:          phase,
		Complex:        complexSpectrum,
		TimeFrames:     numFrames,
		FreqBins:       freqBins,
		SampleRate:     sa.sampleRate,
		WindowSize:     windowSize,
		HopSize:        hopSize,
		FreqResolution: float64(sa.sampleRate) / float64(windowSize),
		TimeResolution: float64(hopSize) / float64(sa.sampleRate),
	}

	logger.Debug("STFT computation completed", logging.Fields{
		"time_frames":     result.TimeFrames,
		"freq_bins":       result.FreqBins,
		"freq_resolution": result.FreqResolution,
		"time_resolution": result.TimeResolution,
		"workers_used":    numWorkers,
	})

	return result, nil
}
