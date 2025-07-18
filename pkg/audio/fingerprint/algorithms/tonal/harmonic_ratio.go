package tonal

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/harmonic"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// HarmonicRatioMethod defines different methods for computing harmonic ratio
type HarmonicRatioMethod int

const (
	HarmonicRatioHNR      HarmonicRatioMethod = iota // Harmonic-to-Noise Ratio
	HarmonicRatioACF                                 // Autocorrelation-based
	HarmonicRatioHPS                                 // Harmonic Product Spectrum
	HarmonicRatioComb                                // Comb filtering
	HarmonicRatioSpectral                            // Spectral peak analysis
	HarmonicRatioYin                                 // YIN-based periodicity
)

// HarmonicRatioParams contains parameters for harmonic ratio computation
type HarmonicRatioParams struct {
	Method     HarmonicRatioMethod `json:"method"`
	SampleRate int                 `json:"sample_rate"`
	WindowSize int                 `json:"window_size"`
	HopSize    int                 `json:"hop_size"`

	// Frequency analysis parameters
	MinFreq           float64 `json:"min_freq"`           // Minimum frequency to analyze
	MaxFreq           float64 `json:"max_freq"`           // Maximum frequency to analyze
	MaxHarmonics      int     `json:"max_harmonics"`      // Maximum number of harmonics
	HarmonicTolerance float64 `json:"harmonic_tolerance"` // Tolerance for harmonic detection

	// Spectral analysis parameters
	MinPeakHeight        float64 `json:"min_peak_height"`        // Minimum peak height threshold
	PeakDetectionWidth   int     `json:"peak_detection_width"`   // Width for peak detection
	SpectralSmoothingLen int     `json:"spectral_smoothing_len"` // Spectral smoothing length

	// Noise floor estimation
	NoiseFloorMethod       string  `json:"noise_floor_method"`        // "median", "percentile", "minimum"
	NoiseFloorPercentile   float64 `json:"noise_floor_percentile"`    // Percentile for noise floor
	NoiseFloorSmoothingLen int     `json:"noise_floor_smoothing_len"` // Smoothing length

	// Temporal analysis
	UseTemporalSmoothing bool `json:"use_temporal_smoothing"` // Enable temporal smoothing
	TemporalSmoothingLen int  `json:"temporal_smoothing_len"` // Temporal smoothing length

	// Advanced parameters
	UseFreqWeighting      bool `json:"use_freq_weighting"`      // Weight by frequency importance
	UsePsychoacousticMask bool `json:"use_psychoacoustic_mask"` // Apply psychoacoustic masking
	AdaptiveThreshold     bool `json:"adaptive_threshold"`      // Use adaptive thresholding
}

// HarmonicRatioResult contains comprehensive harmonic ratio analysis results
type HarmonicRatioResult struct {
	// Primary measures
	HarmonicRatio  float64 `json:"harmonic_ratio"`  // Overall harmonic-to-noise ratio (dB)
	HarmonicEnergy float64 `json:"harmonic_energy"` // Total harmonic energy
	NoiseEnergy    float64 `json:"noise_energy"`    // Total noise energy
	TotalEnergy    float64 `json:"total_energy"`    // Total signal energy

	// Detailed harmonic analysis
	HarmonicFrequencies []float64 `json:"harmonic_frequencies"` // Detected harmonic frequencies
	HarmonicAmplitudes  []float64 `json:"harmonic_amplitudes"`  // Harmonic amplitudes
	HarmonicPhases      []float64 `json:"harmonic_phases"`      // Harmonic phases
	HarmonicRatios      []float64 `json:"harmonic_ratios"`      // Individual harmonic ratios

	// Fundamental frequency analysis
	F0Frequency  float64 `json:"f0_frequency"`  // Fundamental frequency
	F0Confidence float64 `json:"f0_confidence"` // F0 detection confidence
	F0Strength   float64 `json:"f0_strength"`   // F0 strength measure

	// Spectral analysis
	SpectralPeaks      []harmonic.SpectralPeak `json:"spectral_peaks"`        // Detected spectral peaks
	NoiseFloor         []float64               `json:"noise_floor"`           // Estimated noise floor
	SignalToNoiseRatio float64                 `json:"signal_to_noise_ratio"` // Overall SNR

	// Quality metrics
	Periodicity float64 `json:"periodicity"` // Periodicity measure
	Harmonicity float64 `json:"harmonicity"` // Harmonicity measure
	Voicing     float64 `json:"voicing"`     // Voicing probability
	Roughness   float64 `json:"roughness"`   // Spectral roughness

	// Temporal analysis
	TemporalStability float64 `json:"temporal_stability"` // Temporal stability
	TemporalCoherence float64 `json:"temporal_coherence"` // Temporal coherence

	// Analysis metadata
	Method            string     `json:"method"`              // Method used
	ProcessingTime    float64    `json:"processing_time"`     // Processing time (ms)
	NumHarmonics      int        `json:"num_harmonics"`       // Number of harmonics found
	AnalysisFreqRange [2]float64 `json:"analysis_freq_range"` // Frequency range analyzed
}

// HarmonicRatioAnalyzer implements harmonic-to-noise ratio analysis
type HarmonicRatioAnalyzer struct {
	params HarmonicRatioParams

	// Analysis components
	fft           *spectral.FFT
	stft          *spectral.STFT
	peakDetector  *harmonic.SpectralPeaks
	pitchDetector *PitchDetector
	autocorr      *stats.AutoCorrelation

	// Internal buffers
	spectrum   []complex128
	magnitude  []float64
	phase      []float64
	noiseFloor []float64

	// Temporal tracking
	previousHR float64
	hrHistory  []float64
	f0History  []float64

	// Preprocessing
	windowFunc []float64
	freqBins   []float64

	initialized bool
}

// NewHarmonicRatioAnalyzer creates a new harmonic ratio analyzer
func NewHarmonicRatioAnalyzer(sampleRate int) *HarmonicRatioAnalyzer {
	return &HarmonicRatioAnalyzer{
		params: HarmonicRatioParams{
			Method:                 HarmonicRatioHNR,
			SampleRate:             sampleRate,
			WindowSize:             2048,
			HopSize:                512,
			MinFreq:                80.0,
			MaxFreq:                8000.0,
			MaxHarmonics:           20,
			HarmonicTolerance:      0.1,
			MinPeakHeight:          0.001,
			PeakDetectionWidth:     3,
			SpectralSmoothingLen:   5,
			NoiseFloorMethod:       "percentile",
			NoiseFloorPercentile:   0.1,
			NoiseFloorSmoothingLen: 10,
			UseTemporalSmoothing:   true,
			TemporalSmoothingLen:   5,
			UseFreqWeighting:       false,
			UsePsychoacousticMask:  false,
			AdaptiveThreshold:      false,
		},
		fft:           spectral.NewFFT(),
		stft:          spectral.NewSTFT(),
		peakDetector:  harmonic.NewSpectralPeaks(sampleRate, 0.001, 20.0, 100),
		pitchDetector: NewPitchDetector(sampleRate),
		autocorr:      stats.NewAutoCorrelation(2048),
		hrHistory:     make([]float64, 0),
		f0History:     make([]float64, 0),
	}
}

// NewHarmonicRatioAnalyzerWithParams creates analyzer with custom parameters
func NewHarmonicRatioAnalyzerWithParams(params HarmonicRatioParams) *HarmonicRatioAnalyzer {
	hra := &HarmonicRatioAnalyzer{
		params:        params,
		fft:           spectral.NewFFT(),
		stft:          spectral.NewSTFT(),
		peakDetector:  harmonic.NewSpectralPeaks(params.SampleRate, params.MinPeakHeight, 20.0, 100),
		pitchDetector: NewPitchDetector(params.SampleRate),
		autocorr:      stats.NewAutoCorrelation(params.WindowSize),
		hrHistory:     make([]float64, 0),
		f0History:     make([]float64, 0),
	}

	hra.Initialize()
	return hra
}

// Initialize sets up the analyzer
func (hra *HarmonicRatioAnalyzer) Initialize() {
	if hra.initialized {
		return
	}

	// Initialize buffers
	hra.spectrum = make([]complex128, hra.params.WindowSize)
	hra.magnitude = make([]float64, hra.params.WindowSize/2)
	hra.phase = make([]float64, hra.params.WindowSize/2)
	hra.noiseFloor = make([]float64, hra.params.WindowSize/2)

	// Initialize frequency bins
	hra.freqBins = make([]float64, hra.params.WindowSize/2)
	for i := range hra.freqBins {
		hra.freqBins[i] = float64(i) * float64(hra.params.SampleRate) / float64(hra.params.WindowSize)
	}

	// Initialize window function
	hra.windowFunc = hra.createWindowFunction()

	hra.initialized = true
}

// AnalyzeFrame analyzes harmonic ratio for a single audio frame
func (hra *HarmonicRatioAnalyzer) AnalyzeFrame(audioFrame []float64) (HarmonicRatioResult, error) {
	if !hra.initialized {
		hra.Initialize()
	}

	startTime := hra.getCurrentTime()

	// Validate input
	if len(audioFrame) != hra.params.WindowSize {
		return HarmonicRatioResult{}, fmt.Errorf("audio frame size (%d) doesn't match window size (%d)", len(audioFrame), hra.params.WindowSize)
	}

	// Preprocess frame
	processedFrame := hra.preprocessFrame(audioFrame)

	// Compute spectrum
	err := hra.computeSpectrum(processedFrame)
	if err != nil {
		return HarmonicRatioResult{}, err
	}

	// Analyze using selected method
	var result HarmonicRatioResult
	switch hra.params.Method {
	case HarmonicRatioHNR:
		result = hra.analyzeHNR()
	case HarmonicRatioACF:
		result = hra.analyzeACF(processedFrame)
	case HarmonicRatioHPS:
		result = hra.analyzeHPS()
	case HarmonicRatioComb:
		result = hra.analyzeComb()
	case HarmonicRatioSpectral:
		result = hra.analyzeSpectral()
	case HarmonicRatioYin:
		result = hra.analyzeYin(processedFrame)
	default:
		result = hra.analyzeHNR()
	}

	// Post-process results
	result = hra.postProcessResult(result)

	// Update temporal tracking
	hra.updateTemporalTracking(result)

	// Set metadata
	result.Method = hra.getMethodName(hra.params.Method)
	result.ProcessingTime = hra.getCurrentTime() - startTime
	result.AnalysisFreqRange = [2]float64{hra.params.MinFreq, hra.params.MaxFreq}

	return result, nil
}

// preprocessFrame applies preprocessing to audio frame
func (hra *HarmonicRatioAnalyzer) preprocessFrame(audioFrame []float64) []float64 {
	processed := make([]float64, len(audioFrame))
	copy(processed, audioFrame)

	// Apply window function
	for i := range processed {
		processed[i] *= hra.windowFunc[i]
	}

	return processed
}

// computeSpectrum computes the frequency spectrum
func (hra *HarmonicRatioAnalyzer) computeSpectrum(audioFrame []float64) error {
	// Compute FFT
	hra.spectrum = hra.fft.Compute(audioFrame)

	// Extract magnitude and phase
	for i := 0; i < len(hra.magnitude); i++ {
		real := real(hra.spectrum[i])
		imag := imag(hra.spectrum[i])
		hra.magnitude[i] = math.Sqrt(real*real + imag*imag)
		hra.phase[i] = math.Atan2(imag, real)
	}

	// Smooth spectrum if requested
	if hra.params.SpectralSmoothingLen > 1 {
		hra.magnitude = common.MovingAverage(hra.magnitude, hra.params.SpectralSmoothingLen)
	}

	// Estimate noise floor
	hra.estimateNoiseFloor()

	return nil
}

// analyzeHNR analyzes using Harmonic-to-Noise Ratio method
func (hra *HarmonicRatioAnalyzer) analyzeHNR() HarmonicRatioResult {
	// Detect fundamental frequency
	f0, f0Confidence := hra.detectFundamentalFrequency()

	// Find harmonic peaks
	harmonicPeaks := hra.findHarmonicPeaks(f0)

	// Calculate harmonic and noise energies
	harmonicEnergy := 0.0
	noiseEnergy := 0.0
	totalEnergy := 0.0

	// Create harmonic mask
	harmonicMask := make([]bool, len(hra.magnitude))

	for _, peak := range harmonicPeaks {
		// Mark frequency bins around each harmonic as harmonic
		binIdx := int(peak.Frequency * float64(hra.params.WindowSize) / float64(hra.params.SampleRate))
		width := hra.params.PeakDetectionWidth

		for i := max(0, binIdx-width); i < min(len(harmonicMask), binIdx+width+1); i++ {
			harmonicMask[i] = true
		}
	}

	// Calculate energies
	for i := range hra.magnitude {
		freq := hra.freqBins[i]
		if freq >= hra.params.MinFreq && freq <= hra.params.MaxFreq {
			energy := hra.magnitude[i] * hra.magnitude[i]
			totalEnergy += energy

			if harmonicMask[i] {
				harmonicEnergy += energy
			} else {
				noiseEnergy += energy
			}
		}
	}

	// Calculate harmonic ratio
	var harmonicRatio float64
	if noiseEnergy > 0 {
		harmonicRatio = 10.0 * math.Log10(harmonicEnergy/noiseEnergy)
	} else {
		harmonicRatio = 60.0 // Very high HNR
	}

	// Extract harmonic information
	harmonicFreqs := make([]float64, len(harmonicPeaks))
	harmonicAmps := make([]float64, len(harmonicPeaks))
	harmonicPhases := make([]float64, len(harmonicPeaks))

	for i, peak := range harmonicPeaks {
		harmonicFreqs[i] = peak.Frequency
		harmonicAmps[i] = peak.Magnitude
		harmonicPhases[i] = peak.Phase
	}

	// Calculate additional metrics
	periodicity := hra.calculatePeriodicity(f0)
	harmonicity := hra.calculateHarmonicity(harmonicPeaks, f0)
	voicing := hra.calculateVoicing(harmonicRatio)
	roughness := hra.calculateRoughness(harmonicPeaks)

	return HarmonicRatioResult{
		HarmonicRatio:       harmonicRatio,
		HarmonicEnergy:      harmonicEnergy,
		NoiseEnergy:         noiseEnergy,
		TotalEnergy:         totalEnergy,
		HarmonicFrequencies: harmonicFreqs,
		HarmonicAmplitudes:  harmonicAmps,
		HarmonicPhases:      harmonicPhases,
		F0Frequency:         f0,
		F0Confidence:        f0Confidence,
		F0Strength:          f0Confidence,
		SpectralPeaks:       harmonicPeaks,
		NoiseFloor:          hra.noiseFloor,
		SignalToNoiseRatio:  hra.calculateSNR(),
		Periodicity:         periodicity,
		Harmonicity:         harmonicity,
		Voicing:             voicing,
		Roughness:           roughness,
		NumHarmonics:        len(harmonicPeaks),
	}
}

// analyzeACF analyzes using autocorrelation method
func (hra *HarmonicRatioAnalyzer) analyzeACF(audioFrame []float64) HarmonicRatioResult {
	// Compute autocorrelation
	autocorrResult, err := hra.autocorr.Compute(audioFrame)
	if err != nil {
		return HarmonicRatioResult{}
	}

	// Find periodicity from autocorrelation
	periodicity := hra.calculatePeriodicityFromACF(autocorrResult)

	// Estimate harmonic ratio from autocorrelation strength
	maxCorr := 0.0
	for _, corr := range autocorrResult.Correlations {
		if math.Abs(corr) > maxCorr {
			maxCorr = math.Abs(corr)
		}
	}

	// Convert correlation to HNR estimate
	harmonicRatio := 20.0 * math.Log10(maxCorr/(1.0-maxCorr+1e-10))

	return HarmonicRatioResult{
		HarmonicRatio: harmonicRatio,
		Periodicity:   periodicity,
		Voicing:       maxCorr,
	}
}

// analyzeHPS analyzes using Harmonic Product Spectrum method
func (hra *HarmonicRatioAnalyzer) analyzeHPS() HarmonicRatioResult {
	// Compute HPS
	hps := make([]float64, len(hra.magnitude))
	copy(hps, hra.magnitude)

	// Multiply harmonics
	for h := 2; h <= min(5, hra.params.MaxHarmonics); h++ {
		for i := 0; i < len(hps)/h; i++ {
			hps[i] *= hra.magnitude[i*h]
		}
	}

	// Find fundamental frequency from HPS peak
	maxIdx := 0
	maxVal := hps[0]

	minBin := int(hra.params.MinFreq * float64(hra.params.WindowSize) / float64(hra.params.SampleRate))
	maxBin := int(hra.params.MaxFreq * float64(hra.params.WindowSize) / float64(hra.params.SampleRate))

	for i := minBin; i < maxBin && i < len(hps); i++ {
		if hps[i] > maxVal {
			maxVal = hps[i]
			maxIdx = i
		}
	}

	f0 := hra.freqBins[maxIdx]

	// Find harmonic peaks
	harmonicPeaks := hra.findHarmonicPeaks(f0)

	// Calculate harmonic ratio based on HPS strength
	harmonicRatio := 20.0 * math.Log10(maxVal+1e-10)

	return HarmonicRatioResult{
		HarmonicRatio: harmonicRatio,
		F0Frequency:   f0,
		SpectralPeaks: harmonicPeaks,
		NumHarmonics:  len(harmonicPeaks),
	}
}

// analyzeComb analyzes using comb filtering method
func (hra *HarmonicRatioAnalyzer) analyzeComb() HarmonicRatioResult {
	// This would implement comb filtering approach
	// For now, fallback to HNR method
	return hra.analyzeHNR()
}

// analyzeSpectral analyzes using spectral peak analysis
func (hra *HarmonicRatioAnalyzer) analyzeSpectral() HarmonicRatioResult {
	// Detect all spectral peaks
	allPeaks := hra.peakDetector.DetectPeaks(hra.magnitude, hra.params.WindowSize)

	// Filter peaks by frequency range
	validPeaks := make([]harmonic.SpectralPeak, 0)
	for _, peak := range allPeaks {
		if peak.Frequency >= hra.params.MinFreq && peak.Frequency <= hra.params.MaxFreq {
			validPeaks = append(validPeaks, peak)
		}
	}

	// Try to identify harmonic structure
	f0 := hra.estimateF0FromPeaks(validPeaks)
	harmonicPeaks := hra.findHarmonicPeaks(f0)

	// Calculate harmonic ratio
	harmonicEnergy := 0.0
	totalEnergy := 0.0

	for _, peak := range validPeaks {
		energy := peak.Magnitude * peak.Magnitude
		totalEnergy += energy

		// Check if this peak is harmonic
		if hra.isHarmonic(peak.Frequency, f0) {
			harmonicEnergy += energy
		}
	}

	noiseEnergy := totalEnergy - harmonicEnergy

	var harmonicRatio float64
	if noiseEnergy > 0 {
		harmonicRatio = 10.0 * math.Log10(harmonicEnergy/noiseEnergy)
	} else {
		harmonicRatio = 60.0
	}

	return HarmonicRatioResult{
		HarmonicRatio:  harmonicRatio,
		HarmonicEnergy: harmonicEnergy,
		NoiseEnergy:    noiseEnergy,
		TotalEnergy:    totalEnergy,
		F0Frequency:    f0,
		SpectralPeaks:  harmonicPeaks,
		NumHarmonics:   len(harmonicPeaks),
	}
}

// analyzeYin analyzes using YIN-based method
func (hra *HarmonicRatioAnalyzer) analyzeYin(audioFrame []float64) HarmonicRatioResult {
	// Use pitch detector for YIN analysis
	pitchParams := PitchDetectionParams{
		Method:     AutocorrelationYin,
		SampleRate: hra.params.SampleRate,
		WindowSize: hra.params.WindowSize,
		MinFreq:    hra.params.MinFreq,
		MaxFreq:    hra.params.MaxFreq,
	}

	tempPitchDetector := NewPitchDetectorWithParams(pitchParams)
	pitchResult, err := tempPitchDetector.DetectPitch(audioFrame)

	if err != nil {
		return HarmonicRatioResult{}
	}

	// Convert pitch detection results to harmonic ratio
	harmonicRatio := 20.0 * math.Log10(pitchResult.Confidence+1e-10)

	return HarmonicRatioResult{
		HarmonicRatio: harmonicRatio,
		F0Frequency:   pitchResult.Pitch,
		F0Confidence:  pitchResult.Confidence,
		Periodicity:   pitchResult.Periodicity,
		Voicing:       pitchResult.Voicing,
	}
}

// Helper functions

func (hra *HarmonicRatioAnalyzer) detectFundamentalFrequency() (float64, float64) {
	// Simple F0 detection using spectral peaks
	peaks := hra.peakDetector.DetectPeaks(hra.magnitude, hra.params.WindowSize)

	if len(peaks) == 0 {
		return 0.0, 0.0
	}

	// Find the lowest significant peak as potential F0
	minFreq := hra.params.MinFreq
	for _, peak := range peaks {
		if peak.Frequency >= minFreq {
			return peak.Frequency, peak.Magnitude
		}
	}

	return peaks[0].Frequency, peaks[0].Magnitude
}

func (hra *HarmonicRatioAnalyzer) findHarmonicPeaks(f0 float64) []harmonic.SpectralPeak {
	if f0 <= 0 {
		return []harmonic.SpectralPeak{}
	}

	harmonicPeaks := make([]harmonic.SpectralPeak, 0)

	// Look for harmonics up to Nyquist or max frequency
	maxHarmonic := int(math.Min(float64(hra.params.MaxHarmonics), hra.params.MaxFreq/f0))

	for h := 1; h <= maxHarmonic; h++ {
		expectedFreq := f0 * float64(h)
		if expectedFreq > hra.params.MaxFreq {
			break
		}

		// Find the closest peak to expected harmonic frequency
		peak := hra.findNearestPeak(expectedFreq)
		if peak != nil {
			// Check if peak is within tolerance
			tolerance := hra.params.HarmonicTolerance * expectedFreq
			if math.Abs(peak.Frequency-expectedFreq) < tolerance {
				peak.Harmonic = h - 1 // 0-indexed
				harmonicPeaks = append(harmonicPeaks, *peak)
			}
		}
	}

	return harmonicPeaks
}

func (hra *HarmonicRatioAnalyzer) findNearestPeak(targetFreq float64) *harmonic.SpectralPeak {
	// Find the spectral bin closest to target frequency
	targetBin := int(targetFreq * float64(hra.params.WindowSize) / float64(hra.params.SampleRate))

	if targetBin >= len(hra.magnitude) {
		return nil
	}

	// Look for local maximum around target bin
	searchWidth := hra.params.PeakDetectionWidth
	maxMag := 0.0
	maxBin := targetBin

	for i := max(0, targetBin-searchWidth); i < min(len(hra.magnitude), targetBin+searchWidth+1); i++ {
		if hra.magnitude[i] > maxMag {
			maxMag = hra.magnitude[i]
			maxBin = i
		}
	}

	// Check if it's actually a peak
	if maxBin > 0 && maxBin < len(hra.magnitude)-1 {
		if hra.magnitude[maxBin] > hra.magnitude[maxBin-1] &&
			hra.magnitude[maxBin] > hra.magnitude[maxBin+1] {
			return &harmonic.SpectralPeak{
				Frequency: hra.freqBins[maxBin],
				Magnitude: hra.magnitude[maxBin],
				Phase:     hra.phase[maxBin],
				BinIndex:  maxBin,
			}
		}
	}

	return nil
}

func (hra *HarmonicRatioAnalyzer) estimateNoiseFloor() {
	switch hra.params.NoiseFloorMethod {
	case "median":
		hra.noiseFloor = hra.computeMedianNoiseFloor()
	case "percentile":
		hra.noiseFloor = hra.computePercentileNoiseFloor()
	case "minimum":
		hra.noiseFloor = hra.computeMinimumNoiseFloor()
	default:
		hra.noiseFloor = hra.computePercentileNoiseFloor()
	}

	// Smooth noise floor if requested
	if hra.params.NoiseFloorSmoothingLen > 1 {
		hra.noiseFloor = common.MovingAverage(hra.noiseFloor, hra.params.NoiseFloorSmoothingLen)
	}
}

func (hra *HarmonicRatioAnalyzer) computePercentileNoiseFloor() []float64 {
	noiseFloor := make([]float64, len(hra.magnitude))

	// Use local percentile estimation
	windowSize := 20

	for i := range hra.magnitude {
		start := max(0, i-windowSize/2)
		end := min(len(hra.magnitude), i+windowSize/2)

		window := hra.magnitude[start:end]
		percentile := common.Percentile(window, hra.params.NoiseFloorPercentile)
		noiseFloor[i] = percentile
	}

	return noiseFloor
}

func (hra *HarmonicRatioAnalyzer) computeMedianNoiseFloor() []float64 {
	noiseFloor := make([]float64, len(hra.magnitude))

	// Use local median estimation
	windowSize := 20

	for i := range hra.magnitude {
		start := max(0, i-windowSize/2)
		end := min(len(hra.magnitude), i+windowSize/2)

		window := hra.magnitude[start:end]
		noiseFloor[i] = common.Percentile(window, 0.5) // Median
	}

	return noiseFloor
}

func (hra *HarmonicRatioAnalyzer) computeMinimumNoiseFloor() []float64 {
	noiseFloor := make([]float64, len(hra.magnitude))

	// Use local minimum estimation
	windowSize := 20

	for i := range hra.magnitude {
		start := max(0, i-windowSize/2)
		end := min(len(hra.magnitude), i+windowSize/2)

		minVal := hra.magnitude[start]
		for j := start; j < end; j++ {
			if hra.magnitude[j] < minVal {
				minVal = hra.magnitude[j]
			}
		}
		noiseFloor[i] = minVal
	}

	return noiseFloor
}

func (hra *HarmonicRatioAnalyzer) calculatePeriodicity(f0 float64) float64 {
	if f0 <= 0 {
		return 0.0
	}

	// Simple periodicity measure based on harmonic strength
	harmonicStrength := 0.0
	totalStrength := 0.0

	for i := range hra.magnitude {
		freq := hra.freqBins[i]
		if freq >= hra.params.MinFreq && freq <= hra.params.MaxFreq {
			totalStrength += hra.magnitude[i]

			if hra.isHarmonic(freq, f0) {
				harmonicStrength += hra.magnitude[i]
			}
		}
	}

	if totalStrength > 0 {
		return harmonicStrength / totalStrength
	}

	return 0.0
}

func (hra *HarmonicRatioAnalyzer) calculateHarmonicity(harmonicPeaks []harmonic.SpectralPeak, f0 float64) float64 {
	if len(harmonicPeaks) == 0 || f0 <= 0 {
		return 0.0
	}

	// Measure how well peaks align with harmonic series
	totalDeviation := 0.0

	for _, peak := range harmonicPeaks {
		// Find closest harmonic
		harmonic := math.Round(peak.Frequency / f0)
		expectedFreq := f0 * harmonic

		// Calculate relative deviation
		deviation := math.Abs(peak.Frequency-expectedFreq) / expectedFreq
		totalDeviation += deviation
	}

	// Convert to harmonicity measure (0-1)
	avgDeviation := totalDeviation / float64(len(harmonicPeaks))
	harmonicity := math.Exp(-avgDeviation * 10.0) // Exponential decay

	return harmonicity
}

func (hra *HarmonicRatioAnalyzer) calculateVoicing(harmonicRatio float64) float64 {
	// Convert harmonic ratio to voicing probability
	// Using sigmoid transformation
	return 1.0 / (1.0 + math.Exp(-0.1*(harmonicRatio-10.0)))
}

func (hra *HarmonicRatioAnalyzer) calculateRoughness(harmonicPeaks []harmonic.SpectralPeak) float64 {
	if len(harmonicPeaks) < 2 {
		return 0.0
	}

	// Calculate spectral roughness based on peak interactions
	roughness := 0.0

	for i := 0; i < len(harmonicPeaks); i++ {
		for j := i + 1; j < len(harmonicPeaks); j++ {
			peak1 := harmonicPeaks[i]
			peak2 := harmonicPeaks[j]

			// Calculate frequency difference
			freqDiff := math.Abs(peak1.Frequency - peak2.Frequency)

			// Calculate roughness contribution
			// Based on Plomp-Levelt roughness curve
			if freqDiff > 0 {
				roughnessContrib := (peak1.Magnitude * peak2.Magnitude) / (freqDiff + 1.0)
				roughness += roughnessContrib
			}
		}
	}

	return roughness
}

func (hra *HarmonicRatioAnalyzer) calculateSNR() float64 {
	// Calculate overall signal-to-noise ratio
	signalEnergy := 0.0
	noiseEnergy := 0.0

	for i := range hra.magnitude {
		freq := hra.freqBins[i]
		if freq >= hra.params.MinFreq && freq <= hra.params.MaxFreq {
			signalPower := hra.magnitude[i] * hra.magnitude[i]
			noisePower := hra.noiseFloor[i] * hra.noiseFloor[i]

			signalEnergy += signalPower
			noiseEnergy += noisePower
		}
	}

	if noiseEnergy > 0 {
		return 10.0 * math.Log10(signalEnergy/noiseEnergy)
	}

	return 60.0 // Very high SNR
}

func (hra *HarmonicRatioAnalyzer) calculatePeriodicityFromACF(autocorrResult *stats.CorrelationResult) float64 {
	if len(autocorrResult.Correlations) == 0 {
		return 0.0
	}

	// Find maximum correlation (excluding lag 0)
	maxCorr := 0.0
	for i := 1; i < len(autocorrResult.Correlations); i++ {
		if math.Abs(autocorrResult.Correlations[i]) > maxCorr {
			maxCorr = math.Abs(autocorrResult.Correlations[i])
		}
	}

	return maxCorr
}

func (hra *HarmonicRatioAnalyzer) estimateF0FromPeaks(peaks []harmonic.SpectralPeak) float64 {
	if len(peaks) == 0 {
		return 0.0
	}

	// Sort peaks by magnitude
	sortedPeaks := make([]harmonic.SpectralPeak, len(peaks))
	copy(sortedPeaks, peaks)
	sort.Slice(sortedPeaks, func(i, j int) bool {
		return sortedPeaks[i].Magnitude > sortedPeaks[j].Magnitude
	})

	// Try different F0 candidates and find best fit
	bestF0 := 0.0
	bestScore := 0.0

	// Use top peaks as potential F0 candidates
	maxCandidates := min(5, len(sortedPeaks))
	for i := 0; i < maxCandidates; i++ {
		candidate := sortedPeaks[i].Frequency

		// Check how well this candidate explains the peak structure
		score := hra.evaluateF0Candidate(candidate, peaks)

		if score > bestScore {
			bestScore = score
			bestF0 = candidate
		}
	}

	return bestF0
}

func (hra *HarmonicRatioAnalyzer) evaluateF0Candidate(f0 float64, peaks []harmonic.SpectralPeak) float64 {
	if f0 <= 0 {
		return 0.0
	}

	score := 0.0
	maxHarmonic := int(hra.params.MaxFreq / f0)

	for h := 1; h <= maxHarmonic && h <= hra.params.MaxHarmonics; h++ {
		expectedFreq := f0 * float64(h)

		// Find closest peak to expected frequency
		closestPeak := hra.findClosestPeak(expectedFreq, peaks)
		if closestPeak != nil {
			// Calculate score based on proximity and magnitude
			tolerance := hra.params.HarmonicTolerance * expectedFreq
			distance := math.Abs(closestPeak.Frequency - expectedFreq)

			if distance < tolerance {
				proximityScore := 1.0 - (distance / tolerance)
				magnitudeScore := closestPeak.Magnitude
				score += proximityScore * magnitudeScore
			}
		}
	}

	return score
}

func (hra *HarmonicRatioAnalyzer) findClosestPeak(targetFreq float64, peaks []harmonic.SpectralPeak) *harmonic.SpectralPeak {
	if len(peaks) == 0 {
		return nil
	}

	closest := &peaks[0]
	minDistance := math.Abs(peaks[0].Frequency - targetFreq)

	for i := 1; i < len(peaks); i++ {
		distance := math.Abs(peaks[i].Frequency - targetFreq)
		if distance < minDistance {
			minDistance = distance
			closest = &peaks[i]
		}
	}

	return closest
}

func (hra *HarmonicRatioAnalyzer) isHarmonic(freq, f0 float64) bool {
	if f0 <= 0 {
		return false
	}

	// Check if frequency is close to any harmonic
	harmonic := math.Round(freq / f0)
	expectedFreq := f0 * harmonic
	tolerance := hra.params.HarmonicTolerance * expectedFreq

	return math.Abs(freq-expectedFreq) < tolerance
}

func (hra *HarmonicRatioAnalyzer) postProcessResult(result HarmonicRatioResult) HarmonicRatioResult {
	// Apply temporal smoothing if enabled
	if hra.params.UseTemporalSmoothing && len(hra.hrHistory) > 0 {
		result.HarmonicRatio = hra.applyTemporalSmoothing(result.HarmonicRatio)
	}

	// Calculate temporal stability
	result.TemporalStability = hra.calculateTemporalStability()
	result.TemporalCoherence = hra.calculateTemporalCoherence()

	return result
}

func (hra *HarmonicRatioAnalyzer) applyTemporalSmoothing(currentHR float64) float64 {
	if len(hra.hrHistory) == 0 {
		return currentHR
	}

	// Apply exponential smoothing
	alpha := 0.3
	return alpha*currentHR + (1.0-alpha)*hra.previousHR
}

func (hra *HarmonicRatioAnalyzer) calculateTemporalStability() float64 {
	if len(hra.hrHistory) < 3 {
		return 0.0
	}

	// Calculate coefficient of variation
	mean := common.Mean(hra.hrHistory)
	variance := common.Variance(hra.hrHistory)

	if mean > 0 {
		cv := math.Sqrt(variance) / mean
		return math.Max(0.0, 1.0-cv) // Stability = 1 - CV
	}

	return 0.0
}

func (hra *HarmonicRatioAnalyzer) calculateTemporalCoherence() float64 {
	if len(hra.hrHistory) < 2 {
		return 0.0
	}

	// Calculate autocorrelation of HR sequence
	if len(hra.hrHistory) > 1 {
		// Simple coherence based on consecutive differences
		totalDiff := 0.0
		for i := 1; i < len(hra.hrHistory); i++ {
			diff := math.Abs(hra.hrHistory[i] - hra.hrHistory[i-1])
			totalDiff += diff
		}

		avgDiff := totalDiff / float64(len(hra.hrHistory)-1)
		coherence := math.Exp(-avgDiff / 10.0) // Exponential decay
		return coherence
	}

	return 0.0
}

func (hra *HarmonicRatioAnalyzer) updateTemporalTracking(result HarmonicRatioResult) {
	// Update history
	hra.hrHistory = append(hra.hrHistory, result.HarmonicRatio)
	hra.f0History = append(hra.f0History, result.F0Frequency)

	// Keep only recent history
	maxHistory := hra.params.TemporalSmoothingLen * 2
	if len(hra.hrHistory) > maxHistory {
		hra.hrHistory = hra.hrHistory[len(hra.hrHistory)-maxHistory:]
		hra.f0History = hra.f0History[len(hra.f0History)-maxHistory:]
	}

	// Update previous value
	hra.previousHR = result.HarmonicRatio
}

func (hra *HarmonicRatioAnalyzer) createWindowFunction() []float64 {
	window := make([]float64, hra.params.WindowSize)

	// Hann window
	for i := range window {
		window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(hra.params.WindowSize-1)))
	}

	return window
}

func (hra *HarmonicRatioAnalyzer) getMethodName(method HarmonicRatioMethod) string {
	switch method {
	case HarmonicRatioHNR:
		return "Harmonic-to-Noise Ratio"
	case HarmonicRatioACF:
		return "Autocorrelation Function"
	case HarmonicRatioHPS:
		return "Harmonic Product Spectrum"
	case HarmonicRatioComb:
		return "Comb Filtering"
	case HarmonicRatioSpectral:
		return "Spectral Peak Analysis"
	case HarmonicRatioYin:
		return "YIN-based"
	default:
		return "Unknown"
	}
}

func (hra *HarmonicRatioAnalyzer) getCurrentTime() float64 {
	// Placeholder for time measurement
	return 0.0
}

// Public API methods

// AnalyzeSequence analyzes harmonic ratio for a sequence of audio frames
func (hra *HarmonicRatioAnalyzer) AnalyzeSequence(audioFrames [][]float64) ([]HarmonicRatioResult, error) {
	results := make([]HarmonicRatioResult, len(audioFrames))

	for i, frame := range audioFrames {
		result, err := hra.AnalyzeFrame(frame)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

// GetAverageHarmonicRatio computes average harmonic ratio from sequence
func (hra *HarmonicRatioAnalyzer) GetAverageHarmonicRatio(results []HarmonicRatioResult) float64 {
	if len(results) == 0 {
		return 0.0
	}

	sum := 0.0
	count := 0

	for _, result := range results {
		if result.HarmonicRatio > -60.0 { // Filter out very low values
			sum += result.HarmonicRatio
			count++
		}
	}

	if count > 0 {
		return sum / float64(count)
	}

	return 0.0
}

// Reset resets the analyzer state
func (hra *HarmonicRatioAnalyzer) Reset() {
	hra.previousHR = 0.0
	hra.hrHistory = make([]float64, 0)
	hra.f0History = make([]float64, 0)
}

// GetParameters returns current parameters
func (hra *HarmonicRatioAnalyzer) GetParameters() HarmonicRatioParams {
	return hra.params
}

// SetParameters updates parameters
func (hra *HarmonicRatioAnalyzer) SetParameters(params HarmonicRatioParams) {
	hra.params = params
	hra.initialized = false // Force re-initialization
}

// GetHistory returns the harmonic ratio history
func (hra *HarmonicRatioAnalyzer) GetHistory() []float64 {
	return hra.hrHistory
}

// GetF0History returns the F0 history
func (hra *HarmonicRatioAnalyzer) GetF0History() []float64 {
	return hra.f0History
}

// Utility functions

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// GetSupportedMethods returns list of supported harmonic ratio methods
func GetSupportedHarmonicRatioMethods() []string {
	return []string{
		"Harmonic-to-Noise Ratio",
		"Autocorrelation Function",
		"Harmonic Product Spectrum",
		"Comb Filtering",
		"Spectral Peak Analysis",
		"YIN-based",
	}
}

// ClassifyHarmonicRatio classifies harmonic ratio into categories
func ClassifyHarmonicRatio(harmonicRatio float64) string {
	if harmonicRatio >= 20.0 {
		return "Very High"
	} else if harmonicRatio >= 10.0 {
		return "High"
	} else if harmonicRatio >= 5.0 {
		return "Medium"
	} else if harmonicRatio >= 0.0 {
		return "Low"
	} else {
		return "Very Low"
	}
}

// EstimateVoicingQuality estimates voicing quality from harmonic ratio
func EstimateVoicingQuality(harmonicRatio float64) float64 {
	// Convert HNR to voicing quality (0-1)
	return 1.0 / (1.0 + math.Exp(-0.1*(harmonicRatio-5.0)))
}
