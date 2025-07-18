package tonal

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// PitchDetectionMethod represents different pitch detection algorithms
type PitchDetectionMethod int

const (
	// Autocorrelation-based methods
	AutocorrelationYin PitchDetectionMethod = iota
	AutocorrelationACF
	AutocorrelationNSDF

	// Frequency domain methods
	FrequencyDomainHPS // Harmonic Product Spectrum
	FrequencyDomainCepstrum
	FrequencyDomainPeaks

	// Time domain methods
	TimeDomainZeroCrossing
	TimeDomainPeakPicking

	// Hybrid methods
	HybridYinFFT
	HybridMPM    // McLeod Pitch Method
	HybridPRAATT // PRAAT-style algorithm
)

// PitchCandidate represents a potential pitch with confidence
type PitchCandidate struct {
	Frequency  float64 `json:"frequency"`  // Frequency in Hz
	Confidence float64 `json:"confidence"` // Confidence score (0-1)
	Salience   float64 `json:"salience"`   // Perceptual salience
	Harmonic   int     `json:"harmonic"`   // Harmonic number (1=fundamental)
	Method     string  `json:"method"`     // Detection method used
}

// PitchDetectionResult contains comprehensive pitch detection results
type PitchDetectionResult struct {
	// Primary pitch information
	Pitch      float64 `json:"pitch"`      // Best pitch estimate (Hz)
	Confidence float64 `json:"confidence"` // Overall confidence (0-1)
	Salience   float64 `json:"salience"`   // Perceptual salience
	Voicing    float64 `json:"voicing"`    // Voicing probability (0-1)

	// Multiple pitch candidates
	Candidates []PitchCandidate `json:"candidates"`

	// Pitch stability measures
	Stability   float64 `json:"stability"`   // Pitch stability over time
	Periodicity float64 `json:"periodicity"` // Periodicity strength

	// Harmonic analysis
	Harmonics  []PitchCandidate `json:"harmonics"`   // Detected harmonics
	F0Multiple float64          `json:"f0_multiple"` // F0 multiple factor

	// Quality metrics
	SNR      float64 `json:"snr"`      // Signal-to-noise ratio
	Clarity  float64 `json:"clarity"`  // Pitch clarity measure
	Strength float64 `json:"strength"` // Pitch strength

	// Algorithm-specific data
	YinThreshold  float64   `json:"yin_threshold"`  // YIN threshold used
	AutocorrPeaks []float64 `json:"autocorr_peaks"` // Autocorrelation peaks
	CepstralPeak  float64   `json:"cepstral_peak"`  // Cepstral peak location
	HarmonicRatio float64   `json:"harmonic_ratio"` // Harmonic-to-noise ratio

	// Computational details
	Method      PitchDetectionMethod `json:"method"`
	SampleRate  int                  `json:"sample_rate"`
	WindowSize  int                  `json:"window_size"`
	ProcessTime float64              `json:"process_time"` // Processing time in ms
}

// PitchDetectionParams contains parameters for pitch detection
type PitchDetectionParams struct {
	Method     PitchDetectionMethod `json:"method"`
	SampleRate int                  `json:"sample_rate"`
	WindowSize int                  `json:"window_size"`
	HopSize    int                  `json:"hop_size"`

	// Frequency range constraints
	MinFreq float64 `json:"min_freq"` // Minimum frequency (Hz)
	MaxFreq float64 `json:"max_freq"` // Maximum frequency (Hz)

	// Algorithm-specific parameters
	YinThreshold      float64 `json:"yin_threshold"` // YIN threshold (0.1-0.5)
	AutocorrThreshold float64 `json:"autocorr_threshold"`
	CepstralThreshold float64 `json:"cepstral_threshold"`

	// Quality thresholds
	MinConfidence    float64 `json:"min_confidence"`    // Minimum confidence threshold
	MinSalience      float64 `json:"min_salience"`      // Minimum salience threshold
	VoicingThreshold float64 `json:"voicing_threshold"` // Voicing threshold

	// Harmonic analysis
	MaxHarmonics      int     `json:"max_harmonics"`      // Maximum harmonics to analyze
	HarmonicTolerance float64 `json:"harmonic_tolerance"` // Harmonic frequency tolerance

	// Preprocessing options
	PreEmphasis    bool   `json:"pre_emphasis"`    // Apply pre-emphasis
	WindowFunction string `json:"window_function"` // Window function type
	ZeroPadding    int    `json:"zero_padding"`    // Zero padding factor

	// Post-processing options
	MedianFilter      int  `json:"median_filter"` // Median filter length
	TemporalSmoothing bool `json:"temporal_smoothing"`
	OctaveCorrection  bool `json:"octave_correction"`
}

// PitchDetector implements comprehensive pitch detection algorithms
//
// References:
// - de Cheveigné, A., Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music"
// - Boersma, P. (1993). "Accurate short-term analysis of the fundamental frequency"
// - McLeod, P., Wyvill, G. (2005). "A smarter way to find pitch"
// - Rabiner, L.R. (1977). "On the use of autocorrelation analysis for pitch detection"
// - Noll, A.M. (1967). "Cepstrum pitch determination"
// - Schroeder, M.R. (1968). "Period histogram and product spectrum"
// - Talkin, D. (1995). "A robust algorithm for pitch tracking (RAPT)"
//
// Pitch detection is fundamental for:
// - Music transcription and analysis
// - Audio synthesis and processing
// - Speech analysis and recognition
// - Audio effects and processing
// - Music information retrieval
type PitchDetector struct {
	params PitchDetectionParams

	// Analysis components
	fft      *spectral.FFT
	autocorr *stats.AutoCorrelation

	// Internal buffers
	window         []float64
	spectrum       []complex128
	autocorrResult []float64

	// Temporal tracking
	previousPitch     float64
	pitchHistory      []float64
	confidenceHistory []float64

	// Preprocessing
	preEmphasis []float64
	windowFunc  []float64
}

// NewPitchDetector creates a new pitch detector with default parameters
func NewPitchDetector(sampleRate int) *PitchDetector {
	return &PitchDetector{
		params: PitchDetectionParams{
			Method:            AutocorrelationYin,
			SampleRate:        sampleRate,
			WindowSize:        1024,
			HopSize:           512,
			MinFreq:           80.0,   // Low male voice
			MaxFreq:           1000.0, // High female voice
			YinThreshold:      0.15,
			AutocorrThreshold: 0.3,
			CepstralThreshold: 0.3,
			MinConfidence:     0.5,
			MinSalience:       0.1,
			VoicingThreshold:  0.45,
			MaxHarmonics:      10,
			HarmonicTolerance: 0.1,
			PreEmphasis:       true,
			WindowFunction:    "hann",
			ZeroPadding:       2,
			MedianFilter:      3,
			TemporalSmoothing: true,
			OctaveCorrection:  true,
		},
		fft:               spectral.NewFFT(),
		autocorr:          stats.NewAutoCorrelation(2048),
		pitchHistory:      make([]float64, 0),
		confidenceHistory: make([]float64, 0),
	}
}

// NewPitchDetectorWithParams creates a pitch detector with custom parameters
func NewPitchDetectorWithParams(params PitchDetectionParams) *PitchDetector {
	pd := &PitchDetector{
		params:            params,
		fft:               spectral.NewFFT(),
		autocorr:          stats.NewAutoCorrelation(params.WindowSize * 2),
		pitchHistory:      make([]float64, 0),
		confidenceHistory: make([]float64, 0),
	}

	pd.initializeBuffers()
	return pd
}

// initializeBuffers initializes internal buffers and preprocessing
func (pd *PitchDetector) initializeBuffers() {
	pd.window = make([]float64, pd.params.WindowSize)
	pd.spectrum = make([]complex128, pd.params.WindowSize*pd.params.ZeroPadding)
	pd.autocorrResult = make([]float64, pd.params.WindowSize)

	// Initialize window function
	pd.windowFunc = pd.createWindowFunction(pd.params.WindowSize, pd.params.WindowFunction)

	// Initialize pre-emphasis filter
	if pd.params.PreEmphasis {
		pd.preEmphasis = make([]float64, pd.params.WindowSize)
	}
}

// DetectPitch detects pitch in a single audio frame
func (pd *PitchDetector) DetectPitch(audioFrame []float64) (*PitchDetectionResult, error) {
	if len(audioFrame) != pd.params.WindowSize {
		return nil, fmt.Errorf("audio frame size (%d) doesn't match window size (%d)", len(audioFrame), pd.params.WindowSize)
	}

	startTime := pd.getCurrentTime()

	// Preprocess audio frame
	processedFrame := pd.preprocessFrame(audioFrame)

	// Detect pitch using selected method
	var result *PitchDetectionResult
	var err error

	switch pd.params.Method {
	case AutocorrelationYin:
		result, err = pd.detectPitchYin(processedFrame)
	case AutocorrelationACF:
		result, err = pd.detectPitchACF(processedFrame)
	case AutocorrelationNSDF:
		result, err = pd.detectPitchNSDF(processedFrame)
	case FrequencyDomainHPS:
		result, err = pd.detectPitchHPS(processedFrame)
	case FrequencyDomainCepstrum:
		result, err = pd.detectPitchCepstrum(processedFrame)
	case FrequencyDomainPeaks:
		result, err = pd.detectPitchPeaks(processedFrame)
	case TimeDomainZeroCrossing:
		result, err = pd.detectPitchZeroCrossing(processedFrame)
	case HybridYinFFT:
		result, err = pd.detectPitchYinFFT(processedFrame)
	case HybridMPM:
		result, err = pd.detectPitchMPM(processedFrame)
	default:
		return nil, fmt.Errorf("unsupported pitch detection method: %d", pd.params.Method)
	}

	if err != nil {
		return nil, err
	}

	// Post-process results
	result = pd.postProcessResult(result)

	// Update temporal tracking
	pd.updateTemporalTracking(result)

	// Set computational details
	result.Method = pd.params.Method
	result.SampleRate = pd.params.SampleRate
	result.WindowSize = pd.params.WindowSize
	result.ProcessTime = pd.getCurrentTime() - startTime

	return result, nil
}

// preprocessFrame applies preprocessing to the audio frame
func (pd *PitchDetector) preprocessFrame(audioFrame []float64) []float64 {
	processed := make([]float64, len(audioFrame))
	copy(processed, audioFrame)

	// Apply pre-emphasis
	if pd.params.PreEmphasis {
		processed = pd.applyPreEmphasis(processed)
	}

	// Apply window function
	for i := range processed {
		processed[i] *= pd.windowFunc[i]
	}

	return processed
}

// applyPreEmphasis applies pre-emphasis filtering
func (pd *PitchDetector) applyPreEmphasis(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	result := make([]float64, len(signal))
	result[0] = signal[0]

	// Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
	for i := 1; i < len(signal); i++ {
		result[i] = signal[i] - 0.97*signal[i-1]
	}

	return result
}

// createWindowFunction creates a window function of specified type
func (pd *PitchDetector) createWindowFunction(size int, windowType string) []float64 {
	window := make([]float64, size)

	switch windowType {
	case "hann":
		for i := range window {
			window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1)))
		}
	case "hamming":
		for i := range window {
			window[i] = 0.54 - 0.46*math.Cos(2.0*math.Pi*float64(i)/float64(size-1))
		}
	case "blackman":
		for i := range window {
			window[i] = 0.42 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(size-1)) + 0.08*math.Cos(4.0*math.Pi*float64(i)/float64(size-1))
		}
	case "rectangular":
		for i := range window {
			window[i] = 1.0
		}
	default:
		// Default to Hann window
		for i := range window {
			window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1)))
		}
	}

	return window
}

// detectPitchYin implements the YIN pitch detection algorithm
// Reference: de Cheveigné, A., Kawahara, H. (2002)
func (pd *PitchDetector) detectPitchYin(audioFrame []float64) (*PitchDetectionResult, error) {
	n := len(audioFrame)
	halfN := n / 2

	// Calculate difference function
	diff := make([]float64, halfN)
	for tau := range halfN {
		sum := 0.0
		for j := range halfN {
			delta := audioFrame[j] - audioFrame[j+tau]
			sum += delta * delta
		}
		diff[tau] = sum
	}

	// Calculate cumulative mean normalized difference function
	cmndf := make([]float64, halfN)
	cmndf[0] = 1.0

	runningSum := 0.0
	for tau := 1; tau < halfN; tau++ {
		runningSum += diff[tau]
		cmndf[tau] = diff[tau] / (runningSum / float64(tau))
	}

	// Find the first minimum below threshold
	minTau := -1
	for tau := 1; tau < halfN; tau++ {
		if cmndf[tau] < pd.params.YinThreshold {
			// Check if this is a local minimum
			if tau+1 < halfN && cmndf[tau] < cmndf[tau+1] {
				minTau = tau
				break
			}
		}
	}

	// Create result
	result := &PitchDetectionResult{
		YinThreshold: pd.params.YinThreshold,
	}

	if minTau > 0 {
		// Parabolic interpolation for better accuracy
		period := pd.parabolicInterpolation(cmndf, minTau)
		frequency := float64(pd.params.SampleRate) / period

		// Calculate confidence
		confidence := 1.0 - cmndf[minTau]

		// Validate frequency range
		if frequency >= pd.params.MinFreq && frequency <= pd.params.MaxFreq {
			result.Pitch = frequency
			result.Confidence = confidence
			result.Periodicity = confidence
			result.Voicing = confidence

			// Create primary candidate
			result.Candidates = []PitchCandidate{
				{
					Frequency:  frequency,
					Confidence: confidence,
					Salience:   confidence,
					Harmonic:   1,
					Method:     "YIN",
				},
			}
		}
	}

	return result, nil
}

// detectPitchACF implements autocorrelation-based pitch detection
func (pd *PitchDetector) detectPitchACF(audioFrame []float64) (*PitchDetectionResult, error) {
	// Compute autocorrelation
	autocorrResult, err := pd.autocorr.Compute(audioFrame)
	if err != nil {
		return nil, err
	}

	correlations := autocorrResult.Correlations
	lags := autocorrResult.Lags

	// Find peaks in autocorrelation
	candidates := make([]PitchCandidate, 0)

	for i := 1; i < len(correlations)-1; i++ {
		lag := lags[i]
		corr := correlations[i]

		// Skip negative lags and lag 0
		if lag <= 0 {
			continue
		}

		// Check for local maximum
		if corr > correlations[i-1] && corr > correlations[i+1] && corr > pd.params.AutocorrThreshold {
			frequency := float64(pd.params.SampleRate) / float64(lag)

			// Validate frequency range
			if frequency >= pd.params.MinFreq && frequency <= pd.params.MaxFreq {
				candidates = append(candidates, PitchCandidate{
					Frequency:  frequency,
					Confidence: corr,
					Salience:   corr,
					Harmonic:   1,
					Method:     "ACF",
				})
			}
		}
	}

	// Sort candidates by confidence
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Confidence > candidates[j].Confidence
	})

	// Create result
	result := &PitchDetectionResult{
		Candidates:    candidates,
		AutocorrPeaks: correlations,
	}

	if len(candidates) > 0 {
		best := candidates[0]
		result.Pitch = best.Frequency
		result.Confidence = best.Confidence
		result.Periodicity = best.Confidence
		result.Voicing = best.Confidence
	}

	return result, nil
}

// detectPitchNSDF implements Normalized Square Difference Function
func (pd *PitchDetector) detectPitchNSDF(audioFrame []float64) (*PitchDetectionResult, error) {
	n := len(audioFrame)
	halfN := n / 2

	// Calculate NSDF
	nsdf := make([]float64, halfN)

	for tau := range halfN {
		acf := 0.0 // Autocorrelation
		m1 := 0.0  // Mean square at lag 0
		m2 := 0.0  // Mean square at lag tau

		for j := range halfN {
			x1 := audioFrame[j]
			x2 := audioFrame[j+tau]

			acf += x1 * x2
			m1 += x1 * x1
			m2 += x2 * x2
		}

		// NSDF = 2 * ACF / (m1 + m2)
		if m1+m2 > 0 {
			nsdf[tau] = 2.0 * acf / (m1 + m2)
		}
	}

	// Find peaks
	candidates := make([]PitchCandidate, 0)

	for i := 1; i < len(nsdf)-1; i++ {
		if nsdf[i] > nsdf[i-1] && nsdf[i] > nsdf[i+1] && nsdf[i] > pd.params.AutocorrThreshold {
			frequency := float64(pd.params.SampleRate) / float64(i)

			if frequency >= pd.params.MinFreq && frequency <= pd.params.MaxFreq {
				candidates = append(candidates, PitchCandidate{
					Frequency:  frequency,
					Confidence: nsdf[i],
					Salience:   nsdf[i],
					Harmonic:   1,
					Method:     "NSDF",
				})
			}
		}
	}

	// Sort candidates by confidence
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Confidence > candidates[j].Confidence
	})

	// Create result
	result := &PitchDetectionResult{
		Candidates: candidates,
	}

	if len(candidates) > 0 {
		best := candidates[0]
		result.Pitch = best.Frequency
		result.Confidence = best.Confidence
		result.Periodicity = best.Confidence
		result.Voicing = best.Confidence
	}

	return result, nil
}

// detectPitchHPS implements Harmonic Product Spectrum method
func (pd *PitchDetector) detectPitchHPS(audioFrame []float64) (*PitchDetectionResult, error) {
	// Compute FFT
	fftSize := pd.params.WindowSize * pd.params.ZeroPadding
	paddedFrame := make([]float64, fftSize)
	copy(paddedFrame, audioFrame)

	spectrum := pd.fft.Compute(paddedFrame)

	// Compute magnitude spectrum
	magnitude := make([]float64, len(spectrum)/2)
	for i := range magnitude {
		magnitude[i] = math.Sqrt(real(spectrum[i])*real(spectrum[i]) + imag(spectrum[i])*imag(spectrum[i]))
	}

	// Compute harmonic product spectrum
	hps := make([]float64, len(magnitude))
	copy(hps, magnitude)

	maxHarmonics := 5
	for h := 2; h <= maxHarmonics; h++ {
		for i := 0; i < len(hps)/h; i++ {
			hps[i] *= magnitude[i*h]
		}
	}

	// Find peak in HPS
	maxIdx := 0
	maxVal := hps[0]

	minBin := int(pd.params.MinFreq * float64(fftSize) / float64(pd.params.SampleRate))
	maxBin := int(pd.params.MaxFreq * float64(fftSize) / float64(pd.params.SampleRate))

	for i := minBin; i < maxBin && i < len(hps); i++ {
		if hps[i] > maxVal {
			maxVal = hps[i]
			maxIdx = i
		}
	}

	// Convert bin to frequency
	frequency := float64(maxIdx) * float64(pd.params.SampleRate) / float64(fftSize)

	// Calculate confidence based on peak prominence
	confidence := 0.0
	if maxVal > 0 {
		// Simple confidence measure
		confidence = math.Min(maxVal/1000.0, 1.0)
	}

	// Create result
	result := &PitchDetectionResult{
		Pitch:       frequency,
		Confidence:  confidence,
		Periodicity: confidence,
		Voicing:     confidence,
		Candidates: []PitchCandidate{
			{
				Frequency:  frequency,
				Confidence: confidence,
				Salience:   confidence,
				Harmonic:   1,
				Method:     "HPS",
			},
		},
	}

	return result, nil
}

// detectPitchCepstrum implements cepstral pitch detection
func (pd *PitchDetector) detectPitchCepstrum(audioFrame []float64) (*PitchDetectionResult, error) {
	// Compute FFT
	spectrum := pd.fft.Compute(audioFrame)

	logMag := make([]complex128, len(spectrum))
	for i := range spectrum {
		// Correct magnitude calculation: sqrt(real^2 + imag^2)
		magnitude := cmplx.Abs(spectrum[i])

		// Convert log result to complex128 (real part only)
		logMag[i] = complex(math.Log(magnitude+1e-10), 0)
	}

	// Compute inverse FFT to get cepstrum
	cepstrum := pd.fft.ComputeInverse(logMag)

	// Convert to real cepstrum
	realCepstrum := make([]float64, len(cepstrum))
	for i := range cepstrum {
		realCepstrum[i] = real(cepstrum[i])
	}

	// Find peak in cepstrum
	minQuefrency := int(float64(pd.params.SampleRate) / pd.params.MaxFreq)
	maxQuefrency := int(float64(pd.params.SampleRate) / pd.params.MinFreq)

	maxIdx := minQuefrency
	maxVal := realCepstrum[minQuefrency]

	for i := minQuefrency; i < maxQuefrency && i < len(realCepstrum); i++ {
		if realCepstrum[i] > maxVal {
			maxVal = realCepstrum[i]
			maxIdx = i
		}
	}

	// Convert quefrency to frequency
	frequency := float64(pd.params.SampleRate) / float64(maxIdx)

	// Calculate confidence
	confidence := math.Min(maxVal/0.1, 1.0)

	// Create result
	result := &PitchDetectionResult{
		Pitch:        frequency,
		Confidence:   confidence,
		Periodicity:  confidence,
		Voicing:      confidence,
		CepstralPeak: float64(maxIdx),
		Candidates: []PitchCandidate{
			{
				Frequency:  frequency,
				Confidence: confidence,
				Salience:   confidence,
				Harmonic:   1,
				Method:     "Cepstrum",
			},
		},
	}

	return result, nil
}

// detectPitchPeaks implements spectral peak-based pitch detection
func (pd *PitchDetector) detectPitchPeaks(audioFrame []float64) (*PitchDetectionResult, error) {
	// This is a simplified version - would need SpectralPeaks from harmonic package
	// For now, use HPS as fallback
	return pd.detectPitchHPS(audioFrame)
}

// detectPitchZeroCrossing implements zero-crossing rate pitch detection
func (pd *PitchDetector) detectPitchZeroCrossing(audioFrame []float64) (*PitchDetectionResult, error) {
	// Count zero crossings
	crossings := 0
	for i := 1; i < len(audioFrame); i++ {
		if (audioFrame[i] > 0 && audioFrame[i-1] <= 0) || (audioFrame[i] <= 0 && audioFrame[i-1] > 0) {
			crossings++
		}
	}

	// Estimate frequency (very rough)
	frequency := float64(crossings) * float64(pd.params.SampleRate) / (2.0 * float64(len(audioFrame)))

	// Simple confidence based on regularity
	confidence := 0.3 // Low confidence for this simple method

	// Create result
	result := &PitchDetectionResult{
		Pitch:       frequency,
		Confidence:  confidence,
		Periodicity: confidence,
		Voicing:     confidence,
		Candidates: []PitchCandidate{
			{
				Frequency:  frequency,
				Confidence: confidence,
				Salience:   confidence,
				Harmonic:   1,
				Method:     "ZeroCrossing",
			},
		},
	}

	return result, nil
}

// detectPitchYinFFT implements YIN-FFT hybrid method
func (pd *PitchDetector) detectPitchYinFFT(audioFrame []float64) (*PitchDetectionResult, error) {
	// For now, use standard YIN
	return pd.detectPitchYin(audioFrame)
}

// detectPitchMPM implements McLeod Pitch Method
func (pd *PitchDetector) detectPitchMPM(audioFrame []float64) (*PitchDetectionResult, error) {
	// This would implement the full MPM algorithm
	// For now, use NSDF as it's similar
	return pd.detectPitchNSDF(audioFrame)
}

// parabolicInterpolation performs parabolic interpolation for better frequency accuracy
func (pd *PitchDetector) parabolicInterpolation(data []float64, peakIdx int) float64 {
	if peakIdx <= 0 || peakIdx >= len(data)-1 {
		return float64(peakIdx)
	}

	y1 := data[peakIdx-1]
	y2 := data[peakIdx]
	y3 := data[peakIdx+1]

	// Parabolic interpolation formula
	a := (y1 - 2*y2 + y3) / 2
	b := (y3 - y1) / 2

	if a == 0 {
		return float64(peakIdx)
	}

	// Peak location
	xPeak := -b / (2 * a)

	return float64(peakIdx) + xPeak
}

// postProcessResult applies post-processing to the pitch detection result
func (pd *PitchDetector) postProcessResult(result *PitchDetectionResult) *PitchDetectionResult {
	if result == nil {
		return result
	}

	// Apply octave correction
	if pd.params.OctaveCorrection {
		result = pd.applyOctaveCorrection(result)
	}

	// Calculate additional quality metrics
	result.Clarity = pd.calculateClarity(result)
	result.Strength = pd.calculateStrength(result)
	result.Salience = pd.calculateSalience(result)

	// Apply confidence thresholds
	if result.Confidence < pd.params.MinConfidence {
		result.Pitch = 0.0
		result.Confidence = 0.0
		result.Voicing = 0.0
	}

	return result
}

// applyOctaveCorrection attempts to correct octave errors
func (pd *PitchDetector) applyOctaveCorrection(result *PitchDetectionResult) *PitchDetectionResult {
	if result.Pitch == 0.0 || len(pd.pitchHistory) == 0 {
		return result
	}

	// Get recent pitch history
	recentPitches := pd.getRecentPitches(5)
	if len(recentPitches) < 3 {
		return result
	}

	// Calculate median of recent pitches
	medianPitch := pd.calculateMedian(recentPitches)

	// Check for octave errors
	currentPitch := result.Pitch

	// Check if current pitch is roughly an octave off
	ratios := []float64{0.5, 2.0, 1.0 / 3.0, 3.0}
	tolerance := 0.1

	for _, ratio := range ratios {
		expectedPitch := medianPitch * ratio
		if math.Abs(currentPitch-expectedPitch)/expectedPitch < tolerance {
			// Prefer the pitch closer to the median
			if math.Abs(currentPitch-medianPitch) > math.Abs(expectedPitch-medianPitch) {
				result.Pitch = expectedPitch
				result.F0Multiple = ratio
			}
			break
		}
	}

	return result
}

// calculateClarity calculates pitch clarity measure
func (pd *PitchDetector) calculateClarity(result *PitchDetectionResult) float64 {
	if len(result.Candidates) == 0 {
		return 0.0
	}

	// Clarity based on the difference between best and second-best candidates
	if len(result.Candidates) == 1 {
		return result.Candidates[0].Confidence
	}

	best := result.Candidates[0].Confidence
	second := result.Candidates[1].Confidence

	// Clarity = (best - second) / best
	if best > 0 {
		return (best - second) / best
	}

	return 0.0
}

// calculateStrength calculates pitch strength measure
func (pd *PitchDetector) calculateStrength(result *PitchDetectionResult) float64 {
	// Strength based on periodicity and voicing
	return (result.Periodicity + result.Voicing) / 2.0
}

// calculateSalience calculates perceptual salience measure
func (pd *PitchDetector) calculateSalience(result *PitchDetectionResult) float64 {
	// Salience based on confidence and harmonic strength
	baseScore := result.Confidence

	// Boost salience for frequencies in the perceptually important range
	if result.Pitch >= 200 && result.Pitch <= 800 {
		baseScore *= 1.2
	}

	// Reduce salience for very high or low frequencies
	if result.Pitch < 100 || result.Pitch > 1000 {
		baseScore *= 0.8
	}

	return math.Min(baseScore, 1.0)
}

// updateTemporalTracking updates the temporal tracking state
func (pd *PitchDetector) updateTemporalTracking(result *PitchDetectionResult) {
	if result == nil {
		return
	}

	// Update pitch history
	pd.pitchHistory = append(pd.pitchHistory, result.Pitch)
	pd.confidenceHistory = append(pd.confidenceHistory, result.Confidence)

	// Keep only recent history
	maxHistory := 20
	if len(pd.pitchHistory) > maxHistory {
		pd.pitchHistory = pd.pitchHistory[len(pd.pitchHistory)-maxHistory:]
		pd.confidenceHistory = pd.confidenceHistory[len(pd.confidenceHistory)-maxHistory:]
	}

	// Apply temporal smoothing
	if pd.params.TemporalSmoothing && len(pd.pitchHistory) > 1 {
		result.Pitch = pd.applyTemporalSmoothing(result.Pitch)
	}

	// Calculate stability
	result.Stability = pd.calculateStability()

	// Update previous pitch
	pd.previousPitch = result.Pitch
}

// applyTemporalSmoothing applies temporal smoothing to pitch estimates
func (pd *PitchDetector) applyTemporalSmoothing(currentPitch float64) float64 {
	if len(pd.pitchHistory) < 2 {
		return currentPitch
	}

	// Apply median filter if enabled
	if pd.params.MedianFilter > 0 {
		recentPitches := pd.getRecentPitches(pd.params.MedianFilter)
		if len(recentPitches) >= 3 {
			return pd.calculateMedian(recentPitches)
		}
	}

	// Simple exponential smoothing
	alpha := 0.3
	return alpha*currentPitch + (1-alpha)*pd.previousPitch
}

// calculateStability calculates pitch stability measure
func (pd *PitchDetector) calculateStability() float64 {
	if len(pd.pitchHistory) < 3 {
		return 0.0
	}

	// Calculate coefficient of variation
	validPitches := make([]float64, 0)
	for _, pitch := range pd.pitchHistory {
		if pitch > 0 {
			validPitches = append(validPitches, pitch)
		}
	}

	if len(validPitches) < 2 {
		return 0.0
	}

	// Calculate mean and standard deviation
	mean := 0.0
	for _, pitch := range validPitches {
		mean += pitch
	}
	mean /= float64(len(validPitches))

	variance := 0.0
	for _, pitch := range validPitches {
		diff := pitch - mean
		variance += diff * diff
	}
	variance /= float64(len(validPitches) - 1)
	stdDev := math.Sqrt(variance)

	// Stability = 1 - coefficient of variation
	if mean > 0 {
		cv := stdDev / mean
		return math.Max(0.0, 1.0-cv)
	}

	return 0.0
}

// getRecentPitches returns the most recent pitch values
func (pd *PitchDetector) getRecentPitches(count int) []float64 {
	if count <= 0 || len(pd.pitchHistory) == 0 {
		return []float64{}
	}

	start := len(pd.pitchHistory) - count
	start = max(start, 0)

	return pd.pitchHistory[start:]
}

// calculateMedian calculates the median of a slice of values
func (pd *PitchDetector) calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Filter out zero values
	filtered := make([]float64, 0)
	for _, v := range values {
		if v > 0 {
			filtered = append(filtered, v)
		}
	}

	if len(filtered) == 0 {
		return 0.0
	}

	// Sort values
	sorted := make([]float64, len(filtered))
	copy(sorted, filtered)
	sort.Float64s(sorted)

	// Calculate median
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		return sorted[n/2]
	}
}

// getCurrentTime returns current time in milliseconds (placeholder)
func (pd *PitchDetector) getCurrentTime() float64 {
	// In real implementation, use time.Now().UnixNano() / 1e6
	return 0.0
}

// ProcessAudioStream processes a stream of audio frames
func (pd *PitchDetector) ProcessAudioStream(audioFrames [][]float64) ([]*PitchDetectionResult, error) {
	results := make([]*PitchDetectionResult, len(audioFrames))

	for i, frame := range audioFrames {
		result, err := pd.DetectPitch(frame)
		if err != nil {
			return nil, fmt.Errorf("error processing frame %d: %v", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// Reset resets the pitch detector state
func (pd *PitchDetector) Reset() {
	pd.previousPitch = 0.0
	pd.pitchHistory = make([]float64, 0)
	pd.confidenceHistory = make([]float64, 0)
}

// SetParameters updates the detector parameters
func (pd *PitchDetector) SetParameters(params PitchDetectionParams) {
	pd.params = params
	pd.initializeBuffers()
}

// GetParameters returns the current parameters
func (pd *PitchDetector) GetParameters() PitchDetectionParams {
	return pd.params
}

// GetPitchHistory returns the pitch history
func (pd *PitchDetector) GetPitchHistory() []float64 {
	return pd.pitchHistory
}

// GetConfidenceHistory returns the confidence history
func (pd *PitchDetector) GetConfidenceHistory() []float64 {
	return pd.confidenceHistory
}

// AnalyzePitchStability analyzes pitch stability over time
func (pd *PitchDetector) AnalyzePitchStability(pitchSequence []float64) map[string]float64 {
	analysis := make(map[string]float64)

	if len(pitchSequence) < 2 {
		return analysis
	}

	// Filter out zero/unvoiced frames
	validPitches := make([]float64, 0)
	for _, pitch := range pitchSequence {
		if pitch > 0 {
			validPitches = append(validPitches, pitch)
		}
	}

	if len(validPitches) < 2 {
		return analysis
	}

	// Calculate basic statistics
	mean := 0.0
	for _, pitch := range validPitches {
		mean += pitch
	}
	mean /= float64(len(validPitches))

	variance := 0.0
	for _, pitch := range validPitches {
		diff := pitch - mean
		variance += diff * diff
	}
	variance /= float64(len(validPitches) - 1)
	stdDev := math.Sqrt(variance)

	// Calculate jitter (period-to-period variation)
	jitter := 0.0
	for i := 1; i < len(validPitches); i++ {
		diff := math.Abs(validPitches[i] - validPitches[i-1])
		jitter += diff
	}
	jitter /= float64(len(validPitches) - 1)

	// Calculate vibrato rate (approximate)
	vibratoRate := pd.estimateVibratoRate(validPitches)

	analysis["mean_pitch"] = mean
	analysis["pitch_std_dev"] = stdDev
	analysis["coefficient_of_variation"] = stdDev / mean
	analysis["jitter"] = jitter
	analysis["stability"] = 1.0 / (1.0 + stdDev/mean)
	analysis["vibrato_rate"] = vibratoRate
	analysis["voiced_frames_ratio"] = float64(len(validPitches)) / float64(len(pitchSequence))

	return analysis
}

// estimateVibratoRate estimates vibrato rate from pitch sequence
func (pd *PitchDetector) estimateVibratoRate(pitchSequence []float64) float64 {
	if len(pitchSequence) < 10 {
		return 0.0
	}

	// Simple vibrato detection using zero-crossing of detrended pitch
	// Remove linear trend
	detrended := make([]float64, len(pitchSequence))

	// Calculate linear trend
	n := float64(len(pitchSequence))
	sumX := n * (n - 1) / 2
	sumY := 0.0
	sumXY := 0.0
	sumX2 := (n - 1) * n * (2*n - 1) / 6

	for i, pitch := range pitchSequence {
		sumY += pitch
		sumXY += float64(i) * pitch
	}

	// Linear regression
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	intercept := (sumY - slope*sumX) / n

	// Detrend
	for i := range pitchSequence {
		trend := intercept + slope*float64(i)
		detrended[i] = pitchSequence[i] - trend
	}

	// Count zero crossings
	crossings := 0
	for i := 1; i < len(detrended); i++ {
		if (detrended[i] > 0 && detrended[i-1] <= 0) || (detrended[i] <= 0 && detrended[i-1] > 0) {
			crossings++
		}
	}

	// Convert to rate (Hz) - assuming frames are at hop rate
	hopRate := float64(pd.params.SampleRate) / float64(pd.params.HopSize)
	vibratoRate := float64(crossings) / (2.0 * float64(len(detrended)) / hopRate)

	return vibratoRate
}

// GetPitchDetectionMethodName returns the name of the detection method
func GetPitchDetectionMethodName(method PitchDetectionMethod) string {
	switch method {
	case AutocorrelationYin:
		return "YIN (Autocorrelation)"
	case AutocorrelationACF:
		return "Autocorrelation Function"
	case AutocorrelationNSDF:
		return "Normalized Square Difference Function"
	case FrequencyDomainHPS:
		return "Harmonic Product Spectrum"
	case FrequencyDomainCepstrum:
		return "Cepstral Analysis"
	case FrequencyDomainPeaks:
		return "Spectral Peaks"
	case TimeDomainZeroCrossing:
		return "Zero Crossing Rate"
	case TimeDomainPeakPicking:
		return "Peak Picking"
	case HybridYinFFT:
		return "YIN-FFT Hybrid"
	case HybridMPM:
		return "McLeod Pitch Method"
	case HybridPRAATT:
		return "PRAAT Algorithm"
	default:
		return "Unknown"
	}
}
