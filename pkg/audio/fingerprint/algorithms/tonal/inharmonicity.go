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

// InharmonicityMethod defines different methods for computing inharmonicity
type InharmonicityMethod int

const (
	InharmonicityRailsback         InharmonicityMethod = iota // Railsback curve fitting
	InharmonicityFletcherMunson                               // Fletcher-Munson based
	InharmonicitySpectralDeviation                            // Spectral deviation from harmonics
	InharmonicityPartialTracking                              // Partial tracking method
	InharmonicityBeatPattern                                  // Beat pattern analysis
	InharmonicityStatistical                                  // Statistical deviation
)

// InharmonicityModel represents different physical models
type InharmonicityModel int

const (
	ModelString   InharmonicityModel = iota // String instrument model
	ModelPipe                               // Pipe/wind instrument model
	ModelMembrane                           // Membrane/percussion model
	ModelGeneral                            // General inharmonicity model
)

// InharmonicityParams contains parameters for inharmonicity analysis
type InharmonicityParams struct {
	Method     InharmonicityMethod `json:"method"`
	Model      InharmonicityModel  `json:"model"`
	SampleRate int                 `json:"sample_rate"`
	WindowSize int                 `json:"window_size"`

	// Harmonic analysis parameters
	MaxHarmonics        int     `json:"max_harmonics"`         // Maximum harmonics to analyze
	MinHarmonics        int     `json:"min_harmonics"`         // Minimum harmonics required
	HarmonicTolerance   float64 `json:"harmonic_tolerance"`    // Tolerance for harmonic detection
	MinHarmonicStrength float64 `json:"min_harmonic_strength"` // Minimum harmonic strength

	// Frequency analysis
	MinFreq        float64 `json:"min_freq"`        // Minimum fundamental frequency
	MaxFreq        float64 `json:"max_freq"`        // Maximum fundamental frequency
	FreqResolution float64 `json:"freq_resolution"` // Frequency resolution requirement

	// Inharmonicity calculation
	ReferenceHarmonic int  `json:"reference_harmonic"`  // Reference harmonic (usually 1)
	WeightByAmplitude bool `json:"weight_by_amplitude"` // Weight by harmonic amplitude
	UseLogFrequency   bool `json:"use_log_frequency"`   // Use logarithmic frequency scale

	// Physical model parameters
	StringTension   float64 `json:"string_tension"`   // String tension (for string model)
	StringLength    float64 `json:"string_length"`    // String length
	StringStiffness float64 `json:"string_stiffness"` // String stiffness parameter

	// Statistical parameters
	OutlierThreshold    float64 `json:"outlier_threshold"`     // Threshold for outlier removal
	ConfidenceLevel     float64 `json:"confidence_level"`      // Confidence level for statistics
	UseRobustEstimation bool    `json:"use_robust_estimation"` // Use robust statistical methods

	// Temporal analysis
	UseTemporalTracking bool `json:"use_temporal_tracking"` // Track inharmonicity over time
	TemporalWindow      int  `json:"temporal_window"`       // Window for temporal analysis
}

// InharmonicityResult contains comprehensive inharmonicity analysis results
type InharmonicityResult struct {
	// Primary inharmonicity measures
	Inharmonicity           float64 `json:"inharmonicity"`            // Overall inharmonicity coefficient
	InharmonicityStdDev     float64 `json:"inharmonicity_std_dev"`    // Standard deviation
	InharmonicityConfidence float64 `json:"inharmonicity_confidence"` // Confidence in measurement

	// Detailed harmonic analysis
	HarmonicDeviations  []float64 `json:"harmonic_deviations"`  // Deviation for each harmonic
	HarmonicFrequencies []float64 `json:"harmonic_frequencies"` // Measured harmonic frequencies
	IdealFrequencies    []float64 `json:"ideal_frequencies"`    // Ideal harmonic frequencies
	HarmonicAmplitudes  []float64 `json:"harmonic_amplitudes"`  // Harmonic amplitudes
	HarmonicNumbers     []int     `json:"harmonic_numbers"`     // Harmonic numbers

	// Fundamental frequency analysis
	F0Frequency  float64 `json:"f0_frequency"`  // Fundamental frequency
	F0Confidence float64 `json:"f0_confidence"` // F0 detection confidence
	F0Stability  float64 `json:"f0_stability"`  // F0 stability measure

	// Physical model parameters
	EstimatedStiffness float64            `json:"estimated_stiffness"` // Estimated string stiffness
	ModelFitQuality    float64            `json:"model_fit_quality"`   // Quality of model fit
	ModelParameters    map[string]float64 `json:"model_parameters"`    // Model-specific parameters

	// Quality metrics
	HarmonicClarity  float64 `json:"harmonic_clarity"`  // Clarity of harmonic structure
	SpectralPurity   float64 `json:"spectral_purity"`   // Spectral purity measure
	PartialCoherence float64 `json:"partial_coherence"` // Coherence of partials

	// Statistical measures
	RSquared   float64 `json:"r_squared"`   // Coefficient of determination
	ChiSquared float64 `json:"chi_squared"` // Chi-squared goodness of fit
	PValue     float64 `json:"p_value"`     // Statistical significance

	// Temporal analysis
	TemporalStability  float64 `json:"temporal_stability"`  // Temporal stability
	InharmonicityTrend float64 `json:"inharmonicity_trend"` // Trend over time

	// Analysis metadata
	Method          string  `json:"method"`           // Method used
	Model           string  `json:"model"`            // Physical model used
	NumHarmonics    int     `json:"num_harmonics"`    // Number of harmonics analyzed
	ProcessingTime  float64 `json:"processing_time"`  // Processing time (ms)
	AnalysisQuality string  `json:"analysis_quality"` // Quality assessment
}

// InharmonicityAnalyzer implements inharmonicity analysis
type InharmonicityAnalyzer struct {
	params InharmonicityParams

	// Analysis components
	harmonicAnalyzer *HarmonicRatioAnalyzer
	pitchDetector    *PitchDetector
	peakDetector     *harmonic.SpectralPeaks
	fft              *spectral.FFT

	// Internal state
	spectrum  []complex128
	magnitude []float64
	freqBins  []float64

	// Temporal tracking
	inharmonicityHistory []float64
	f0History            []float64

	// Model fitting
	modelCoefficients []float64
	fitResiduals      []float64

	initialized bool
}

// NewInharmonicityAnalyzer creates a new inharmonicity analyzer
func NewInharmonicityAnalyzer(sampleRate int) *InharmonicityAnalyzer {
	return &InharmonicityAnalyzer{
		params: InharmonicityParams{
			Method:              InharmonicitySpectralDeviation,
			Model:               ModelGeneral,
			SampleRate:          sampleRate,
			WindowSize:          4096,
			MaxHarmonics:        20,
			MinHarmonics:        3,
			HarmonicTolerance:   0.05,
			MinHarmonicStrength: 0.01,
			MinFreq:             80.0,
			MaxFreq:             2000.0,
			FreqResolution:      1.0,
			ReferenceHarmonic:   1,
			WeightByAmplitude:   true,
			UseLogFrequency:     false,
			StringTension:       1.0,
			StringLength:        1.0,
			StringStiffness:     0.001,
			OutlierThreshold:    2.0,
			ConfidenceLevel:     0.95,
			UseRobustEstimation: true,
			UseTemporalTracking: false,
			TemporalWindow:      10,
		},
		harmonicAnalyzer:     NewHarmonicRatioAnalyzer(sampleRate),
		pitchDetector:        NewPitchDetector(sampleRate),
		peakDetector:         harmonic.NewSpectralPeaks(sampleRate, 0.001, 20.0, 100),
		fft:                  spectral.NewFFT(),
		inharmonicityHistory: make([]float64, 0),
		f0History:            make([]float64, 0),
	}
}

// NewInharmonicityAnalyzerWithParams creates analyzer with custom parameters
func NewInharmonicityAnalyzerWithParams(params InharmonicityParams) *InharmonicityAnalyzer {
	ia := &InharmonicityAnalyzer{
		params:               params,
		harmonicAnalyzer:     NewHarmonicRatioAnalyzer(params.SampleRate),
		pitchDetector:        NewPitchDetector(params.SampleRate),
		peakDetector:         harmonic.NewSpectralPeaks(params.SampleRate, 0.001, 20.0, 100),
		fft:                  spectral.NewFFT(),
		inharmonicityHistory: make([]float64, 0),
		f0History:            make([]float64, 0),
	}

	ia.Initialize()
	return ia
}

// Initialize sets up the analyzer
func (ia *InharmonicityAnalyzer) Initialize() {
	if ia.initialized {
		return
	}

	// Initialize buffers
	ia.spectrum = make([]complex128, ia.params.WindowSize)
	ia.magnitude = make([]float64, ia.params.WindowSize/2)

	// Initialize frequency bins
	ia.freqBins = make([]float64, ia.params.WindowSize/2)
	for i := range ia.freqBins {
		ia.freqBins[i] = float64(i) * float64(ia.params.SampleRate) / float64(ia.params.WindowSize)
	}

	ia.initialized = true
}

// AnalyzeFrame analyzes inharmonicity for a single audio frame
func (ia *InharmonicityAnalyzer) AnalyzeFrame(audioFrame []float64) (InharmonicityResult, error) {
	if !ia.initialized {
		ia.Initialize()
	}

	startTime := ia.getCurrentTime()

	// Validate input
	if len(audioFrame) != ia.params.WindowSize {
		return InharmonicityResult{}, fmt.Errorf("audio frame size (%d) doesn't match window size (%d)", len(audioFrame), ia.params.WindowSize)
	}

	// Detect fundamental frequency
	f0, f0Confidence, err := ia.detectFundamentalFrequency(audioFrame)
	if err != nil {
		return InharmonicityResult{}, err
	}

	if f0 <= 0 || f0 < ia.params.MinFreq || f0 > ia.params.MaxFreq {
		return InharmonicityResult{
			F0Frequency:     f0,
			F0Confidence:    f0Confidence,
			AnalysisQuality: "Poor - Invalid F0",
		}, nil
	}

	// Analyze harmonic structure
	harmonicData, err := ia.analyzeHarmonicStructure(audioFrame, f0)
	if err != nil {
		return InharmonicityResult{}, err
	}

	// Calculate inharmonicity using selected method
	var result InharmonicityResult
	switch ia.params.Method {
	case InharmonicityRailsback:
		result = ia.calculateRailsbackInharmonicity(harmonicData, f0)
	case InharmonicityFletcherMunson:
		result = ia.calculateFletcherMunsonInharmonicity(harmonicData, f0)
	case InharmonicitySpectralDeviation:
		result = ia.calculateSpectralDeviationInharmonicity(harmonicData, f0)
	case InharmonicityPartialTracking:
		result = ia.calculatePartialTrackingInharmonicity(harmonicData, f0)
	case InharmonicityBeatPattern:
		result = ia.calculateBeatPatternInharmonicity(audioFrame, f0)
	case InharmonicityStatistical:
		result = ia.calculateStatisticalInharmonicity(harmonicData, f0)
	default:
		result = ia.calculateSpectralDeviationInharmonicity(harmonicData, f0)
	}

	// Set basic information
	result.F0Frequency = f0
	result.F0Confidence = f0Confidence
	result.NumHarmonics = len(harmonicData.frequencies)

	// Apply physical model if specified
	if ia.params.Model != ModelGeneral {
		result = ia.applyPhysicalModel(result, harmonicData)
	}

	// Calculate quality metrics
	result = ia.calculateQualityMetrics(result, harmonicData)

	// Post-process results
	result = ia.postProcessResult(result)

	// Update temporal tracking
	ia.updateTemporalTracking(result)

	// Set metadata
	result.Method = ia.getMethodName(ia.params.Method)
	result.Model = ia.getModelName(ia.params.Model)
	result.ProcessingTime = ia.getCurrentTime() - startTime
	result.AnalysisQuality = ia.assessAnalysisQuality(result)

	return result, nil
}

// HarmonicData represents analyzed harmonic structure
type HarmonicData struct {
	frequencies  []float64 // Measured harmonic frequencies
	amplitudes   []float64 // Harmonic amplitudes
	phases       []float64 // Harmonic phases
	harmonicNums []int     // Harmonic numbers
	idealFreqs   []float64 // Ideal harmonic frequencies
	deviations   []float64 // Frequency deviations
}

// detectFundamentalFrequency detects the fundamental frequency
func (ia *InharmonicityAnalyzer) detectFundamentalFrequency(audioFrame []float64) (float64, float64, error) {
	// Use pitch detector for F0 estimation
	pitchResult, err := ia.pitchDetector.DetectPitch(audioFrame)
	if err != nil {
		return 0.0, 0.0, err
	}

	return pitchResult.Pitch, pitchResult.Confidence, nil
}

// analyzeHarmonicStructure analyzes the harmonic structure
func (ia *InharmonicityAnalyzer) analyzeHarmonicStructure(audioFrame []float64, f0 float64) (*HarmonicData, error) {
	// Compute spectrum
	ia.spectrum = ia.fft.Compute(audioFrame)

	// Extract magnitude
	for i := 0; i < len(ia.magnitude); i++ {
		real := real(ia.spectrum[i])
		imag := imag(ia.spectrum[i])
		ia.magnitude[i] = math.Sqrt(real*real + imag*imag)
	}

	// Find harmonic peaks
	harmonicData := &HarmonicData{
		frequencies:  make([]float64, 0),
		amplitudes:   make([]float64, 0),
		phases:       make([]float64, 0),
		harmonicNums: make([]int, 0),
		idealFreqs:   make([]float64, 0),
		deviations:   make([]float64, 0),
	}

	// Search for harmonics
	for h := 1; h <= ia.params.MaxHarmonics; h++ {
		idealFreq := f0 * float64(h)

		// Skip if beyond analysis range
		if idealFreq > ia.params.MaxFreq {
			break
		}

		// Find peak near ideal harmonic frequency
		peak := ia.findHarmonicPeak(idealFreq)
		if peak != nil && peak.Magnitude > ia.params.MinHarmonicStrength {
			// Calculate deviation
			deviation := (peak.Frequency - idealFreq) / idealFreq

			// Check if within tolerance
			if math.Abs(deviation) < ia.params.HarmonicTolerance {
				harmonicData.frequencies = append(harmonicData.frequencies, peak.Frequency)
				harmonicData.amplitudes = append(harmonicData.amplitudes, peak.Magnitude)
				harmonicData.phases = append(harmonicData.phases, peak.Phase)
				harmonicData.harmonicNums = append(harmonicData.harmonicNums, h)
				harmonicData.idealFreqs = append(harmonicData.idealFreqs, idealFreq)
				harmonicData.deviations = append(harmonicData.deviations, deviation)
			}
		}
	}

	// Check if we have enough harmonics
	if len(harmonicData.frequencies) < ia.params.MinHarmonics {
		return nil, fmt.Errorf("insufficient harmonics found: %d < %d", len(harmonicData.frequencies), ia.params.MinHarmonics)
	}

	return harmonicData, nil
}

// findHarmonicPeak finds a harmonic peak near the expected frequency
func (ia *InharmonicityAnalyzer) findHarmonicPeak(expectedFreq float64) *harmonic.SpectralPeak {
	// Convert frequency to bin index
	expectedBin := expectedFreq * float64(ia.params.WindowSize) / float64(ia.params.SampleRate)

	// Search in a window around expected bin
	tolerance := ia.params.HarmonicTolerance * expectedFreq
	toleranceBins := tolerance * float64(ia.params.WindowSize) / float64(ia.params.SampleRate)

	startBin := int(math.Max(0, expectedBin-toleranceBins))
	endBin := int(math.Min(float64(len(ia.magnitude)-1), expectedBin+toleranceBins))

	// Find maximum in search window
	maxMag := 0.0
	maxBin := startBin

	for i := startBin; i <= endBin; i++ {
		if ia.magnitude[i] > maxMag {
			maxMag = ia.magnitude[i]
			maxBin = i
		}
	}

	// Verify it's a local peak
	if maxBin > 0 && maxBin < len(ia.magnitude)-1 {
		if ia.magnitude[maxBin] > ia.magnitude[maxBin-1] &&
			ia.magnitude[maxBin] > ia.magnitude[maxBin+1] {

			freq := ia.freqBins[maxBin]
			phase := math.Atan2(imag(ia.spectrum[maxBin]), real(ia.spectrum[maxBin]))

			return &harmonic.SpectralPeak{
				Frequency: freq,
				Magnitude: maxMag,
				Phase:     phase,
				BinIndex:  maxBin,
			}
		}
	}

	return nil
}

// calculateSpectralDeviationInharmonicity calculates inharmonicity using spectral deviation
func (ia *InharmonicityAnalyzer) calculateSpectralDeviationInharmonicity(harmonicData *HarmonicData, f0 float64) InharmonicityResult {
	if len(harmonicData.deviations) == 0 {
		return InharmonicityResult{}
	}

	// Calculate weighted inharmonicity coefficient
	var numerator, denominator float64

	for i, deviation := range harmonicData.deviations {
		harmonicNum := float64(harmonicData.harmonicNums[i])
		weight := 1.0

		if ia.params.WeightByAmplitude {
			weight = harmonicData.amplitudes[i]
		}

		// Inharmonicity formula: B * n^2 where B is inharmonicity coefficient
		expectedDeviation := ia.calculateExpectedDeviation(harmonicNum)

		numerator += weight * (deviation - expectedDeviation) * harmonicNum * harmonicNum
		denominator += weight * harmonicNum * harmonicNum * harmonicNum * harmonicNum
	}

	var inharmonicity float64
	if denominator > 0 {
		inharmonicity = numerator / denominator
	}

	// Calculate standard deviation
	deviationVariance := common.Variance(harmonicData.deviations)
	inharmonicityStdDev := math.Sqrt(deviationVariance)

	// Calculate confidence based on data quality
	confidence := ia.calculateConfidence(harmonicData, inharmonicity)

	return InharmonicityResult{
		Inharmonicity:           inharmonicity,
		InharmonicityStdDev:     inharmonicityStdDev,
		InharmonicityConfidence: confidence,
		HarmonicDeviations:      harmonicData.deviations,
		HarmonicFrequencies:     harmonicData.frequencies,
		IdealFrequencies:        harmonicData.idealFreqs,
		HarmonicAmplitudes:      harmonicData.amplitudes,
		HarmonicNumbers:         harmonicData.harmonicNums,
	}
}

// calculateRailsbackInharmonicity calculates inharmonicity using Railsback curve fitting
func (ia *InharmonicityAnalyzer) calculateRailsbackInharmonicity(harmonicData *HarmonicData, f0 float64) InharmonicityResult {
	// Railsback curve: f_n = n * f_0 * sqrt(1 + B * n^2)
	// Where B is the inharmonicity coefficient

	if len(harmonicData.frequencies) < 3 {
		return InharmonicityResult{}
	}

	// Fit B parameter using least squares
	bestB := ia.fitRailsbackCurve(harmonicData, f0)

	// Calculate goodness of fit
	rSquared := ia.calculateRSquared(harmonicData, f0, bestB)

	// Calculate deviations with fitted model
	deviations := make([]float64, len(harmonicData.frequencies))
	for i := range harmonicData.frequencies {
		n := float64(harmonicData.harmonicNums[i])
		expectedFreq := n * f0 * math.Sqrt(1+bestB*n*n)
		deviations[i] = (harmonicData.frequencies[i] - expectedFreq) / expectedFreq
	}

	confidence := math.Max(0.0, rSquared)

	result := InharmonicityResult{
		Inharmonicity:           bestB,
		InharmonicityStdDev:     math.Sqrt(common.Variance(deviations)),
		InharmonicityConfidence: confidence,
		HarmonicDeviations:      deviations,
		HarmonicFrequencies:     harmonicData.frequencies,
		IdealFrequencies:        harmonicData.idealFreqs,
		HarmonicAmplitudes:      harmonicData.amplitudes,
		HarmonicNumbers:         harmonicData.harmonicNums,
		RSquared:                rSquared,
		ModelFitQuality:         rSquared,
	}

	result.ModelParameters = map[string]float64{
		"railsback_coefficient": bestB,
		"fundamental_frequency": f0,
	}

	return result
}

// calculateFletcherMunsonInharmonicity calculates inharmonicity using Fletcher-Munson weighting
func (ia *InharmonicityAnalyzer) calculateFletcherMunsonInharmonicity(harmonicData *HarmonicData, f0 float64) InharmonicityResult {
	// Apply Fletcher-Munson equal loudness weighting to harmonic analysis

	weightedDeviations := make([]float64, len(harmonicData.deviations))

	for i, deviation := range harmonicData.deviations {
		freq := harmonicData.frequencies[i]

		// Calculate Fletcher-Munson weighting
		weight := ia.calculateFletcherMunsonWeight(freq)
		weightedDeviations[i] = deviation * weight
	}

	// Calculate weighted inharmonicity
	inharmonicity := common.Mean(weightedDeviations)
	inharmonicityStdDev := math.Sqrt(common.Variance(weightedDeviations))

	confidence := ia.calculateConfidence(harmonicData, inharmonicity)

	return InharmonicityResult{
		Inharmonicity:           inharmonicity,
		InharmonicityStdDev:     inharmonicityStdDev,
		InharmonicityConfidence: confidence,
		HarmonicDeviations:      weightedDeviations,
		HarmonicFrequencies:     harmonicData.frequencies,
		IdealFrequencies:        harmonicData.idealFreqs,
		HarmonicAmplitudes:      harmonicData.amplitudes,
		HarmonicNumbers:         harmonicData.harmonicNums,
	}
}

// calculatePartialTrackingInharmonicity calculates inharmonicity using partial tracking
func (ia *InharmonicityAnalyzer) calculatePartialTrackingInharmonicity(harmonicData *HarmonicData, f0 float64) InharmonicityResult {
	// This would implement partial tracking across time
	// For now, fallback to spectral deviation method
	return ia.calculateSpectralDeviationInharmonicity(harmonicData, f0)
}

// calculateBeatPatternInharmonicity calculates inharmonicity using beat pattern analysis
func (ia *InharmonicityAnalyzer) calculateBeatPatternInharmonicity(audioFrame []float64, f0 float64) InharmonicityResult {
	// This would analyze beat patterns between near-harmonic partials
	// For now, use a simplified approach

	// Compute autocorrelation to detect beating
	autocorr := stats.NewAutoCorrelation(len(audioFrame))
	autocorrResult, err := autocorr.Compute(audioFrame)
	if err != nil {
		return InharmonicityResult{}
	}

	// Analyze periodicity variations
	periodicityVariation := ia.analyzePeriodicityVariation(autocorrResult)

	// Convert to inharmonicity estimate
	inharmonicity := periodicityVariation * 0.001 // Rough conversion

	return InharmonicityResult{
		Inharmonicity: inharmonicity,
		F0Frequency:   f0,
	}
}

// calculateStatisticalInharmonicity calculates inharmonicity using statistical methods
func (ia *InharmonicityAnalyzer) calculateStatisticalInharmonicity(harmonicData *HarmonicData, f0 float64) InharmonicityResult {
	if len(harmonicData.deviations) == 0 {
		return InharmonicityResult{}
	}

	// Remove outliers if requested
	deviations := harmonicData.deviations
	if ia.params.UseRobustEstimation {
		deviations = ia.removeOutliers(deviations)
	}

	// Calculate robust statistics
	mean := common.Mean(deviations)
	variance := common.Variance(deviations)
	stdDev := math.Sqrt(variance)

	// Calculate confidence interval
	confidence := ia.calculateStatisticalConfidence(deviations)

	// Perform goodness of fit test
	chiSquared, pValue := ia.performGoodnessOfFitTest(deviations)

	return InharmonicityResult{
		Inharmonicity:           mean,
		InharmonicityStdDev:     stdDev,
		InharmonicityConfidence: confidence,
		HarmonicDeviations:      deviations,
		HarmonicFrequencies:     harmonicData.frequencies,
		IdealFrequencies:        harmonicData.idealFreqs,
		HarmonicAmplitudes:      harmonicData.amplitudes,
		HarmonicNumbers:         harmonicData.harmonicNums,
		ChiSquared:              chiSquared,
		PValue:                  pValue,
	}
}

// Helper functions

func (ia *InharmonicityAnalyzer) calculateExpectedDeviation(harmonicNum float64) float64 {
	// For ideal harmonics, expected deviation is 0
	// This could be modified for specific physical models
	return 0.0
}

func (ia *InharmonicityAnalyzer) calculateConfidence(harmonicData *HarmonicData, inharmonicity float64) float64 {
	// Confidence based on number of harmonics and their amplitudes
	numHarmonics := float64(len(harmonicData.frequencies))
	harmonicStrength := common.Mean(harmonicData.amplitudes)

	// Normalize confidence
	confidence := (numHarmonics / float64(ia.params.MaxHarmonics)) * harmonicStrength
	confidence = math.Min(1.0, confidence)

	return confidence
}

func (ia *InharmonicityAnalyzer) fitRailsbackCurve(harmonicData *HarmonicData, f0 float64) float64 {
	// Fit B parameter in: f_n = n * f_0 * sqrt(1 + B * n^2)
	// Using least squares optimization

	bestB := 0.0
	minError := math.Inf(1)

	// Grid search for B parameter
	for b := -0.01; b <= 0.01; b += 0.0001 {
		error := 0.0

		for i := range harmonicData.frequencies {
			n := float64(harmonicData.harmonicNums[i])
			expectedFreq := n * f0 * math.Sqrt(1+b*n*n)
			diff := harmonicData.frequencies[i] - expectedFreq
			error += diff * diff
		}

		if error < minError {
			minError = error
			bestB = b
		}
	}

	return bestB
}

func (ia *InharmonicityAnalyzer) calculateRSquared(harmonicData *HarmonicData, f0 float64, B float64) float64 {
	// Calculate R-squared for model fit

	// Calculate predicted values
	var ssRes, ssTot float64
	meanObserved := common.Mean(harmonicData.frequencies)

	for i := range harmonicData.frequencies {
		n := float64(harmonicData.harmonicNums[i])
		predicted := n * f0 * math.Sqrt(1+B*n*n)
		observed := harmonicData.frequencies[i]

		ssRes += (observed - predicted) * (observed - predicted)
		ssTot += (observed - meanObserved) * (observed - meanObserved)
	}

	if ssTot > 0 {
		return 1.0 - (ssRes / ssTot)
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) calculateFletcherMunsonWeight(freq float64) float64 {
	// Simplified Fletcher-Munson equal loudness weighting
	// Based on 40-phon curve approximation

	if freq < 20 || freq > 20000 {
		return 0.0
	}

	// Logarithmic frequency
	// TODO: unused
	// logFreq := math.Log10(freq)

	// Simplified weighting function
	// Peak sensitivity around 3-4 kHz
	weight := 1.0
	if freq < 1000 {
		weight = 0.5 + 0.5*(freq/1000)
	} else if freq > 4000 {
		weight = 1.0 - 0.3*math.Log10(freq/4000)
	}

	return math.Max(0.1, weight)
}

func (ia *InharmonicityAnalyzer) analyzePeriodicityVariation(autocorrResult *stats.CorrelationResult) float64 {
	if len(autocorrResult.Correlations) < 10 {
		return 0.0
	}

	// Find peaks in autocorrelation
	peaks := common.FindPeaks(autocorrResult.Correlations, 0.1, 1.0)

	if len(peaks) < 2 {
		return 0.0
	}

	// Calculate variation in peak positions (indicates inharmonicity)
	peakPositions := make([]float64, len(peaks))
	for i, peakIdx := range peaks {
		peakPositions[i] = float64(autocorrResult.Lags[peakIdx])
	}

	// Calculate coefficient of variation
	mean := common.Mean(peakPositions)
	variance := common.Variance(peakPositions)

	if mean > 0 {
		return math.Sqrt(variance) / mean
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) removeOutliers(data []float64) []float64 {
	if len(data) < 3 {
		return data
	}

	// Calculate Q1, Q3, and IQR
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	q1 := common.Percentile(sorted, 0.25)
	q3 := common.Percentile(sorted, 0.75)
	iqr := q3 - q1

	// Remove outliers beyond threshold * IQR
	threshold := ia.params.OutlierThreshold
	lowerBound := q1 - threshold*iqr
	upperBound := q3 + threshold*iqr

	filtered := make([]float64, 0)
	for _, val := range data {
		if val >= lowerBound && val <= upperBound {
			filtered = append(filtered, val)
		}
	}

	return filtered
}

func (ia *InharmonicityAnalyzer) calculateStatisticalConfidence(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Calculate confidence based on sample size and variance
	n := float64(len(data))
	variance := common.Variance(data)

	// Standard error
	standardError := math.Sqrt(variance / n)

	// T-value for confidence level (approximation)
	tValue := 2.0 // Rough approximation for 95% confidence

	// Confidence interval width
	marginOfError := tValue * standardError
	mean := common.Mean(data)

	// Convert to confidence measure (0-1)
	if mean != 0 {
		relativeError := marginOfError / math.Abs(mean)
		confidence := 1.0 / (1.0 + relativeError)
		return math.Min(1.0, confidence)
	}

	return 0.5
}

func (ia *InharmonicityAnalyzer) performGoodnessOfFitTest(data []float64) (float64, float64) {
	// Simplified chi-squared goodness of fit test
	// Testing against normal distribution

	if len(data) < 5 {
		return 0.0, 1.0
	}

	// Calculate expected frequencies for normal distribution
	mean := common.Mean(data)
	stdDev := math.Sqrt(common.Variance(data))

	// Create bins
	numBins := int(math.Min(10, float64(len(data))/2))
	minVal := mean - 3*stdDev
	maxVal := mean + 3*stdDev
	binWidth := (maxVal - minVal) / float64(numBins)

	// Count observed frequencies
	observed := make([]int, numBins)
	for _, val := range data {
		binIndex := int((val - minVal) / binWidth)
		if binIndex >= 0 && binIndex < numBins {
			observed[binIndex]++
		}
	}

	// Calculate expected frequencies (normal distribution)
	expected := make([]float64, numBins)
	for i := range expected {
		binCenter := minVal + (float64(i)+0.5)*binWidth
		// Normal probability density
		z := (binCenter - mean) / stdDev
		density := math.Exp(-0.5*z*z) / (stdDev * math.Sqrt(2*math.Pi))
		expected[i] = density * binWidth * float64(len(data))
	}

	// Calculate chi-squared statistic
	chiSquared := 0.0
	for i := range observed {
		if expected[i] > 0 {
			diff := float64(observed[i]) - expected[i]
			chiSquared += (diff * diff) / expected[i]
		}
	}

	// Calculate p-value (rough approximation)
	degreesOfFreedom := float64(numBins - 3)                // -3 for estimated parameters
	pValue := 1.0 - (chiSquared / (degreesOfFreedom * 2.0)) // Very rough approximation
	pValue = math.Max(0.0, math.Min(1.0, pValue))

	return chiSquared, pValue
}

func (ia *InharmonicityAnalyzer) applyPhysicalModel(result InharmonicityResult, harmonicData *HarmonicData) InharmonicityResult {
	switch ia.params.Model {
	case ModelString:
		return ia.applyStringModel(result, harmonicData)
	case ModelPipe:
		return ia.applyPipeModel(result, harmonicData)
	case ModelMembrane:
		return ia.applyMembraneModel(result, harmonicData)
	default:
		return result
	}
}

func (ia *InharmonicityAnalyzer) applyStringModel(result InharmonicityResult, harmonicData *HarmonicData) InharmonicityResult {
	// String inharmonicity model: B = (π³ * E * d⁴) / (64 * T * L²)
	// Where E = Young's modulus, d = string diameter, T = tension, L = length

	// Estimate stiffness parameter from measured inharmonicity
	B := result.Inharmonicity

	// Calculate model parameters
	estimatedStiffness := B * ia.params.StringTension * ia.params.StringLength * ia.params.StringLength

	// Calculate model fit quality
	modelFitQuality := ia.calculateStringModelFit(harmonicData, B)

	result.EstimatedStiffness = estimatedStiffness
	result.ModelFitQuality = modelFitQuality

	if result.ModelParameters == nil {
		result.ModelParameters = make(map[string]float64)
	}
	result.ModelParameters["string_stiffness"] = estimatedStiffness
	result.ModelParameters["string_tension"] = ia.params.StringTension
	result.ModelParameters["string_length"] = ia.params.StringLength

	return result
}

func (ia *InharmonicityAnalyzer) applyPipeModel(result InharmonicityResult, harmonicData *HarmonicData) InharmonicityResult {
	// Pipe inharmonicity model (simplified)
	// For now, use general model
	return result
}

func (ia *InharmonicityAnalyzer) applyMembraneModel(result InharmonicityResult, harmonicData *HarmonicData) InharmonicityResult {
	// Membrane inharmonicity model (simplified)
	// For now, use general model
	return result
}

func (ia *InharmonicityAnalyzer) calculateStringModelFit(harmonicData *HarmonicData, B float64) float64 {
	if len(harmonicData.frequencies) == 0 {
		return 0.0
	}

	// Calculate expected frequencies for string model
	f0 := harmonicData.idealFreqs[0] / float64(harmonicData.harmonicNums[0])

	totalError := 0.0
	for i := range harmonicData.frequencies {
		n := float64(harmonicData.harmonicNums[i])
		expectedFreq := n * f0 * math.Sqrt(1+B*n*n)
		error := math.Abs(harmonicData.frequencies[i] - expectedFreq)
		totalError += error
	}

	avgError := totalError / float64(len(harmonicData.frequencies))

	// Convert to quality measure (0-1)
	quality := math.Exp(-avgError / f0)
	return quality
}

func (ia *InharmonicityAnalyzer) calculateQualityMetrics(result InharmonicityResult, harmonicData *HarmonicData) InharmonicityResult {
	// Calculate harmonic clarity
	result.HarmonicClarity = ia.calculateHarmonicClarity(harmonicData)

	// Calculate spectral purity
	result.SpectralPurity = ia.calculateSpectralPurity(harmonicData)

	// Calculate partial coherence
	result.PartialCoherence = ia.calculatePartialCoherence(harmonicData)

	return result
}

func (ia *InharmonicityAnalyzer) calculateHarmonicClarity(harmonicData *HarmonicData) float64 {
	if len(harmonicData.amplitudes) == 0 {
		return 0.0
	}

	// Clarity based on amplitude distribution
	maxAmp := 0.0
	for _, amp := range harmonicData.amplitudes {
		if amp > maxAmp {
			maxAmp = amp
		}
	}

	// Calculate dynamic range
	minAmp := maxAmp
	for _, amp := range harmonicData.amplitudes {
		if amp < minAmp && amp > 0 {
			minAmp = amp
		}
	}

	if minAmp > 0 {
		dynamicRange := 20 * math.Log10(maxAmp/minAmp)
		clarity := dynamicRange / 60.0 // Normalize to 0-1
		return math.Min(1.0, clarity)
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) calculateSpectralPurity(harmonicData *HarmonicData) float64 {
	if len(harmonicData.deviations) == 0 {
		return 0.0
	}

	// Purity based on how close harmonics are to ideal positions
	avgDeviation := 0.0
	for _, dev := range harmonicData.deviations {
		avgDeviation += math.Abs(dev)
	}
	avgDeviation /= float64(len(harmonicData.deviations))

	// Convert to purity measure
	purity := math.Exp(-avgDeviation * 20)
	return purity
}

func (ia *InharmonicityAnalyzer) calculatePartialCoherence(harmonicData *HarmonicData) float64 {
	if len(harmonicData.frequencies) < 2 {
		return 0.0
	}

	// Coherence based on regularity of harmonic spacing
	spacings := make([]float64, len(harmonicData.frequencies)-1)
	for i := 1; i < len(harmonicData.frequencies); i++ {
		spacings[i-1] = harmonicData.frequencies[i] - harmonicData.frequencies[i-1]
	}

	// Calculate coefficient of variation of spacings
	if len(spacings) > 0 {
		mean := common.Mean(spacings)
		variance := common.Variance(spacings)

		if mean > 0 {
			cv := math.Sqrt(variance) / mean
			coherence := 1.0 / (1.0 + cv)
			return coherence
		}
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) postProcessResult(result InharmonicityResult) InharmonicityResult {
	// Apply temporal smoothing if enabled
	if ia.params.UseTemporalTracking && len(ia.inharmonicityHistory) > 0 {
		result.Inharmonicity = ia.applyTemporalSmoothing(result.Inharmonicity)
	}

	// Calculate temporal stability
	result.TemporalStability = ia.calculateTemporalStability()
	result.InharmonicityTrend = ia.calculateInharmonicityTrend()

	// Calculate F0 stability
	result.F0Stability = ia.calculateF0Stability()

	return result
}

func (ia *InharmonicityAnalyzer) applyTemporalSmoothing(currentInharmonicity float64) float64 {
	if len(ia.inharmonicityHistory) == 0 {
		return currentInharmonicity
	}

	// Apply exponential smoothing
	alpha := 0.3
	previousInharmonicity := ia.inharmonicityHistory[len(ia.inharmonicityHistory)-1]
	return alpha*currentInharmonicity + (1.0-alpha)*previousInharmonicity
}

func (ia *InharmonicityAnalyzer) calculateTemporalStability() float64 {
	if len(ia.inharmonicityHistory) < 3 {
		return 0.0
	}

	// Calculate coefficient of variation
	mean := common.Mean(ia.inharmonicityHistory)
	variance := common.Variance(ia.inharmonicityHistory)

	if mean != 0 {
		cv := math.Sqrt(variance) / math.Abs(mean)
		stability := 1.0 / (1.0 + cv)
		return stability
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) calculateInharmonicityTrend() float64 {
	if len(ia.inharmonicityHistory) < 3 {
		return 0.0
	}

	// Calculate linear trend using simple regression
	n := len(ia.inharmonicityHistory)
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}

	slope, _, _ := common.LinRegression(x, ia.inharmonicityHistory)
	return slope
}

func (ia *InharmonicityAnalyzer) calculateF0Stability() float64 {
	if len(ia.f0History) < 3 {
		return 0.0
	}

	// Calculate coefficient of variation for F0
	mean := common.Mean(ia.f0History)
	variance := common.Variance(ia.f0History)

	if mean > 0 {
		cv := math.Sqrt(variance) / mean
		stability := 1.0 / (1.0 + cv)
		return stability
	}

	return 0.0
}

func (ia *InharmonicityAnalyzer) updateTemporalTracking(result InharmonicityResult) {
	if !ia.params.UseTemporalTracking {
		return
	}

	// Update history
	ia.inharmonicityHistory = append(ia.inharmonicityHistory, result.Inharmonicity)
	ia.f0History = append(ia.f0History, result.F0Frequency)

	// Keep only recent history
	maxHistory := ia.params.TemporalWindow
	if len(ia.inharmonicityHistory) > maxHistory {
		ia.inharmonicityHistory = ia.inharmonicityHistory[len(ia.inharmonicityHistory)-maxHistory:]
		ia.f0History = ia.f0History[len(ia.f0History)-maxHistory:]
	}
}

func (ia *InharmonicityAnalyzer) assessAnalysisQuality(result InharmonicityResult) string {
	// Assess overall quality of the analysis

	score := 0.0

	// Factor 1: Number of harmonics
	harmonicScore := float64(result.NumHarmonics) / float64(ia.params.MaxHarmonics)
	score += harmonicScore * 0.3

	// Factor 2: F0 confidence
	score += result.F0Confidence * 0.3

	// Factor 3: Inharmonicity confidence
	score += result.InharmonicityConfidence * 0.2

	// Factor 4: Model fit quality (if available)
	if result.ModelFitQuality > 0 {
		score += result.ModelFitQuality * 0.2
	} else {
		score += 0.1 // Neutral contribution
	}

	// Classify quality
	if score >= 0.8 {
		return "Excellent"
	} else if score >= 0.6 {
		return "Good"
	} else if score >= 0.4 {
		return "Fair"
	} else if score >= 0.2 {
		return "Poor"
	} else {
		return "Very Poor"
	}
}

func (ia *InharmonicityAnalyzer) getMethodName(method InharmonicityMethod) string {
	switch method {
	case InharmonicityRailsback:
		return "Railsback Curve"
	case InharmonicityFletcherMunson:
		return "Fletcher-Munson Weighted"
	case InharmonicitySpectralDeviation:
		return "Spectral Deviation"
	case InharmonicityPartialTracking:
		return "Partial Tracking"
	case InharmonicityBeatPattern:
		return "Beat Pattern Analysis"
	case InharmonicityStatistical:
		return "Statistical Analysis"
	default:
		return "Unknown"
	}
}

func (ia *InharmonicityAnalyzer) getModelName(model InharmonicityModel) string {
	switch model {
	case ModelString:
		return "String Instrument"
	case ModelPipe:
		return "Pipe/Wind Instrument"
	case ModelMembrane:
		return "Membrane/Percussion"
	case ModelGeneral:
		return "General"
	default:
		return "Unknown"
	}
}

func (ia *InharmonicityAnalyzer) getCurrentTime() float64 {
	// Placeholder for time measurement
	return 0.0
}

// Public API methods

// AnalyzeSequence analyzes inharmonicity for a sequence of audio frames
func (ia *InharmonicityAnalyzer) AnalyzeSequence(audioFrames [][]float64) ([]InharmonicityResult, error) {
	results := make([]InharmonicityResult, len(audioFrames))

	for i, frame := range audioFrames {
		result, err := ia.AnalyzeFrame(frame)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

// GetAverageInharmonicity computes average inharmonicity from sequence
func (ia *InharmonicityAnalyzer) GetAverageInharmonicity(results []InharmonicityResult) float64 {
	if len(results) == 0 {
		return 0.0
	}

	sum := 0.0
	count := 0

	for _, result := range results {
		if result.InharmonicityConfidence > 0.5 { // Only use confident measurements
			sum += result.Inharmonicity
			count++
		}
	}

	if count > 0 {
		return sum / float64(count)
	}

	return 0.0
}

// Reset resets the analyzer state
func (ia *InharmonicityAnalyzer) Reset() {
	ia.inharmonicityHistory = make([]float64, 0)
	ia.f0History = make([]float64, 0)
	ia.modelCoefficients = make([]float64, 0)
	ia.fitResiduals = make([]float64, 0)
}

// GetParameters returns current parameters
func (ia *InharmonicityAnalyzer) GetParameters() InharmonicityParams {
	return ia.params
}

// SetParameters updates parameters
func (ia *InharmonicityAnalyzer) SetParameters(params InharmonicityParams) {
	ia.params = params
	ia.initialized = false // Force re-initialization
}

// GetHistory returns the inharmonicity history
func (ia *InharmonicityAnalyzer) GetHistory() []float64 {
	return ia.inharmonicityHistory
}

// GetF0History returns the F0 history
func (ia *InharmonicityAnalyzer) GetF0History() []float64 {
	return ia.f0History
}

// Utility functions

// GetSupportedMethods returns list of supported inharmonicity methods
func GetSupportedInharmonicityMethods() []string {
	return []string{
		"Railsback Curve",
		"Fletcher-Munson Weighted",
		"Spectral Deviation",
		"Partial Tracking",
		"Beat Pattern Analysis",
		"Statistical Analysis",
	}
}

// GetSupportedModels returns list of supported physical models
func GetSupportedInharmonicityModels() []string {
	return []string{
		"String Instrument",
		"Pipe/Wind Instrument",
		"Membrane/Percussion",
		"General",
	}
}

// ClassifyInharmonicity classifies inharmonicity level
func ClassifyInharmonicity(inharmonicity float64) string {
	absInh := math.Abs(inharmonicity)

	if absInh < 0.0001 {
		return "Very Low"
	} else if absInh < 0.001 {
		return "Low"
	} else if absInh < 0.005 {
		return "Moderate"
	} else if absInh < 0.01 {
		return "High"
	} else {
		return "Very High"
	}
}

// EstimateInstrumentType estimates instrument type from inharmonicity
func EstimateInstrumentType(inharmonicity float64, f0 float64) string {
	absInh := math.Abs(inharmonicity)

	// Piano strings typically have higher inharmonicity
	if absInh > 0.002 && f0 < 500 {
		return "Piano/String"
	}

	// Wind instruments typically have very low inharmonicity
	if absInh < 0.0005 {
		return "Wind/Brass"
	}

	// Guitar and other plucked strings
	if absInh > 0.0005 && absInh < 0.002 {
		return "Plucked String"
	}

	// Vocal or other harmonic sources
	if absInh < 0.001 && f0 > 100 && f0 < 800 {
		return "Vocal/Harmonic"
	}

	return "Unknown"
}
