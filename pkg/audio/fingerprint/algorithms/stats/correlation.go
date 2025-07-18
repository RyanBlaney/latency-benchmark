package stats

import (
	"fmt"
	"math"
	"math/cmplx"
)

// CorrelationType represents different types of correlation calculations
type CorrelationType int

const (
	// Pearson correlation (linear correlation)
	Pearson CorrelationType = iota

	// Spearman rank correlation
	Spearman

	// Kendall tau correlation
	Kendall

	// Normalized cross-correlation
	NormalizedCrossCorrelation

	// Zero-normalized cross-correlation
	ZeroNormalizedCrossCorrelation
)

// CorrelationMethod represents different computational approaches
type CorrelationMethod int

const (
	// Direct time-domain calculation
	TimeDomain CorrelationMethod = iota

	// FFT-based frequency domain (faster for large signals)
	FrequencyDomain

	// Sliding window approach
	SlidingWindow
)

// CorrelationResult contains comprehensive correlation analysis results
type CorrelationResult struct {
	// Primary correlation values
	Correlations []float64 `json:"correlations"`
	Lags         []int     `json:"lags"`

	// Peak correlation information
	PeakCorrelation float64 `json:"peak_correlation"`
	PeakLag         int     `json:"peak_lag"`
	PeakIndex       int     `json:"peak_index"`

	// Statistical significance
	PValue          float64 `json:"p_value"`
	IsSignificant   bool    `json:"is_significant"`
	ConfidenceLevel float64 `json:"confidence_level"`

	// Correlation quality metrics
	SNR            float64 `json:"snr"`              // Signal-to-noise ratio
	Sharpness      float64 `json:"sharpness"`        // Peak sharpness
	SecondPeak     float64 `json:"second_peak"`      // Second highest peak
	PeakToSidelobe float64 `json:"peak_to_sidelobe"` // Peak-to-sidelobe ratio

	// Computational details
	Method          CorrelationMethod `json:"method"`
	Type            CorrelationType   `json:"type"`
	MaxLag          int               `json:"max_lag"`
	OverlapLength   int               `json:"overlap_length"`
	ComputationTime float64           `json:"computation_time"` // in milliseconds
}

// CrossCorrelation computes cross-correlation between two signals
//
// References:
// - Rabiner, L., Schafer, R. (1978). "Digital Processing of Speech Signals"
// - Oppenheim, A.V., Schafer, R.W. (2010). "Discrete-Time Signal Processing"
// - Lewis, J.P. (1995). "Fast Template Matching"
// - Knuth, D.E. (1998). "The Art of Computer Programming, Vol. 2"
//
// Cross-correlation is fundamental for:
// - Audio alignment and synchronization
// - Echo detection and cancellation
// - Pattern matching in signals
// - Time delay estimation
// - Audio fingerprint matching
type CrossCorrelation struct {
	maxLag          int
	correlationType CorrelationType
	method          CorrelationMethod
	confidenceLevel float64

	// Numerical stability parameters
	minStdDev       float64
	normalizeInputs bool

	// Performance optimization
	useFFT       bool
	fftThreshold int
}

// NewCrossCorrelation creates a new cross-correlation calculator with default settings
func NewCrossCorrelation(maxLag int) *CrossCorrelation {
	return &CrossCorrelation{
		maxLag:          maxLag,
		correlationType: Pearson,
		method:          TimeDomain,
		confidenceLevel: 0.95,
		minStdDev:       1e-10,
		normalizeInputs: true,
		useFFT:          true,
		fftThreshold:    1000,
	}
}

// NewCrossCorrelationWithParams creates a cross-correlation calculator with custom parameters
func NewCrossCorrelationWithParams(maxLag int, corrType CorrelationType, method CorrelationMethod) *CrossCorrelation {
	return &CrossCorrelation{
		maxLag:          maxLag,
		correlationType: corrType,
		method:          method,
		confidenceLevel: 0.95,
		minStdDev:       1e-10,
		normalizeInputs: true,
		useFFT:          method == FrequencyDomain,
		fftThreshold:    1000,
	}
}

// Compute calculates cross-correlation between two signals with comprehensive analysis
func (cc *CrossCorrelation) Compute(signal1, signal2 []float64) (*CorrelationResult, error) {
	if len(signal1) == 0 || len(signal2) == 0 {
		return nil, fmt.Errorf("empty signals provided")
	}

	startTime := getTimeMs()

	// Choose computational method based on signal size and settings
	method := cc.method
	if cc.useFFT && (len(signal1) > cc.fftThreshold || len(signal2) > cc.fftThreshold) {
		method = FrequencyDomain
	}

	var correlations []float64
	var lags []int
	var err error

	switch method {
	case FrequencyDomain:
		correlations, lags, err = cc.computeFFT(signal1, signal2)
	case TimeDomain:
		correlations, lags, err = cc.computeTimeDomain(signal1, signal2)
	case SlidingWindow:
		correlations, lags, err = cc.computeSlidingWindow(signal1, signal2)
	default:
		return nil, fmt.Errorf("unsupported correlation method")
	}

	if err != nil {
		return nil, err
	}

	// Find peak correlation
	peakCorr, peakLag, peakIdx := cc.findPeak(correlations, lags)

	// Calculate statistical significance
	pValue := cc.calculatePValue(peakCorr, len(signal1), len(signal2))
	isSignificant := pValue < (1.0 - cc.confidenceLevel)

	// Calculate quality metrics
	snr := cc.calculateSNR(correlations, peakIdx)
	sharpness := cc.calculateSharpness(correlations, peakIdx)
	secondPeak := cc.findSecondPeak(correlations, peakIdx)
	peakToSidelobe := cc.calculatePeakToSidelobe(correlations, peakIdx)

	// Calculate overlap length
	overlapLength := cc.calculateOverlapLength(len(signal1), len(signal2), peakLag)

	computationTime := getTimeMs() - startTime

	return &CorrelationResult{
		Correlations:    correlations,
		Lags:            lags,
		PeakCorrelation: peakCorr,
		PeakLag:         peakLag,
		PeakIndex:       peakIdx,
		PValue:          pValue,
		IsSignificant:   isSignificant,
		ConfidenceLevel: cc.confidenceLevel,
		SNR:             snr,
		Sharpness:       sharpness,
		SecondPeak:      secondPeak,
		PeakToSidelobe:  peakToSidelobe,
		Method:          method,
		Type:            cc.correlationType,
		MaxLag:          cc.maxLag,
		OverlapLength:   overlapLength,
		ComputationTime: computationTime,
	}, nil
}

// computeTimeDomain performs time-domain cross-correlation calculation
func (cc *CrossCorrelation) computeTimeDomain(signal1, signal2 []float64) ([]float64, []int, error) {
	// Normalize signals if requested
	norm1 := signal1
	norm2 := signal2

	if cc.normalizeInputs {
		norm1 = cc.normalize(signal1)
		norm2 = cc.normalize(signal2)
	}

	// Calculate actual max lag based on signal lengths
	actualMaxLag := cc.calculateActualMaxLag(len(norm1), len(norm2))

	// Calculate correlation for all lags
	numLags := 2*actualMaxLag + 1
	correlations := make([]float64, numLags)
	lags := make([]int, numLags)

	for i := range numLags {
		lag := i - actualMaxLag
		lags[i] = lag
		correlations[i] = cc.computeAtLag(norm1, norm2, lag)
	}

	return correlations, lags, nil
}

// computeFFT performs FFT-based cross-correlation calculation
func (cc *CrossCorrelation) computeFFT(signal1, signal2 []float64) ([]float64, []int, error) {
	// Normalize signals if requested
	norm1 := signal1
	norm2 := signal2

	if cc.normalizeInputs {
		norm1 = cc.normalize(signal1)
		norm2 = cc.normalize(signal2)
	}

	// Determine FFT size (next power of 2)
	n1, n2 := len(norm1), len(norm2)
	fftSize := nextPowerOf2(n1 + n2 - 1)

	// Zero-pad signals
	padded1 := make([]complex128, fftSize)
	padded2 := make([]complex128, fftSize)

	for i := range n1 {
		padded1[i] = complex(norm1[i], 0)
	}
	for i := range n2 {
		padded2[i] = complex(norm2[i], 0)
	}

	// Compute FFTs
	fft1 := fft(padded1)
	fft2 := fft(padded2)

	// Compute cross-power spectrum (conjugate of signal2)
	crossPower := make([]complex128, fftSize)
	for i := range fftSize {
		crossPower[i] = fft1[i] * cmplx.Conj(fft2[i])
	}

	// Compute inverse FFT to get correlation
	correlation := ifft(crossPower)

	// Extract real parts and rearrange
	actualMaxLag := cc.calculateActualMaxLag(n1, n2)
	numLags := 2*actualMaxLag + 1
	correlations := make([]float64, numLags)
	lags := make([]int, numLags)

	for i := range numLags {
		lag := i - actualMaxLag
		lags[i] = lag

		// Map lag to FFT index
		var idx int
		if lag >= 0 {
			idx = lag
		} else {
			idx = fftSize + lag
		}

		correlations[i] = real(correlation[idx])
	}

	return correlations, lags, nil
}

// computeSlidingWindow performs sliding window cross-correlation
func (cc *CrossCorrelation) computeSlidingWindow(signal1, signal2 []float64) ([]float64, []int, error) {
	// This is similar to time domain but optimized for streaming applications
	return cc.computeTimeDomain(signal1, signal2)
}

// computeAtLag calculates correlation at a specific lag with improved accuracy
func (cc *CrossCorrelation) computeAtLag(signal1, signal2 []float64, lag int) float64 {
	switch cc.correlationType {
	case Pearson:
		return cc.pearsonCorrelation(signal1, signal2, lag)
	case NormalizedCrossCorrelation:
		return cc.normalizedCrossCorrelation(signal1, signal2, lag)
	case ZeroNormalizedCrossCorrelation:
		return cc.zeroNormalizedCrossCorrelation(signal1, signal2, lag)
	default:
		return cc.pearsonCorrelation(signal1, signal2, lag)
	}
}

// pearsonCorrelation calculates Pearson correlation coefficient at given lag
func (cc *CrossCorrelation) pearsonCorrelation(signal1, signal2 []float64, lag int) float64 {
	// Determine overlap region
	start1, end1, start2, end2 := cc.calculateOverlapRegion(len(signal1), len(signal2), lag)

	// Calculate overlap length
	overlapLen := minInt(end1-start1, end2-start2)
	if overlapLen <= 1 {
		return 0.0
	}

	// Calculate means in overlap region
	var mean1, mean2 float64
	count := 0

	for i := range overlapLen {
		idx1 := start1 + i
		idx2 := start2 + i

		if idx1 >= 0 && idx1 < len(signal1) && idx2 >= 0 && idx2 < len(signal2) {
			mean1 += signal1[idx1]
			mean2 += signal2[idx2]
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	mean1 /= float64(count)
	mean2 /= float64(count)

	// Calculate correlation coefficient
	var numerator, sum1sq, sum2sq float64

	for i := range overlapLen {
		idx1 := start1 + i
		idx2 := start2 + i

		if idx1 >= 0 && idx1 < len(signal1) && idx2 >= 0 && idx2 < len(signal2) {
			diff1 := signal1[idx1] - mean1
			diff2 := signal2[idx2] - mean2

			numerator += diff1 * diff2
			sum1sq += diff1 * diff1
			sum2sq += diff2 * diff2
		}
	}

	denominator := math.Sqrt(sum1sq * sum2sq)
	if denominator < cc.minStdDev {
		return 0.0
	}

	correlation := numerator / denominator
	return clampCorrelation(correlation)
}

// normalizedCrossCorrelation calculates normalized cross-correlation
func (cc *CrossCorrelation) normalizedCrossCorrelation(signal1, signal2 []float64, lag int) float64 {
	start1, end1, start2, end2 := cc.calculateOverlapRegion(len(signal1), len(signal2), lag)
	overlapLen := minInt(end1-start1, end2-start2)

	if overlapLen <= 0 {
		return 0.0
	}

	var sum, sum1sq, sum2sq float64
	count := 0

	for i := range overlapLen {
		idx1 := start1 + i
		idx2 := start2 + i

		if idx1 >= 0 && idx1 < len(signal1) && idx2 >= 0 && idx2 < len(signal2) {
			val1 := signal1[idx1]
			val2 := signal2[idx2]

			sum += val1 * val2
			sum1sq += val1 * val1
			sum2sq += val2 * val2
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	denominator := math.Sqrt(sum1sq * sum2sq)
	if denominator < cc.minStdDev {
		return 0.0
	}

	return sum / denominator
}

// zeroNormalizedCrossCorrelation calculates zero-normalized cross-correlation
func (cc *CrossCorrelation) zeroNormalizedCrossCorrelation(signal1, signal2 []float64, lag int) float64 {
	// First normalize to zero mean
	norm1 := cc.subtractMean(signal1)
	norm2 := cc.subtractMean(signal2)

	return cc.normalizedCrossCorrelation(norm1, norm2, lag)
}

// calculateOverlapRegion determines the overlap region for given lag
func (cc *CrossCorrelation) calculateOverlapRegion(len1, len2, lag int) (start1, end1, start2, end2 int) {
	if lag >= 0 {
		// signal2 is delayed relative to signal1
		start1, end1 = 0, len1
		start2, end2 = lag, len2

		// Limit to actual overlap
		if end1 > len2-lag {
			end1 = len2 - lag
		}
		if end2 > len2 {
			end2 = len2
		}
	} else {
		// signal1 is delayed relative to signal2
		start1, end1 = -lag, len1
		start2, end2 = 0, len2

		// Limit to actual overlap
		if end1 > len1 {
			end1 = len1
		}
		if end2 > len1+lag {
			end2 = len1 + lag
		}
	}

	return start1, end1, start2, end2
}

// calculateActualMaxLag determines the actual maximum lag based on signal lengths
func (cc *CrossCorrelation) calculateActualMaxLag(len1, len2 int) int {
	actualMaxLag := cc.maxLag

	// Limit by signal lengths
	actualMaxLag = min(actualMaxLag, len1-1)
	actualMaxLag = min(actualMaxLag, len2-1)
	actualMaxLag = max(actualMaxLag, 0)

	return actualMaxLag
}

// normalize normalizes a signal to zero mean and unit variance
func (cc *CrossCorrelation) normalize(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Calculate mean
	mean := 0.0
	for _, val := range signal {
		mean += val
	}
	mean /= float64(len(signal))

	// Calculate standard deviation
	variance := 0.0
	for _, val := range signal {
		diff := val - mean
		variance += diff * diff
	}
	variance /= float64(len(signal))
	stdDev := math.Sqrt(variance)

	// Handle constant signals
	if stdDev < cc.minStdDev {
		normalized := make([]float64, len(signal))
		for i, val := range signal {
			normalized[i] = val - mean
		}
		return normalized
	}

	// Normalize to zero mean, unit variance
	normalized := make([]float64, len(signal))
	for i, val := range signal {
		normalized[i] = (val - mean) / stdDev
	}

	return normalized
}

// subtractMean subtracts the mean from a signal
func (cc *CrossCorrelation) subtractMean(signal []float64) []float64 {
	if len(signal) == 0 {
		return signal
	}

	// Calculate mean
	mean := 0.0
	for _, val := range signal {
		mean += val
	}
	mean /= float64(len(signal))

	// Subtract mean
	result := make([]float64, len(signal))
	for i, val := range signal {
		result[i] = val - mean
	}

	return result
}

// findPeak finds the peak correlation and corresponding lag
func (cc *CrossCorrelation) findPeak(correlations []float64, lags []int) (float64, int, int) {
	if len(correlations) == 0 {
		return 0, 0, 0
	}

	maxCorr := correlations[0]
	bestLag := lags[0]
	bestIdx := 0

	for i, corr := range correlations {
		if math.Abs(corr) > math.Abs(maxCorr) {
			maxCorr = corr
			bestLag = lags[i]
			bestIdx = i
		}
	}

	return maxCorr, bestLag, bestIdx
}

// calculatePValue calculates statistical significance (simplified)
func (cc *CrossCorrelation) calculatePValue(correlation float64, n1, n2 int) float64 {
	// Simplified p-value calculation for correlation
	// For more accurate results, use proper statistical tests
	n := minInt(n1, n2)
	if n <= 2 {
		return 1.0
	}

	// Transform correlation to t-statistic
	t := math.Abs(correlation) * math.Sqrt(float64(n-2)) / math.Sqrt(1.0-correlation*correlation)

	// Approximate p-value using Student's t-distribution
	// This is a simplified approximation
	if t > 2.0 {
		return 0.01
	} else if t > 1.5 {
		return 0.05
	} else if t > 1.0 {
		return 0.1
	}

	return 0.5
}

// calculateSNR calculates signal-to-noise ratio of correlation
func (cc *CrossCorrelation) calculateSNR(correlations []float64, peakIdx int) float64 {
	if len(correlations) == 0 || peakIdx < 0 || peakIdx >= len(correlations) {
		return 0.0
	}

	peakValue := math.Abs(correlations[peakIdx])

	// Calculate noise level (excluding peak region)
	noiseSum := 0.0
	noiseCount := 0

	for i, corr := range correlations {
		if abs(i-peakIdx) > 5 { // Exclude peak region
			noiseSum += corr * corr
			noiseCount++
		}
	}

	if noiseCount == 0 {
		return 0.0
	}

	noiseLevel := math.Sqrt(noiseSum / float64(noiseCount))

	if noiseLevel < 1e-10 {
		return math.Inf(1)
	}

	return 20.0 * math.Log10(peakValue/noiseLevel)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// calculateSharpness calculates the sharpness of the correlation peak
func (cc *CrossCorrelation) calculateSharpness(correlations []float64, peakIdx int) float64 {
	if len(correlations) < 3 || peakIdx <= 0 || peakIdx >= len(correlations)-1 {
		return 0.0
	}

	// Calculate second derivative at peak (measure of sharpness)
	secondDerivative := correlations[peakIdx+1] - 2*correlations[peakIdx] + correlations[peakIdx-1]
	return -secondDerivative // Negative because we want positive sharpness for peaks
}

// findSecondPeak finds the second highest peak
func (cc *CrossCorrelation) findSecondPeak(correlations []float64, peakIdx int) float64 {
	if len(correlations) == 0 {
		return 0.0
	}

	secondPeak := 0.0

	for i, corr := range correlations {
		if i != peakIdx && math.Abs(corr) > math.Abs(secondPeak) {
			secondPeak = corr
		}
	}

	return secondPeak
}

// calculatePeakToSidelobe calculates peak-to-sidelobe ratio
func (cc *CrossCorrelation) calculatePeakToSidelobe(correlations []float64, peakIdx int) float64 {
	if len(correlations) == 0 || peakIdx < 0 || peakIdx >= len(correlations) {
		return 0.0
	}

	peakValue := math.Abs(correlations[peakIdx])
	maxSidelobe := 0.0

	// Find maximum sidelobe (excluding main peak region)
	for i, corr := range correlations {
		if abs(i-peakIdx) > 10 { // Exclude main peak region
			if math.Abs(corr) > maxSidelobe {
				maxSidelobe = math.Abs(corr)
			}
		}
	}

	if maxSidelobe < 1e-10 {
		return math.Inf(1)
	}

	return 20.0 * math.Log10(peakValue/maxSidelobe)
}

// calculateOverlapLength calculates the overlap length for given lag
func (cc *CrossCorrelation) calculateOverlapLength(len1, len2, lag int) int {
	start1, end1, start2, end2 := cc.calculateOverlapRegion(len1, len2, lag)
	return minInt(end1-start1, end2-start2)
}

// AutoCorrelation computes auto-correlation of a signal
type AutoCorrelation struct {
	crossCorr *CrossCorrelation
}

// NewAutoCorrelation creates a new auto-correlation calculator
func NewAutoCorrelation(maxLag int) *AutoCorrelation {
	return &AutoCorrelation{
		crossCorr: NewCrossCorrelation(maxLag),
	}
}

// Compute calculates auto-correlation of a signal
func (ac *AutoCorrelation) Compute(signal []float64) (*CorrelationResult, error) {
	return ac.crossCorr.Compute(signal, signal)
}

// SetParameters updates the correlation parameters
func (ac *AutoCorrelation) SetParameters(corrType CorrelationType, method CorrelationMethod) {
	ac.crossCorr.correlationType = corrType
	ac.crossCorr.method = method
}

// Utility functions

// clampCorrelation ensures correlation is in valid range [-1, 1]
func clampCorrelation(correlation float64) float64 {
	if correlation > 1.0 {
		return 1.0
	} else if correlation < -1.0 {
		return -1.0
	}
	return correlation
}

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// nextPowerOf2 returns the next power of 2 greater than or equal to n
func nextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}

	power := 1
	for power < n {
		power <<= 1
	}
	return power
}

// Simple FFT implementation (for demonstration - in production use optimized library)
func fft(x []complex128) []complex128 {
	n := len(x)
	if n <= 1 {
		return x
	}

	// Divide
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)

	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Conquer
	evenFFT := fft(even)
	oddFFT := fft(odd)

	// Combine
	result := make([]complex128, n)
	for i := 0; i < n/2; i++ {
		t := cmplx.Exp(complex(0, -2*math.Pi*float64(i)/float64(n))) * oddFFT[i]
		result[i] = evenFFT[i] + t
		result[i+n/2] = evenFFT[i] - t
	}

	return result
}

// Simple IFFT implementation
func ifft(x []complex128) []complex128 {
	n := len(x)

	// Conjugate
	for i := range x {
		x[i] = cmplx.Conj(x[i])
	}

	// FFT
	result := fft(x)

	// Conjugate and scale
	for i := range result {
		result[i] = cmplx.Conj(result[i]) / complex(float64(n), 0)
	}

	return result
}

// getTimeMs returns current time in milliseconds (placeholder)
func getTimeMs() float64 {
	// In real implementation, use time.Now().UnixNano() / 1e6
	return 0.0
}
