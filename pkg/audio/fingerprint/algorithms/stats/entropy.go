package stats

import (
	"fmt"
	"math"
	"sort"
)

// EntropyType represents different types of entropy measures
type EntropyType int

const (
	// Shannon entropy (information entropy)
	Shannon EntropyType = iota

	// Rényi entropy (generalized entropy)
	Renyi

	// Tsallis entropy (non-extensive entropy)
	Tsallis

	// Hartley entropy (max entropy)
	Hartley

	// Collision entropy (order-2 Rényi entropy)
	Collision

	// Min entropy (worst-case entropy)
	MinEntropy
)

// HistogramMethod represents different histogram construction methods
type HistogramMethod int

const (
	// Fixed number of bins
	FixedBins HistogramMethod = iota

	// Sturges' rule: k = log2(n) + 1
	Sturges

	// Scott's rule: bin width = 3.49σn^(-1/3)
	Scott

	// Freedman-Diaconis rule: bin width = 2*IQR*n^(-1/3)
	FreedmanDiaconis

	// Square root rule: k = sqrt(n)
	SquareRoot

	// Doane's rule: extension of Sturges for non-normal data
	Doane
)

// EntropyResult contains comprehensive entropy analysis results
type EntropyResult struct {
	// Primary entropy measures
	ShannonEntropy   float64 `json:"shannon_entropy"`
	RenyiEntropy     float64 `json:"renyi_entropy"`
	TsallisEntropy   float64 `json:"tsallis_entropy"`
	HartleyEntropy   float64 `json:"hartley_entropy"`
	CollisionEntropy float64 `json:"collision_entropy"`
	MinEntropy       float64 `json:"min_entropy"`

	// Normalized entropy measures (0-1 range)
	NormalizedShannon float64 `json:"normalized_shannon"`
	NormalizedRenyi   float64 `json:"normalized_renyi"`

	// Entropy rate and conditional entropy
	EntropyRate        float64 `json:"entropy_rate"`
	ConditionalEntropy float64 `json:"conditional_entropy"`

	// Histogram information
	Histogram       []float64       `json:"histogram"`
	BinEdges        []float64       `json:"bin_edges"`
	BinCenters      []float64       `json:"bin_centers"`
	NumBins         int             `json:"num_bins"`
	HistogramMethod HistogramMethod `json:"histogram_method"`

	// Statistical measures
	Variance float64 `json:"variance"`
	Skewness float64 `json:"skewness"`
	Kurtosis float64 `json:"kurtosis"`

	// Parameters used
	RenyiAlpha float64 `json:"renyi_alpha"`
	TsallisQ   float64 `json:"tsallis_q"`

	// Data characteristics
	NumSamples     int     `json:"num_samples"`
	DataRange      float64 `json:"data_range"`
	EffectiveRange float64 `json:"effective_range"` // Range containing 95% of data
}

// EntropyParams contains parameters for entropy calculation
type EntropyParams struct {
	NumBins         int             `json:"num_bins"`
	HistogramMethod HistogramMethod `json:"histogram_method"`
	RenyiAlpha      float64         `json:"renyi_alpha"` // Parameter for Rényi entropy
	TsallisQ        float64         `json:"tsallis_q"`   // Parameter for Tsallis entropy

	// Normalization options
	NormalizeEntropy bool    `json:"normalize_entropy"`
	BaseLog          float64 `json:"base_log"` // Base for logarithm (2, e, 10)

	// Smoothing options
	SmoothHistogram bool    `json:"smooth_histogram"`
	SmoothingKernel string  `json:"smoothing_kernel"` // "gaussian", "uniform"
	SmoothingWidth  float64 `json:"smoothing_width"`

	// Boundary handling
	BoundaryMethod string  `json:"boundary_method"` // "reflect", "extend", "zero"
	MinProbability float64 `json:"min_probability"` // Minimum probability for log calculation
}

// Entropy implements various entropy measures for statistical analysis of audio signals
//
// References:
// - Shannon, C.E. (1948). "A Mathematical Theory of Communication"
// - Rényi, A. (1961). "On Measures of Entropy and Information"
// - Tsallis, C. (1988). "Possible generalization of Boltzmann-Gibbs statistics"
// - Cover, T.M., Thomas, J.A. (2006). "Elements of Information Theory"
// - Beirlant, J., et al. (1997). "Nonparametric entropy estimation: An overview"
// - Paninski, L. (2003). "Estimation of entropy and mutual information"
//
// Entropy measures the amount of information or uncertainty in a dataset:
// - Higher entropy = more unpredictable/random
// - Lower entropy = more predictable/structured
//
// Applications in audio processing:
// - Signal complexity analysis
// - Audio texture characterization
// - Noise level estimation
// - Compression efficiency prediction
// - Audio event detection
type Entropy struct {
	params EntropyParams
}

// NewEntropy creates a new entropy analyzer with default parameters
func NewEntropy() *Entropy {
	return &Entropy{
		params: EntropyParams{
			NumBins:          50,
			HistogramMethod:  Scott,
			RenyiAlpha:       2.0,
			TsallisQ:         2.0,
			NormalizeEntropy: true,
			BaseLog:          2.0, // Information theory standard
			SmoothHistogram:  false,
			SmoothingKernel:  "gaussian",
			SmoothingWidth:   1.0,
			BoundaryMethod:   "reflect",
			MinProbability:   1e-12,
		},
	}
}

// NewEntropyWithParams creates an entropy analyzer with custom parameters
func NewEntropyWithParams(params EntropyParams) *Entropy {
	return &Entropy{params: params}
}

// Analyze computes comprehensive entropy analysis for the input data
func (e *Entropy) Analyze(data []float64) (*EntropyResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}

	n := len(data)

	// Calculate data statistics
	_, variance, skewness, kurtosis := e.calculateMoments(data)
	dataRange := e.calculateRange(data)
	effectiveRange := e.calculateEffectiveRange(data)

	// Determine optimal number of bins
	numBins := e.determineOptimalBins(data, n)

	// Build histogram
	histogram, binEdges, binCenters := e.buildHistogram(data, numBins)

	// Smooth histogram if requested
	if e.params.SmoothHistogram {
		histogram = e.smoothHistogram(histogram)
	}

	// Normalize histogram to probabilities
	probabilities := e.normalizeToProbabilities(histogram)

	// Calculate various entropy measures
	shannonEntropy := e.calculateShannonEntropy(probabilities)
	renyiEntropy := e.calculateRenyiEntropy(probabilities, e.params.RenyiAlpha)
	tsallisEntropy := e.calculateTsallisEntropy(probabilities, e.params.TsallisQ)
	hartleyEntropy := e.calculateHartleyEntropy(probabilities)
	collisionEntropy := e.calculateRenyiEntropy(probabilities, 2.0) // α = 2
	minEntropy := e.calculateMinEntropy(probabilities)

	// Calculate normalized entropy measures
	maxEntropy := math.Log(float64(numBins)) / math.Log(e.params.BaseLog)
	normalizedShannon := shannonEntropy / maxEntropy
	normalizedRenyi := renyiEntropy / maxEntropy

	// Calculate entropy rate and conditional entropy
	entropyRate := e.calculateEntropyRate(data)
	conditionalEntropy := e.calculateConditionalEntropy(data)

	return &EntropyResult{
		ShannonEntropy:     shannonEntropy,
		RenyiEntropy:       renyiEntropy,
		TsallisEntropy:     tsallisEntropy,
		HartleyEntropy:     hartleyEntropy,
		CollisionEntropy:   collisionEntropy,
		MinEntropy:         minEntropy,
		NormalizedShannon:  normalizedShannon,
		NormalizedRenyi:    normalizedRenyi,
		EntropyRate:        entropyRate,
		ConditionalEntropy: conditionalEntropy,
		Histogram:          histogram,
		BinEdges:           binEdges,
		BinCenters:         binCenters,
		NumBins:            numBins,
		HistogramMethod:    e.params.HistogramMethod,
		Variance:           variance,
		Skewness:           skewness,
		Kurtosis:           kurtosis,
		RenyiAlpha:         e.params.RenyiAlpha,
		TsallisQ:           e.params.TsallisQ,
		NumSamples:         n,
		DataRange:          dataRange,
		EffectiveRange:     effectiveRange,
	}, nil
}

// calculateMoments computes statistical moments of the data
func (e *Entropy) calculateMoments(data []float64) (mean, variance, skewness, kurtosis float64) {
	n := len(data)
	if n == 0 {
		return 0, 0, 0, 0
	}

	// Calculate mean
	sum := 0.0
	for _, x := range data {
		sum += x
	}
	mean = sum / float64(n)

	// Calculate variance
	sumSq := 0.0
	for _, x := range data {
		diff := x - mean
		sumSq += diff * diff
	}
	variance = sumSq / float64(n-1)

	// Calculate skewness and kurtosis
	if variance > 0 {
		stdDev := math.Sqrt(variance)
		sumCubed := 0.0
		sumFourth := 0.0

		for _, x := range data {
			standardized := (x - mean) / stdDev
			sumCubed += standardized * standardized * standardized
			sumFourth += standardized * standardized * standardized * standardized
		}

		skewness = sumCubed / float64(n)
		kurtosis = sumFourth/float64(n) - 3.0 // Excess kurtosis
	}

	return mean, variance, skewness, kurtosis
}

// calculateRange computes the range of the data
func (e *Entropy) calculateRange(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	min := data[0]
	max := data[0]

	for _, x := range data {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}

	return max - min
}

// calculateEffectiveRange computes the range containing 95% of the data
func (e *Entropy) calculateEffectiveRange(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)
	lowerIdx := int(0.025 * float64(n))
	upperIdx := int(0.975 * float64(n))

	if upperIdx >= n {
		upperIdx = n - 1
	}

	return sorted[upperIdx] - sorted[lowerIdx]
}

// determineOptimalBins calculates the optimal number of bins using the specified method
func (e *Entropy) determineOptimalBins(data []float64, n int) int {
	if e.params.HistogramMethod == FixedBins {
		return e.params.NumBins
	}

	switch e.params.HistogramMethod {
	case Sturges:
		return int(math.Log2(float64(n))) + 1

	case Scott:
		// Scott's rule: bin width = 3.49σn^(-1/3)
		_, variance, _, _ := e.calculateMoments(data)
		stdDev := math.Sqrt(variance)
		dataRange := e.calculateRange(data)
		binWidth := 3.49 * stdDev * math.Pow(float64(n), -1.0/3.0)
		if binWidth > 0 {
			return int(math.Ceil(dataRange / binWidth))
		}
		return 10

	case FreedmanDiaconis:
		// Freedman-Diaconis rule: bin width = 2*IQR*n^(-1/3)
		sorted := make([]float64, len(data))
		copy(sorted, data)
		sort.Float64s(sorted)

		q1Idx := int(0.25 * float64(n))
		q3Idx := int(0.75 * float64(n))
		iqr := sorted[q3Idx] - sorted[q1Idx]

		binWidth := 2.0 * iqr * math.Pow(float64(n), -1.0/3.0)
		dataRange := e.calculateRange(data)
		if binWidth > 0 {
			return int(math.Ceil(dataRange / binWidth))
		}
		return 10

	case SquareRoot:
		return int(math.Ceil(math.Sqrt(float64(n))))

	case Doane:
		// Doane's rule: extension of Sturges for non-normal data
		_, _, skewness, _ := e.calculateMoments(data)
		sigma := math.Sqrt(6.0 * float64(n-2) / (float64(n+1) * float64(n+3)))
		return int(1.0 + math.Log2(float64(n)) + math.Log2(1.0+math.Abs(skewness)/sigma))

	default:
		return e.params.NumBins
	}
}

// buildHistogram constructs a histogram from the data
func (e *Entropy) buildHistogram(data []float64, numBins int) ([]float64, []float64, []float64) {
	if len(data) == 0 || numBins <= 0 {
		return nil, nil, nil
	}

	min := data[0]
	max := data[0]

	// Find data range
	for _, x := range data {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}

	// Handle edge case where all values are the same
	if min == max {
		histogram := make([]float64, 1)
		histogram[0] = float64(len(data))
		binEdges := []float64{min, max}
		binCenters := []float64{min}
		return histogram, binEdges, binCenters
	}

	// Create bin edges
	binWidth := (max - min) / float64(numBins)
	binEdges := make([]float64, numBins+1)
	binCenters := make([]float64, numBins)

	for i := 0; i <= numBins; i++ {
		binEdges[i] = min + float64(i)*binWidth
	}

	for i := range numBins {
		binCenters[i] = binEdges[i] + binWidth/2.0
	}

	// Fill histogram
	histogram := make([]float64, numBins)
	for _, x := range data {
		binIdx := int((x - min) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		if binIdx < 0 {
			binIdx = 0
		}
		histogram[binIdx]++
	}

	return histogram, binEdges, binCenters
}

// smoothHistogram applies smoothing to the histogram
func (e *Entropy) smoothHistogram(histogram []float64) []float64 {
	if len(histogram) == 0 {
		return histogram
	}

	smoothed := make([]float64, len(histogram))
	width := int(e.params.SmoothingWidth)

	switch e.params.SmoothingKernel {
	case "gaussian":
		// Gaussian smoothing
		for i := range histogram {
			sum := 0.0
			weightSum := 0.0

			for j := -width; j <= width; j++ {
				idx := i + j
				if idx >= 0 && idx < len(histogram) {
					weight := math.Exp(-float64(j*j) / (2.0 * e.params.SmoothingWidth * e.params.SmoothingWidth))
					sum += histogram[idx] * weight
					weightSum += weight
				}
			}

			if weightSum > 0 {
				smoothed[i] = sum / weightSum
			}
		}

	case "uniform":
		// Uniform smoothing
		for i := range histogram {
			sum := 0.0
			count := 0

			for j := -width; j <= width; j++ {
				idx := i + j
				if idx >= 0 && idx < len(histogram) {
					sum += histogram[idx]
					count++
				}
			}

			if count > 0 {
				smoothed[i] = sum / float64(count)
			}
		}

	default:
		copy(smoothed, histogram)
	}

	return smoothed
}

// normalizeToProbabilities converts histogram counts to probabilities
func (e *Entropy) normalizeToProbabilities(histogram []float64) []float64 {
	if len(histogram) == 0 {
		return histogram
	}

	// Calculate total count
	total := 0.0
	for _, count := range histogram {
		total += count
	}

	if total == 0 {
		return histogram
	}

	// Normalize to probabilities
	probabilities := make([]float64, len(histogram))
	for i, count := range histogram {
		probabilities[i] = count / total

		// Apply minimum probability threshold
		if probabilities[i] < e.params.MinProbability {
			probabilities[i] = e.params.MinProbability
		}
	}

	return probabilities
}

// calculateShannonEntropy computes Shannon entropy
// H(X) = -∑ p(x) * log(p(x))
func (e *Entropy) calculateShannonEntropy(probabilities []float64) float64 {
	entropy := 0.0
	logBase := math.Log(e.params.BaseLog)

	for _, p := range probabilities {
		if p > 0 {
			entropy -= p * math.Log(p) / logBase
		}
	}

	return entropy
}

// calculateRenyiEntropy computes Rényi entropy of order α
// H_α(X) = (1/(1-α)) * log(∑ p(x)^α)
func (e *Entropy) calculateRenyiEntropy(probabilities []float64, alpha float64) float64 {
	if alpha == 1.0 {
		return e.calculateShannonEntropy(probabilities)
	}

	if alpha <= 0 {
		return 0.0
	}

	sum := 0.0
	for _, p := range probabilities {
		if p > 0 {
			sum += math.Pow(p, alpha)
		}
	}

	if sum <= 0 {
		return 0.0
	}

	logBase := math.Log(e.params.BaseLog)
	return math.Log(sum) / (logBase * (1.0 - alpha))
}

// calculateTsallisEntropy computes Tsallis entropy of order q
// S_q(X) = (1/(q-1)) * (1 - ∑ p(x)^q)
func (e *Entropy) calculateTsallisEntropy(probabilities []float64, q float64) float64 {
	if q == 1.0 {
		return e.calculateShannonEntropy(probabilities)
	}

	sum := 0.0
	for _, p := range probabilities {
		if p > 0 {
			sum += math.Pow(p, q)
		}
	}

	return (1.0 - sum) / (q - 1.0)
}

// calculateHartleyEntropy computes Hartley entropy (max entropy)
// H₀(X) = log(|supp(X)|) where |supp(X)| is the support size
func (e *Entropy) calculateHartleyEntropy(probabilities []float64) float64 {
	supportSize := 0
	for _, p := range probabilities {
		if p > 0 {
			supportSize++
		}
	}

	if supportSize <= 0 {
		return 0.0
	}

	logBase := math.Log(e.params.BaseLog)
	return math.Log(float64(supportSize)) / logBase
}

// calculateMinEntropy computes min entropy
// H_∞(X) = -log(max p(x))
func (e *Entropy) calculateMinEntropy(probabilities []float64) float64 {
	maxProb := 0.0
	for _, p := range probabilities {
		if p > maxProb {
			maxProb = p
		}
	}

	if maxProb <= 0 {
		return 0.0
	}

	logBase := math.Log(e.params.BaseLog)
	return -math.Log(maxProb) / logBase
}

// calculateEntropyRate computes the entropy rate of the signal
// Simplified implementation using first-order differences
func (e *Entropy) calculateEntropyRate(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Calculate first-order differences
	diffs := make([]float64, len(data)-1)
	for i := 1; i < len(data); i++ {
		diffs[i-1] = data[i] - data[i-1]
	}

	// Calculate entropy of differences
	result, err := e.Analyze(diffs)
	if err != nil {
		return 0.0
	}

	return result.ShannonEntropy
}

// calculateConditionalEntropy computes conditional entropy H(X|Y)
// Simplified implementation using lag-1 conditioning
func (e *Entropy) calculateConditionalEntropy(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Create joint distribution of (X_t, X_{t-1})
	numBins := int(math.Sqrt(float64(len(data))))
	numBins = max(numBins, 2)

	min := data[0]
	max := data[0]
	for _, x := range data {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}

	if min == max {
		return 0.0
	}

	binWidth := (max - min) / float64(numBins)

	// Build joint histogram
	jointHist := make([][]float64, numBins)
	for i := range jointHist {
		jointHist[i] = make([]float64, numBins)
	}

	for i := 1; i < len(data); i++ {
		xBin := int((data[i] - min) / binWidth)
		yBin := int((data[i-1] - min) / binWidth)

		if xBin >= numBins {
			xBin = numBins - 1
		}
		if yBin >= numBins {
			yBin = numBins - 1
		}

		jointHist[xBin][yBin]++
	}

	// Calculate conditional entropy
	total := float64(len(data) - 1)
	conditionalEntropy := 0.0
	logBase := math.Log(e.params.BaseLog)

	for i := 0; i < numBins; i++ {
		// Calculate marginal P(Y=y)
		marginalY := 0.0
		for j := 0; j < numBins; j++ {
			marginalY += jointHist[j][i]
		}

		if marginalY > 0 {
			// Calculate H(X|Y=y)
			entropyGivenY := 0.0
			for j := 0; j < numBins; j++ {
				if jointHist[j][i] > 0 {
					condProb := jointHist[j][i] / marginalY
					entropyGivenY -= condProb * math.Log(condProb) / logBase
				}
			}

			conditionalEntropy += (marginalY / total) * entropyGivenY
		}
	}

	return conditionalEntropy
}

// CalculateSpecificEntropy computes a specific type of entropy
func (e *Entropy) CalculateSpecificEntropy(data []float64, entropyType EntropyType) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("empty data")
	}

	// Build histogram
	numBins := e.determineOptimalBins(data, len(data))
	histogram, _, _ := e.buildHistogram(data, numBins)
	probabilities := e.normalizeToProbabilities(histogram)

	switch entropyType {
	case Shannon:
		return e.calculateShannonEntropy(probabilities), nil
	case Renyi:
		return e.calculateRenyiEntropy(probabilities, e.params.RenyiAlpha), nil
	case Tsallis:
		return e.calculateTsallisEntropy(probabilities, e.params.TsallisQ), nil
	case Hartley:
		return e.calculateHartleyEntropy(probabilities), nil
	case Collision:
		return e.calculateRenyiEntropy(probabilities, 2.0), nil
	case MinEntropy:
		return e.calculateMinEntropy(probabilities), nil
	default:
		return 0, fmt.Errorf("unknown entropy type")
	}
}

// SetParameters updates the entropy calculation parameters
func (e *Entropy) SetParameters(params EntropyParams) {
	e.params = params
}

// GetParameters returns the current parameters
func (e *Entropy) GetParameters() EntropyParams {
	return e.params
}

// GetEntropyTypeName returns the human-readable name of the entropy type
func GetEntropyTypeName(entropyType EntropyType) string {
	switch entropyType {
	case Shannon:
		return "Shannon Entropy"
	case Renyi:
		return "Rényi Entropy"
	case Tsallis:
		return "Tsallis Entropy"
	case Hartley:
		return "Hartley Entropy"
	case Collision:
		return "Collision Entropy"
	case MinEntropy:
		return "Min Entropy"
	default:
		return "Unknown"
	}
}
