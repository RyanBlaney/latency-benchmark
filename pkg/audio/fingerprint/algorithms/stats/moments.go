package stats

import (
	"fmt"
	"math"
	"sort"
)

// MomentType represents different types of moments
type MomentType int

const (
	// Raw moments about the origin
	RawMoment MomentType = iota

	// Central moments about the mean
	CentralMoment

	// Standardized moments (normalized by standard deviation)
	StandardizedMoment

	// Absolute moments
	AbsoluteMoment

	// Logarithmic moments
	LogarithmicMoment
)

// MomentResult contains comprehensive moment analysis results
type MomentResult struct {
	// Basic descriptive statistics
	Mean     float64 `json:"mean"`     // First raw moment (μ₁)
	Variance float64 `json:"variance"` // Second central moment (σ²)
	StdDev   float64 `json:"std_dev"`  // Standard deviation (σ)
	Skewness float64 `json:"skewness"` // Third standardized moment
	Kurtosis float64 `json:"kurtosis"` // Fourth standardized moment (excess)

	// Raw moments about origin
	RawMoments []float64 `json:"raw_moments"` // μ'ᵣ = E[Xʳ]

	// Central moments about mean
	CentralMoments []float64 `json:"central_moments"` // μᵣ = E[(X-μ)ʳ]

	// Standardized moments
	StandardizedMoments []float64 `json:"standardized_moments"` // μᵣ/σʳ

	// Absolute moments
	AbsoluteMoments []float64 `json:"absolute_moments"` // E[|X-μ|ʳ]

	// Higher order moment statistics
	Hyperskewness float64 `json:"hyperskewness"` // Fifth standardized moment
	Hyperkurtosis float64 `json:"hyperkurtosis"` // Sixth standardized moment

	// Distribution shape measures
	CoefficientOfVariation float64 `json:"coefficient_of_variation"` // σ/μ
	StandardError          float64 `json:"standard_error"`           // σ/√n

	// Moment-based distribution measures
	PearsonMomentSkewness float64 `json:"pearson_moment_skewness"` // 3(μ-median)/σ
	BowleySkewness        float64 `json:"bowley_skewness"`         // Quartile-based skewness

	// L-moments (linear combinations of order statistics)
	LMoments      []float64 `json:"l_moments"`       // L₁, L₂, L₃, L₄, ...
	LMomentRatios []float64 `json:"l_moment_ratios"` // τ₂, τ₃, τ₄, ... (L-CV, L-skew, L-kurt)

	// Cumulants (moment-generating function derivatives)
	Cumulants []float64 `json:"cumulants"` // κ₁, κ₂, κ₃, κ₄, ...

	// Sample properties
	NumSamples  int     `json:"num_samples"`
	SampleRange float64 `json:"sample_range"`

	// Moment convergence information
	MomentOrder int    `json:"moment_order"` // Highest computed moment order
	Convergence []bool `json:"convergence"`  // Whether each moment converged
}

// MomentParams contains parameters for moment calculation
type MomentParams struct {
	MaxOrder         int  `json:"max_order"`         // Maximum moment order to compute
	ComputeLMoments  bool `json:"compute_l_moments"` // Whether to compute L-moments
	ComputeCumulants bool `json:"compute_cumulants"` // Whether to compute cumulants

	// Numerical stability options
	UseWelfordMethod  bool    `json:"use_welford_method"`  // Use Welford's algorithm for numerical stability
	ClampLargeMoments bool    `json:"clamp_large_moments"` // Clamp extremely large moments
	MaxMomentValue    float64 `json:"max_moment_value"`    // Maximum allowed moment value

	// Distribution fitting options
	FitDistribution bool `json:"fit_distribution"` // Attempt to fit known distributions
	TestNormality   bool `json:"test_normality"`   // Test for normality using moments

	// Robust estimation options
	UseRobustEstimation bool    `json:"use_robust_estimation"` // Use robust moment estimation
	TrimProportion      float64 `json:"trim_proportion"`       // Proportion to trim for robust estimation
}

// Moments implements comprehensive statistical moment analysis for audio signals
//
// References:
// - Kendall, M., Stuart, A. (1977). "The Advanced Theory of Statistics, Volume 1"
// - Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of Distributions"
// - Serfling, R., Xiao, P. (2007). "A contribution to multivariate L-moments"
// - Cramér, H. (1946). "Mathematical Methods of Statistics"
// - Pearson, K. (1895). "Contributions to the Mathematical Theory of Evolution"
// - Fisher, R.A. (1930). "The moments of the distribution for normal samples"
// - Welford, B.P. (1962). "Note on a method for calculating corrected sums of squares"
//
// Moments provide fundamental characterization of probability distributions:
// - 1st moment: Location (mean)
// - 2nd moment: Spread (variance)
// - 3rd moment: Asymmetry (skewness)
// - 4th moment: Tail behavior (kurtosis)
// - Higher moments: Fine distribution details
//
// Applications in audio processing:
// - Signal characterization and classification
// - Noise analysis and detection
// - Audio texture analysis
// - Distribution fitting and modeling
// - Anomaly detection in audio streams
type Moments struct {
	params MomentParams
}

// NewMoments creates a new moment analyzer with default parameters
func NewMoments() *Moments {
	return &Moments{
		params: MomentParams{
			MaxOrder:            6,
			ComputeLMoments:     true,
			ComputeCumulants:    true,
			UseWelfordMethod:    true,
			ClampLargeMoments:   true,
			MaxMomentValue:      1e12,
			FitDistribution:     false,
			TestNormality:       true,
			UseRobustEstimation: false,
			TrimProportion:      0.1,
		},
	}
}

// NewMomentsWithParams creates a moment analyzer with custom parameters
func NewMomentsWithParams(params MomentParams) *Moments {
	return &Moments{params: params}
}

// Analyze computes comprehensive moment analysis for the input data
func (m *Moments) Analyze(data []float64) (*MomentResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}

	n := len(data)

	// Robust estimation if requested
	workingData := data
	if m.params.UseRobustEstimation {
		workingData = m.trimData(data, m.params.TrimProportion)
	}

	// Calculate basic statistics
	var mean, variance, stdDev float64
	if m.params.UseWelfordMethod {
		mean, variance = m.welfordVariance(workingData)
	} else {
		mean, variance = m.classicVariance(workingData)
	}
	stdDev = math.Sqrt(variance)

	// Calculate sample range
	sampleRange := m.calculateRange(workingData)

	// Calculate raw moments
	rawMoments := m.calculateRawMoments(workingData, m.params.MaxOrder)

	// Calculate central moments
	centralMoments := m.calculateCentralMoments(workingData, mean, m.params.MaxOrder)

	// Calculate standardized moments
	standardizedMoments := m.calculateStandardizedMoments(centralMoments, stdDev)

	// Calculate absolute moments
	absoluteMoments := m.calculateAbsoluteMoments(workingData, mean, m.params.MaxOrder)

	// Extract standard moment measures
	skewness := 0.0
	kurtosis := 0.0
	hyperskewness := 0.0
	hyperkurtosis := 0.0

	if len(standardizedMoments) > 3 {
		skewness = standardizedMoments[3]
	}
	if len(standardizedMoments) > 4 {
		kurtosis = standardizedMoments[4] - 3.0 // Excess kurtosis
	}
	if len(standardizedMoments) > 5 {
		hyperskewness = standardizedMoments[5]
	}
	if len(standardizedMoments) > 6 {
		hyperkurtosis = standardizedMoments[6]
	}

	// Calculate derived statistics
	coefficientOfVariation := 0.0
	if mean != 0 {
		coefficientOfVariation = stdDev / math.Abs(mean)
	}

	standardError := stdDev / math.Sqrt(float64(n))

	// Calculate Pearson moment skewness
	pearsonMomentSkewness := m.calculatePearsonMomentSkewness(workingData, mean, stdDev)

	// Calculate Bowley skewness
	bowleySkewness := m.calculateBowleySkewness(workingData)

	// Calculate L-moments if requested
	var lMoments, lMomentRatios []float64
	if m.params.ComputeLMoments {
		lMoments = m.calculateLMoments(workingData, m.params.MaxOrder)
		lMomentRatios = m.calculateLMomentRatios(lMoments)
	}

	// Calculate cumulants if requested
	var cumulants []float64
	if m.params.ComputeCumulants {
		cumulants = m.calculateCumulants(centralMoments, m.params.MaxOrder)
	}

	// Check moment convergence
	convergence := m.checkMomentConvergence(rawMoments, centralMoments)

	return &MomentResult{
		Mean:                   mean,
		Variance:               variance,
		StdDev:                 stdDev,
		Skewness:               skewness,
		Kurtosis:               kurtosis,
		RawMoments:             rawMoments,
		CentralMoments:         centralMoments,
		StandardizedMoments:    standardizedMoments,
		AbsoluteMoments:        absoluteMoments,
		Hyperskewness:          hyperskewness,
		Hyperkurtosis:          hyperkurtosis,
		CoefficientOfVariation: coefficientOfVariation,
		StandardError:          standardError,
		PearsonMomentSkewness:  pearsonMomentSkewness,
		BowleySkewness:         bowleySkewness,
		LMoments:               lMoments,
		LMomentRatios:          lMomentRatios,
		Cumulants:              cumulants,
		NumSamples:             n,
		SampleRange:            sampleRange,
		MomentOrder:            m.params.MaxOrder,
		Convergence:            convergence,
	}, nil
}

// trimData removes extreme values for robust estimation
func (m *Moments) trimData(data []float64, trimProportion float64) []float64 {
	if trimProportion <= 0 || trimProportion >= 0.5 {
		return data
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)
	trimCount := int(trimProportion * float64(n))

	if trimCount >= n/2 {
		return data
	}

	return sorted[trimCount : n-trimCount]
}

// welfordVariance calculates mean and variance using Welford's algorithm for numerical stability
// Reference: Welford, B.P. (1962). "Note on a method for calculating corrected sums of squares"
func (m *Moments) welfordVariance(data []float64) (mean, variance float64) {
	n := len(data)
	if n == 0 {
		return 0, 0
	}

	mean = 0.0
	m2 := 0.0

	for i, x := range data {
		delta := x - mean
		mean += delta / float64(i+1)
		delta2 := x - mean
		m2 += delta * delta2
	}

	if n > 1 {
		variance = m2 / float64(n-1) // Sample variance
	}

	return mean, variance
}

// classicVariance calculates mean and variance using classic formulas
func (m *Moments) classicVariance(data []float64) (mean, variance float64) {
	n := len(data)
	if n == 0 {
		return 0, 0
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

	if n > 1 {
		variance = sumSq / float64(n-1) // Sample variance
	}

	return mean, variance
}

// calculateRange computes the range of the data
func (m *Moments) calculateRange(data []float64) float64 {
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

// calculateRawMoments computes raw moments about the origin
// Raw moment of order r: μ'ᵣ = E[Xʳ] = (1/n) ∑ xᵢʳ
func (m *Moments) calculateRawMoments(data []float64, maxOrder int) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}

	moments := make([]float64, maxOrder+1)

	for r := 0; r <= maxOrder; r++ {
		sum := 0.0
		for _, x := range data {
			if r == 0 {
				sum += 1.0
			} else {
				power := math.Pow(x, float64(r))
				if m.params.ClampLargeMoments && math.Abs(power) > m.params.MaxMomentValue {
					power = math.Copysign(m.params.MaxMomentValue, power)
				}
				sum += power
			}
		}
		moments[r] = sum / float64(n)
	}

	return moments
}

// calculateCentralMoments computes central moments about the mean
// Central moment of order r: μᵣ = E[(X-μ)ʳ] = (1/n) ∑ (xᵢ-μ)ʳ
func (m *Moments) calculateCentralMoments(data []float64, mean float64, maxOrder int) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}

	moments := make([]float64, maxOrder+1)

	for r := 0; r <= maxOrder; r++ {
		sum := 0.0
		for _, x := range data {
			switch r {
			case 0:
				sum += 1.0
			case 1:
				sum += 0.0 // Central moment of order 1 is always 0
			default:
				power := math.Pow(x-mean, float64(r))
				if m.params.ClampLargeMoments && math.Abs(power) > m.params.MaxMomentValue {
					power = math.Copysign(m.params.MaxMomentValue, power)
				}
				sum += power
			}
		}
		moments[r] = sum / float64(n)
	}

	return moments
}

// calculateStandardizedMoments computes standardized moments
// Standardized moment of order r: βᵣ = μᵣ/σʳ
func (m *Moments) calculateStandardizedMoments(centralMoments []float64, stdDev float64) []float64 {
	if len(centralMoments) == 0 || stdDev == 0 {
		return nil
	}

	standardized := make([]float64, len(centralMoments))

	for r := range len(centralMoments) {
		switch r {
		case 0:
			standardized[r] = 1.0
		case 1:
			standardized[r] = 0.0
		case 2:
			standardized[r] = 1.0
		default:
			standardized[r] = centralMoments[r] / math.Pow(stdDev, float64(r))
		}
	}

	return standardized
}

// calculateAbsoluteMoments computes absolute moments about the mean
// Absolute moment of order r: E[|X-μ|ʳ] = (1/n) ∑ |xᵢ-μ|ʳ
func (m *Moments) calculateAbsoluteMoments(data []float64, mean float64, maxOrder int) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}

	moments := make([]float64, maxOrder+1)

	for r := 0; r <= maxOrder; r++ {
		sum := 0.0
		for _, x := range data {
			if r == 0 {
				sum += 1.0
			} else {
				power := math.Pow(math.Abs(x-mean), float64(r))
				if m.params.ClampLargeMoments && power > m.params.MaxMomentValue {
					power = m.params.MaxMomentValue
				}
				sum += power
			}
		}
		moments[r] = sum / float64(n)
	}

	return moments
}

// calculatePearsonMomentSkewness computes Pearson's moment coefficient of skewness
// Pearson skewness = 3(mean - median) / standard_deviation
func (m *Moments) calculatePearsonMomentSkewness(data []float64, mean, stdDev float64) float64 {
	if len(data) == 0 || stdDev == 0 {
		return 0.0
	}

	// Calculate median
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)
	var median float64

	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		median = sorted[n/2]
	}

	return 3.0 * (mean - median) / stdDev
}

// calculateBowleySkewness computes Bowley's coefficient of skewness (quartile skewness)
// Bowley skewness = (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
func (m *Moments) calculateBowleySkewness(data []float64) float64 {
	if len(data) < 4 {
		return 0.0
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)

	// Calculate quartiles
	q1Idx := int(0.25 * float64(n))
	q2Idx := int(0.50 * float64(n))
	q3Idx := int(0.75 * float64(n))

	if q1Idx >= n {
		q1Idx = n - 1
	}
	if q2Idx >= n {
		q2Idx = n - 1
	}
	if q3Idx >= n {
		q3Idx = n - 1
	}

	q1 := sorted[q1Idx]
	q2 := sorted[q2Idx]
	q3 := sorted[q3Idx]

	denominator := q3 - q1
	if denominator == 0 {
		return 0.0
	}

	return (q3 + q1 - 2.0*q2) / denominator
}

// calculateLMoments computes L-moments (linear combinations of order statistics)
// Reference: Hosking, J.R.M. (1990). "L-moments: Analysis and Estimation of Distributions"
func (m *Moments) calculateLMoments(data []float64, maxOrder int) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}

	// Sort data
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	lMoments := make([]float64, maxOrder+1)

	// L₁ (location) = mean
	sum := 0.0
	for _, x := range sorted {
		sum += x
	}
	lMoments[1] = sum / float64(n)

	// Higher order L-moments
	for r := 2; r <= maxOrder && r <= n; r++ {
		sum := 0.0
		for i := range n {
			// Calculate binomial coefficient and weight
			weight := 0.0
			for k := range r {
				binomCoeff := m.binomialCoeff(i, k) * m.binomialCoeff(n-1-i, r-1-k)
				weight += float64(binomCoeff) * math.Pow(-1.0, float64(r-1-k))
			}
			sum += weight * sorted[i]
		}
		lMoments[r] = sum / (float64(r) * float64(m.binomialCoeff(n, r)))
	}

	return lMoments
}

// calculateLMomentRatios computes L-moment ratios
func (m *Moments) calculateLMomentRatios(lMoments []float64) []float64 {
	if len(lMoments) < 2 {
		return nil
	}

	ratios := make([]float64, len(lMoments))

	// τ₂ = L₂/L₁ (L-CV)
	if len(lMoments) > 2 && lMoments[1] != 0 {
		ratios[2] = lMoments[2] / lMoments[1]
	}

	// τᵣ = Lᵣ/L₂ for r ≥ 3
	for r := 3; r < len(lMoments); r++ {
		if lMoments[2] != 0 {
			ratios[r] = lMoments[r] / lMoments[2]
		}
	}

	return ratios
}

// calculateCumulants computes cumulants from central moments
// Reference: Cramér, H. (1946). "Mathematical Methods of Statistics"
func (m *Moments) calculateCumulants(centralMoments []float64, maxOrder int) []float64 {
	if len(centralMoments) == 0 {
		return nil
	}

	cumulants := make([]float64, maxOrder+1)

	// Copy available central moments
	for i := 0; i < len(centralMoments) && i <= maxOrder; i++ {
		cumulants[i] = centralMoments[i]
	}

	// Apply cumulant-moment relationships
	if maxOrder >= 4 && len(centralMoments) > 4 {
		// κ₄ = μ₄ - 3μ₂²
		cumulants[4] = centralMoments[4] - 3.0*centralMoments[2]*centralMoments[2]
	}

	// Higher order cumulants can be computed using Bell polynomials
	// For simplicity, we'll use the central moments as approximations

	return cumulants
}

// checkMomentConvergence checks if moments have converged to reasonable values
func (m *Moments) checkMomentConvergence(rawMoments, centralMoments []float64) []bool {
	maxLen := len(rawMoments)
	maxLen = max(maxLen, len(centralMoments))

	convergence := make([]bool, maxLen)

	for i := 0; i < maxLen; i++ {
		converged := true

		// Check raw moments
		if i < len(rawMoments) {
			if math.IsNaN(rawMoments[i]) || math.IsInf(rawMoments[i], 0) {
				converged = false
			}
			if m.params.ClampLargeMoments && math.Abs(rawMoments[i]) > m.params.MaxMomentValue {
				converged = false
			}
		}

		// Check central moments
		if i < len(centralMoments) {
			if math.IsNaN(centralMoments[i]) || math.IsInf(centralMoments[i], 0) {
				converged = false
			}
			if m.params.ClampLargeMoments && math.Abs(centralMoments[i]) > m.params.MaxMomentValue {
				converged = false
			}
		}

		convergence[i] = converged
	}

	return convergence
}

// binomialCoeff calculates binomial coefficient (n choose k)
func (m *Moments) binomialCoeff(n, k int) int {
	if k > n || k < 0 {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}

	result := 1
	for i := 0; i < k; i++ {
		result = result * (n - i) / (i + 1)
	}

	return result
}

// CalculateSpecificMoment computes a specific moment of given type and order
func (m *Moments) CalculateSpecificMoment(data []float64, momentType MomentType, order int) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("empty data")
	}

	if order < 0 {
		return 0, fmt.Errorf("moment order must be non-negative")
	}

	mean, _ := m.welfordVariance(data)

	switch momentType {
	case RawMoment:
		moments := m.calculateRawMoments(data, order)
		if order < len(moments) {
			return moments[order], nil
		}
		return 0, fmt.Errorf("order %d exceeds calculated moments", order)

	case CentralMoment:
		moments := m.calculateCentralMoments(data, mean, order)
		if order < len(moments) {
			return moments[order], nil
		}
		return 0, fmt.Errorf("order %d exceeds calculated moments", order)

	case StandardizedMoment:
		centralMoments := m.calculateCentralMoments(data, mean, order)
		_, variance := m.welfordVariance(data)
		stdDev := math.Sqrt(variance)
		standardized := m.calculateStandardizedMoments(centralMoments, stdDev)
		if order < len(standardized) {
			return standardized[order], nil
		}
		return 0, fmt.Errorf("order %d exceeds calculated moments", order)

	case AbsoluteMoment:
		moments := m.calculateAbsoluteMoments(data, mean, order)
		if order < len(moments) {
			return moments[order], nil
		}
		return 0, fmt.Errorf("order %d exceeds calculated moments", order)

	default:
		return 0, fmt.Errorf("unsupported moment type")
	}
}

// SetParameters updates the moment calculation parameters
func (m *Moments) SetParameters(params MomentParams) {
	m.params = params
}

// GetParameters returns the current parameters
func (m *Moments) GetParameters() MomentParams {
	return m.params
}

// GetMomentTypeName returns the human-readable name of the moment type
func GetMomentTypeName(momentType MomentType) string {
	switch momentType {
	case RawMoment:
		return "Raw Moment"
	case CentralMoment:
		return "Central Moment"
	case StandardizedMoment:
		return "Standardized Moment"
	case AbsoluteMoment:
		return "Absolute Moment"
	case LogarithmicMoment:
		return "Logarithmic Moment"
	default:
		return "Unknown"
	}
}
