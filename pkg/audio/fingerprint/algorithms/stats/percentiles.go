package stats

import (
	"fmt"
	"math"
	"sort"
)

// PercentileMethod represents different methods for calculating percentiles
type PercentileMethod int

const (
	// Linear interpolation between closest ranks (R-6, Excel, most common)
	Linear PercentileMethod = iota

	// Lower value of the two closest ranks (R-1)
	Lower

	// Higher value of the two closest ranks (R-3)
	Higher

	// Midpoint of the two closest ranks (R-2)
	Midpoint

	// Weighted average giving more weight to closer rank (R-4)
	Weighted

	// Method used by R default (R-7)
	RDefault

	// Median-unbiased method (R-8)
	MedianUnbiased

	// Approximately normal unbiased method (R-9)
	NormalUnbiased
)

// PercentileResult contains comprehensive percentile statistics
type PercentileResult struct {
	Values      []float64           `json:"values"`      // Original sorted values
	Percentiles map[float64]float64 `json:"percentiles"` // Percentile -> value mapping
	Quartiles   QuartileInfo        `json:"quartiles"`   // Q1, Q2, Q3 information
	Extremes    ExtremeInfo         `json:"extremes"`    // Min, max, outliers
	Summary     SummaryStats        `json:"summary"`     // Basic statistics
	Method      PercentileMethod    `json:"method"`      // Calculation method used
}

// QuartileInfo contains quartile-specific information
type QuartileInfo struct {
	Q1  float64 `json:"q1"`  // First quartile (25th percentile)
	Q2  float64 `json:"q2"`  // Second quartile (50th percentile, median)
	Q3  float64 `json:"q3"`  // Third quartile (75th percentile)
	IQR float64 `json:"iqr"` // Interquartile range (Q3 - Q1)
}

// ExtremeInfo contains information about extreme values
type ExtremeInfo struct {
	Min           float64   `json:"min"`
	Max           float64   `json:"max"`
	Range         float64   `json:"range"`
	LowerOutliers []float64 `json:"lower_outliers"` // Values < Q1 - 1.5*IQR
	UpperOutliers []float64 `json:"upper_outliers"` // Values > Q3 + 1.5*IQR
	LowerExtreme  []float64 `json:"lower_extreme"`  // Values < Q1 - 3*IQR
	UpperExtreme  []float64 `json:"upper_extreme"`  // Values > Q3 + 3*IQR
}

// SummaryStats contains basic summary statistics
type SummaryStats struct {
	Count    int     `json:"count"`
	Mean     float64 `json:"mean"`
	Variance float64 `json:"variance"`
	StdDev   float64 `json:"std_dev"`
	Skewness float64 `json:"skewness"`
	Kurtosis float64 `json:"kurtosis"`
}

// Percentiles implements various percentile calculation methods for statistical analysis
//
// References:
//   - Hyndman, R.J., Fan, Y. (1996). "Sample Quantiles in Statistical Packages"
//     The American Statistician, 50(4), 361-365
//   - NIST/SEMATECH e-Handbook of Statistical Methods
//   - Tukey, J.W. (1977). "Exploratory Data Analysis"
//   - McGill, R., Tukey, J.W., Larsen, W.A. (1978). "Variations of box plots"
//
// This implementation provides multiple percentile calculation methods:
// 1. Linear interpolation (most common, used by Excel)
// 2. Lower/Higher value methods
// 3. R-compatible methods (R-1 through R-9)
// 4. Outlier detection using IQR method
// 5. Comprehensive statistical summaries
type Percentiles struct {
	method   PercentileMethod
	outlierK float64 // Outlier detection multiplier (default 1.5)
	extremeK float64 // Extreme value multiplier (default 3.0)
}

// NewPercentiles creates a new percentile analyzer with linear interpolation method
func NewPercentiles() *Percentiles {
	return &Percentiles{
		method:   Linear,
		outlierK: 1.5,
		extremeK: 3.0,
	}
}

// NewPercentilesWithMethod creates a percentile analyzer with specified method
func NewPercentilesWithMethod(method PercentileMethod) *Percentiles {
	return &Percentiles{
		method:   method,
		outlierK: 1.5,
		extremeK: 3.0,
	}
}

// NewPercentilesWithOutlierThreshold creates analyzer with custom outlier detection
func NewPercentilesWithOutlierThreshold(method PercentileMethod, outlierK, extremeK float64) *Percentiles {
	return &Percentiles{
		method:   method,
		outlierK: outlierK,
		extremeK: extremeK,
	}
}

// Analyze computes comprehensive percentile statistics for the input data
func (p *Percentiles) Analyze(data []float64) (*PercentileResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}

	// Create a copy and sort
	values := make([]float64, len(data))
	copy(values, data)
	sort.Float64s(values)

	// Calculate standard percentiles
	percentiles := make(map[float64]float64)
	standardPercentiles := []float64{
		0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100,
	}

	for _, pct := range standardPercentiles {
		val, err := p.calculatePercentile(values, pct)
		if err != nil {
			return nil, err
		}
		percentiles[pct] = val
	}

	// Calculate quartiles
	q1 := percentiles[25]
	q2 := percentiles[50]
	q3 := percentiles[75]
	iqr := q3 - q1

	quartiles := QuartileInfo{
		Q1:  q1,
		Q2:  q2,
		Q3:  q3,
		IQR: iqr,
	}

	// Find extremes and outliers
	extremes := p.findExtremes(values, quartiles)

	// Calculate summary statistics
	summary := p.calculateSummaryStats(values)

	return &PercentileResult{
		Values:      values,
		Percentiles: percentiles,
		Quartiles:   quartiles,
		Extremes:    extremes,
		Summary:     summary,
		Method:      p.method,
	}, nil
}

// CalculatePercentile computes a single percentile value
func (p *Percentiles) CalculatePercentile(data []float64, percentile float64) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("empty data")
	}

	if percentile < 0 || percentile > 100 {
		return 0, fmt.Errorf("percentile must be between 0 and 100")
	}

	// Create a copy and sort
	values := make([]float64, len(data))
	copy(values, data)
	sort.Float64s(values)

	return p.calculatePercentile(values, percentile)
}

// calculatePercentile implements various percentile calculation methods
func (p *Percentiles) calculatePercentile(sortedData []float64, percentile float64) (float64, error) {
	n := len(sortedData)
	if n == 0 {
		return 0, fmt.Errorf("empty data")
	}

	if n == 1 {
		return sortedData[0], nil
	}

	// Convert percentile to quantile (0-1 range)
	q := percentile / 100.0

	switch p.method {
	case Linear:
		return p.linearInterpolation(sortedData, q), nil
	case Lower:
		return p.lowerValue(sortedData, q), nil
	case Higher:
		return p.higherValue(sortedData, q), nil
	case Midpoint:
		return p.midpointValue(sortedData, q), nil
	case Weighted:
		return p.weightedAverage(sortedData, q), nil
	case RDefault:
		return p.rDefaultMethod(sortedData, q), nil
	case MedianUnbiased:
		return p.medianUnbiased(sortedData, q), nil
	case NormalUnbiased:
		return p.normalUnbiased(sortedData, q), nil
	default:
		return p.linearInterpolation(sortedData, q), nil
	}
}

// linearInterpolation implements linear interpolation method (R-6, Excel default)
// Formula: h = (n-1) * p + 1, where p is the quantile
func (p *Percentiles) linearInterpolation(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n-1)*q + 1.0

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	// Linear interpolation between floor and ceiling
	lower := int(math.Floor(h)) - 1 // Convert to 0-based index
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	fraction := h - math.Floor(h)
	return data[lower] + fraction*(data[upper]-data[lower])
}

// lowerValue returns the lower of two closest values (R-1)
func (p *Percentiles) lowerValue(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n) * q

	if h <= 1.0 {
		return data[0]
	}

	index := int(math.Ceil(h)) - 1
	if index >= n {
		index = n - 1
	}

	return data[index]
}

// higherValue returns the higher of two closest values (R-3)
func (p *Percentiles) higherValue(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n) * q

	if h < 1.0 {
		return data[0]
	}

	index := int(math.Floor(h))
	if index >= n {
		index = n - 1
	}

	return data[index]
}

// midpointValue returns the midpoint of two closest values (R-2)
func (p *Percentiles) midpointValue(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n) * q

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	lower := int(math.Floor(h)) - 1
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	return (data[lower] + data[upper]) / 2.0
}

// weightedAverage implements weighted average method (R-4)
func (p *Percentiles) weightedAverage(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n) * q

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	lower := int(math.Floor(h)) - 1
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	fraction := h - math.Floor(h)
	return data[lower] + fraction*(data[upper]-data[lower])
}

// rDefaultMethod implements R's default quantile method (R-7)
func (p *Percentiles) rDefaultMethod(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n-1)*q + 1.0

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	lower := int(math.Floor(h)) - 1
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	fraction := h - math.Floor(h)
	return data[lower] + fraction*(data[upper]-data[lower])
}

// medianUnbiased implements median-unbiased method (R-8)
func (p *Percentiles) medianUnbiased(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n+1)/3.0 + float64(n-1)/3.0*q

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	lower := int(math.Floor(h)) - 1
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	fraction := h - math.Floor(h)
	return data[lower] + fraction*(data[upper]-data[lower])
}

// normalUnbiased implements approximately normal unbiased method (R-9)
func (p *Percentiles) normalUnbiased(data []float64, q float64) float64 {
	n := len(data)
	h := float64(n)/4.0 + 0.25 + q*(float64(n)+0.5)

	if h <= 1.0 {
		return data[0]
	}
	if h >= float64(n) {
		return data[n-1]
	}

	lower := int(math.Floor(h)) - 1
	upper := int(math.Ceil(h)) - 1

	if lower == upper {
		return data[lower]
	}

	fraction := h - math.Floor(h)
	return data[lower] + fraction*(data[upper]-data[lower])
}

// findExtremes identifies outliers and extreme values using IQR method
func (p *Percentiles) findExtremes(data []float64, quartiles QuartileInfo) ExtremeInfo {
	n := len(data)
	if n == 0 {
		return ExtremeInfo{}
	}

	min := data[0]
	max := data[n-1]

	// Calculate outlier boundaries
	lowerOutlierBound := quartiles.Q1 - p.outlierK*quartiles.IQR
	upperOutlierBound := quartiles.Q3 + p.outlierK*quartiles.IQR

	// Calculate extreme value boundaries
	lowerExtremeBound := quartiles.Q1 - p.extremeK*quartiles.IQR
	upperExtremeBound := quartiles.Q3 + p.extremeK*quartiles.IQR

	var lowerOutliers, upperOutliers []float64
	var lowerExtreme, upperExtreme []float64

	for _, value := range data {
		if value < lowerOutlierBound {
			lowerOutliers = append(lowerOutliers, value)
			if value < lowerExtremeBound {
				lowerExtreme = append(lowerExtreme, value)
			}
		} else if value > upperOutlierBound {
			upperOutliers = append(upperOutliers, value)
			if value > upperExtremeBound {
				upperExtreme = append(upperExtreme, value)
			}
		}
	}

	return ExtremeInfo{
		Min:           min,
		Max:           max,
		Range:         max - min,
		LowerOutliers: lowerOutliers,
		UpperOutliers: upperOutliers,
		LowerExtreme:  lowerExtreme,
		UpperExtreme:  upperExtreme,
	}
}

// calculateSummaryStats computes basic summary statistics
func (p *Percentiles) calculateSummaryStats(data []float64) SummaryStats {
	n := len(data)
	if n == 0 {
		return SummaryStats{}
	}

	// Calculate mean
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	mean := sum / float64(n)

	// Calculate variance
	variance := 0.0
	for _, value := range data {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(n - 1) // Sample variance

	stdDev := math.Sqrt(variance)

	// Calculate skewness (third central moment)
	skewness := 0.0
	if stdDev > 0 {
		for _, value := range data {
			standardized := (value - mean) / stdDev
			skewness += standardized * standardized * standardized
		}
		skewness /= float64(n)
	}

	// Calculate kurtosis (fourth central moment)
	kurtosis := 0.0
	if stdDev > 0 {
		for _, value := range data {
			standardized := (value - mean) / stdDev
			kurtosis += standardized * standardized * standardized * standardized
		}
		kurtosis /= float64(n)
		kurtosis -= 3.0 // Excess kurtosis (subtract 3 for normal distribution)
	}

	return SummaryStats{
		Count:    n,
		Mean:     mean,
		Variance: variance,
		StdDev:   stdDev,
		Skewness: skewness,
		Kurtosis: kurtosis,
	}
}

// CalculateCustomPercentiles computes percentiles for custom percentile values
func (p *Percentiles) CalculateCustomPercentiles(data []float64, percentiles []float64) (map[float64]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}

	// Create a copy and sort
	values := make([]float64, len(data))
	copy(values, data)
	sort.Float64s(values)

	result := make(map[float64]float64)

	for _, pct := range percentiles {
		if pct < 0 || pct > 100 {
			return nil, fmt.Errorf("percentile %f must be between 0 and 100", pct)
		}

		val, err := p.calculatePercentile(values, pct)
		if err != nil {
			return nil, err
		}
		result[pct] = val
	}

	return result, nil
}

// GetMethodName returns the human-readable name of the percentile method
func (p *Percentiles) GetMethodName() string {
	switch p.method {
	case Linear:
		return "Linear Interpolation"
	case Lower:
		return "Lower Value"
	case Higher:
		return "Higher Value"
	case Midpoint:
		return "Midpoint"
	case Weighted:
		return "Weighted Average"
	case RDefault:
		return "R Default (R-7)"
	case MedianUnbiased:
		return "Median Unbiased (R-8)"
	case NormalUnbiased:
		return "Normal Unbiased (R-9)"
	default:
		return "Unknown"
	}
}

// SetMethod changes the percentile calculation method
func (p *Percentiles) SetMethod(method PercentileMethod) {
	p.method = method
}

// SetOutlierThresholds sets custom outlier detection thresholds
func (p *Percentiles) SetOutlierThresholds(outlierK, extremeK float64) {
	p.outlierK = outlierK
	p.extremeK = extremeK
}

// GetOutlierThresholds returns current outlier detection thresholds
func (p *Percentiles) GetOutlierThresholds() (float64, float64) {
	return p.outlierK, p.extremeK
}

// CalculateBoxPlotStatistics computes the five-number summary for box plots
// Returns: min, Q1, median, Q3, max
func (p *Percentiles) CalculateBoxPlotStatistics(data []float64) (float64, float64, float64, float64, float64, error) {
	if len(data) == 0 {
		return 0, 0, 0, 0, 0, fmt.Errorf("empty data")
	}

	values := make([]float64, len(data))
	copy(values, data)
	sort.Float64s(values)

	min := values[0]
	max := values[len(values)-1]

	q1, err := p.calculatePercentile(values, 25)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}

	median, err := p.calculatePercentile(values, 50)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}

	q3, err := p.calculatePercentile(values, 75)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}

	return min, q1, median, q3, max, nil
}

// CalculatePercentileRank computes the percentile rank of a given value
// Returns the percentage of values that are less than or equal to the given value
func (p *Percentiles) CalculatePercentileRank(data []float64, value float64) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("empty data")
	}

	count := 0
	for _, v := range data {
		if v <= value {
			count++
		}
	}

	return float64(count) / float64(len(data)) * 100.0, nil
}
