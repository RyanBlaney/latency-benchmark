package common

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

// Basic statistical functions used across algorithms using gonum for robustness

// Mean calculates the arithmetic mean of a slice using gonum
func Mean(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	return stat.Mean(data, nil)
}

// Variance calculates the sample variance of a slice using gonum
func Variance(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}
	return stat.Variance(data, nil)
}

// StandardDeviation calculates the sample standard deviation
func StandardDeviation(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}
	return math.Sqrt(Variance(data))
}

// Percentile calculates the p-th percentile (p between 0 and 1)
func Percentile(data []float64, p float64) float64 {
	if len(data) == 0 || p < 0 || p > 1 {
		return 0.0
	}

	// Make a copy and sort
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	// Use gonum's quantile function
	return stat.Quantile(p, stat.Empirical, sorted, nil)
}

// RMS calculates root mean square
func RMS(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}

	sumSquares := 0.0
	for _, val := range data {
		sumSquares += val * val
	}

	return math.Sqrt(sumSquares / float64(len(data)))
}

// Normalize normalizes data to zero mean and unit variance
func Normalize(data []float64) []float64 {
	if len(data) == 0 {
		return data
	}

	mean := Mean(data)
	std := StandardDeviation(data)

	if std < 1e-10 {
		// Handle constant data
		normalized := make([]float64, len(data))
		for i, val := range data {
			normalized[i] = val - mean
		}
		return normalized
	}

	normalized := make([]float64, len(data))
	for i, val := range data {
		normalized[i] = (val - mean) / std
	}

	return normalized
}

// MinMaxNormalize normalizes data to [0, 1] range
func MinMaxNormalize(data []float64) []float64 {
	if len(data) == 0 {
		return data
	}

	min := floats.Min(data)
	max := floats.Max(data)

	if math.Abs(max-min) < 1e-10 {
		// Handle constant data
		normalized := make([]float64, len(data))
		return normalized // All zeros
	}

	normalized := make([]float64, len(data))
	for i, val := range data {
		normalized[i] = (val - min) / (max - min)
	}

	return normalized
}

// EnergyNormalize normalizes by total energy
func EnergyNormalize(data []float64) []float64 {
	if len(data) == 0 {
		return data
	}

	energy := 0.0
	for _, val := range data {
		energy += val * val
	}

	if energy < 1e-10 {
		return data // Return unchanged if no energy
	}

	energyNorm := math.Sqrt(energy)
	normalized := make([]float64, len(data))
	for i, val := range data {
		normalized[i] = val / energyNorm
	}

	return normalized
}

// MovingAverage calculates simple moving average with given window size
func MovingAverage(data []float64, windowSize int) []float64 {
	if len(data) == 0 || windowSize <= 0 || windowSize > len(data) {
		return data
	}

	result := make([]float64, len(data))

	// Handle initial window
	for i := 0; i < windowSize; i++ {
		sum := 0.0
		for j := 0; j <= i; j++ {
			sum += data[j]
		}
		result[i] = sum / float64(i+1)
	}

	// Sliding window for the rest
	for i := windowSize; i < len(data); i++ {
		sum := 0.0
		for j := i - windowSize + 1; j <= i; j++ {
			sum += data[j]
		}
		result[i] = sum / float64(windowSize)
	}

	return result
}

// MedianFilter applies median filtering with given window size
func MedianFilter(data []float64, windowSize int) []float64 {
	if len(data) == 0 || windowSize <= 0 {
		return data
	}

	if windowSize > len(data) {
		windowSize = len(data)
	}

	result := make([]float64, len(data))
	halfWindow := windowSize / 2

	for i := range data {
		// Define window bounds
		start := i - halfWindow
		end := i + halfWindow + 1

		if start < 0 {
			start = 0
		}
		if end > len(data) {
			end = len(data)
		}

		// Extract window and find median
		window := make([]float64, end-start)
		copy(window, data[start:end])
		sort.Float64s(window)

		// Get median
		mid := len(window) / 2
		if len(window)%2 == 0 {
			result[i] = (window[mid-1] + window[mid]) / 2.0
		} else {
			result[i] = window[mid]
		}
	}

	return result
}

// Correlation calculates Pearson correlation coefficient between two series
func Correlation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	return stat.Correlation(x, y, nil)
}

// Covariance calculates sample covariance between two series
func Covariance(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	meanX := Mean(x)
	meanY := Mean(y)

	sum := 0.0
	for i := range x {
		sum += (x[i] - meanX) * (y[i] - meanY)
	}

	return sum / float64(len(x)-1)
}

// LinRegression performs simple linear regression and returns slope, intercept, r²
func LinRegression(x, y []float64) (slope, intercept, rSquared float64) {
	if len(x) != len(y) || len(x) < 2 {
		return 0, 0, 0
	}

	// Use gonum's linear regression
	alpha, beta := stat.LinearRegression(x, y, nil, false)

	// Calculate R²
	yMean := Mean(y)
	ssTotal := 0.0
	ssResidual := 0.0

	for i := range x {
		predicted := alpha + beta*x[i]
		ssTotal += (y[i] - yMean) * (y[i] - yMean)
		ssResidual += (y[i] - predicted) * (y[i] - predicted)
	}

	rSquared = 1.0 - (ssResidual / ssTotal)
	if math.IsNaN(rSquared) || math.IsInf(rSquared, 0) {
		rSquared = 0.0
	}

	return beta, alpha, rSquared
}

// FindPeaks finds local maxima in data
func FindPeaks(data []float64, minHeight, minDistance float64) []int {
	if len(data) < 3 {
		return []int{}
	}

	var peaks []int

	for i := 1; i < len(data)-1; i++ {
		// Check if it's a local maximum
		if data[i] > data[i-1] && data[i] > data[i+1] && data[i] >= minHeight {
			// Check minimum distance constraint
			validPeak := true
			for _, existingPeak := range peaks {
				if math.Abs(float64(i-existingPeak)) < minDistance {
					// If new peak is higher, replace the old one
					if data[i] > data[existingPeak] {
						// Remove the old peak
						for j, peak := range peaks {
							if peak == existingPeak {
								peaks = append(peaks[:j], peaks[j+1:]...)
								break
							}
						}
					} else {
						validPeak = false
					}
					break
				}
			}

			if validPeak {
				peaks = append(peaks, i)
			}
		}
	}

	return peaks
}

// Interpolate performs linear interpolation
func Interpolate(x, y []float64, xi float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0.0
	}

	// Find the interval
	if xi <= x[0] {
		return y[0]
	}
	if xi >= x[len(x)-1] {
		return y[len(y)-1]
	}

	// Binary search for the interval
	left := 0
	right := len(x) - 1

	for right-left > 1 {
		mid := (left + right) / 2
		if x[mid] <= xi {
			left = mid
		} else {
			right = mid
		}
	}

	// Linear interpolation
	t := (xi - x[left]) / (x[right] - x[left])
	return y[left] + t*(y[right]-y[left])
}

// Clamp constrains a value to a range
func Clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

// Lerp performs linear interpolation between two values
func Lerp(a, b, t float64) float64 {
	return a + t*(b-a)
}

// IsPowerOfTwo checks if n is a power of 2
func IsPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// NextPowerOfTwo finds the next power of 2 >= n
func NextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}

	power := 1
	for power < n {
		power <<= 1
	}
	return power
}
