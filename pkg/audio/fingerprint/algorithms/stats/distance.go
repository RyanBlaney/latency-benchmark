package stats

import (
	"math"
)

// DistanceFunction is a function type for computing distance between two vectors
type DistanceFunction func(a, b []float64) float64

// GetDistanceFunction returns the appropriate distance function for the given metric
func GetDistanceFunction(metric DistanceMetric) DistanceFunction {
	switch metric {
	case EuclideanDistance:
		return EuclideanDistanceFunc
	case ManhattanDistance:
		return ManhattanDistanceFunc
	case CosineDistance:
		return CosineDistanceFunc
	case PearsonDistance:
		return PearsonDistanceFunc
	case MahalanobisDistance:
		return MahalanobisDistanceFunc
	default:
		return EuclideanDistanceFunc
	}
}

// EuclideanDistanceFunc calculates Euclidean distance between two points
func EuclideanDistanceFunc(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// ManhattanDistanceFunc calculates Manhattan (L1) distance between two points
func ManhattanDistanceFunc(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

// CosineDistanceFunc calculates cosine distance (1 - cosine similarity)
func CosineDistanceFunc(a, b []float64) float64 {
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - similarity
}

// CosineSimilarityFunc calculates cosine similarity between two vectors
func CosineSimilarityFunc(a, b []float64) float64 {
	return 1.0 - CosineDistanceFunc(a, b)
}

// PearsonDistanceFunc calculates Pearson correlation distance (1 - |correlation|)
func PearsonDistanceFunc(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 1.0
	}

	// Calculate means
	meanA := 0.0
	meanB := 0.0
	for i := range a {
		meanA += a[i]
		meanB += b[i]
	}
	meanA /= float64(n)
	meanB /= float64(n)

	// Calculate correlation coefficient
	numerator := 0.0
	sumSqA := 0.0
	sumSqB := 0.0

	for i := range a {
		diffA := a[i] - meanA
		diffB := b[i] - meanB
		numerator += diffA * diffB
		sumSqA += diffA * diffA
		sumSqB += diffB * diffB
	}

	if sumSqA == 0 || sumSqB == 0 {
		return 1.0
	}

	correlation := numerator / math.Sqrt(sumSqA*sumSqB)
	return 1.0 - math.Abs(correlation)
}

// PearsonCorrelationFunc calculates Pearson correlation coefficient
func PearsonCorrelationFunc(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0.0
	}

	// Calculate means
	meanA := 0.0
	meanB := 0.0
	for i := range a {
		meanA += a[i]
		meanB += b[i]
	}
	meanA /= float64(n)
	meanB /= float64(n)

	// Calculate correlation coefficient
	numerator := 0.0
	sumSqA := 0.0
	sumSqB := 0.0

	for i := range a {
		diffA := a[i] - meanA
		diffB := b[i] - meanB
		numerator += diffA * diffB
		sumSqA += diffA * diffA
		sumSqB += diffB * diffB
	}

	if sumSqA == 0 || sumSqB == 0 {
		return 0.0
	}

	return numerator / math.Sqrt(sumSqA*sumSqB)
}

// MahalanobisDistanceFunc calculates Mahalanobis distance
// Note: This is a simplified version that assumes identity covariance matrix
// For full implementation, pass covariance matrix as additional parameter
func MahalanobisDistanceFunc(a, b []float64) float64 {
	// Simplified implementation - same as Euclidean for identity covariance
	return EuclideanDistanceFunc(a, b)
}

// ChebyshevDistanceFunc calculates Chebyshev distance (L∞ norm)
func ChebyshevDistanceFunc(a, b []float64) float64 {
	maxDiff := 0.0
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

// MinkowskiDistanceFunc calculates Minkowski distance with parameter p
func MinkowskiDistanceFunc(a, b []float64, p float64) float64 {
	if p == 1.0 {
		return ManhattanDistanceFunc(a, b)
	}
	if p == 2.0 {
		return EuclideanDistanceFunc(a, b)
	}
	if math.IsInf(p, 1) {
		return ChebyshevDistanceFunc(a, b)
	}

	sum := 0.0
	for i := range a {
		sum += math.Pow(math.Abs(a[i]-b[i]), p)
	}
	return math.Pow(sum, 1.0/p)
}

// HammingDistanceFunc calculates Hamming distance (for binary vectors)
func HammingDistanceFunc(a, b []float64) float64 {
	count := 0.0
	for i := range a {
		if a[i] != b[i] {
			count++
		}
	}
	return count
}

// JaccardDistanceFunc calculates Jaccard distance (1 - Jaccard similarity)
func JaccardDistanceFunc(a, b []float64) float64 {
	intersection := 0.0
	union := 0.0

	for i := range a {
		if a[i] > 0 || b[i] > 0 {
			union++
			if a[i] > 0 && b[i] > 0 {
				intersection++
			}
		}
	}

	if union == 0 {
		return 0.0
	}

	return 1.0 - (intersection / union)
}

// CanberraDistanceFunc calculates Canberra distance
func CanberraDistanceFunc(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		denominator := math.Abs(a[i]) + math.Abs(b[i])
		if denominator > 0 {
			sum += math.Abs(a[i]-b[i]) / denominator
		}
	}
	return sum
}

// BrayCurtisDistanceFunc calculates Bray-Curtis distance
func BrayCurtisDistanceFunc(a, b []float64) float64 {
	numerator := 0.0
	denominator := 0.0

	for i := range a {
		numerator += math.Abs(a[i] - b[i])
		denominator += a[i] + b[i]
	}

	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

// KLDivergenceFunc calculates Kullback-Leibler divergence (requires probability distributions)
func KLDivergenceFunc(p, q []float64) float64 {
	// Normalize to ensure probability distributions
	pNorm := normalizeToProbability(p)
	qNorm := normalizeToProbability(q)

	kl := 0.0
	for i := range pNorm {
		if pNorm[i] > 0 && qNorm[i] > 0 {
			kl += pNorm[i] * math.Log(pNorm[i]/qNorm[i])
		}
	}

	return kl
}

// JensenShannonDistanceFunc calculates Jensen-Shannon distance
func JensenShannonDistanceFunc(p, q []float64) float64 {
	// Normalize to probability distributions
	pNorm := normalizeToProbability(p)
	qNorm := normalizeToProbability(q)

	// Compute average distribution
	m := make([]float64, len(pNorm))
	for i := range pNorm {
		m[i] = (pNorm[i] + qNorm[i]) / 2.0
	}

	// Compute JS divergence
	js := 0.5*KLDivergenceFunc(pNorm, m) + 0.5*KLDivergenceFunc(qNorm, m)

	// Convert to distance
	return math.Sqrt(js)
}

// HellingerDistanceFunc calculates Hellinger distance
func HellingerDistanceFunc(p, q []float64) float64 {
	pNorm := normalizeToProbability(p)
	qNorm := normalizeToProbability(q)

	sum := 0.0
	for i := range pNorm {
		diff := math.Sqrt(pNorm[i]) - math.Sqrt(qNorm[i])
		sum += diff * diff
	}

	return math.Sqrt(sum) / math.Sqrt(2)
}

// BhattacharyyaDistanceFunc calculates Bhattacharyya distance
func BhattacharyyaDistanceFunc(p, q []float64) float64 {
	pNorm := normalizeToProbability(p)
	qNorm := normalizeToProbability(q)

	bc := 0.0 // Bhattacharyya coefficient
	for i := range pNorm {
		bc += math.Sqrt(pNorm[i] * qNorm[i])
	}

	if bc <= 0 {
		return math.Inf(1)
	}

	return -math.Log(bc)
}

// EarthMoversDistanceFunc calculates Earth Mover's Distance (Wasserstein distance)
// This is a simplified 1D implementation
func EarthMoversDistanceFunc(a, b []float64) float64 {
	// Normalize to probability distributions
	aNorm := normalizeToProbability(a)
	bNorm := normalizeToProbability(b)

	// Compute cumulative distributions
	aCum := make([]float64, len(aNorm))
	bCum := make([]float64, len(bNorm))

	aCum[0] = aNorm[0]
	bCum[0] = bNorm[0]

	for i := 1; i < len(aNorm); i++ {
		aCum[i] = aCum[i-1] + aNorm[i]
		bCum[i] = bCum[i-1] + bNorm[i]
	}

	// Compute EMD as sum of absolute differences between CDFs
	emd := 0.0
	for i := range aCum {
		emd += math.Abs(aCum[i] - bCum[i])
	}

	return emd
}

// Helper function to normalize a vector to a probability distribution
func normalizeToProbability(values []float64) []float64 {
	sum := 0.0
	for _, val := range values {
		if val > 0 { // Only sum positive values
			sum += val
		}
	}

	if sum == 0 {
		// Return uniform distribution if all values are zero/negative
		prob := make([]float64, len(values))
		uniformVal := 1.0 / float64(len(values))
		for i := range prob {
			prob[i] = uniformVal
		}
		return prob
	}

	prob := make([]float64, len(values))
	for i, val := range values {
		if val > 0 {
			prob[i] = val / sum
		}
		// Leave negative values as 0
	}

	return prob
}

// Additional utility functions

// DistanceMatrix computes pairwise distances between all vectors
func DistanceMatrix(data [][]float64, metric DistanceMetric) [][]float64 {
	n := len(data)
	matrix := make([][]float64, n)
	distFunc := GetDistanceFunction(metric)

	for i := range n {
		matrix[i] = make([]float64, n)
		for j := range n {
			if i == j {
				matrix[i][j] = 0.0
			} else if j > i {
				// Compute distance only for upper triangle
				matrix[i][j] = distFunc(data[i], data[j])
			} else {
				// Copy from upper triangle (symmetric)
				matrix[i][j] = matrix[j][i]
			}
		}
	}

	return matrix
}

// NearestNeighbors finds k nearest neighbors for a query point
func NearestNeighbors(query []float64, data [][]float64, k int, metric DistanceMetric) []int {
	if k <= 0 || k > len(data) {
		return []int{}
	}

	distFunc := GetDistanceFunction(metric)
	type neighbor struct {
		index    int
		distance float64
	}

	neighbors := make([]neighbor, len(data))
	for i, point := range data {
		neighbors[i] = neighbor{
			index:    i,
			distance: distFunc(query, point),
		}
	}

	// Sort by distance
	for i := 0; i < len(neighbors)-1; i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[i].distance > neighbors[j].distance {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	// Return indices of k nearest neighbors
	result := make([]int, k)
	for i := range k {
		result[i] = neighbors[i].index
	}

	return result
}

// IsValidDistance checks if two vectors can be compared with the given metric
func IsValidDistance(a, b []float64, metric DistanceMetric) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}

	// Check for probability distribution metrics
	switch metric {
	case EuclideanDistance, ManhattanDistance, CosineDistance, PearsonDistance:
		return true
	default:
		return true
	}
}

// GetDistanceMetricName returns human-readable name for distance metric
func GetDistanceMetricName(metric DistanceMetric) string {
	switch metric {
	case EuclideanDistance:
		return "Euclidean"
	case ManhattanDistance:
		return "Manhattan"
	case CosineDistance:
		return "Cosine"
	case PearsonDistance:
		return "Pearson"
	case MahalanobisDistance:
		return "Mahalanobis"
	default:
		return "Unknown"
	}
}
