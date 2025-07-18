package chroma

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// ChromaVector represents a chroma feature vector with utilities
type ChromaVector struct {
	Values     []float64 `json:"values"`     // Chroma values (typically 12 elements)
	Size       int       `json:"size"`       // Vector size (12, 24, or 36)
	Normalized bool      `json:"normalized"` // Whether vector is normalized
	Energy     float64   `json:"energy"`     // Total energy
	Centroid   float64   `json:"centroid"`   // Spectral centroid in chroma space
	Entropy    float64   `json:"entropy"`    // Entropy of distribution
}

// ChromaVectorStats contains statistics about a chroma vector
type ChromaVectorStats struct {
	Mean           float64 `json:"mean"`
	Variance       float64 `json:"variance"`
	StdDev         float64 `json:"std_dev"`
	Skewness       float64 `json:"skewness"`
	Kurtosis       float64 `json:"kurtosis"`
	Range          float64 `json:"range"`
	MaxValue       float64 `json:"max_value"`
	MinValue       float64 `json:"min_value"`
	MaxIndex       int     `json:"max_index"`       // Index of maximum value
	DominantChroma int     `json:"dominant_chroma"` // Strongest chroma class
	NumPeaks       int     `json:"num_peaks"`       // Number of local peaks
	Sparsity       float64 `json:"sparsity"`        // Measure of sparsity (0-1)
	Uniformity     float64 `json:"uniformity"`      // How uniform the distribution is
}

// ChromaDistanceMetric defines distance/similarity metrics for chroma vectors
type ChromaDistanceMetric int

const (
	ChromaDistanceEuclidean ChromaDistanceMetric = iota
	ChromaDistanceManhattan
	ChromaDistanceCosine
	ChromaDistancePearson
	// Chroma-specific distances
	ChromaDistanceKLDivergence
	ChromaDistanceJSDistance
	ChromaDistanceHellinger
)

// ChromaVectorAnalyzer provides utilities for chroma vector analysis
type ChromaVectorAnalyzer struct {
	normalizer *common.Normalizer
	moments    *stats.Moments
}

// NewChromaVectorAnalyzer creates a new chroma vector analyzer
func NewChromaVectorAnalyzer() *ChromaVectorAnalyzer {
	return &ChromaVectorAnalyzer{
		normalizer: common.NewNormalizer(common.Energy),
		moments:    stats.NewMoments(),
	}
}

// CreateChromaVector creates a ChromaVector from values
func (cva *ChromaVectorAnalyzer) CreateChromaVector(values []float64) ChromaVector {
	cv := ChromaVector{
		Values:     make([]float64, len(values)),
		Size:       len(values),
		Normalized: false,
	}

	copy(cv.Values, values)
	cv.Energy = cva.computeEnergy(cv.Values)
	cv.Centroid = cva.computeCentroid(cv.Values)
	cv.Entropy = cva.computeEntropy(cv.Values)

	return cv
}

// Normalize normalizes a chroma vector using specified method
func (cva *ChromaVectorAnalyzer) Normalize(cv ChromaVector, method common.NormalizationType) ChromaVector {
	normalizer := common.NewNormalizer(method)
	normalized := normalizer.Normalize(cv.Values)

	result := cv
	result.Values = normalized
	result.Normalized = true
	result.Energy = cva.computeEnergy(normalized)

	return result
}

// ComputeStats computes comprehensive statistics for a chroma vector
func (cva *ChromaVectorAnalyzer) ComputeStats(cv ChromaVector) ChromaVectorStats {
	values := cv.Values

	// Use existing moments computation
	momentResult, err := cva.moments.Analyze(values)
	if err != nil {
		// TODO: handle properly
		fmt.Printf("failed to analyze moments in chroma vector analyzer")
	}

	// Find min/max
	minVal, maxVal := values[0], values[0]
	maxIndex := 0
	for i, val := range values {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
			maxIndex = i
		}
	}

	// Count peaks using existing peak detection
	peaks := common.FindPeaks(values, maxVal*0.1, 1.0) // 10% of max as threshold

	// Compute sparsity (ratio of zero/near-zero elements)
	sparsity := cva.computeSparsity(values)

	// Compute uniformity (how evenly distributed the energy is)
	uniformity := cva.computeUniformity(values)

	return ChromaVectorStats{
		Mean:           momentResult.Mean,
		Variance:       momentResult.Variance,
		StdDev:         momentResult.StdDev,
		Skewness:       momentResult.Skewness,
		Kurtosis:       momentResult.Kurtosis,
		Range:          maxVal - minVal,
		MaxValue:       maxVal,
		MinValue:       minVal,
		MaxIndex:       maxIndex,
		DominantChroma: maxIndex,
		NumPeaks:       len(peaks),
		Sparsity:       sparsity,
		Uniformity:     uniformity,
	}
}

// Distance computes distance between two chroma vectors
func (cva *ChromaVectorAnalyzer) Distance(cv1, cv2 ChromaVector, metric ChromaDistanceMetric) float64 {
	if len(cv1.Values) != len(cv2.Values) {
		return math.Inf(1) // Invalid comparison
	}

	switch metric {
	case ChromaDistanceEuclidean:
		return stats.EuclideanDistanceFunc(cv1.Values, cv2.Values)
	case ChromaDistanceCosine:
		return stats.CosineDistanceFunc(cv1.Values, cv2.Values)
	case ChromaDistanceManhattan:
		return stats.ManhattanDistanceFunc(cv1.Values, cv2.Values)
	case ChromaDistancePearson:
		return stats.PearsonDistanceFunc(cv1.Values, cv2.Values)
	case ChromaDistanceKLDivergence:
		return stats.KLDivergenceFunc(cv1.Values, cv2.Values)
	case ChromaDistanceJSDistance:
		return stats.JensenShannonDistanceFunc(cv1.Values, cv2.Values)
	case ChromaDistanceHellinger:
		return stats.HellingerDistanceFunc(cv1.Values, cv2.Values)
	default:
		return stats.EuclideanDistanceFunc(cv1.Values, cv2.Values)
	}
}

// Similarity computes similarity between two chroma vectors (0-1, higher = more similar)
func (cva *ChromaVectorAnalyzer) Similarity(cv1, cv2 ChromaVector, metric ChromaDistanceMetric) float64 {
	distance := cva.Distance(cv1, cv2, metric)

	switch metric {
	case ChromaDistanceCosine:
		// Cosine distance is already 0-2, convert to similarity
		return 1.0 - (distance / 2.0)
	case ChromaDistancePearson:
		// Correlation distance is 0-2, convert to similarity
		return 1.0 - (distance / 2.0)
	default:
		// For other metrics, use exponential decay
		return math.Exp(-distance)
	}
}

// ShiftOptimal finds optimal circular shift to maximize similarity
func (cva *ChromaVectorAnalyzer) ShiftOptimal(cv1, cv2 ChromaVector, metric ChromaDistanceMetric) (int, float64) {
	bestShift := 0
	bestSimilarity := 0.0

	for shift := 0; shift < cv1.Size; shift++ {
		shifted := cva.CircularShift(cv1, shift)
		similarity := cva.Similarity(shifted, cv2, metric)

		if similarity > bestSimilarity {
			bestSimilarity = similarity
			bestShift = shift
		}
	}

	return bestShift, bestSimilarity
}

// CircularShift performs circular shift of chroma vector
func (cva *ChromaVectorAnalyzer) CircularShift(cv ChromaVector, shift int) ChromaVector {
	shifted := cv
	shifted.Values = make([]float64, len(cv.Values))

	for i := 0; i < len(cv.Values); i++ {
		shifted.Values[i] = cv.Values[(i+shift)%len(cv.Values)]
	}

	return shifted
}

// Interpolate interpolates between two chroma vectors
func (cva *ChromaVectorAnalyzer) Interpolate(cv1, cv2 ChromaVector, t float64) ChromaVector {
	if len(cv1.Values) != len(cv2.Values) {
		return cv1 // Return first vector if sizes don't match
	}

	// Clamp t to [0, 1]
	t = common.Clamp(t, 0.0, 1.0)

	result := ChromaVector{
		Values: make([]float64, len(cv1.Values)),
		Size:   cv1.Size,
	}

	// Linear interpolation using existing Lerp function
	for i := 0; i < len(cv1.Values); i++ {
		result.Values[i] = common.Lerp(cv1.Values[i], cv2.Values[i], t)
	}

	result.Energy = cva.computeEnergy(result.Values)
	result.Centroid = cva.computeCentroid(result.Values)
	result.Entropy = cva.computeEntropy(result.Values)

	return result
}

// Smooth applies smoothing to a sequence of chroma vectors
func (cva *ChromaVectorAnalyzer) Smooth(vectors []ChromaVector, windowSize int) []ChromaVector {
	if len(vectors) == 0 || windowSize <= 1 {
		return vectors
	}

	smoothed := make([]ChromaVector, len(vectors))

	for i := 0; i < len(vectors); i++ {
		// Determine window bounds
		start := common.Clamp(float64(i-windowSize/2), 0, float64(len(vectors)-1))
		end := common.Clamp(float64(i+windowSize/2), 0, float64(len(vectors)-1))

		// Compute average over window
		avgValues := make([]float64, vectors[i].Size)
		count := 0

		for j := int(start); j <= int(end); j++ {
			for k := 0; k < vectors[j].Size; k++ {
				avgValues[k] += vectors[j].Values[k]
			}
			count++
		}

		// Normalize by count
		for k := 0; k < len(avgValues); k++ {
			avgValues[k] /= float64(count)
		}

		smoothed[i] = cva.CreateChromaVector(avgValues)
	}

	return smoothed
}

// FindDominantChroma finds the dominant chroma class
func (cva *ChromaVectorAnalyzer) FindDominantChroma(cv ChromaVector) (int, float64) {
	maxVal := 0.0
	maxIdx := 0

	for i, val := range cv.Values {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx, maxVal
}

// ComputeChromaTemplate computes a template from multiple chroma vectors
func (cva *ChromaVectorAnalyzer) ComputeChromaTemplate(vectors []ChromaVector) ChromaVector {
	if len(vectors) == 0 {
		return ChromaVector{}
	}

	size := vectors[0].Size
	template := make([]float64, size)

	// Compute element-wise mean
	for _, cv := range vectors {
		for i := 0; i < size && i < len(cv.Values); i++ {
			template[i] += cv.Values[i]
		}
	}

	// Normalize by count
	for i := 0; i < size; i++ {
		template[i] /= float64(len(vectors))
	}

	return cva.CreateChromaVector(template)
}

// Helper functions

func (cva *ChromaVectorAnalyzer) computeEnergy(values []float64) float64 {
	energy := 0.0
	for _, val := range values {
		energy += val * val
	}
	return math.Sqrt(energy)
}

func (cva *ChromaVectorAnalyzer) computeCentroid(values []float64) float64 {
	numerator := 0.0
	denominator := 0.0

	for i, val := range values {
		numerator += float64(i) * val
		denominator += val
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

func (cva *ChromaVectorAnalyzer) computeEntropy(values []float64) float64 {
	// Normalize to probability distribution
	sum := 0.0
	for _, val := range values {
		sum += val
	}

	if sum == 0 {
		return 0
	}

	entropy := 0.0
	for _, val := range values {
		if val > 0 {
			prob := val / sum
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

func (cva *ChromaVectorAnalyzer) computeSparsity(values []float64) float64 {
	threshold := 0.01 // 1% of max
	maxVal := 0.0
	for _, val := range values {
		if val > maxVal {
			maxVal = val
		}
	}

	zeroCount := 0
	for _, val := range values {
		if val < threshold*maxVal {
			zeroCount++
		}
	}

	return float64(zeroCount) / float64(len(values))
}

func (cva *ChromaVectorAnalyzer) computeUniformity(values []float64) float64 {
	// Compute how close the distribution is to uniform
	sum := 0.0
	for _, val := range values {
		sum += val
	}

	if sum == 0 {
		return 1.0 // Perfectly uniform (all zeros)
	}

	expectedVal := sum / float64(len(values))
	variance := 0.0

	for _, val := range values {
		diff := val - expectedVal
		variance += diff * diff
	}

	variance /= float64(len(values))

	// Convert variance to uniformity measure (lower variance = higher uniformity)
	return 1.0 / (1.0 + variance)
}

func (cva *ChromaVectorAnalyzer) correlationDistance(v1, v2 []float64) float64 {
	// Use existing correlation function from common package
	corr := common.Correlation(v1, v2)
	return 1.0 - corr
}

func (cva *ChromaVectorAnalyzer) klDivergence(v1, v2 []float64) float64 {
	// Normalize to probability distributions
	p := cva.normalizeToProbability(v1)
	q := cva.normalizeToProbability(v2)

	kl := 0.0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 && q[i] > 0 {
			kl += p[i] * math.Log(p[i]/q[i])
		}
	}

	return kl
}

func (cva *ChromaVectorAnalyzer) jensenShannonDistance(v1, v2 []float64) float64 {
	// JS distance is symmetric version of KL divergence
	p := cva.normalizeToProbability(v1)
	q := cva.normalizeToProbability(v2)

	// Compute average distribution
	m := make([]float64, len(p))
	for i := 0; i < len(p); i++ {
		m[i] = (p[i] + q[i]) / 2.0
	}

	// Compute JS divergence
	js := 0.5*cva.klDivergence(p, m) + 0.5*cva.klDivergence(q, m)

	// Convert to distance
	return math.Sqrt(js)
}

func (cva *ChromaVectorAnalyzer) hellingerDistance(v1, v2 []float64) float64 {
	p := cva.normalizeToProbability(v1)
	q := cva.normalizeToProbability(v2)

	sum := 0.0
	for i := 0; i < len(p); i++ {
		diff := math.Sqrt(p[i]) - math.Sqrt(q[i])
		sum += diff * diff
	}

	return math.Sqrt(sum) / math.Sqrt(2)
}

func (cva *ChromaVectorAnalyzer) normalizeToProbability(values []float64) []float64 {
	sum := 0.0
	for _, val := range values {
		sum += val
	}

	if sum == 0 {
		// Return uniform distribution
		prob := make([]float64, len(values))
		for i := range prob {
			prob[i] = 1.0 / float64(len(values))
		}
		return prob
	}

	prob := make([]float64, len(values))
	for i, val := range values {
		prob[i] = val / sum
	}

	return prob
}
