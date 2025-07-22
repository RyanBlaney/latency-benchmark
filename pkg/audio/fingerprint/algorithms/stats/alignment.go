package stats

import (
	"fmt"
	"math"
	"sort"
)

// AlignmentMethod defines different alignment approaches
type AlignmentMethod int

const (
	AlignmentDTW AlignmentMethod = iota
	AlignmentCrossCorrelation
	AlignmentPhaseCorrelation
	AlignmentHybrid
)

// AlignmentAnalyzer provides comprehensive audio alignment capabilities
// WHY: Audio alignment is crucial for fingerprint matching, synchronization,
// and determining similarity between audio segments of different lengths
type AlignmentAnalyzer struct {
	method           AlignmentMethod
	maxLag           int
	dtwAnalyzer      *DTWAlignment
	crossCorr        *CrossCorrelation
	stepSize         int
	hopSize          int
	windowSize       int
	confidenceThresh float64
}

// AlignmentResult contains comprehensive alignment analysis
type AlignmentResult struct {
	// Primary alignment information
	Method        AlignmentMethod `json:"method"`
	Offset        int             `json:"offset"`         // Sample offset (negative = query is delayed)
	OffsetSeconds float64         `json:"offset_seconds"` // Offset in seconds
	Confidence    float64         `json:"confidence"`     // Alignment confidence (0-1)
	Similarity    float64         `json:"similarity"`     // Overall similarity score

	// DTW-specific results
	DTWResult *DTWResult `json:"dtw_result,omitempty"`

	// Cross-correlation results
	CrossCorrResult *CorrelationResult `json:"cross_corr_result,omitempty"`

	// Quality metrics
	AlignmentQuality float64 `json:"alignment_quality"` // Quality of alignment
	NoiseLevel       float64 `json:"noise_level"`       // Estimated noise level
	Stability        float64 `json:"stability"`         // Stability of alignment

	// Analysis metadata
	QueryLength     int     `json:"query_length"`
	ReferenceLength int     `json:"reference_length"`
	ProcessingTime  float64 `json:"processing_time"` // Processing time (ms)
	SampleRate      int     `json:"sample_rate"`
}

func NewAlignmentAnalyzer(method AlignmentMethod, maxLag, sampleRate, hopSize, windowSize int, confidenceThresh float64) *AlignmentAnalyzer {
	// Create cross-correlation with explicit normalized correlation
	crossCorr := NewCrossCorrelationWithParams(
		maxLag,
		NormalizedCrossCorrelation, // Use this instead of Pearson for raw features
		TimeDomain,
	)

	// Ensure normalization is enabled
	crossCorr.normalizeInputs = true

	return &AlignmentAnalyzer{
		method:           method,
		maxLag:           maxLag,
		dtwAnalyzer:      NewDTWAlignment(),
		crossCorr:        crossCorr,
		stepSize:         1, // TODO: get from config
		hopSize:          hopSize,
		windowSize:       windowSize,
		confidenceThresh: confidenceThresh,
	}
}

// AlignFeatures aligns two feature sequences (e.g., MFCC, chroma)
func (aa *AlignmentAnalyzer) AlignFeatures(query, reference [][]float64, sampleRate int) (*AlignmentResult, error) {
	if len(query) == 0 || len(reference) == 0 {
		return nil, fmt.Errorf("empty feature sequences provided")
	}

	result := &AlignmentResult{
		Method:          aa.method,
		QueryLength:     len(query),
		ReferenceLength: len(reference),
		SampleRate:      sampleRate,
	}

	switch aa.method {
	case AlignmentDTW:
		return aa.alignWithDTW(query, reference, result)
	case AlignmentCrossCorrelation:
		return aa.alignWithCrossCorrelation(query, reference, result)
	case AlignmentHybrid:
		return aa.alignWithHybrid(query, reference, result)
	default:
		return nil, fmt.Errorf("unsupported alignment method: %d", aa.method)
	}
}

// AlignAudio aligns two audio signals using energy-based features
func (aa *AlignmentAnalyzer) AlignAudio(queryPCM, referencePCM []float64, sampleRate int) (*AlignmentResult, error) {
	// Convert audio to energy features for alignment
	queryFeatures := aa.extractEnergyFeatures(queryPCM, sampleRate)
	refFeatures := aa.extractEnergyFeatures(referencePCM, sampleRate)

	// Convert 1D features to 2D for compatibility
	query2D := make([][]float64, len(queryFeatures))
	ref2D := make([][]float64, len(refFeatures))

	for i, v := range queryFeatures {
		query2D[i] = []float64{v}
	}
	for i, v := range refFeatures {
		ref2D[i] = []float64{v}
	}

	return aa.AlignFeatures(query2D, ref2D, sampleRate)
}

// alignWithDTW performs DTW-based alignment
func (aa *AlignmentAnalyzer) alignWithDTW(query, reference [][]float64, result *AlignmentResult) (*AlignmentResult, error) {
	dtwResult, err := aa.dtwAnalyzer.Align(query, reference)
	if err != nil {
		return nil, fmt.Errorf("DTW alignment failed: %w", err)
	}

	result.DTWResult = dtwResult
	result.Similarity = aa.calculateSimilarityFromDTW(dtwResult)
	result.Confidence = aa.calculateDTWConfidence(dtwResult)

	// Calculate average offset from DTW path
	result.Offset = aa.calculateAverageOffset(dtwResult.Path)
	result.OffsetSeconds = float64(result.Offset) / float64(result.SampleRate)

	// Calculate quality metrics
	result.AlignmentQuality = aa.calculateDTWQuality(dtwResult)
	result.Stability = aa.calculatePathStability(dtwResult.Path)

	return result, nil
}

// alignWithCrossCorrelation performs cross-correlation based alignment
func (aa *AlignmentAnalyzer) alignWithCrossCorrelation(query, reference [][]float64, result *AlignmentResult) (*AlignmentResult, error) {
	queryVec := aa.flatten2DFeatures(query)
	refVec := aa.flatten2DFeatures(reference)

	corrResult, err := aa.crossCorr.Compute(queryVec, refVec)
	if err != nil {
		return nil, fmt.Errorf("cross-correlation failed: %w", err)
	}

	fmt.Printf("DEBUG ALIGNMENT:\n")
	fmt.Printf("  Raw peak lag (frames): %d\n", corrResult.PeakLag)
	fmt.Printf("  Hop size: %d samples\n", aa.hopSize)
	fmt.Printf("  Sample rate: %d Hz\n", result.SampleRate)
	fmt.Printf("  Frame offset in samples: %d\n", corrResult.PeakLag*aa.hopSize)
	fmt.Printf("  Frame offset in seconds: %.3f\n", float64(corrResult.PeakLag*aa.hopSize)/float64(result.SampleRate))

	result.CrossCorrResult = corrResult

	// Convert frame lag back to sample lag
	hopSize := aa.hopSize
	result.Offset = corrResult.PeakLag * hopSize // Convert frames to samples
	result.OffsetSeconds = float64(result.Offset) / float64(result.SampleRate)

	// Fix floating-point precision issues
	similarity := corrResult.PeakCorrelation
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}
	result.Similarity = similarity

	result.Confidence = aa.calculateCorrelationConfidence(corrResult)
	result.AlignmentQuality = corrResult.Sharpness
	result.NoiseLevel = 1.0 - corrResult.SNR/20.0

	fmt.Printf("DEBUG CORRELATION:\n")
	fmt.Printf("  Peak correlation: %.3f\n", corrResult.PeakCorrelation)
	fmt.Printf("  SNR: %.3f\n", corrResult.SNR)
	fmt.Printf("  Sharpness: %.3f\n", corrResult.Sharpness)
	fmt.Printf("  Calculated confidence: %.3f\n", aa.calculateCorrelationConfidence(corrResult))

	return result, nil
}

// alignWithHybrid combines DTW and cross-correlation
func (aa *AlignmentAnalyzer) alignWithHybrid(query, reference [][]float64, result *AlignmentResult) (*AlignmentResult, error) {
	// First, use cross-correlation for coarse alignment
	corrResult, err := aa.alignWithCrossCorrelation(query, reference, result)
	if err != nil {
		return nil, err
	}

	// If cross-correlation confidence is high, use it
	if corrResult.Confidence > 0.7 {
		return corrResult, nil
	}

	// Otherwise, use DTW for fine alignment
	dtwResult, err := aa.alignWithDTW(query, reference, result)
	if err != nil {
		// Fall back to cross-correlation result
		return corrResult, nil
	}

	// Combine results - prefer DTW path but use correlation metrics
	result.Method = AlignmentHybrid
	result.DTWResult = dtwResult.DTWResult
	result.CrossCorrResult = corrResult.CrossCorrResult

	// Weighted combination of confidences
	result.Confidence = 0.6*dtwResult.Confidence + 0.4*corrResult.Confidence
	result.Similarity = 0.7*dtwResult.Similarity + 0.3*corrResult.Similarity

	return result, nil
}

// Helper methods

func (aa *AlignmentAnalyzer) extractEnergyFeatures(pcm []float64, sampleRate int) []float64 {
	frameSize := aa.windowSize
	hopSize := aa.hopSize
	numFrames := (len(pcm)-frameSize)/hopSize + 1

	energy := make([]float64, numFrames)
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		// Calculate RMS energy
		sum := 0.0
		for j := start; j < end; j++ {
			sum += pcm[j] * pcm[j]
		}
		energy[i] = math.Sqrt(sum / float64(end-start))
	}

	return energy
}

func (aa *AlignmentAnalyzer) flatten2DFeatures(features [][]float64) []float64 {
	if len(features) == 0 {
		return []float64{}
	}

	result := make([]float64, len(features))

	// Option A: Use first component (often most important)
	for i, frame := range features {
		if len(frame) > 0 {
			result[i] = frame[0]
		}
	}

	return result
}

func (aa *AlignmentAnalyzer) calculateSimilarityFromDTW(dtwResult *DTWResult) float64 {
	// Convert DTW distance to similarity (lower distance = higher similarity)
	// This is a heuristic transformation
	maxDistance := math.Sqrt(float64(dtwResult.QueryLength + dtwResult.RefLength))
	normalizedDist := dtwResult.Distance / maxDistance
	return math.Max(0, 1.0-normalizedDist)
}

func (aa *AlignmentAnalyzer) calculateDTWConfidence(dtwResult *DTWResult) float64 {
	// Confidence based on path characteristics and cost distribution
	if len(dtwResult.Path) == 0 {
		return 0.0
	}

	// Calculate cost variance along path
	costs := make([]float64, len(dtwResult.Path))
	sumCost := 0.0
	for i, point := range dtwResult.Path {
		costs[i] = point.Cost
		sumCost += point.Cost
	}

	meanCost := sumCost / float64(len(costs))
	variance := 0.0
	for _, cost := range costs {
		diff := cost - meanCost
		variance += diff * diff
	}
	variance /= float64(len(costs))

	// Lower variance = higher confidence
	confidence := 1.0 / (1.0 + variance)
	return math.Min(1.0, confidence)
}

func (aa *AlignmentAnalyzer) calculateCorrelationConfidence(corrResult *CorrelationResult) float64 {
	// Confidence based on peak sharpness and SNR
	sharpnessWeight := 0.6
	snrWeight := 0.4

	sharpnessConf := math.Min(1.0, corrResult.Sharpness)
	snrConf := math.Min(1.0, corrResult.SNR/20.0) // Normalize SNR

	return sharpnessWeight*sharpnessConf + snrWeight*snrConf
}

func (aa *AlignmentAnalyzer) calculateAverageOffset(path []AlignPoint) int {
	if len(path) == 0 {
		return 0
	}

	sumOffset := 0
	for _, point := range path {
		sumOffset += point.RefIndex - point.QueryIndex
	}

	return sumOffset / len(path)
}

func (aa *AlignmentAnalyzer) calculateDTWQuality(dtwResult *DTWResult) float64 {
	qualityMetrics := GetAlignmentQuality(dtwResult)

	// Combine various quality metrics
	efficiency := qualityMetrics["path_efficiency"]
	diagonal := qualityMetrics["diagonal_ratio"]

	return 0.5*efficiency + 0.5*diagonal
}

func (aa *AlignmentAnalyzer) calculatePathStability(path []AlignPoint) float64 {
	if len(path) < 3 {
		return 0.0
	}

	// Calculate stability based on path smoothness
	directionChanges := 0
	prevDirection := [2]int{0, 0}

	for i := 1; i < len(path); i++ {
		currDirection := [2]int{
			path[i].QueryIndex - path[i-1].QueryIndex,
			path[i].RefIndex - path[i-1].RefIndex,
		}

		if i > 1 && (currDirection[0] != prevDirection[0] || currDirection[1] != prevDirection[1]) {
			directionChanges++
		}

		prevDirection = currDirection
	}

	// Lower direction changes = higher stability
	stability := 1.0 - float64(directionChanges)/float64(len(path)-1)
	return math.Max(0.0, stability)
}

// GetAlignmentQuality calculates quality metrics (wrapper for DTW function)
func GetAlignmentQuality(dtwResult *DTWResult) map[string]float64 {
	// This would call the DTW analyzer's quality function
	// For now, return basic metrics
	quality := make(map[string]float64)

	if dtwResult == nil || len(dtwResult.Path) == 0 {
		return quality
	}

	// Path efficiency
	expectedLength := math.Max(float64(dtwResult.QueryLength), float64(dtwResult.RefLength))
	quality["path_efficiency"] = expectedLength / float64(len(dtwResult.Path))

	// Diagonal ratio
	diagonalSteps := 0
	for i := 1; i < len(dtwResult.Path); i++ {
		if dtwResult.Path[i].QueryIndex > dtwResult.Path[i-1].QueryIndex &&
			dtwResult.Path[i].RefIndex > dtwResult.Path[i-1].RefIndex {
			diagonalSteps++
		}
	}
	quality["diagonal_ratio"] = float64(diagonalSteps) / float64(len(dtwResult.Path)-1)

	return quality
}

// FindBestAlignment tries multiple methods and returns the best result
func (aa *AlignmentAnalyzer) FindBestAlignment(query, reference [][]float64, sampleRate int) (*AlignmentResult, error) {
	methods := []AlignmentMethod{AlignmentCrossCorrelation, AlignmentDTW}
	var bestResult *AlignmentResult
	bestScore := 0.0

	for _, method := range methods {
		aa.method = method
		result, err := aa.AlignFeatures(query, reference, sampleRate)
		if err != nil {
			continue
		}

		// Score based on confidence and similarity
		score := 0.6*result.Confidence + 0.4*result.Similarity
		if score > bestScore {
			bestScore = score
			bestResult = result
		}
	}

	if bestResult == nil {
		return nil, fmt.Errorf("all alignment methods failed")
	}

	return bestResult, nil
}

// AlignmentStats provides statistical analysis of alignment results
type AlignmentStats struct {
	MeanOffset   float64 `json:"mean_offset"`
	StdDevOffset float64 `json:"stddev_offset"`
	MedianOffset float64 `json:"median_offset"`
	OffsetRange  float64 `json:"offset_range"`
	Consistency  float64 `json:"consistency"` // How consistent are multiple alignments
}

// AnalyzeAlignmentConsistency analyzes consistency across multiple alignment attempts
func (aa *AlignmentAnalyzer) AnalyzeAlignmentConsistency(query, reference [][]float64, sampleRate int, numTrials int) (*AlignmentStats, error) {
	if numTrials < 2 {
		numTrials = 5
	}

	offsets := make([]float64, 0, numTrials)

	// Perform multiple alignments with slight variations
	for i := 0; i < numTrials; i++ {
		// Add small random perturbations to test robustness
		perturbedQuery := aa.addNoise(query, 0.01) // 1% noise

		result, err := aa.AlignFeatures(perturbedQuery, reference, sampleRate)
		if err != nil {
			continue
		}

		offsets = append(offsets, float64(result.Offset))
	}

	if len(offsets) == 0 {
		return nil, fmt.Errorf("no successful alignments")
	}

	return aa.calculateOffsetStats(offsets), nil
}

func (aa *AlignmentAnalyzer) addNoise(features [][]float64, noiseLevel float64) [][]float64 {
	// Add small amount of Gaussian noise for robustness testing
	noisy := make([][]float64, len(features))
	for i, frame := range features {
		noisy[i] = make([]float64, len(frame))
		for j, val := range frame {
			// Simple pseudo-random noise (not cryptographically secure)
			noise := (math.Sin(float64(i*j+i+j)) * noiseLevel * val)
			noisy[i][j] = val + noise
		}
	}
	return noisy
}

func (aa *AlignmentAnalyzer) calculateOffsetStats(offsets []float64) *AlignmentStats {
	if len(offsets) == 0 {
		return &AlignmentStats{}
	}

	// Calculate mean
	sum := 0.0
	for _, offset := range offsets {
		sum += offset
	}
	mean := sum / float64(len(offsets))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, offset := range offsets {
		diff := offset - mean
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(len(offsets)))

	// Calculate median
	sorted := make([]float64, len(offsets))
	copy(sorted, offsets)
	sort.Float64s(sorted)

	var median float64
	n := len(sorted)
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	// Calculate range
	offsetRange := sorted[n-1] - sorted[0]

	// Calculate consistency (inverse of coefficient of variation)
	consistency := 1.0
	if mean != 0 {
		cv := stdDev / math.Abs(mean)
		consistency = 1.0 / (1.0 + cv)
	}

	return &AlignmentStats{
		MeanOffset:   mean,
		StdDevOffset: stdDev,
		MedianOffset: median,
		OffsetRange:  offsetRange,
		Consistency:  consistency,
	}
}
