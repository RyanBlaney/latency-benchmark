package chroma

import (
	"math"
)

// ChromaSimilarityMethod defines methods for comparing chromagram sequences
type ChromaSimilarityMethod int

const (
	ChromaSimilarityDirect ChromaSimilarityMethod = iota
	ChromaSimilarityBinary
	ChromaSimilaritySmithWaterman
	ChromaSimilarityDTW
	ChromaSimilarityQMax
	ChromaSimilarityOTI
)

// ChromaSimilarityParams holds parameters for chromagram similarity computation
type ChromaSimilarityParams struct {
	Method                 ChromaSimilarityMethod `json:"method"`
	DistanceMetric         ChromaDistanceMetric   `json:"distance_metric"`
	BinaryThreshold        float64                `json:"binary_threshold"`   // Threshold for binary similarity
	FrameStackSize         int                    `json:"frame_stack_size"`   // Number of frames to stack
	FrameStackStride       int                    `json:"frame_stack_stride"` // Stride between stacked frames
	OTIRadius              int                    `json:"oti_radius"`         // Radius for OTI method
	DTWBandRadius          int                    `json:"dtw_band_radius"`    // Band radius for DTW
	NormalizeSimilarity    bool                   `json:"normalize_similarity"`
	TranspositionInvariant bool                   `json:"transposition_invariant"` // Enable key-invariant comparison
}

// ChromaSimilarityResult contains the result of chromagram similarity computation
type ChromaSimilarityResult struct {
	SimilarityMatrix  [][]float64            `json:"similarity_matrix"`  // Cross-similarity matrix
	OverallSimilarity float64                `json:"overall_similarity"` // Global similarity measure
	OptimalPath       []AlignmentPoint       `json:"optimal_path"`       // Alignment path (for DTW/SW)
	BestTransposition int                    `json:"best_transposition"` // Best transposition shift
	Method            ChromaSimilarityMethod `json:"method"`
	QueryFrames       int                    `json:"query_frames"`
	ReferenceFrames   int                    `json:"reference_frames"`
	ComputationTime   float64                `json:"computation_time"` // Processing time in ms
}

// AlignmentPoint represents a point in the optimal alignment path
type AlignmentPoint struct {
	QueryFrame     int     `json:"query_frame"`
	ReferenceFrame int     `json:"reference_frame"`
	Score          float64 `json:"score"`
}

// ChromaSequenceSimilarity computes similarity between chromagram sequences
// This operates on time series of chroma vectors (sequences), not individual vectors
type ChromaSequenceSimilarity struct {
	params         ChromaSimilarityParams
	vectorAnalyzer *ChromaVectorAnalyzer // For individual frame comparisons
}

// NewChromaSequenceSimilarity creates a new chromagram sequence similarity analyzer
func NewChromaSequenceSimilarity() *ChromaSequenceSimilarity {
	return &ChromaSequenceSimilarity{
		params: ChromaSimilarityParams{
			Method:                 ChromaSimilarityDirect,
			DistanceMetric:         ChromaDistanceCosine,
			BinaryThreshold:        0.4,
			FrameStackSize:         1,
			FrameStackStride:       1,
			OTIRadius:              10,
			DTWBandRadius:          50,
			NormalizeSimilarity:    true,
			TranspositionInvariant: false,
		},
		vectorAnalyzer: NewChromaVectorAnalyzer(),
	}
}

// NewChromaSequenceSimilarityWithParams creates analyzer with custom parameters
func NewChromaSequenceSimilarityWithParams(params ChromaSimilarityParams) *ChromaSequenceSimilarity {
	return &ChromaSequenceSimilarity{
		params:         params,
		vectorAnalyzer: NewChromaVectorAnalyzer(),
	}
}

// ComputeSimilarity computes similarity between two chromagram sequences
// queryChroma and refChroma are sequences of chroma vectors [time][chroma_bins]
func (css *ChromaSequenceSimilarity) ComputeSimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	switch css.params.Method {
	case ChromaSimilarityDirect:
		return css.computeDirectSimilarity(queryChroma, refChroma)
	case ChromaSimilarityBinary:
		return css.computeBinarySimilarity(queryChroma, refChroma)
	case ChromaSimilaritySmithWaterman:
		return css.computeSmithWatermanSimilarity(queryChroma, refChroma)
	case ChromaSimilarityDTW:
		return css.computeDTWSimilarity(queryChroma, refChroma)
	case ChromaSimilarityQMax:
		return css.computeQMaxSimilarity(queryChroma, refChroma)
	case ChromaSimilarityOTI:
		return css.computeOTISimilarity(queryChroma, refChroma)
	default:
		return css.computeDirectSimilarity(queryChroma, refChroma)
	}
}

// computeDirectSimilarity computes direct cross-similarity matrix
func (css *ChromaSequenceSimilarity) computeDirectSimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	queryLen := len(queryChroma)
	refLen := len(refChroma)

	// Create cross-similarity matrix
	simMatrix := make([][]float64, queryLen)
	for i := range simMatrix {
		simMatrix[i] = make([]float64, refLen)
	}

	totalSimilarity := 0.0
	count := 0

	// Handle transposition invariance if enabled
	var bestShift int
	var maxSimilarity float64

	if css.params.TranspositionInvariant {
		bestShift, maxSimilarity = css.findBestTransposition(queryChroma, refChroma)
	}

	// Compute pairwise similarities
	for i := 0; i < queryLen; i++ {
		for j := 0; j < refLen; j++ {
			var sim float64

			if css.params.TranspositionInvariant {
				// Use optimal transposition
				shiftedQuery := css.vectorAnalyzer.CircularShift(queryChroma[i], bestShift)
				sim = css.vectorAnalyzer.Similarity(shiftedQuery, refChroma[j], css.params.DistanceMetric)
			} else {
				sim = css.vectorAnalyzer.Similarity(queryChroma[i], refChroma[j], css.params.DistanceMetric)
			}

			simMatrix[i][j] = sim
			totalSimilarity += sim
			count++
		}
	}

	overallSim := totalSimilarity / float64(count)
	if css.params.TranspositionInvariant {
		overallSim = maxSimilarity
	}

	return ChromaSimilarityResult{
		SimilarityMatrix:  simMatrix,
		OverallSimilarity: overallSim,
		BestTransposition: bestShift,
		Method:            css.params.Method,
		QueryFrames:       queryLen,
		ReferenceFrames:   refLen,
	}
}

// computeBinarySimilarity computes binary cross-similarity for cover song detection
func (css *ChromaSequenceSimilarity) computeBinarySimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	// First compute direct similarity
	directResult := css.computeDirectSimilarity(queryChroma, refChroma)

	// Convert to binary matrix
	binaryMatrix := make([][]float64, len(directResult.SimilarityMatrix))
	for i := range binaryMatrix {
		binaryMatrix[i] = make([]float64, len(directResult.SimilarityMatrix[i]))
		for j := range binaryMatrix[i] {
			if directResult.SimilarityMatrix[i][j] > css.params.BinaryThreshold {
				binaryMatrix[i][j] = 1.0
			} else {
				binaryMatrix[i][j] = 0.0
			}
		}
	}

	// Compute overall binary similarity (percentage of matches)
	matches := 0.0
	total := 0.0
	for i := range binaryMatrix {
		for j := range binaryMatrix[i] {
			matches += binaryMatrix[i][j]
			total += 1.0
		}
	}

	directResult.SimilarityMatrix = binaryMatrix
	directResult.OverallSimilarity = matches / total
	directResult.Method = ChromaSimilarityBinary

	return directResult
}

// computeSmithWatermanSimilarity computes local alignment similarity
func (css *ChromaSequenceSimilarity) computeSmithWatermanSimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	queryLen := len(queryChroma)
	refLen := len(refChroma)

	// Initialize scoring matrix
	scores := make([][]float64, queryLen+1)
	for i := range scores {
		scores[i] = make([]float64, refLen+1)
	}

	// Initialize traceback matrix
	traceback := make([][]int, queryLen+1)
	for i := range traceback {
		traceback[i] = make([]int, refLen+1)
	}

	maxScore := 0.0
	maxI, maxJ := 0, 0

	// Fill scoring matrix
	for i := 1; i <= queryLen; i++ {
		for j := 1; j <= refLen; j++ {
			// Compute similarity between frames
			similarity := css.vectorAnalyzer.Similarity(queryChroma[i-1], refChroma[j-1], css.params.DistanceMetric)

			// Smith-Waterman scoring
			match := scores[i-1][j-1] + similarity
			delete := scores[i-1][j] - 0.1 // Gap penalty
			insert := scores[i][j-1] - 0.1 // Gap penalty

			maxVal := math.Max(0, math.Max(match, math.Max(delete, insert)))
			scores[i][j] = maxVal

			// Track maximum for local alignment
			if maxVal > maxScore {
				maxScore = maxVal
				maxI, maxJ = i, j
			}

			// Traceback direction
			if maxVal == match {
				traceback[i][j] = 1 // diagonal
			} else if maxVal == delete {
				traceback[i][j] = 2 // up
			} else if maxVal == insert {
				traceback[i][j] = 3 // left
			}
		}
	}

	// Traceback to find optimal path
	path := css.tracebackAlignment(traceback, scores, maxI, maxJ)

	// Create similarity matrix from scores
	simMatrix := make([][]float64, queryLen)
	for i := range simMatrix {
		simMatrix[i] = make([]float64, refLen)
		for j := range simMatrix[i] {
			simMatrix[i][j] = scores[i+1][j+1]
		}
	}

	// Normalize score by alignment length
	normalizedScore := maxScore / float64(len(path))

	return ChromaSimilarityResult{
		SimilarityMatrix:  simMatrix,
		OverallSimilarity: normalizedScore,
		OptimalPath:       path,
		Method:            ChromaSimilaritySmithWaterman,
		QueryFrames:       queryLen,
		ReferenceFrames:   refLen,
	}
}

// computeDTWSimilarity computes Dynamic Time Warping similarity
func (css *ChromaSequenceSimilarity) computeDTWSimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	queryLen := len(queryChroma)
	refLen := len(refChroma)

	// Initialize cost matrix
	cost := make([][]float64, queryLen)
	for i := range cost {
		cost[i] = make([]float64, refLen)
		for j := range cost[i] {
			cost[i][j] = math.Inf(1)
		}
	}

	// Initialize accumulated cost matrix
	accCost := make([][]float64, queryLen)
	for i := range accCost {
		accCost[i] = make([]float64, refLen)
	}

	// Compute local costs (distance matrix)
	for i := 0; i < queryLen; i++ {
		for j := 0; j < refLen; j++ {
			cost[i][j] = css.vectorAnalyzer.Distance(queryChroma[i], refChroma[j], css.params.DistanceMetric)
		}
	}

	// Initialize first element
	accCost[0][0] = cost[0][0]

	// Initialize first row and column
	for i := 1; i < queryLen; i++ {
		accCost[i][0] = accCost[i-1][0] + cost[i][0]
	}
	for j := 1; j < refLen; j++ {
		accCost[0][j] = accCost[0][j-1] + cost[0][j]
	}

	// Fill accumulated cost matrix with band constraint if specified
	for i := 1; i < queryLen; i++ {
		for j := 1; j < refLen; j++ {
			// Apply band constraint if specified
			if css.params.DTWBandRadius > 0 {
				expectedJ := int(float64(j*queryLen) / float64(refLen))
				if abs(j-expectedJ) > css.params.DTWBandRadius {
					continue // Skip cells outside band
				}
			}

			// Find minimum cost path
			minPrev := math.Min(accCost[i-1][j], math.Min(accCost[i][j-1], accCost[i-1][j-1]))
			accCost[i][j] = cost[i][j] + minPrev
		}
	}

	// Traceback to find optimal path
	path := css.tracebackDTW(accCost, queryLen-1, refLen-1)

	// Convert distances to similarities
	simMatrix := make([][]float64, queryLen)
	for i := range simMatrix {
		simMatrix[i] = make([]float64, refLen)
		for j := range simMatrix[i] {
			simMatrix[i][j] = math.Exp(-cost[i][j]) // Convert distance to similarity
		}
	}

	// Overall similarity is inverse of normalized DTW distance
	dtwDistance := accCost[queryLen-1][refLen-1] / float64(len(path))
	overallSimilarity := math.Exp(-dtwDistance)

	return ChromaSimilarityResult{
		SimilarityMatrix:  simMatrix,
		OverallSimilarity: overallSimilarity,
		OptimalPath:       path,
		Method:            ChromaSimilarityDTW,
		QueryFrames:       queryLen,
		ReferenceFrames:   refLen,
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// computeQMaxSimilarity computes Qmax similarity (maximum similarity along diagonals)
func (css *ChromaSequenceSimilarity) computeQMaxSimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	// First compute direct similarity matrix
	directResult := css.computeDirectSimilarity(queryChroma, refChroma)
	simMatrix := directResult.SimilarityMatrix

	queryLen := len(queryChroma)
	refLen := len(refChroma)

	// Compute Qmax by finding maximum similarity along each diagonal
	maxSim := 0.0
	diagonalCount := 0

	// Main diagonals (positive slope)
	for d := -(refLen - 1); d < queryLen; d++ {
		diagonalMax := 0.0
		diagonalCount++

		for i := 0; i < queryLen; i++ {
			j := i - d
			if j >= 0 && j < refLen {
				if simMatrix[i][j] > diagonalMax {
					diagonalMax = simMatrix[i][j]
				}
			}
		}

		if diagonalMax > maxSim {
			maxSim = diagonalMax
		}
	}

	directResult.OverallSimilarity = maxSim
	directResult.Method = ChromaSimilarityQMax

	return directResult
}

// computeOTISimilarity computes Optimal Transposition Index similarity
func (css *ChromaSequenceSimilarity) computeOTISimilarity(queryChroma, refChroma []ChromaVector) ChromaSimilarityResult {
	queryLen := len(queryChroma)
	refLen := len(refChroma)

	maxSimilarity := 0.0
	bestTransposition := 0
	var bestMatrix [][]float64

	// Try all possible transpositions (0-11 semitones)
	for shift := 0; shift < 12; shift++ {
		totalSim := 0.0
		simMatrix := make([][]float64, queryLen)

		for i := range simMatrix {
			simMatrix[i] = make([]float64, refLen)
		}

		// Compute similarity with this transposition
		for i := 0; i < queryLen; i++ {
			// Apply circular shift to query chroma
			shiftedQuery := css.vectorAnalyzer.CircularShift(queryChroma[i], shift)

			for j := math.Max(0, float64(i-css.params.OTIRadius)); j < math.Min(float64(refLen), float64(i+css.params.OTIRadius+1)); j++ {
				sim := css.vectorAnalyzer.Similarity(shiftedQuery, refChroma[int(j)], css.params.DistanceMetric)
				simMatrix[i][int(j)] = sim
				totalSim += sim
			}
		}

		avgSim := totalSim / float64(queryLen*refLen)
		if avgSim > maxSimilarity {
			maxSimilarity = avgSim
			bestTransposition = shift
			bestMatrix = simMatrix
		}
	}

	return ChromaSimilarityResult{
		SimilarityMatrix:  bestMatrix,
		OverallSimilarity: maxSimilarity,
		BestTransposition: bestTransposition,
		Method:            ChromaSimilarityOTI,
		QueryFrames:       queryLen,
		ReferenceFrames:   refLen,
	}
}

// Helper functions

// findBestTransposition finds optimal circular shift for transposition invariance
func (css *ChromaSequenceSimilarity) findBestTransposition(queryChroma, refChroma []ChromaVector) (int, float64) {
	bestShift := 0
	maxSimilarity := 0.0

	for shift := 0; shift < 12; shift++ {
		totalSim := 0.0
		count := 0

		// Sample a subset of frames for efficiency
		sampleStep := math.Max(1, float64(len(queryChroma))/50) // Sample ~50 frames

		for i := 0; i < len(queryChroma); i += int(sampleStep) {
			shiftedQuery := css.vectorAnalyzer.CircularShift(queryChroma[i], shift)

			for j := 0; j < len(refChroma); j += int(sampleStep) {
				sim := css.vectorAnalyzer.Similarity(shiftedQuery, refChroma[j], css.params.DistanceMetric)
				totalSim += sim
				count++
			}
		}

		avgSim := totalSim / float64(count)
		if avgSim > maxSimilarity {
			maxSimilarity = avgSim
			bestShift = shift
		}
	}

	return bestShift, maxSimilarity
}

// tracebackAlignment performs traceback for Smith-Waterman alignment
func (css *ChromaSequenceSimilarity) tracebackAlignment(traceback [][]int, scores [][]float64, startI, startJ int) []AlignmentPoint {
	var path []AlignmentPoint
	i, j := startI, startJ

	for i > 0 && j > 0 && scores[i][j] > 0 {
		path = append([]AlignmentPoint{{
			QueryFrame:     i - 1,
			ReferenceFrame: j - 1,
			Score:          scores[i][j],
		}}, path...)

		switch traceback[i][j] {
		case 1: // diagonal
			i--
			j--
		case 2: // up
			i--
		case 3: // left
			j--
		default:
			return path
		}
	}

	return path
}

// tracebackDTW performs traceback for DTW alignment
func (css *ChromaSequenceSimilarity) tracebackDTW(cost [][]float64, i, j int) []AlignmentPoint {
	var path []AlignmentPoint

	for i > 0 || j > 0 {
		path = append([]AlignmentPoint{{
			QueryFrame:     i,
			ReferenceFrame: j,
			Score:          cost[i][j],
		}}, path...)

		if i == 0 {
			j--
		} else if j == 0 {
			i--
		} else {
			// Choose minimum cost predecessor
			if cost[i-1][j-1] <= cost[i-1][j] && cost[i-1][j-1] <= cost[i][j-1] {
				i--
				j--
			} else if cost[i-1][j] <= cost[i][j-1] {
				i--
			} else {
				j--
			}
		}
	}

	return path
}

// GetParams returns current parameters
func (css *ChromaSequenceSimilarity) GetParams() ChromaSimilarityParams {
	return css.params
}

// SetParams updates parameters
func (css *ChromaSequenceSimilarity) SetParams(params ChromaSimilarityParams) {
	css.params = params
}
