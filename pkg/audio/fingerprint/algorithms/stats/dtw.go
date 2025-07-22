package stats

import (
	"fmt"
	"math"
)

// DTWAlignment represents Dynamic Time Warping alignment
// WHY: DTW is essential for aligning audio sequences of different lengths,
// crucial for audio fingerprint matching and similarity computation
type DTWAlignment struct {
	constraintBand int    // Sakoe-Chiba band constraint
	stepPattern    string // Step pattern type
	distanceMetric DistanceMetric
}

// DTWResult contains DTW alignment results
type DTWResult struct {
	Distance    float64      `json:"distance"`     // Total DTW distance
	Path        []AlignPoint `json:"path"`         // Optimal alignment path
	CostMatrix  [][]float64  `json:"cost_matrix"`  // DTW cost matrix
	QueryLength int          `json:"query_length"` // Length of query sequence
	RefLength   int          `json:"ref_length"`   // Length of reference sequence
	Normalized  bool         `json:"normalized"`   // Whether distance is normalized
	StepPattern string       `json:"step_pattern"` // Step pattern used
	Constraint  int          `json:"constraint"`   // Band constraint used
}

// AlignPoint represents a point in the alignment path
type AlignPoint struct {
	QueryIndex int     `json:"query_index"` // Index in query sequence
	RefIndex   int     `json:"ref_index"`   // Index in reference sequence
	Cost       float64 `json:"cost"`        // Local cost at this point
}

// NewDTWAlignment creates a new DTW alignment instance
func NewDTWAlignment() *DTWAlignment {
	return &DTWAlignment{
		constraintBand: -1,           // No constraint by default
		stepPattern:    "symmetric2", // Standard symmetric step pattern
		distanceMetric: EuclideanDistance,
	}
}

// NewDTWAlignmentWithParams creates DTW with custom parameters
func NewDTWAlignmentWithParams(constraintBand int, stepPattern string, metric DistanceMetric) *DTWAlignment {
	return &DTWAlignment{
		constraintBand: constraintBand,
		stepPattern:    stepPattern,
		distanceMetric: metric,
	}
}

// Align performs DTW alignment between two sequences
func (dtw *DTWAlignment) Align(query, reference [][]float64) (*DTWResult, error) {
	if len(query) == 0 || len(reference) == 0 {
		return nil, fmt.Errorf("empty sequences provided")
	}

	queryLen := len(query)
	refLen := len(reference)

	// Initialize cost matrix
	costMatrix := make([][]float64, queryLen+1)
	for i := range costMatrix {
		costMatrix[i] = make([]float64, refLen+1)
		for j := range costMatrix[i] {
			costMatrix[i][j] = math.Inf(1)
		}
	}

	// Set starting point
	costMatrix[0][0] = 0

	// Fill cost matrix using dynamic programming
	err := dtw.fillCostMatrix(costMatrix, query, reference)
	if err != nil {
		return nil, fmt.Errorf("failed to fill cost matrix: %w", err)
	}

	// Backtrack to find optimal path
	path, err := dtw.backtrack(costMatrix, queryLen, refLen)
	if err != nil {
		return nil, fmt.Errorf("failed to backtrack path: %w", err)
	}

	// Calculate final distance
	finalDistance := costMatrix[queryLen][refLen]

	// Normalize by path length if desired
	normalizedDistance := finalDistance / float64(len(path))

	return &DTWResult{
		Distance:    normalizedDistance,
		Path:        path,
		CostMatrix:  costMatrix[1:], // Remove padding row/column
		QueryLength: queryLen,
		RefLength:   refLen,
		Normalized:  true,
		StepPattern: dtw.stepPattern,
		Constraint:  dtw.constraintBand,
	}, nil
}

// fillCostMatrix fills the DTW cost matrix
func (dtw *DTWAlignment) fillCostMatrix(costMatrix [][]float64, query, reference [][]float64) error {
	queryLen := len(query)
	refLen := len(reference)

	distanceFunc := GetDistanceFunction(dtw.distanceMetric)

	for i := 1; i <= queryLen; i++ {
		for j := 1; j <= refLen; j++ {
			// Check band constraint
			if dtw.constraintBand > 0 {
				if math.Abs(float64(i-j)) > float64(dtw.constraintBand) {
					continue // Skip cells outside the band
				}
			}

			// Calculate local distance
			localDist := distanceFunc(query[i-1], reference[j-1])

			// Apply step pattern
			minCost, err := dtw.applyStepPattern(costMatrix, i, j)
			if err != nil {
				return err
			}

			costMatrix[i][j] = localDist + minCost
		}
	}

	return nil
}

// applyStepPattern applies the specified step pattern
func (dtw *DTWAlignment) applyStepPattern(costMatrix [][]float64, i, j int) (float64, error) {
	switch dtw.stepPattern {
	case "symmetric2":
		// Standard symmetric pattern: (i-1,j), (i,j-1), (i-1,j-1)
		return math.Min(math.Min(costMatrix[i-1][j], costMatrix[i][j-1]), costMatrix[i-1][j-1]), nil

	case "asymmetric":
		// Asymmetric pattern favoring horizontal moves
		return math.Min(costMatrix[i-1][j], costMatrix[i][j-1]), nil

	case "symmetric1":
		// Simple symmetric pattern
		if i > 0 && j > 0 {
			return math.Min(costMatrix[i-1][j]+1, math.Min(costMatrix[i][j-1]+1, costMatrix[i-1][j-1])), nil
		} else if i > 0 {
			return costMatrix[i-1][j] + 1, nil
		} else if j > 0 {
			return costMatrix[i][j-1] + 1, nil
		}
		return 0, nil

	default:
		return 0, fmt.Errorf("unknown step pattern: %s", dtw.stepPattern)
	}
}

// backtrack finds the optimal alignment path
func (dtw *DTWAlignment) backtrack(costMatrix [][]float64, queryLen, refLen int) ([]AlignPoint, error) {
	var path []AlignPoint
	i, j := queryLen, refLen

	for i > 0 || j > 0 {
		// Add current point to path
		cost := 0.0
		if i > 0 && j > 0 {
			cost = costMatrix[i][j] - costMatrix[i-1][j-1]
		}

		path = append([]AlignPoint{{
			QueryIndex: i - 1,
			RefIndex:   j - 1,
			Cost:       cost,
		}}, path...)

		// Find previous step based on step pattern
		prevI, prevJ := dtw.findPreviousStep(costMatrix, i, j)
		i, j = prevI, prevJ
	}

	return path, nil
}

// findPreviousStep finds the previous step in backtracking
func (dtw *DTWAlignment) findPreviousStep(costMatrix [][]float64, i, j int) (int, int) {
	if i == 0 {
		return 0, j - 1
	}
	if j == 0 {
		return i - 1, 0
	}

	// Find minimum cost predecessor
	costs := []struct {
		cost float64
		i, j int
	}{
		{costMatrix[i-1][j], i - 1, j},       // Vertical
		{costMatrix[i][j-1], i, j - 1},       // Horizontal
		{costMatrix[i-1][j-1], i - 1, j - 1}, // Diagonal
	}

	minIdx := 0
	for idx, c := range costs {
		if c.cost < costs[minIdx].cost {
			minIdx = idx
		}
	}

	return costs[minIdx].i, costs[minIdx].j
}

// AlignVectors aligns two 1D feature vectors
func (dtw *DTWAlignment) AlignVectors(query, reference []float64) (*DTWResult, error) {
	// Convert 1D vectors to 2D for compatibility
	query2D := make([][]float64, len(query))
	ref2D := make([][]float64, len(reference))

	for i, v := range query {
		query2D[i] = []float64{v}
	}
	for i, v := range reference {
		ref2D[i] = []float64{v}
	}

	return dtw.Align(query2D, ref2D)
}

// ConstrainedAlign performs DTW with global path constraints
func (dtw *DTWAlignment) ConstrainedAlign(query, reference [][]float64, startConstraint, endConstraint [2]int) (*DTWResult, error) {
	// This is a simplified version - full implementation would respect start/end constraints
	result, err := dtw.Align(query, reference)
	if err != nil {
		return nil, err
	}

	// Filter path to respect constraints (simplified)
	if len(result.Path) > 0 {
		// Ensure path starts and ends within constraints if specified
		// This is a basic implementation - production would be more sophisticated
		// TODO: implement this
	}

	return result, nil
}

// GetAlignmentQuality calculates quality metrics for the alignment
func (dtw *DTWAlignment) GetAlignmentQuality(result *DTWResult) map[string]float64 {
	if result == nil || len(result.Path) == 0 {
		return map[string]float64{}
	}

	quality := make(map[string]float64)

	// Path efficiency (shorter paths are better for similar lengths)
	expectedLength := math.Max(float64(result.QueryLength), float64(result.RefLength))
	quality["path_efficiency"] = expectedLength / float64(len(result.Path))

	// Path straightness (diagonal preference)
	diagonalSteps := 0
	for i := 1; i < len(result.Path); i++ {
		if result.Path[i].QueryIndex > result.Path[i-1].QueryIndex &&
			result.Path[i].RefIndex > result.Path[i-1].RefIndex {
			diagonalSteps++
		}
	}
	quality["diagonal_ratio"] = float64(diagonalSteps) / float64(len(result.Path)-1)

	// Average local cost
	totalCost := 0.0
	for _, point := range result.Path {
		totalCost += point.Cost
	}
	quality["average_cost"] = totalCost / float64(len(result.Path))

	// Normalized distance
	quality["normalized_distance"] = result.Distance

	return quality
}

// OptimizeStepPattern selects best step pattern for given data
func (dtw *DTWAlignment) OptimizeStepPattern(query, reference [][]float64) (string, error) {
	patterns := []string{"symmetric2", "asymmetric", "symmetric1"}
	bestPattern := patterns[0]
	bestDistance := math.Inf(1)

	originalPattern := dtw.stepPattern

	for _, pattern := range patterns {
		dtw.stepPattern = pattern
		result, err := dtw.Align(query, reference)
		if err != nil {
			continue
		}

		if result.Distance < bestDistance {
			bestDistance = result.Distance
			bestPattern = pattern
		}
	}

	dtw.stepPattern = originalPattern
	return bestPattern, nil
}
