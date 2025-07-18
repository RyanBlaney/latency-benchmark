package chroma

import (
	"math"
)

// TonnetzPoint represents a point in the Tonnetz (harmonic network)
type TonnetzPoint struct {
	X          float64 // Horizontal coordinate (fifth dimension)
	Y          float64 // Vertical coordinate (major third dimension)
	Weight     float64 // Weight/energy at this point
	PitchClass int     // Associated pitch class (0-11)
}

// TonnetzCentroid represents the centroid of energy in Tonnetz space
type TonnetzCentroid struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// TonnetzAnalyzer provides Tonnetz (harmonic network) analysis
//
// TONNETZ EXPLANATION:
// The Tonnetz is a conceptual lattice that represents harmonic relationships
// in music theory. It maps pitch classes onto a 2D plane where:
// - Horizontal axis represents perfect fifths (7 semitones)
// - Vertical axis represents major thirds (4 semitones)
// - Close proximity in the lattice = strong harmonic relationship
// - Triangular regions represent major/minor triads
// - Movement patterns reveal harmonic progressions
type TonnetzAnalyzer struct {
	// Tonnetz coordinates for each pitch class
	coordinates map[int]TonnetzPoint
}

// NewTonnetzAnalyzer creates a new Tonnetz analyzer
func NewTonnetzAnalyzer() *TonnetzAnalyzer {
	ta := &TonnetzAnalyzer{
		coordinates: make(map[int]TonnetzPoint),
	}

	// Initialize Tonnetz coordinates for all 12 pitch classes
	ta.initializeTonnetzCoordinates()

	return ta
}

// initializeTonnetzCoordinates sets up the Tonnetz coordinate system
func (ta *TonnetzAnalyzer) initializeTonnetzCoordinates() {
	// Tonnetz coordinates based on perfect fifths (x) and major thirds (y)
	// Starting from C (0,0) and following circle of fifths and major thirds

	// Circle of fifths coordinates (x-axis)
	fifthsCoords := map[int]float64{
		0:  0,  // C
		7:  1,  // G
		2:  2,  // D
		9:  3,  // A
		4:  4,  // E
		11: 5,  // B
		6:  6,  // F#
		1:  -5, // C# (wrapping around)
		8:  -4, // G#
		3:  -3, // D#
		10: -2, // A#
		5:  -1, // F
	}

	// Major thirds create the y-axis displacement
	for pc := range 12 {
		x := fifthsCoords[pc]

		// Y coordinate based on major third relationships
		// Each major third (4 semitones) moves up in y
		y := 0.0

		// Calculate y based on major third cycles
		majorThirdCycle := pc
		for majorThirdCycle >= 4 {
			y += 1.0
			majorThirdCycle -= 4
		}
		for majorThirdCycle < 0 {
			y -= 1.0
			majorThirdCycle += 4
		}

		// Adjust y coordinate for better lattice representation
		switch pc {
		case 4, 8, 0: // E, G#, C
			y = 0.0
		case 7, 11, 3: // G, B, D#
			y = math.Sqrt(3.0) / 2.0
		case 10, 2, 6: // A#, D, F#
			y = -math.Sqrt(3.0) / 2.0
		case 1, 5, 9: // C#, F, A
			y = math.Sqrt(3.0)
		}

		ta.coordinates[pc] = TonnetzPoint{
			X:          x,
			Y:          y,
			Weight:     0.0,
			PitchClass: pc,
		}
	}
}

// ComputeTonnetz computes Tonnetz representation from chromagram
func (ta *TonnetzAnalyzer) ComputeTonnetz(chromagram [][]float64) []TonnetzPoint {
	if len(chromagram) == 0 || len(chromagram[0]) != 12 {
		return []TonnetzPoint{}
	}

	// Calculate mean energy for each pitch class across time
	meanEnergy := make([]float64, 12)
	for t := range chromagram {
		for pc := range 12 {
			meanEnergy[pc] += chromagram[t][pc]
		}
	}
	for pc := range meanEnergy {
		meanEnergy[pc] /= float64(len(chromagram))
	}

	// Create Tonnetz points with energy weights
	tonnetzPoints := make([]TonnetzPoint, 12)
	for pc := range 12 {
		point := ta.coordinates[pc]
		point.Weight = meanEnergy[pc]
		tonnetzPoints[pc] = point
	}

	return tonnetzPoints
}

// ComputeTonnetzCentroid calculates the centroid of energy in Tonnetz space
func (ta *TonnetzAnalyzer) ComputeTonnetzCentroid(tonnetzPoints []TonnetzPoint) TonnetzCentroid {
	if len(tonnetzPoints) == 0 {
		return TonnetzCentroid{X: 0.0, Y: 0.0}
	}

	weightedX := 0.0
	weightedY := 0.0
	totalWeight := 0.0

	for _, point := range tonnetzPoints {
		weightedX += point.X * point.Weight
		weightedY += point.Y * point.Weight
		totalWeight += point.Weight
	}

	if totalWeight > 1e-10 {
		return TonnetzCentroid{
			X: weightedX / totalWeight,
			Y: weightedY / totalWeight,
		}
	}

	return TonnetzCentroid{X: 0.0, Y: 0.0}
}

// ComputeTonnetzTrajectory computes trajectory through Tonnetz space over time
func (ta *TonnetzAnalyzer) ComputeTonnetzTrajectory(chromagram [][]float64) [][]float64 {
	if len(chromagram) == 0 {
		return [][]float64{}
	}

	trajectory := make([][]float64, len(chromagram))

	for t := range chromagram {
		// Create Tonnetz points for this time frame
		tonnetzPoints := make([]TonnetzPoint, 12)
		for pc := range 12 {
			point := ta.coordinates[pc]
			point.Weight = chromagram[t][pc]
			tonnetzPoints[pc] = point
		}

		// Calculate centroid for this frame
		centroid := ta.ComputeTonnetzCentroid(tonnetzPoints)
		trajectory[t] = []float64{centroid.X, centroid.Y}
	}

	return trajectory
}

// AnalyzeTonnetzMovement analyzes movement patterns in Tonnetz space
func (ta *TonnetzAnalyzer) AnalyzeTonnetzMovement(trajectory [][]float64) map[string]float64 {
	analysis := make(map[string]float64)

	if len(trajectory) < 2 {
		return analysis
	}

	// Calculate movement statistics
	totalDistance := 0.0
	maxDistance := 0.0
	velocities := make([]float64, len(trajectory)-1)

	for i := 1; i < len(trajectory); i++ {
		dx := trajectory[i][0] - trajectory[i-1][0]
		dy := trajectory[i][1] - trajectory[i-1][1]
		distance := math.Sqrt(dx*dx + dy*dy)

		velocities[i-1] = distance
		totalDistance += distance

		if distance > maxDistance {
			maxDistance = distance
		}
	}

	analysis["total_distance"] = totalDistance
	analysis["max_velocity"] = maxDistance
	analysis["mean_velocity"] = totalDistance / float64(len(velocities))

	// Calculate velocity variance (measure of harmonic stability)
	meanVelocity := analysis["mean_velocity"]
	velocityVariance := 0.0
	for _, vel := range velocities {
		diff := vel - meanVelocity
		velocityVariance += diff * diff
	}
	velocityVariance /= float64(len(velocities))
	analysis["velocity_variance"] = velocityVariance
	analysis["harmonic_stability"] = 1.0 / (1.0 + velocityVariance)

	// Calculate path efficiency (displacement / total distance)
	if totalDistance > 1e-10 && len(trajectory) > 1 {
		startX, startY := trajectory[0][0], trajectory[0][1]
		endX, endY := trajectory[len(trajectory)-1][0], trajectory[len(trajectory)-1][1]
		displacement := math.Sqrt((endX-startX)*(endX-startX) + (endY-startY)*(endY-startY))
		analysis["path_efficiency"] = displacement / totalDistance
	}

	return analysis
}

// DetectHarmonicRegions detects regions of harmonic activity in Tonnetz
func (ta *TonnetzAnalyzer) DetectHarmonicRegions(tonnetzPoints []TonnetzPoint, threshold float64) []map[string]any {
	var regions []map[string]any

	// Group points by proximity and energy
	for _, point := range tonnetzPoints {
		if point.Weight < threshold {
			continue
		}

		// Find nearby pitch classes (harmonic neighbors)
		neighbors := ta.findHarmonicNeighbors(point.PitchClass, tonnetzPoints, 2.0)

		if len(neighbors) >= 2 { // At least 3 pitch classes (including self)
			region := map[string]any{
				"center_pc":     point.PitchClass,
				"center_x":      point.X,
				"center_y":      point.Y,
				"total_energy":  point.Weight,
				"pitch_classes": []int{point.PitchClass},
				"type":          ta.classifyHarmonicRegion(append(neighbors, point.PitchClass)),
			}

			for _, neighbor := range neighbors {
				region["total_energy"] = region["total_energy"].(float64) + tonnetzPoints[neighbor].Weight
				region["pitch_classes"] = append(region["pitch_classes"].([]int), neighbor)
			}

			regions = append(regions, region)
		}
	}

	return regions
}

// findHarmonicNeighbors finds pitch classes within distance threshold in Tonnetz
func (ta *TonnetzAnalyzer) findHarmonicNeighbors(centerPC int, tonnetzPoints []TonnetzPoint, maxDistance float64) []int {
	var neighbors []int
	centerPoint := ta.coordinates[centerPC]

	for pc, point := range ta.coordinates {
		if pc == centerPC {
			continue
		}

		distance := math.Sqrt((point.X-centerPoint.X)*(point.X-centerPoint.X) +
			(point.Y-centerPoint.Y)*(point.Y-centerPoint.Y))

		if distance <= maxDistance && tonnetzPoints[pc].Weight > 1e-10 {
			neighbors = append(neighbors, pc)
		}
	}

	return neighbors
}

// classifyHarmonicRegion classifies a group of pitch classes
func (ta *TonnetzAnalyzer) classifyHarmonicRegion(pitchClasses []int) string {
	if len(pitchClasses) < 3 {
		return "incomplete"
	}

	// Sort pitch classes
	for i := 0; i < len(pitchClasses)-1; i++ {
		for j := i + 1; j < len(pitchClasses); j++ {
			if pitchClasses[j] < pitchClasses[i] {
				pitchClasses[i], pitchClasses[j] = pitchClasses[j], pitchClasses[i]
			}
		}
	}

	// Check for common chord types
	if len(pitchClasses) >= 3 {
		// Check all possible root positions for triads
		for root := 0; root < len(pitchClasses)-2; root++ {
			pc1 := pitchClasses[root]
			pc2 := pitchClasses[root+1]
			pc3 := pitchClasses[root+2]

			// Calculate intervals (handle wraparound)
			interval1 := (pc2 - pc1 + 12) % 12
			interval2 := (pc3 - pc2 + 12) % 12
			// Removed unused interval3

			// Major triad: 4-3 semitones (major third + minor third)
			if (interval1 == 4 && interval2 == 3) ||
				(interval1 == 3 && interval2 == 5) ||
				(interval1 == 5 && interval2 == 4) {
				return "major_triad"
			}

			// Minor triad: 3-4 semitones (minor third + major third)
			if (interval1 == 3 && interval2 == 4) ||
				(interval1 == 4 && interval2 == 5) ||
				(interval1 == 5 && interval2 == 3) {
				return "minor_triad"
			}

			// Diminished triad: 3-3 semitones
			if interval1 == 3 && interval2 == 3 {
				return "diminished_triad"
			}

			// Augmented triad: 4-4 semitones
			if interval1 == 4 && interval2 == 4 {
				return "augmented_triad"
			}
		}
	}

	// Check for seventh chords
	if len(pitchClasses) >= 4 {
		// Simplified seventh chord detection
		return "seventh_chord"
	}

	// Check for perfect fifth
	if len(pitchClasses) == 2 {
		interval := (pitchClasses[1] - pitchClasses[0] + 12) % 12
		if interval == 7 || interval == 5 { // Perfect fifth or perfect fourth
			return "perfect_fifth"
		}
	}

	return "complex"
}

// ComputeHarmonicTension calculates harmonic tension in Tonnetz space
func (ta *TonnetzAnalyzer) ComputeHarmonicTension(tonnetzPoints []TonnetzPoint) float64 {
	if len(tonnetzPoints) < 2 {
		return 0.0
	}

	totalTension := 0.0
	totalWeight := 0.0

	// Calculate tension as weighted sum of distances between active pitch classes
	for i := range len(tonnetzPoints) {
		for j := i + 1; j < len(tonnetzPoints); j++ {
			weight := tonnetzPoints[i].Weight * tonnetzPoints[j].Weight
			if weight > 1e-10 {
				distance := math.Sqrt(
					(tonnetzPoints[i].X-tonnetzPoints[j].X)*(tonnetzPoints[i].X-tonnetzPoints[j].X) +
						(tonnetzPoints[i].Y-tonnetzPoints[j].Y)*(tonnetzPoints[i].Y-tonnetzPoints[j].Y),
				)

				totalTension += weight * distance
				totalWeight += weight
			}
		}
	}

	if totalWeight > 1e-10 {
		return totalTension / totalWeight
	}

	return 0.0
}

// AnalyzeVoiceLeading analyzes voice leading patterns in Tonnetz movement
func (ta *TonnetzAnalyzer) AnalyzeVoiceLeading(trajectory [][]float64) map[string]float64 {
	analysis := make(map[string]float64)

	if len(trajectory) < 2 {
		return analysis
	}

	// Analyze step sizes (smaller steps = smoother voice leading)
	stepSizes := make([]float64, len(trajectory)-1)
	for i := 1; i < len(trajectory); i++ {
		dx := trajectory[i][0] - trajectory[i-1][0]
		dy := trajectory[i][1] - trajectory[i-1][1]
		stepSizes[i-1] = math.Sqrt(dx*dx + dy*dy)
	}

	// Calculate smoothness metrics
	totalStepSize := 0.0
	for _, step := range stepSizes {
		totalStepSize += step
	}

	analysis["mean_step_size"] = totalStepSize / float64(len(stepSizes))

	// Count small steps (smooth voice leading)
	smallSteps := 0
	for _, step := range stepSizes {
		if step < 1.0 { // Threshold for "small" step
			smallSteps++
		}
	}

	analysis["smooth_voice_leading_ratio"] = float64(smallSteps) / float64(len(stepSizes))

	// Calculate directional consistency
	if len(trajectory) >= 3 {
		consistentDirection := 0
		for i := 2; i < len(trajectory); i++ {
			// Calculate direction vectors
			dx1 := trajectory[i-1][0] - trajectory[i-2][0]
			dy1 := trajectory[i-1][1] - trajectory[i-2][1]
			dx2 := trajectory[i][0] - trajectory[i-1][0]
			dy2 := trajectory[i][1] - trajectory[i-1][1]

			// Calculate dot product (measures direction similarity)
			dotProduct := dx1*dx2 + dy1*dy2
			magnitude1 := math.Sqrt(dx1*dx1 + dy1*dy1)
			magnitude2 := math.Sqrt(dx2*dx2 + dy2*dy2)

			if magnitude1 > 1e-10 && magnitude2 > 1e-10 {
				cosine := dotProduct / (magnitude1 * magnitude2)
				if cosine > 0.5 { // Similar direction
					consistentDirection++
				}
			}
		}

		analysis["directional_consistency"] = float64(consistentDirection) / float64(len(trajectory)-2)
	}

	return analysis
}

// ComputeConsonanceDissonance calculates consonance/dissonance based on Tonnetz positions
func (ta *TonnetzAnalyzer) ComputeConsonanceDissonance(tonnetzPoints []TonnetzPoint) map[string]float64 {
	result := make(map[string]float64)

	// Consonant intervals have shorter distances in Tonnetz
	consonantDistance := 0.0
	dissonantDistance := 0.0
	totalEnergy := 0.0

	for i := range len(tonnetzPoints) {
		for j := i + 1; j < len(tonnetzPoints); j++ {
			weight := tonnetzPoints[i].Weight * tonnetzPoints[j].Weight
			if weight > 1e-10 {
				distance := math.Sqrt(
					(tonnetzPoints[i].X-tonnetzPoints[j].X)*(tonnetzPoints[i].X-tonnetzPoints[j].X) +
						(tonnetzPoints[i].Y-tonnetzPoints[j].Y)*(tonnetzPoints[i].Y-tonnetzPoints[j].Y),
				)

				// Classify as consonant or dissonant based on distance
				if distance <= 1.5 { // Close neighbors = consonant
					consonantDistance += weight * distance
				} else { // Distant = dissonant
					dissonantDistance += weight * distance
				}

				totalEnergy += weight
			}
		}
	}

	if totalEnergy > 1e-10 {
		result["consonance"] = consonantDistance / totalEnergy
		result["dissonance"] = dissonantDistance / totalEnergy
		result["consonance_ratio"] = consonantDistance / (consonantDistance + dissonantDistance + 1e-10)
	} else {
		result["consonance"] = 0.0
		result["dissonance"] = 0.0
		result["consonance_ratio"] = 0.5
	}

	return result
}

// GetTonnetzVisualizationData returns data for visualizing the Tonnetz
func (ta *TonnetzAnalyzer) GetTonnetzVisualizationData(tonnetzPoints []TonnetzPoint) map[string]any {
	visualization := map[string]any{
		"points":      tonnetzPoints,
		"coordinates": ta.coordinates,
		"grid_lines":  ta.generateGridLines(),
	}

	return visualization
}

// generateGridLines generates grid lines for Tonnetz visualization
func (ta *TonnetzAnalyzer) generateGridLines() map[string][][]float64 {
	gridLines := map[string][][]float64{
		"fifths":       make([][]float64, 0), // Horizontal lines (perfect fifths)
		"major_thirds": make([][]float64, 0), // Diagonal lines (major thirds)
		"minor_thirds": make([][]float64, 0), // Other diagonal lines (minor thirds)
	}

	// Generate some example grid lines for visualization
	// This would typically be more comprehensive in a full implementation
	// TODO: refine this
	for i := -3; i <= 8; i++ {
		// Perfect fifths (horizontal) - fixed the float64 conversion
		gridLines["fifths"] = append(gridLines["fifths"], []float64{float64(i), -2.0})
		gridLines["fifths"] = append(gridLines["fifths"], []float64{float64(i), 2.0})
	}

	return gridLines
}

// GetPitchClassNames returns pitch class names for display
func (ta *TonnetzAnalyzer) GetPitchClassNames() []string {
	return []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
}

// ComputeTonnetzStatistics computes comprehensive statistics for Tonnetz analysis
func (ta *TonnetzAnalyzer) ComputeTonnetzStatistics(chromagram [][]float64) map[string]any {
	tonnetzPoints := ta.ComputeTonnetz(chromagram)
	trajectory := ta.ComputeTonnetzTrajectory(chromagram)

	stats := map[string]any{
		"centroid":            ta.ComputeTonnetzCentroid(tonnetzPoints), // Fixed: now returns TonnetzCentroid struct
		"harmonic_tension":    ta.ComputeHarmonicTension(tonnetzPoints),
		"movement_analysis":   ta.AnalyzeMovement(trajectory),
		"voice_leading":       ta.AnalyzeVoiceLeading(trajectory),
		"consonance_analysis": ta.ComputeConsonanceDissonance(tonnetzPoints),
		"harmonic_regions":    ta.DetectHarmonicRegions(tonnetzPoints, 0.1),
		"total_energy":        ta.calculateTotalEnergy(tonnetzPoints),
	}

	return stats
}

// AnalyzeMovement is an alias for AnalyzeTonnetzMovement for consistency
func (ta *TonnetzAnalyzer) AnalyzeMovement(trajectory [][]float64) map[string]float64 {
	return ta.AnalyzeTonnetzMovement(trajectory)
}

// calculateTotalEnergy calculates total energy across all Tonnetz points
func (ta *TonnetzAnalyzer) calculateTotalEnergy(tonnetzPoints []TonnetzPoint) float64 {
	total := 0.0
	for _, point := range tonnetzPoints {
		total += point.Weight
	}
	return total
}
