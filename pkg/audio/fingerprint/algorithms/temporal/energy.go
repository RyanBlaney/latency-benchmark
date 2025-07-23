package temporal

import (
	"math"
	"sort"
)

// Energy computes various energy-based temporal features
// This focuses on temporal energy patterns, not duplicating spectral algorithms
type Energy struct {
	frameSize  int
	hopSize    int
	sampleRate int
}

// NewEnergy creates a new energy calculator
func NewEnergy(frameSize, hopSize, sampleRate int) *Energy {
	return &Energy{
		frameSize:  frameSize,
		hopSize:    hopSize,
		sampleRate: sampleRate,
	}
}

// ComputeShortTimeEnergy calculates short-time energy for overlapping frames
// This is the critical function for your alignment needs
func (e *Energy) ComputeShortTimeEnergy(signal []float64) []float64 {
	if len(signal) < e.frameSize || e.hopSize <= 0 || e.frameSize <= 0 {
		return []float64{}
	}

	numFrames := (len(signal)-e.frameSize)/e.hopSize + 1
	energies := make([]float64, numFrames)

	for i := range numFrames {
		startIdx := i * e.hopSize
		endIdx := startIdx + e.frameSize

		if endIdx > len(signal) {
			break
		}

		// Calculate RMS energy for this frame
		sumSquares := 0.0
		for j := startIdx; j < endIdx; j++ {
			sumSquares += signal[j] * signal[j]
		}
		energies[i] = math.Sqrt(sumSquares / float64(e.frameSize))
	}

	return energies
}

// ComputeLogEnergy calculates log energy in dB scale
func (e *Energy) ComputeLogEnergy(signal []float64, floor float64) []float64 {
	energies := e.ComputeShortTimeEnergy(signal)
	logEnergies := make([]float64, len(energies))

	for i, energy := range energies {
		if energy < floor {
			energy = floor
		}
		logEnergies[i] = 20.0 * math.Log10(energy)
	}

	return logEnergies
}

// ComputeEnergyEntropy calculates energy entropy for texture analysis
// Useful for distinguishing speech from music
func (e *Energy) ComputeEnergyEntropy(energies []float64) float64 {
	if len(energies) == 0 {
		return 0.0
	}

	// Normalize energies to probabilities
	totalEnergy := 0.0
	for _, energy := range energies {
		totalEnergy += energy
	}

	if totalEnergy == 0.0 {
		return 0.0
	}

	entropy := 0.0
	for _, energy := range energies {
		if energy > 0.0 {
			prob := energy / totalEnergy
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

// ComputeEnergyVariance calculates energy variance
// High variance indicates dynamic content (speech), low variance indicates steady content
func (e *Energy) ComputeEnergyVariance(energies []float64) float64 {
	if len(energies) < 2 {
		return 0.0
	}

	// Calculate mean
	mean := 0.0
	for _, energy := range energies {
		mean += energy
	}
	mean /= float64(len(energies))

	// Calculate variance
	variance := 0.0
	for _, energy := range energies {
		diff := energy - mean
		variance += diff * diff
	}
	variance /= float64(len(energies) - 1)

	return variance
}

// ComputeEnergyDerivative calculates first derivative of energy
// Useful for onset detection and transient analysis
func (e *Energy) ComputeEnergyDerivative(energies []float64) []float64 {
	if len(energies) < 2 {
		return []float64{}
	}

	derivative := make([]float64, len(energies)-1)
	for i := range len(derivative) {
		derivative[i] = energies[i+1] - energies[i]
	}

	return derivative
}

// ComputeEnergyRatio calculates ratio between energy bands or frames
func (e *Energy) ComputeEnergyRatio(energies1, energies2 []float64) []float64 {
	minLen := len(energies1)
	minLen = min(minLen, len(energies2))

	if minLen == 0 {
		return []float64{}
	}

	ratios := make([]float64, minLen)
	for i := 0; i < minLen; i++ {
		if energies2[i] > 1e-10 {
			ratios[i] = energies1[i] / energies2[i]
		} else {
			ratios[i] = 0.0
		}
	}

	return ratios
}

// ComputeLoudnessRange calculates loudness range (EBU R128 style)
func (e *Energy) ComputeLoudnessRange(signal []float64) float64 {
	if len(signal) == 0 || e.sampleRate <= 0 {
		return 0.0
	}

	// Calculate momentary loudness values (400ms windows)
	windowSize := int(0.4 * float64(e.sampleRate)) // 400ms
	hopSize := windowSize / 4                      // 25% overlap
	if hopSize <= 0 {
		hopSize = 1
	}

	// Temporarily adjust parameters for loudness calculation
	origFrameSize := e.frameSize
	origHopSize := e.hopSize
	e.frameSize = windowSize
	e.hopSize = hopSize

	loudnessValues := e.ComputeShortTimeEnergy(signal)

	// Restore original parameters
	e.frameSize = origFrameSize
	e.hopSize = origHopSize

	if len(loudnessValues) == 0 {
		return 0.0
	}

	// Convert to loudness units (simplified)
	for i := range loudnessValues {
		if loudnessValues[i] > 0 {
			loudnessValues[i] = -0.691 + 10.0*math.Log10(loudnessValues[i]*loudnessValues[i])
		} else {
			loudnessValues[i] = -70.0 // Silence threshold
		}
	}

	// Calculate loudness range (10th to 95th percentile)
	return e.calculatePercentileRange(loudnessValues, 0.10, 0.95)
}

// calculatePercentileRange calculates range between percentiles
func (e *Energy) calculatePercentileRange(values []float64, lowPercentile, highPercentile float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Sort values for percentile calculation
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	// Calculate percentile indices
	lowIdx := int(lowPercentile * float64(len(sorted)-1))
	highIdx := int(highPercentile * float64(len(sorted)-1))

	lowValue := sorted[lowIdx]
	highValue := sorted[highIdx]

	// Return range in dB
	if lowValue <= 0.0 {
		lowValue = 1e-10 // Avoid log(0)
	}
	if highValue <= 0.0 {
		return 0.0
	}

	return 20.0 * math.Log10(highValue/lowValue)
}

// ComputePeakEnergy finds peak energy values and their positions
func (e *Energy) ComputePeakEnergy(energies []float64, threshold float64) ([]float64, []int) {
	if len(energies) < 3 {
		return []float64{}, []int{}
	}

	var peaks []float64
	var positions []int

	for i := 1; i < len(energies)-1; i++ {
		// Check if it's a local maximum above threshold
		if energies[i] > energies[i-1] &&
			energies[i] > energies[i+1] &&
			energies[i] >= threshold {
			peaks = append(peaks, energies[i])
			positions = append(positions, i)
		}
	}

	return peaks, positions
}

// ComputeEnergyStatistics calculates comprehensive energy statistics
func (e *Energy) ComputeEnergyStatistics(signal []float64) map[string]float64 {
	stats := make(map[string]float64)

	energies := e.ComputeShortTimeEnergy(signal)
	if len(energies) == 0 {
		return stats
	}

	// Basic statistics
	mean := 0.0
	for _, energy := range energies {
		mean += energy
	}
	mean /= float64(len(energies))

	variance := e.ComputeEnergyVariance(energies)
	entropy := e.ComputeEnergyEntropy(energies)

	// Peak statistics
	maxEnergy := energies[0]
	minEnergy := energies[0]
	for _, energy := range energies {
		if energy > maxEnergy {
			maxEnergy = energy
		}
		if energy < minEnergy {
			minEnergy = energy
		}
	}

	stats["mean_energy"] = mean
	stats["energy_variance"] = variance
	stats["energy_entropy"] = entropy
	stats["max_energy"] = maxEnergy
	stats["min_energy"] = minEnergy
	stats["energy_range"] = maxEnergy - minEnergy
	stats["loudness_range"] = e.ComputeLoudnessRange(signal)

	return stats
}
