package chroma

import (
	"math"
	"sort"
)

// PitchClass represents a pitch class (0-11) with associated data
type PitchClass struct {
	Class      int     // Pitch class number (0=C, 1=C#, ..., 11=B)
	Name       string  // Pitch class name
	Energy     float64 // Total energy in this pitch class
	Salience   float64 // Salience measure (prominence)
	Confidence float64 // Confidence of detection
}

// PitchClassProfile represents a pitch class distribution
type PitchClassProfile struct {
	Profile    []float64 // 12-element pitch class distribution
	Entropy    float64   // Entropy of the distribution
	Centroid   float64   // Weighted centroid of distribution
	Spread     float64   // Spread around centroid
	Uniformity float64   // How uniform the distribution is
}

// PitchClassAnalyzer provides pitch class profiling and analysis
type PitchClassAnalyzer struct {
	pitchClassNames []string
}

// NewPitchClassAnalyzer creates a new pitch class analyzer
func NewPitchClassAnalyzer() *PitchClassAnalyzer {
	return &PitchClassAnalyzer{
		pitchClassNames: []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"},
	}
}

// CreateProfile creates a pitch class profile from chromagram
func (pca *PitchClassAnalyzer) CreateProfile(chromagram [][]float64) *PitchClassProfile {
	if len(chromagram) == 0 || len(chromagram[0]) != 12 {
		return &PitchClassProfile{
			Profile: make([]float64, 12),
		}
	}

	// Sum across all time frames
	profile := make([]float64, 12)
	for t := range chromagram {
		for pc := 0; pc < 12; pc++ {
			profile[pc] += chromagram[t][pc]
		}
	}

	// Normalize to sum = 1
	pca.normalizeProfile(profile)

	// Calculate derived measures
	entropy := pca.calculateEntropy(profile)
	centroid := pca.calculateCentroid(profile)
	spread := pca.calculateSpread(profile, centroid)
	uniformity := pca.calculateUniformity(profile)

	return &PitchClassProfile{
		Profile:    profile,
		Entropy:    entropy,
		Centroid:   centroid,
		Spread:     spread,
		Uniformity: uniformity,
	}
}

// ExtractPitchClasses extracts prominent pitch classes from chromagram
func (pca *PitchClassAnalyzer) ExtractPitchClasses(chromagram [][]float64, threshold float64) []PitchClass {
	profile := pca.CreateProfile(chromagram)
	var pitchClasses []PitchClass

	for pc := 0; pc < 12; pc++ {
		if profile.Profile[pc] >= threshold {
			// Calculate salience as relative prominence
			salience := pca.calculateSalience(profile.Profile, pc)

			// Simple confidence based on energy and salience
			confidence := math.Min(1.0, profile.Profile[pc]*salience)

			pitchClass := PitchClass{
				Class:      pc,
				Name:       pca.pitchClassNames[pc],
				Energy:     profile.Profile[pc],
				Salience:   salience,
				Confidence: confidence,
			}
			pitchClasses = append(pitchClasses, pitchClass)
		}
	}

	// Sort by energy (descending)
	sort.Slice(pitchClasses, func(i, j int) bool {
		return pitchClasses[i].Energy > pitchClasses[j].Energy
	})

	return pitchClasses
}

// ComparePitchClassProfiles compares two pitch class profiles
func (pca *PitchClassAnalyzer) ComparePitchClassProfiles(profile1, profile2 []float64) map[string]float64 {
	metrics := make(map[string]float64)

	if len(profile1) != 12 || len(profile2) != 12 {
		return metrics
	}

	// Cosine similarity
	metrics["cosine_similarity"] = pca.cosineSimilarity(profile1, profile2)

	// Euclidean distance
	metrics["euclidean_distance"] = pca.euclideanDistance(profile1, profile2)

	// Manhattan distance
	metrics["manhattan_distance"] = pca.manhattanDistance(profile1, profile2)

	// Correlation coefficient
	metrics["correlation"] = pca.correlation(profile1, profile2)

	// Kullback-Leibler divergence (symmetric)
	kl1 := pca.klDivergence(profile1, profile2)
	kl2 := pca.klDivergence(profile2, profile1)
	metrics["kl_divergence"] = (kl1 + kl2) / 2.0

	return metrics
}

// AnalyzeKeyRelationships analyzes relationships between pitch classes
func (pca *PitchClassAnalyzer) AnalyzeKeyRelationships(profile []float64) map[string]float64 {
	analysis := make(map[string]float64)

	if len(profile) != 12 {
		return analysis
	}

	// Circle of fifths relationships
	analysis["fifth_correlation"] = pca.calculateCircleOfFifthsCorrelation(profile)

	// Tonic-dominant relationship (perfect fifth)
	analysis["tonic_dominant_strength"] = pca.analyzeTonicDominant(profile)

	// Major/minor triadic content
	analysis["major_triad_strength"] = pca.analyzeTriadicContent(profile, []int{0, 4, 7}) // Major triad intervals
	analysis["minor_triad_strength"] = pca.analyzeTriadicContent(profile, []int{0, 3, 7}) // Minor triad intervals

	// Diatonic vs chromatic content
	analysis["diatonic_strength"] = pca.analyzeDiatonicContent(profile)

	return analysis
}

// TransposeProfile transposes a pitch class profile by semitones
func (pca *PitchClassAnalyzer) TransposeProfile(profile []float64, semitones int) []float64 {
	if len(profile) != 12 {
		return profile
	}

	transposed := make([]float64, 12)
	for i := 0; i < 12; i++ {
		newIndex := (i + semitones + 12) % 12
		transposed[newIndex] = profile[i]
	}

	return transposed
}

// FindBestTransposition finds the transposition that best matches a template
func (pca *PitchClassAnalyzer) FindBestTransposition(profile, template []float64) (int, float64) {
	if len(profile) != 12 || len(template) != 12 {
		return 0, 0.0
	}

	bestTransposition := 0
	bestCorrelation := -1.0

	for t := 0; t < 12; t++ {
		transposed := pca.TransposeProfile(template, t)
		correlation := pca.correlation(profile, transposed)

		if correlation > bestCorrelation {
			bestCorrelation = correlation
			bestTransposition = t
		}
	}

	return bestTransposition, bestCorrelation
}

// Helper functions

// normalizeProfile normalizes a profile to sum = 1
func (pca *PitchClassAnalyzer) normalizeProfile(profile []float64) {
	sum := 0.0
	for _, val := range profile {
		sum += val
	}

	if sum > 1e-10 {
		for i := range profile {
			profile[i] /= sum
		}
	}
}

// calculateEntropy calculates Shannon entropy of pitch class distribution
func (pca *PitchClassAnalyzer) calculateEntropy(profile []float64) float64 {
	entropy := 0.0
	for _, prob := range profile {
		if prob > 1e-10 {
			entropy -= prob * math.Log2(prob)
		}
	}
	return entropy
}

// calculateCentroid calculates weighted centroid of pitch class distribution
func (pca *PitchClassAnalyzer) calculateCentroid(profile []float64) float64 {
	// Use circular mean for pitch classes
	sumSin := 0.0
	sumCos := 0.0

	for pc, weight := range profile {
		angle := 2.0 * math.Pi * float64(pc) / 12.0
		sumSin += weight * math.Sin(angle)
		sumCos += weight * math.Cos(angle)
	}

	centroidAngle := math.Atan2(sumSin, sumCos)
	if centroidAngle < 0 {
		centroidAngle += 2.0 * math.Pi
	}

	return centroidAngle * 12.0 / (2.0 * math.Pi)
}

// calculateSpread calculates spread around centroid
func (pca *PitchClassAnalyzer) calculateSpread(profile []float64, centroid float64) float64 {
	sumWeightedDistance := 0.0
	totalWeight := 0.0

	for pc, weight := range profile {
		// Circular distance
		distance := math.Min(
			math.Abs(float64(pc)-centroid),
			12.0-math.Abs(float64(pc)-centroid),
		)
		sumWeightedDistance += weight * distance * distance
		totalWeight += weight
	}

	if totalWeight > 1e-10 {
		return math.Sqrt(sumWeightedDistance / totalWeight)
	}
	return 0.0
}

// calculateUniformity calculates how uniform the distribution is
func (pca *PitchClassAnalyzer) calculateUniformity(profile []float64) float64 {
	// Uniformity = 1 - normalized standard deviation
	mean := 1.0 / 12.0 // Uniform distribution mean
	variance := 0.0

	for _, val := range profile {
		diff := val - mean
		variance += diff * diff
	}
	variance /= 12.0

	maxVariance := mean * mean // Maximum possible variance
	if maxVariance > 1e-10 {
		return 1.0 - math.Sqrt(variance/maxVariance)
	}
	return 1.0
}

// calculateSalience calculates salience of a pitch class
func (pca *PitchClassAnalyzer) calculateSalience(profile []float64, pc int) float64 {
	// Salience as ratio to average of neighboring pitch classes
	neighbors := []int{(pc + 11) % 12, (pc + 1) % 12}
	neighborAvg := 0.0
	for _, neighbor := range neighbors {
		neighborAvg += profile[neighbor]
	}
	neighborAvg /= float64(len(neighbors))

	if neighborAvg > 1e-10 {
		return profile[pc] / neighborAvg
	}
	return profile[pc]
}

// Distance and similarity functions

// cosineSimilarity calculates cosine similarity
func (pca *PitchClassAnalyzer) cosineSimilarity(a, b []float64) float64 {
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA > 1e-10 && normB > 1e-10 {
		return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	}
	return 0.0
}

// euclideanDistance calculates Euclidean distance
func (pca *PitchClassAnalyzer) euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// manhattanDistance calculates Manhattan distance
func (pca *PitchClassAnalyzer) manhattanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

// correlation calculates Pearson correlation
func (pca *PitchClassAnalyzer) correlation(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0.0
	}

	meanA := 0.0
	meanB := 0.0
	for i := range a {
		meanA += a[i]
		meanB += b[i]
	}
	meanA /= float64(n)
	meanB /= float64(n)

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

	denominator := math.Sqrt(sumSqA * sumSqB)
	if denominator > 1e-10 {
		return numerator / denominator
	}
	return 0.0
}

// klDivergence calculates Kullback-Leibler divergence
func (pca *PitchClassAnalyzer) klDivergence(p, q []float64) float64 {
	kl := 0.0
	for i := range p {
		if p[i] > 1e-10 && q[i] > 1e-10 {
			kl += p[i] * math.Log(p[i]/q[i])
		}
	}
	return kl
}

// Musical analysis functions

// calculateCircleOfFifthsCorrelation analyzes circle of fifths relationships
func (pca *PitchClassAnalyzer) calculateCircleOfFifthsCorrelation(profile []float64) float64 {
	// Create circle of fifths template (C-G-D-A-E-B-F#-C#-G#-D#-A#-F)
	fifthsOrder := []int{0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5}

	// Calculate correlation between profile and fifths ordering
	orderedProfile := make([]float64, 12)
	for i, pc := range fifthsOrder {
		orderedProfile[i] = profile[pc]
	}

	// Compare with declining exponential (strong fifths relationship)
	template := make([]float64, 12)
	for i := range template {
		template[i] = math.Exp(-float64(i) * 0.3)
	}

	return pca.correlation(orderedProfile, template)
}

// analyzeTonicDominant analyzes tonic-dominant relationships
func (pca *PitchClassAnalyzer) analyzeTonicDominant(profile []float64) float64 {
	maxStrength := 0.0

	// Test all possible tonic-dominant pairs
	for tonic := 0; tonic < 12; tonic++ {
		dominant := (tonic + 7) % 12
		strength := profile[tonic] * profile[dominant]
		if strength > maxStrength {
			maxStrength = strength
		}
	}

	return maxStrength
}

// analyzeTriadicContent analyzes triadic content
func (pca *PitchClassAnalyzer) analyzeTriadicContent(profile []float64, intervals []int) float64 {
	maxStrength := 0.0

	// Test all transpositions
	for root := 0; root < 12; root++ {
		strength := 1.0
		for _, interval := range intervals {
			pc := (root + interval) % 12
			strength *= profile[pc]
		}
		strength = math.Pow(strength, 1.0/float64(len(intervals))) // Geometric mean

		if strength > maxStrength {
			maxStrength = strength
		}
	}

	return maxStrength
}

// analyzeDiatonicContent analyzes diatonic vs chromatic content
func (pca *PitchClassAnalyzer) analyzeDiatonicContent(profile []float64) float64 {
	// Major scale template: C D E F G A B (semitones: 0 2 4 5 7 9 11)
	diatonicPCs := []int{0, 2, 4, 5, 7, 9, 11}
	chromaticPCs := []int{1, 3, 6, 8, 10}

	maxDiatonicStrength := 0.0

	// Test all transpositions
	for root := 0; root < 12; root++ {
		diatonicEnergy := 0.0
		chromaticEnergy := 0.0

		for _, interval := range diatonicPCs {
			pc := (root + interval) % 12
			diatonicEnergy += profile[pc]
		}

		for _, interval := range chromaticPCs {
			pc := (root + interval) % 12
			chromaticEnergy += profile[pc]
		}

		totalEnergy := diatonicEnergy + chromaticEnergy
		if totalEnergy > 1e-10 {
			diatonicStrength := diatonicEnergy / totalEnergy
			if diatonicStrength > maxDiatonicStrength {
				maxDiatonicStrength = diatonicStrength
			}
		}
	}

	return maxDiatonicStrength
}
