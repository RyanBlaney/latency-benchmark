package tonal

import (
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/chroma"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// KeyProfile represents different key detection profiles
type KeyProfile int

const (
	KeyProfileKrumhansl KeyProfile = iota
	KeyProfileTemperley
	KeyProfileShaath
	KeyProfileEDMA
	KeyProfileBgate
	KeyProfileDiatonic
	KeyProfileTonicTriad
)

// KeyMode represents major or minor mode
type KeyMode int

const (
	KeyModeMajor KeyMode = iota
	KeyModeMinor
)

// KeyEstimationMethod defines different key estimation approaches
type KeyEstimationMethod int

const (
	KeyMethodProfile KeyEstimationMethod = iota
	KeyMethodCorrelation
	KeyMethodBayesian
	KeyMethodHMM
	KeyMethodDeepLearning
)

// KeyCandidate represents a potential key with confidence
type KeyCandidate struct {
	Key        int     `json:"key"`        // Key number (0=C, 1=C#, ..., 11=B)
	Mode       KeyMode `json:"mode"`       // Major or Minor
	KeyName    string  `json:"key_name"`   // Human-readable key name
	Confidence float64 `json:"confidence"` // Confidence score (0-1)
	Strength   float64 `json:"strength"`   // Key strength measure
	Profile    string  `json:"profile"`    // Profile used for detection
}

// KeyEstimationResult contains comprehensive key estimation results
type KeyEstimationResult struct {
	// Primary key information
	Key        int     `json:"key"`        // Best key estimate (0-11)
	Mode       KeyMode `json:"mode"`       // Major or Minor
	KeyName    string  `json:"key_name"`   // Human-readable name (e.g., "C major")
	Confidence float64 `json:"confidence"` // Overall confidence (0-1)
	Strength   float64 `json:"strength"`   // Key strength measure

	// Multiple key candidates
	Candidates []KeyCandidate `json:"candidates"`

	// Analysis details
	ChromaVector      []float64 `json:"chroma_vector"`      // Input chroma profile
	KeyProfile        string    `json:"key_profile"`        // Profile type used
	Method            string    `json:"method"`             // Estimation method
	CorrelationScores []float64 `json:"correlation_scores"` // Correlation with each key

	// Quality metrics
	Clarity   float64 `json:"clarity"`   // Key clarity measure
	Ambiguity float64 `json:"ambiguity"` // Key ambiguity measure
	Stability float64 `json:"stability"` // Temporal stability

	// Additional analysis
	RelatedKeys []KeyCandidate `json:"related_keys"` // Closely related keys
	Modulations []int          `json:"modulations"`  // Detected modulation points
	Tonality    float64        `json:"tonality"`     // Overall tonality strength

	// Computational details
	ProcessTime float64 `json:"process_time"` // Processing time in ms
	HPCPSize    int     `json:"hpcp_size"`    // HPCP vector size used
}

// KeyEstimationParams contains parameters for key estimation
type KeyEstimationParams struct {
	Method          KeyEstimationMethod `json:"method"`
	Profile         KeyProfile          `json:"profile"`
	HPCPSize        int                 `json:"hpcp_size"`        // Size of HPCP vector (12, 24, 36)
	UseHarmonics    bool                `json:"use_harmonics"`    // Consider harmonic content
	WeightHarmonics bool                `json:"weight_harmonics"` // Weight harmonics differently

	// Preprocessing
	NormalizeChroma bool `json:"normalize_chroma"` // Normalize input chroma
	RemoveMean      bool `json:"remove_mean"`      // Remove mean from chroma
	UsePolyphony    bool `json:"use_polyphony"`    // Consider polyphonic content

	// Analysis parameters
	MinConfidence   float64 `json:"min_confidence"`   // Minimum confidence threshold
	MaxCandidates   int     `json:"max_candidates"`   // Maximum candidates to return
	ProfileStrength float64 `json:"profile_strength"` // Profile weighting strength

	// Temporal analysis
	UseTemporalSmoothing bool `json:"use_temporal_smoothing"` // Enable temporal smoothing
	TemporalWindow       int  `json:"temporal_window"`        // Frames for temporal analysis

	// Advanced options
	UseDetuningCorrection  bool `json:"use_detuning_correction"` // Correct for detuning
	TranspositionInvariant bool `json:"transposition_invariant"` // Find best transposition
	BinaryMode             bool `json:"binary_mode"`             // Use binary chroma vectors
}

// KeyProfileTemplate contains template for key profile
type KeyProfileTemplate struct {
	MajorProfile []float64 `json:"major_profile"`
	MinorProfile []float64 `json:"minor_profile"`
	Name         string    `json:"name"`
	Description  string    `json:"description"`
}

// KeyEstimator implements musical key estimation algorithms
type KeyEstimator struct {
	params         KeyEstimationParams
	chromaAnalyzer *chroma.ChromaVectorAnalyzer
	hpcp           *chroma.HPCP

	// Key profiles
	profiles map[KeyProfile]*KeyProfileTemplate

	// Temporal tracking
	keyHistory      []KeyCandidate
	stabilityWindow []float64

	// Internal state
	initialized bool
}

// NewKeyEstimator creates a new key estimator with default parameters
func NewKeyEstimator(sampleRate int) *KeyEstimator {
	return &KeyEstimator{
		params: KeyEstimationParams{
			Method:                 KeyMethodProfile,
			Profile:                KeyProfileKrumhansl,
			HPCPSize:               12,
			UseHarmonics:           true,
			WeightHarmonics:        false,
			NormalizeChroma:        true,
			RemoveMean:             false,
			UsePolyphony:           true,
			MinConfidence:          0.5,
			MaxCandidates:          5,
			ProfileStrength:        1.0,
			UseTemporalSmoothing:   false,
			TemporalWindow:         5,
			UseDetuningCorrection:  false,
			TranspositionInvariant: false,
			BinaryMode:             false,
		},
		chromaAnalyzer:  chroma.NewChromaVectorAnalyzer(),
		hpcp:            chroma.NewHPCP(sampleRate),
		keyHistory:      make([]KeyCandidate, 0),
		stabilityWindow: make([]float64, 0),
	}
}

// NewKeyEstimatorWithParams creates a key estimator with custom parameters
func NewKeyEstimatorWithParams(sampleRate int, params KeyEstimationParams) *KeyEstimator {
	ke := &KeyEstimator{
		params:          params,
		chromaAnalyzer:  chroma.NewChromaVectorAnalyzer(),
		hpcp:            chroma.NewHPCP(sampleRate),
		keyHistory:      make([]KeyCandidate, 0),
		stabilityWindow: make([]float64, 0),
	}

	ke.Initialize()
	return ke
}

// Initialize sets up the key estimator
func (ke *KeyEstimator) Initialize() {
	if ke.initialized {
		return
	}

	// Initialize key profiles
	ke.profiles = make(map[KeyProfile]*KeyProfileTemplate)
	ke.initializeKeyProfiles()

	ke.initialized = true
}

// EstimateKey estimates musical key from chroma vector
func (ke *KeyEstimator) EstimateKey(chromaVector chroma.ChromaVector) KeyEstimationResult {
	if !ke.initialized {
		ke.Initialize()
	}

	startTime := ke.getCurrentTime()

	// Preprocess chroma vector
	processedChroma := ke.preprocessChroma(chromaVector)

	// Estimate key using selected method
	var result KeyEstimationResult
	switch ke.params.Method {
	case KeyMethodProfile:
		result = ke.estimateKeyProfile(processedChroma)
	case KeyMethodCorrelation:
		result = ke.estimateKeyCorrelation(processedChroma)
	case KeyMethodBayesian:
		result = ke.estimateKeyBayesian(processedChroma)
	default:
		result = ke.estimateKeyProfile(processedChroma)
	}

	// Post-process results
	result = ke.postProcessResult(result, processedChroma)

	// Update temporal tracking
	ke.updateTemporalTracking(result)

	// Set computational details
	result.Method = ke.getMethodName(ke.params.Method)
	result.KeyProfile = ke.getProfileName(ke.params.Profile)
	result.ProcessTime = ke.getCurrentTime() - startTime
	result.HPCPSize = len(processedChroma.Values)
	result.ChromaVector = processedChroma.Values

	return result
}

// EstimateKeyFromHPCP estimates key directly from HPCP result
func (ke *KeyEstimator) EstimateKeyFromHPCP(hpcpResult chroma.HPCPResult) KeyEstimationResult {
	// Convert HPCP to ChromaVector
	chromaVec := chroma.ChromaVector{
		Values:     hpcpResult.HPCP,
		Size:       hpcpResult.Size,
		Normalized: true,
		Energy:     hpcpResult.Energy,
		Entropy:    hpcpResult.Entropy,
	}

	return ke.EstimateKey(chromaVec)
}

// EstimateKeySequence estimates key from a sequence of chroma vectors
func (ke *KeyEstimator) EstimateKeySequence(chromaSequence []chroma.ChromaVector) KeyEstimationResult {
	if len(chromaSequence) == 0 {
		return KeyEstimationResult{}
	}

	// Compute average chroma profile
	avgChroma := ke.computeAverageChroma(chromaSequence)

	// Estimate key from average
	result := ke.EstimateKey(avgChroma)

	// Analyze temporal stability
	result.Stability = ke.analyzeTemporalStability(chromaSequence)

	// Detect modulations if requested
	if len(chromaSequence) > 10 {
		result.Modulations = ke.detectModulations(chromaSequence)
	}

	return result
}

// preprocessChroma preprocesses the input chroma vector
func (ke *KeyEstimator) preprocessChroma(chromaVector chroma.ChromaVector) chroma.ChromaVector {
	processed := chromaVector

	// Resize to target HPCP size if needed
	if len(chromaVector.Values) != ke.params.HPCPSize {
		processed = ke.resizeChromaVector(chromaVector, ke.params.HPCPSize)
	}

	// Normalize if requested
	if ke.params.NormalizeChroma {
		processed = ke.chromaAnalyzer.Normalize(processed, common.Energy)
	}

	// Remove mean if requested
	if ke.params.RemoveMean {
		processed = ke.removeMean(processed)
	}

	// Apply binary thresholding if requested
	if ke.params.BinaryMode {
		processed = ke.applyBinaryThreshold(processed)
	}

	return processed
}

// estimateKeyProfile estimates key using profile correlation method
func (ke *KeyEstimator) estimateKeyProfile(chromaVector chroma.ChromaVector) KeyEstimationResult {
	profile := ke.profiles[ke.params.Profile]
	if profile == nil {
		// Fallback to default profile
		profile = ke.profiles[KeyProfileKrumhansl]
	}

	candidates := make([]KeyCandidate, 0)
	correlationScores := make([]float64, 24) // 12 major + 12 minor keys

	// Test all 24 keys (12 major + 12 minor)
	for key := 0; key < 12; key++ {
		// Major key
		majorCorr := ke.correlateWithProfile(chromaVector.Values, profile.MajorProfile, key)
		majorIdx := key
		correlationScores[majorIdx] = majorCorr

		candidates = append(candidates, KeyCandidate{
			Key:        key,
			Mode:       KeyModeMajor,
			KeyName:    ke.getKeyName(key, KeyModeMajor),
			Confidence: majorCorr,
			Strength:   majorCorr,
			Profile:    profile.Name,
		})

		// Minor key
		minorCorr := ke.correlateWithProfile(chromaVector.Values, profile.MinorProfile, key)
		minorIdx := key + 12
		correlationScores[minorIdx] = minorCorr

		candidates = append(candidates, KeyCandidate{
			Key:        key,
			Mode:       KeyModeMinor,
			KeyName:    ke.getKeyName(key, KeyModeMinor),
			Confidence: minorCorr,
			Strength:   minorCorr,
			Profile:    profile.Name,
		})
	}

	// Sort candidates by confidence
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Confidence > candidates[j].Confidence
	})

	// Limit candidates
	if len(candidates) > ke.params.MaxCandidates {
		candidates = candidates[:ke.params.MaxCandidates]
	}

	// Get best candidate
	best := candidates[0]

	// Calculate quality metrics
	clarity := ke.calculateClarity(correlationScores)
	ambiguity := ke.calculateAmbiguity(correlationScores)

	return KeyEstimationResult{
		Key:               best.Key,
		Mode:              best.Mode,
		KeyName:           best.KeyName,
		Confidence:        best.Confidence,
		Strength:          best.Strength,
		Candidates:        candidates,
		CorrelationScores: correlationScores,
		Clarity:           clarity,
		Ambiguity:         ambiguity,
		RelatedKeys:       ke.findRelatedKeys(candidates),
	}
}

// estimateKeyCorrelation estimates key using direct correlation
func (ke *KeyEstimator) estimateKeyCorrelation(chromaVector chroma.ChromaVector) KeyEstimationResult {
	// This would implement alternative correlation-based methods
	// For now, fallback to profile method
	return ke.estimateKeyProfile(chromaVector)
}

// estimateKeyBayesian estimates key using Bayesian approach
func (ke *KeyEstimator) estimateKeyBayesian(chromaVector chroma.ChromaVector) KeyEstimationResult {
	// This would implement Bayesian key estimation
	// For now, fallback to profile method
	return ke.estimateKeyProfile(chromaVector)
}

// correlateWithProfile correlates chroma vector with key profile
func (ke *KeyEstimator) correlateWithProfile(chroma, profile []float64, keyShift int) float64 {
	if len(chroma) != len(profile) {
		return 0.0
	}

	// Shift profile to match key
	shiftedProfile := make([]float64, len(profile))
	for i := range profile {
		shiftedIdx := (i + keyShift) % len(profile)
		shiftedProfile[i] = profile[shiftedIdx]
	}

	// Calculate correlation using stats package
	return stats.PearsonCorrelationFunc(chroma, shiftedProfile)
}

// initializeKeyProfiles initializes all key profile templates
func (ke *KeyEstimator) initializeKeyProfiles() {
	// Krumhansl-Schmuckler profiles (empirically derived)
	ke.profiles[KeyProfileKrumhansl] = &KeyProfileTemplate{
		MajorProfile: []float64{6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88},
		MinorProfile: []float64{6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17},
		Name:         "Krumhansl-Schmuckler",
		Description:  "Empirical profiles based on listener ratings",
	}

	// Temperley profiles (corpus-based)
	ke.profiles[KeyProfileTemperley] = &KeyProfileTemplate{
		MajorProfile: []float64{5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0},
		MinorProfile: []float64{5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0},
		Name:         "Temperley",
		Description:  "Statistical profiles from musical corpora",
	}

	// Shaath profiles (optimized for electronic music)
	ke.profiles[KeyProfileShaath] = &KeyProfileTemplate{
		MajorProfile: []float64{6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4},
		MinorProfile: []float64{6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 4.7, 4.0, 2.7, 3.4, 3.2},
		Name:         "Shaath",
		Description:  "Optimized for electronic dance music",
	}

	// EDMA profiles (electronic dance music analysis)
	ke.profiles[KeyProfileEDMA] = &KeyProfileTemplate{
		MajorProfile: []float64{17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122},
		MinorProfile: []float64{18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623},
		Name:         "EDMA",
		Description:  "Electronic Dance Music Analysis profiles",
	}

	// Bgate profiles (balanced gate)
	ke.profiles[KeyProfileBgate] = &KeyProfileTemplate{
		MajorProfile: []float64{16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57},
		MinorProfile: []float64{18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 6.24, 1.61},
		Name:         "Bgate",
		Description:  "Balanced gate profiles for modern music",
	}

	// Diatonic profiles (simple diatonic weights)
	ke.profiles[KeyProfileDiatonic] = &KeyProfileTemplate{
		MajorProfile: []float64{5.0, 0.0, 3.0, 0.0, 4.0, 3.5, 0.0, 4.5, 0.0, 3.0, 0.0, 2.0},
		MinorProfile: []float64{5.0, 0.0, 3.0, 3.5, 0.0, 3.5, 0.0, 4.5, 3.0, 0.0, 2.0, 0.0},
		Name:         "Diatonic",
		Description:  "Simple diatonic scale weights",
	}

	// Tonic Triad profiles (emphasize tonic chord)
	ke.profiles[KeyProfileTonicTriad] = &KeyProfileTemplate{
		MajorProfile: []float64{5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0},
		MinorProfile: []float64{5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0},
		Name:         "Tonic Triad",
		Description:  "Emphasizes tonic triad notes only",
	}
}

// Helper functions

func (ke *KeyEstimator) resizeChromaVector(chromaVector chroma.ChromaVector, targetSize int) chroma.ChromaVector {
	// Simple interpolation to resize chroma vector
	if len(chromaVector.Values) == targetSize {
		return chromaVector
	}

	resized := make([]float64, targetSize)
	ratio := float64(len(chromaVector.Values)) / float64(targetSize)

	for i := 0; i < targetSize; i++ {
		srcIdx := float64(i) * ratio
		if int(srcIdx) < len(chromaVector.Values) {
			resized[i] = chromaVector.Values[int(srcIdx)]
		}
	}

	return chroma.ChromaVector{
		Values:     resized,
		Size:       targetSize,
		Normalized: chromaVector.Normalized,
	}
}

func (ke *KeyEstimator) removeMean(chromaVector chroma.ChromaVector) chroma.ChromaVector {
	mean := common.Mean(chromaVector.Values)

	adjusted := make([]float64, len(chromaVector.Values))
	for i, val := range chromaVector.Values {
		adjusted[i] = val - mean
	}

	result := chromaVector
	result.Values = adjusted
	return result
}

func (ke *KeyEstimator) applyBinaryThreshold(chromaVector chroma.ChromaVector) chroma.ChromaVector {
	threshold := common.Mean(chromaVector.Values)

	binary := make([]float64, len(chromaVector.Values))
	for i, val := range chromaVector.Values {
		if val > threshold {
			binary[i] = 1.0
		} else {
			binary[i] = 0.0
		}
	}

	result := chromaVector
	result.Values = binary
	return result
}

func (ke *KeyEstimator) calculateClarity(scores []float64) float64 {
	if len(scores) < 2 {
		return 0.0
	}

	// Sort scores in descending order
	sorted := make([]float64, len(scores))
	copy(sorted, scores)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))

	// Clarity = (best - second_best) / best
	if sorted[0] > 0 {
		return (sorted[0] - sorted[1]) / sorted[0]
	}

	return 0.0
}

func (ke *KeyEstimator) calculateAmbiguity(scores []float64) float64 {
	// Ambiguity = entropy of normalized scores
	sum := 0.0
	for _, score := range scores {
		if score > 0 {
			sum += score
		}
	}

	if sum == 0 {
		return 0.0
	}

	entropy := 0.0
	for _, score := range scores {
		if score > 0 {
			prob := score / sum
			entropy -= prob * math.Log2(prob)
		}
	}

	// Normalize entropy
	maxEntropy := math.Log2(float64(len(scores)))
	return entropy / maxEntropy
}

func (ke *KeyEstimator) findRelatedKeys(candidates []KeyCandidate) []KeyCandidate {
	if len(candidates) < 2 {
		return []KeyCandidate{}
	}

	// Return top related keys (excluding the best one)
	related := make([]KeyCandidate, 0)
	for i := 1; i < len(candidates) && i < 4; i++ {
		related = append(related, candidates[i])
	}

	return related
}

func (ke *KeyEstimator) computeAverageChroma(chromaSequence []chroma.ChromaVector) chroma.ChromaVector {
	if len(chromaSequence) == 0 {
		return chroma.ChromaVector{}
	}

	size := chromaSequence[0].Size
	avgValues := make([]float64, size)

	for _, chromaVec := range chromaSequence {
		for i := 0; i < size && i < len(chromaVec.Values); i++ {
			avgValues[i] += chromaVec.Values[i]
		}
	}

	// Normalize by count
	count := float64(len(chromaSequence))
	for i := range avgValues {
		avgValues[i] /= count
	}

	return ke.chromaAnalyzer.CreateChromaVector(avgValues)
}

func (ke *KeyEstimator) analyzeTemporalStability(chromaSequence []chroma.ChromaVector) float64 {
	if len(chromaSequence) < 3 {
		return 0.0
	}

	// Estimate key for each frame and measure stability
	keyEstimates := make([]int, len(chromaSequence))

	for i, chromaVec := range chromaSequence {
		result := ke.EstimateKey(chromaVec)
		keyEstimates[i] = result.Key*2 + int(result.Mode) // Combined key+mode
	}

	// Calculate stability as consistency of estimates
	mode := ke.findMode(keyEstimates)
	consistent := 0

	for _, estimate := range keyEstimates {
		if estimate == mode {
			consistent++
		}
	}

	return float64(consistent) / float64(len(keyEstimates))
}

func (ke *KeyEstimator) detectModulations(chromaSequence []chroma.ChromaVector) []int {
	// Simple modulation detection - look for key changes
	modulations := make([]int, 0)
	windowSize := 10

	if len(chromaSequence) < windowSize*2 {
		return modulations
	}

	prevKey := -1
	for i := windowSize; i < len(chromaSequence)-windowSize; i++ {
		// Analyze window around this point
		start := i - windowSize/2
		end := i + windowSize/2

		if start >= 0 && end < len(chromaSequence) {
			windowChroma := chromaSequence[start:end]
			avgChroma := ke.computeAverageChroma(windowChroma)
			result := ke.EstimateKey(avgChroma)

			currentKey := result.Key*2 + int(result.Mode)

			if prevKey != -1 && currentKey != prevKey && result.Confidence > 0.7 {
				modulations = append(modulations, i)
			}

			prevKey = currentKey
		}
	}

	return modulations
}

func (ke *KeyEstimator) findMode(values []int) int {
	counts := make(map[int]int)
	for _, val := range values {
		counts[val]++
	}

	maxCount := 0
	mode := 0
	for val, count := range counts {
		if count > maxCount {
			maxCount = count
			mode = val
		}
	}

	return mode
}

func (ke *KeyEstimator) postProcessResult(result KeyEstimationResult, chromaVector chroma.ChromaVector) KeyEstimationResult {
	// Calculate tonality strength
	result.Tonality = ke.calculateTonality(chromaVector)

	// Apply confidence thresholds
	if result.Confidence < ke.params.MinConfidence {
		result.Confidence = 0.0
		result.KeyName = "Unknown"
	}

	return result
}

func (ke *KeyEstimator) calculateTonality(chromaVector chroma.ChromaVector) float64 {
	// Simple tonality measure based on chroma vector entropy
	entropy := ke.chromaAnalyzer.ComputeStats(chromaVector).Uniformity
	return 1.0 - entropy // Higher uniformity = lower tonality
}

func (ke *KeyEstimator) updateTemporalTracking(result KeyEstimationResult) {
	// Add to history
	if len(result.Candidates) > 0 {
		ke.keyHistory = append(ke.keyHistory, result.Candidates[0])
	}

	// Keep only recent history
	maxHistory := 20
	if len(ke.keyHistory) > maxHistory {
		ke.keyHistory = ke.keyHistory[len(ke.keyHistory)-maxHistory:]
	}
}

func (ke *KeyEstimator) getKeyName(key int, mode KeyMode) string {
	keyNames := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

	name := keyNames[key%12]
	if mode == KeyModeMajor {
		return name + " major"
	} else {
		return name + " minor"
	}
}

func (ke *KeyEstimator) getMethodName(method KeyEstimationMethod) string {
	switch method {
	case KeyMethodProfile:
		return "Profile Correlation"
	case KeyMethodCorrelation:
		return "Direct Correlation"
	case KeyMethodBayesian:
		return "Bayesian"
	case KeyMethodHMM:
		return "Hidden Markov Model"
	case KeyMethodDeepLearning:
		return "Deep Learning"
	default:
		return "Unknown"
	}
}

func (ke *KeyEstimator) getProfileName(profile KeyProfile) string {
	switch profile {
	case KeyProfileKrumhansl:
		return "Krumhansl-Schmuckler"
	case KeyProfileTemperley:
		return "Temperley"
	case KeyProfileShaath:
		return "Shaath"
	case KeyProfileEDMA:
		return "EDMA"
	case KeyProfileBgate:
		return "Bgate"
	case KeyProfileDiatonic:
		return "Diatonic"
	case KeyProfileTonicTriad:
		return "Tonic Triad"
	default:
		return "Unknown"
	}
}

func (ke *KeyEstimator) getCurrentTime() float64 {
	// Placeholder for time measurement
	return 0.0
}

// Public utility functions

// GetKeyName returns human-readable key name
func GetKeyName(key int, mode KeyMode) string {
	keyNames := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

	name := keyNames[key%12]
	if mode == KeyModeMajor {
		return name + " major"
	} else {
		return name + " minor"
	}
}

// GetRelativeKey returns the relative major/minor key
func GetRelativeKey(key int, mode KeyMode) (int, KeyMode) {
	if mode == KeyModeMajor {
		// Relative minor is 3 semitones down
		relativeKey := (key - 3 + 12) % 12
		return relativeKey, KeyModeMinor
	} else {
		// Relative major is 3 semitones up
		relativeKey := (key + 3) % 12
		return relativeKey, KeyModeMajor
	}
}

// GetParallelKey returns the parallel major/minor key
func GetParallelKey(key int, mode KeyMode) (int, KeyMode) {
	if mode == KeyModeMajor {
		return key, KeyModeMinor
	} else {
		return key, KeyModeMajor
	}
}

// GetDominantKey returns the dominant key (5th above)
func GetDominantKey(key int, mode KeyMode) (int, KeyMode) {
	dominantKey := (key + 7) % 12
	return dominantKey, mode
}

// GetSubdominantKey returns the subdominant key (5th below)
func GetSubdominantKey(key int, mode KeyMode) (int, KeyMode) {
	subdominantKey := (key - 7 + 12) % 12
	return subdominantKey, mode
}

// IsKeyCompatible checks if two keys are compatible/related
func IsKeyCompatible(key1 int, mode1 KeyMode, key2 int, mode2 KeyMode) bool {
	// Check for exact match
	if key1 == key2 && mode1 == mode2 {
		return true
	}

	// Check for relative keys
	relKey, relMode := GetRelativeKey(key1, mode1)
	if key2 == relKey && mode2 == relMode {
		return true
	}

	// Check for parallel keys
	parKey, parMode := GetParallelKey(key1, mode1)
	if key2 == parKey && mode2 == parMode {
		return true
	}

	// Check for dominant/subdominant
	domKey, domMode := GetDominantKey(key1, mode1)
	if key2 == domKey && mode2 == domMode {
		return true
	}

	subKey, subMode := GetSubdominantKey(key1, mode1)
	if key2 == subKey && mode2 == subMode {
		return true
	}

	return false
}

// AnalyzeKeyTransition analyzes the transition between two keys
func AnalyzeKeyTransition(fromKey int, fromMode KeyMode, toKey int, toMode KeyMode) map[string]interface{} {
	analysis := make(map[string]interface{})

	// Calculate semitone distance
	distance := (toKey - fromKey + 12) % 12
	analysis["semitone_distance"] = distance

	// Determine transition type
	if fromKey == toKey && fromMode == toMode {
		analysis["transition_type"] = "same_key"
	} else if fromKey == toKey {
		analysis["transition_type"] = "parallel"
	} else {
		relKey, relMode := GetRelativeKey(fromKey, fromMode)
		if toKey == relKey && toMode == relMode {
			analysis["transition_type"] = "relative"
		} else {
			domKey, domMode := GetDominantKey(fromKey, fromMode)
			if toKey == domKey && toMode == domMode {
				analysis["transition_type"] = "dominant"
			} else {
				subKey, subMode := GetSubdominantKey(fromKey, fromMode)
				if toKey == subKey && toMode == subMode {
					analysis["transition_type"] = "subdominant"
				} else {
					analysis["transition_type"] = "distant"
				}
			}
		}
	}

	// Calculate transition strength (based on circle of fifths)
	fifthsDistance := 0
	switch analysis["transition_type"] {
	case "same_key":
		fifthsDistance = 0
	case "parallel":
		fifthsDistance = 0
	case "relative":
		fifthsDistance = 1
	case "dominant", "subdominant":
		fifthsDistance = 1
	default:
		// Calculate actual distance on circle of fifths
		fifthsDistance = min(distance, 12-distance)
	}

	analysis["fifths_distance"] = fifthsDistance
	analysis["transition_strength"] = 1.0 / (1.0 + float64(fifthsDistance))

	return analysis
}

// KeyEstimationBatch processes multiple chroma vectors for batch key estimation
type KeyEstimationBatch struct {
	estimator *KeyEstimator
	results   []KeyEstimationResult
}

// NewKeyEstimationBatch creates a new batch processor
func NewKeyEstimationBatch(sampleRate int) *KeyEstimationBatch {
	return &KeyEstimationBatch{
		estimator: NewKeyEstimator(sampleRate),
		results:   make([]KeyEstimationResult, 0),
	}
}

// ProcessBatch processes a batch of chroma vectors
func (keb *KeyEstimationBatch) ProcessBatch(chromaSequence []chroma.ChromaVector) []KeyEstimationResult {
	results := make([]KeyEstimationResult, len(chromaSequence))

	for i, chromaVec := range chromaSequence {
		results[i] = keb.estimator.EstimateKey(chromaVec)
	}

	keb.results = results
	return results
}

// GetGlobalKey estimates the global key from batch results
func (keb *KeyEstimationBatch) GetGlobalKey() KeyEstimationResult {
	if len(keb.results) == 0 {
		return KeyEstimationResult{}
	}

	// Count key votes
	keyVotes := make(map[string]float64)
	totalConfidence := 0.0

	for _, result := range keb.results {
		if result.Confidence > 0.5 {
			keyVotes[result.KeyName] += result.Confidence
			totalConfidence += result.Confidence
		}
	}

	// Find most voted key
	bestKey := ""
	bestScore := 0.0

	for key, score := range keyVotes {
		if score > bestScore {
			bestScore = score
			bestKey = key
		}
	}

	// Create global result
	globalResult := KeyEstimationResult{
		KeyName:    bestKey,
		Confidence: bestScore / totalConfidence,
		Strength:   bestScore / float64(len(keb.results)),
		Method:     "Batch Voting",
	}

	return globalResult
}

// GetKeyProgression analyzes the key progression through the batch
func (keb *KeyEstimationBatch) GetKeyProgression() []KeyTransition {
	if len(keb.results) < 2 {
		return []KeyTransition{}
	}

	transitions := make([]KeyTransition, 0)

	for i := 1; i < len(keb.results); i++ {
		from := keb.results[i-1]
		to := keb.results[i]

		if from.Confidence > 0.5 && to.Confidence > 0.5 {
			transition := KeyTransition{
				FromKey:        from.Key,
				FromMode:       from.Mode,
				ToKey:          to.Key,
				ToMode:         to.Mode,
				Frame:          i,
				Confidence:     (from.Confidence + to.Confidence) / 2.0,
				TransitionType: keb.classifyTransition(from.Key, from.Mode, to.Key, to.Mode),
			}

			transitions = append(transitions, transition)
		}
	}

	return transitions
}

// KeyTransition represents a key transition in the progression
type KeyTransition struct {
	FromKey        int     `json:"from_key"`
	FromMode       KeyMode `json:"from_mode"`
	ToKey          int     `json:"to_key"`
	ToMode         KeyMode `json:"to_mode"`
	Frame          int     `json:"frame"`
	Confidence     float64 `json:"confidence"`
	TransitionType string  `json:"transition_type"`
}

func (keb *KeyEstimationBatch) classifyTransition(fromKey int, fromMode KeyMode, toKey int, toMode KeyMode) string {
	analysis := AnalyzeKeyTransition(fromKey, fromMode, toKey, toMode)
	return analysis["transition_type"].(string)
}

// Reset resets the estimator state
func (ke *KeyEstimator) Reset() {
	ke.keyHistory = make([]KeyCandidate, 0)
	ke.stabilityWindow = make([]float64, 0)
}

// GetParameters returns current parameters
func (ke *KeyEstimator) GetParameters() KeyEstimationParams {
	return ke.params
}

// SetParameters updates parameters
func (ke *KeyEstimator) SetParameters(params KeyEstimationParams) {
	ke.params = params
	ke.initialized = false // Force re-initialization
}

// GetKeyHistory returns the key estimation history
func (ke *KeyEstimator) GetKeyHistory() []KeyCandidate {
	return ke.keyHistory
}

// GetSupportedProfiles returns list of supported key profiles
func GetSupportedProfiles() []string {
	return []string{
		"Krumhansl-Schmuckler",
		"Temperley",
		"Shaath",
		"EDMA",
		"Bgate",
		"Diatonic",
		"Tonic Triad",
	}
}

// GetSupportedMethods returns list of supported estimation methods
func GetSupportedMethods() []string {
	return []string{
		"Profile Correlation",
		"Direct Correlation",
		"Bayesian",
		"Hidden Markov Model",
		"Deep Learning",
	}
}

// Utility function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
