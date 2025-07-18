package tonal

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/chroma"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// ChordDetectionMethod specifies the algorithm for chord detection
type ChordDetectionMethod int

const (
	ChordTemplateMatching ChordDetectionMethod = iota
	ChordHarmonicAnalysis
	ChordCQTBased
	ChordStatistical
	ChordMLBased
	ChordHybrid
)

// ChordQuality represents the quality/type of a chord
type ChordQuality int

const (
	ChordMajor ChordQuality = iota
	ChordMinor
	ChordDiminished
	ChordAugmented
	ChordSus2
	ChordSus4
	ChordMaj7
	ChordMin7
	ChordDom7
	ChordMinMaj7
	ChordAug7
	ChordDim7
	ChordHalfDim7
	ChordAdd9
	ChordMaj9
	ChordMin9
	ChordDom9
	ChordMaj11
	ChordMin11
	ChordDom11
	ChordMaj13
	ChordMin13
	ChordDom13
	ChordPowerChord // 🤘
	ChordUnknown
)

// ChordInversion represents the inversion of a chord
type ChordInversion int

const (
	ChordRoot ChordInversion = iota
	ChordFirst
	ChordSecond
	ChordThird
	ChordFourth
)

// ChordCandidate represents a possible chord detection
type ChordCandidate struct {
	Root       int            `json:"root"`       // Root note (0=C, 1=C#, ..., 11=B)
	Quality    ChordQuality   `json:"quality"`    // Chord quality/type
	Inversion  ChordInversion `json:"inversion"`  // Chord inversion
	RootName   string         `json:"root_name"`  // Human-readable root name
	ChordName  string         `json:"chord_name"` // Full chord name
	Confidence float64        `json:"confidence"` // Detection confidence (0-1)
	Strength   float64        `json:"strength"`   // Chord strength measure
	Salience   float64        `json:"salience"`   // Perceptual salience
	Method     string         `json:"method"`     // Detection method used
}

// ChordDetectionResult contains the output of chord detection analysis
type ChordDetectionResult struct {
	// Primary chord information
	Root       int            `json:"root"`       // Best root estimate (0-11)
	Quality    ChordQuality   `json:"quality"`    // Best chord quality
	Inversion  ChordInversion `json:"inversion"`  // Best inversion
	RootName   string         `json:"root_name"`  // Human-readable root name
	ChordName  string         `json:"chord_name"` // Full chord name
	Confidence float64        `json:"confidence"` // Overall confidence (0-1)
	Strength   float64        `json:"strength"`   // Chord strength measure

	// Multiple chord candidates
	Candidates []ChordCandidate `json:"candidates"`

	// Harmonic analysis
	ChromaVector    []float64 `json:"chroma_vector"`    // Input chroma profile
	HarmonicProfile []float64 `json:"harmonic_profile"` // Harmonic content analysis
	BassNote        int       `json:"bass_note"`        // Bass note (for inversions)
	BassConfidence  float64   `json:"bass_confidence"`  // Bass note confidence

	// Quality metrics
	Clarity    float64 `json:"clarity"`    // Chord clarity measure
	Ambiguity  float64 `json:"ambiguity"`  // Chord ambiguity measure
	Consonance float64 `json:"consonance"` // Consonance/dissonance measure
	Tension    float64 `json:"tension"`    // Harmonic tension
	Stability  float64 `json:"stability"`  // Temporal stability

	// Template matching scores
	TemplateScores []float64 `json:"template_scores"` // Scores for each chord template

	// Statistical analysis
	ChordProbability float64            `json:"chord_probability"` // Probability this is a chord
	KeyCompatibility map[string]float64 `json:"key_compatibility"` // Compatibility with keys
	FunctionalRole   string             `json:"functional_role"`   // Tonic, subdominant, dominant, etc.

	// Extended harmony analysis
	Extensions   []int `json:"extensions"`    // Additional notes (9, 11, 13, etc.)
	Alterations  []int `json:"alterations"`   // Altered notes (b5, #11, etc.)
	AddedNotes   []int `json:"added_notes"`   // Added notes (add9, add11, etc.)
	OmittedNotes []int `json:"omitted_notes"` // Missing chord tones

	// Voice leading analysis
	VoiceLeading []float64 `json:"voice_leading"` // Voice leading smoothness
	ChordSpacing string    `json:"chord_spacing"` // Close, open, spread
	Doubling     []int     `json:"doubling"`      // Doubled notes

	// Computational details
	Method         string  `json:"method"`          // Detection method used
	ProcessTime    float64 `json:"process_time"`    // Processing time in ms
	ChromaSize     int     `json:"chroma_size"`     // Chroma vector size used
	AnalysisFrames int     `json:"analysis_frames"` // Number of frames analyzed
}

// ChordDetectionParams contains parameters for chord detection
type ChordDetectionParams struct {
	Method          ChordDetectionMethod `json:"method"`
	ChromaSize      int                  `json:"chroma_size"`      // Size of chroma vector (12, 24, 36)
	UseHarmonics    bool                 `json:"use_harmonics"`    // Consider harmonic content
	WeightHarmonics bool                 `json:"weight_harmonics"` // Weight harmonics differently

	// Template matching parameters
	NormalizeTemplates bool    `json:"normalize_templates"` // Normalize chord templates
	TemplateStrength   float64 `json:"template_strength"`   // Template weighting strength
	UseInversions      bool    `json:"use_inversions"`      // Detect chord inversions

	// Quality thresholds
	MinConfidence    float64 `json:"min_confidence"`     // Minimum confidence threshold
	MaxCandidates    int     `json:"max_candidates"`     // Maximum candidates to return
	MinChordStrength float64 `json:"min_chord_strength"` // Minimum chord strength

	// Bass detection
	UseBassDetection bool       `json:"use_bass_detection"` // Enable bass note detection
	BassFreqRange    [2]float64 `json:"bass_freq_range"`    // Bass frequency range [min, max]
	BassWeight       float64    `json:"bass_weight"`        // Weight for bass in detection

	// Extended harmony
	DetectExtensions  bool `json:"detect_extensions"`  // Detect 7ths, 9ths, etc.
	DetectAlterations bool `json:"detect_alterations"` // Detect altered chords
	MaxExtension      int  `json:"max_extension"`      // Maximum extension to detect (7, 9, 11, 13)

	// Temporal analysis
	UseTemporalSmoothing bool `json:"use_temporal_smoothing"` // Enable temporal smoothing
	TemporalWindow       int  `json:"temporal_window"`        // Frames for temporal analysis

	// Advanced options
	UseFunctionalAnalysis bool    `json:"use_functional_analysis"` // Analyze functional harmony
	UseVoiceLeading       bool    `json:"use_voice_leading"`       // Analyze voice leading
	ContextWeight         float64 `json:"context_weight"`          // Weight for harmonic context

	// Machine learning parameters
	ModelPath       string    `json:"model_path"`       // Path to ML model (if available)
	UseEnsemble     bool      `json:"use_ensemble"`     // Use ensemble of methods
	EnsembleWeights []float64 `json:"ensemble_weights"` // Weights for ensemble methods
}

// ChordTemplate represents a chord template for matching
type ChordTemplate struct {
	Quality    ChordQuality `json:"quality"`
	Pattern    []float64    `json:"pattern"`    // Chroma pattern for the chord
	Name       string       `json:"name"`       // Chord name
	Intervals  []int        `json:"intervals"`  // Intervals from root
	Extensions []int        `json:"extensions"` // Possible extensions
	Inversions [][]float64  `json:"inversions"` // Inversion patterns
	Weight     float64      `json:"weight"`     // Template importance weight
	Consonance float64      `json:"consonance"` // Consonance rating
}

// ChordDetector performs chord detection analysis
type ChordDetector struct {
	params ChordDetectionParams

	// Analysis components
	chromaAnalyzer *chroma.ChromaVectorAnalyzer
	hpcp           *chroma.HPCP
	chromaCQT      *chroma.ChromaCQT
	pitchDetector  *PitchDetector

	// Chord templates
	templates map[ChordQuality]*ChordTemplate

	// Temporal tracking
	chordHistory      []ChordCandidate
	stabilityWindow   []float64
	confidenceHistory []float64

	// Statistical analysis
	keyEstimator *KeyEstimator
	moments      *stats.Moments

	// Internal state
	sampleRate  int
	initialized bool
}

// NewChordDetector creates a new chord detector with default parameters
func NewChordDetector(sampleRate int) *ChordDetector {
	params := ChordDetectionParams{
		Method:                ChordTemplateMatching,
		ChromaSize:            12,
		UseHarmonics:          true,
		WeightHarmonics:       false,
		NormalizeTemplates:    true,
		TemplateStrength:      1.0,
		UseInversions:         true,
		MinConfidence:         0.3,
		MaxCandidates:         5,
		MinChordStrength:      0.2,
		UseBassDetection:      true,
		BassFreqRange:         [2]float64{80.0, 350.0},
		BassWeight:            0.3,
		DetectExtensions:      true,
		DetectAlterations:     false,
		MaxExtension:          13,
		UseTemporalSmoothing:  true,
		TemporalWindow:        5,
		UseFunctionalAnalysis: false,
		UseVoiceLeading:       false,
		ContextWeight:         0.1,
		UseEnsemble:           false,
	}

	return NewChordDetectorWithParams(sampleRate, params)
}

// NewChordDetectorWithParams creates a new chord detector with custom parameters
func NewChordDetectorWithParams(sampleRate int, params ChordDetectionParams) *ChordDetector {
	cd := &ChordDetector{
		params:     params,
		sampleRate: sampleRate,
		templates:  make(map[ChordQuality]*ChordTemplate),
	}

	cd.chromaAnalyzer = chroma.NewChromaVectorAnalyzer()
	cd.hpcp = chroma.NewHPCP(sampleRate)
	cd.chromaCQT = chroma.NewChromaCQTDefault(sampleRate)
	cd.pitchDetector = NewPitchDetector(sampleRate)
	cd.keyEstimator = NewKeyEstimator(sampleRate)
	cd.moments = stats.NewMoments()

	cd.initializeTemplates()
	cd.initialized = true

	return cd
}

// initializeTemplates sets up the chord templates
func (cd *ChordDetector) initializeTemplates() {
	// Major chord template
	cd.templates[ChordMajor] = &ChordTemplate{
		Quality:    ChordMajor,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "major",
		Intervals:  []int{0, 4, 7},
		Weight:     1.0,
		Consonance: 0.9,
	}

	// Minor chord template
	cd.templates[ChordMinor] = &ChordTemplate{
		Quality:    ChordMinor,
		Pattern:    []float64{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "minor",
		Intervals:  []int{0, 3, 7},
		Weight:     1.0,
		Consonance: 0.85,
	}

	// Diminished chord template
	cd.templates[ChordDiminished] = &ChordTemplate{
		Quality:    ChordDiminished,
		Pattern:    []float64{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "diminished",
		Intervals:  []int{0, 3, 6},
		Weight:     0.8,
		Consonance: 0.3,
	}

	// Augmented chord template
	cd.templates[ChordAugmented] = &ChordTemplate{
		Quality:    ChordAugmented,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		Name:       "augmented",
		Intervals:  []int{0, 4, 8},
		Weight:     0.7,
		Consonance: 0.4,
	}

	// Dominant 7th chord template
	cd.templates[ChordDom7] = &ChordTemplate{
		Quality:    ChordDom7,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0},
		Name:       "dominant7",
		Intervals:  []int{0, 4, 7, 10},
		Weight:     0.9,
		Consonance: 0.7,
	}

	// Major 7th chord template
	cd.templates[ChordMaj7] = &ChordTemplate{
		Quality:    ChordMaj7,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
		Name:       "major7",
		Intervals:  []int{0, 4, 7, 11},
		Weight:     0.85,
		Consonance: 0.8,
	}

	// Minor 7th chord template
	cd.templates[ChordMin7] = &ChordTemplate{
		Quality:    ChordMin7,
		Pattern:    []float64{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0},
		Name:       "minor7",
		Intervals:  []int{0, 3, 7, 10},
		Weight:     0.85,
		Consonance: 0.75,
	}

	// Sus2 chord template
	cd.templates[ChordSus2] = &ChordTemplate{
		Quality:    ChordSus2,
		Pattern:    []float64{1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "sus2",
		Intervals:  []int{0, 2, 7},
		Weight:     0.7,
		Consonance: 0.6,
	}

	// Sus4 chord template
	cd.templates[ChordSus4] = &ChordTemplate{
		Quality:    ChordSus4,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "sus4",
		Intervals:  []int{0, 5, 7},
		Weight:     0.7,
		Consonance: 0.6,
	}

	// Power chord template (root and fifth only)
	cd.templates[ChordPowerChord] = &ChordTemplate{
		Quality:    ChordPowerChord,
		Pattern:    []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		Name:       "power",
		Intervals:  []int{0, 7},
		Weight:     0.6,
		Consonance: 0.8,
	}

	// Generate inversion patterns for each template
	for _, template := range cd.templates {
		cd.generateInversions(template)
	}
}

// generateInversions creates inversion patterns for a chord template
func (cd *ChordDetector) generateInversions(template *ChordTemplate) {
	numIntervals := len(template.Intervals)
	template.Inversions = make([][]float64, numIntervals)

	// Root position is the original pattern
	template.Inversions[0] = make([]float64, len(template.Pattern))
	copy(template.Inversions[0], template.Pattern)

	// Generate inversions by moving bass note
	for inv := 1; inv < numIntervals; inv++ {
		invPattern := make([]float64, 12)

		// Get bass note for this inversion
		// TODO: unused
		// bassInterval := template.Intervals[inv]

		// Weight bass note more heavily
		for i, interval := range template.Intervals {
			notePos := (interval + 12) % 12
			if i == inv {
				// Bass note gets higher weight
				invPattern[notePos] = 1.5
			} else {
				invPattern[notePos] = template.Pattern[notePos]
			}
		}

		template.Inversions[inv] = invPattern
	}
}

// DetectChord performs chord detection on audio data
func (cd *ChordDetector) DetectChord(audioData []float64) (*ChordDetectionResult, error) {
	if !cd.initialized {
		return nil, fmt.Errorf("chord detector not initialized")
	}

	startTime := time.Now()

	// Extract chroma features
	chromaResult, err := cd.extractChromaFeatures(audioData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract chroma features: %v", err)
	}

	// Detect bass note if enabled
	var bassNote int
	var bassConfidence float64
	if cd.params.UseBassDetection {
		bassNote, bassConfidence = cd.detectBassNote(audioData)
	}

	// Perform chord detection based on selected method
	var candidates []ChordCandidate
	var templateScores []float64

	switch cd.params.Method {
	case ChordTemplateMatching:
		candidates, templateScores = cd.templateMatching(chromaResult, bassNote, bassConfidence)
	case ChordHarmonicAnalysis:
		candidates, templateScores = cd.harmonicAnalysis(audioData, chromaResult)
	case ChordCQTBased:
		candidates, templateScores = cd.cqtBasedDetection(audioData)
	case ChordStatistical:
		candidates, templateScores = cd.statisticalAnalysis(chromaResult)
	case ChordHybrid:
		candidates, templateScores = cd.hybridAnalysis(audioData, chromaResult, bassNote, bassConfidence)
	default:
		candidates, templateScores = cd.templateMatching(chromaResult, bassNote, bassConfidence)
	}

	// Apply temporal smoothing if enabled
	if cd.params.UseTemporalSmoothing {
		candidates = cd.applyTemporalSmoothing(candidates)
	}

	// Sort candidates by confidence
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Confidence > candidates[j].Confidence
	})

	// Limit number of candidates
	if len(candidates) > cd.params.MaxCandidates {
		candidates = candidates[:cd.params.MaxCandidates]
	}

	// Create result
	result := &ChordDetectionResult{
		ChromaVector:   chromaResult,
		BassNote:       bassNote,
		BassConfidence: bassConfidence,
		Candidates:     candidates,
		TemplateScores: templateScores,
		Method:         cd.getMethodName(),
		ProcessTime:    float64(time.Since(startTime).Nanoseconds()) / 1e6,
		ChromaSize:     cd.params.ChromaSize,
	}

	// Set primary chord information from best candidate
	if len(candidates) > 0 {
		best := candidates[0]
		result.Root = best.Root
		result.Quality = best.Quality
		result.Inversion = best.Inversion
		result.RootName = best.RootName
		result.ChordName = best.ChordName
		result.Confidence = best.Confidence
		result.Strength = best.Strength
	}

	// Calculate additional metrics
	cd.calculateQualityMetrics(result, chromaResult)

	// Analyze extensions and alterations if enabled
	if cd.params.DetectExtensions {
		cd.analyzeExtensions(result, chromaResult)
	}

	// Functional analysis if enabled
	if cd.params.UseFunctionalAnalysis {
		cd.analyzeFunctionalRole(result)
	}

	// Update temporal tracking
	cd.updateTemporalTracking(candidates)

	return result, nil
}

// extractChromaFeatures extracts chroma features from audio
func (cd *ChordDetector) extractChromaFeatures(audioData []float64) ([]float64, error) {
	switch cd.params.ChromaSize {
	case 12:
		// Use HPCP for 12-bin chroma
		// First compute FFT to get magnitude spectrum
		fft := spectral.NewFFT()
		windowSize := 2048
		if len(audioData) < windowSize {
			windowSize = len(audioData)
		}

		// Pad or truncate audio to window size
		windowedAudio := make([]float64, windowSize)
		copy(windowedAudio, audioData)

		spectrum := fft.Compute(windowedAudio)

		// Convert to magnitude
		magnitude := make([]float64, len(spectrum))
		for i, c := range spectrum {
			magnitude[i] = math.Sqrt(real(c)*real(c) + imag(c)*imag(c))
		}

		hpcpResult := cd.hpcp.ComputeFromSpectrum(magnitude, windowSize)
		return hpcpResult.HPCP, nil
	case 24, 36:
		// Use CQT for higher resolution
		windowSize := 2048
		chromaResult, err := cd.chromaCQT.ComputeChroma(audioData, windowSize)
		if err != nil {
			return nil, err
		}

		// chromaResult is [][]float64 (time x frequency), we need to average over time
		if len(chromaResult) == 0 {
			return make([]float64, cd.params.ChromaSize), nil
		}

		// Average over time frames
		avgChroma := make([]float64, len(chromaResult[0]))
		for _, frame := range chromaResult {
			for i, val := range frame {
				avgChroma[i] += val
			}
		}

		// Normalize by number of frames
		numFrames := float64(len(chromaResult))
		for i := range avgChroma {
			avgChroma[i] /= numFrames
		}

		return avgChroma, nil
	default:
		return nil, fmt.Errorf("unsupported chroma size: %d", cd.params.ChromaSize)
	}
}

// detectBassNote detects the bass note from audio
func (cd *ChordDetector) detectBassNote(audioData []float64) (int, float64) {
	// Use pitch detection on low-passed audio
	// This is a simplified implementation - in practice you might want
	// to use more sophisticated bass detection

	pitchResult, err := cd.pitchDetector.DetectPitch(audioData)
	if err != nil || pitchResult.Confidence < 0.3 {
		return 0, 0.0
	}

	// Check if pitch is in bass range
	if pitchResult.Pitch < cd.params.BassFreqRange[0] || pitchResult.Pitch > cd.params.BassFreqRange[1] {
		return 0, 0.0
	}

	// Convert frequency to chroma class
	bassNote := cd.frequencyToChroma(pitchResult.Pitch)
	return bassNote, pitchResult.Confidence
}

// templateMatching performs chord detection using template matching
func (cd *ChordDetector) templateMatching(chroma []float64, bassNote int, bassConfidence float64) ([]ChordCandidate, []float64) {
	var candidates []ChordCandidate
	templateScores := make([]float64, len(cd.templates)*12) // 12 roots × templates
	scoreIndex := 0

	// Normalize chroma vector if needed
	normalizedChroma := cd.normalizeChroma(chroma)

	// Test each chord quality at each root
	for quality, template := range cd.templates {
		for root := 0; root < 12; root++ {
			// Rotate template to match root
			rotatedPattern := cd.rotatePattern(template.Pattern, root)

			// Calculate correlation score
			score := cd.calculateTemplateScore(normalizedChroma, rotatedPattern, template.Weight)
			templateScores[scoreIndex] = score
			scoreIndex++

			// Apply bass weighting if bass note detected
			if cd.params.UseBassDetection && bassConfidence > 0.3 {
				bassBonus := cd.calculateBassBonus(root, bassNote, bassConfidence, quality)
				score += bassBonus
			}

			// Check if score meets minimum threshold
			if score >= cd.params.MinChordStrength {
				candidate := ChordCandidate{
					Root:       root,
					Quality:    quality,
					Inversion:  ChordRoot, // Default to root position
					RootName:   cd.getNoteName(root),
					ChordName:  cd.getChordName(root, quality, ChordRoot),
					Confidence: math.Min(score, 1.0),
					Strength:   score,
					Method:     "template_matching",
				}

				// Check for inversions if enabled
				if cd.params.UseInversions && bassConfidence > 0.3 {
					inversion, invScore := cd.detectInversion(normalizedChroma, template, root, bassNote)
					if invScore > score {
						candidate.Inversion = inversion
						candidate.ChordName = cd.getChordName(root, quality, inversion)
						candidate.Confidence = math.Min(invScore, 1.0)
						candidate.Strength = invScore
					}
				}

				candidates = append(candidates, candidate)
			}
		}
	}

	return candidates, templateScores
}

// harmonicAnalysis performs chord detection using harmonic analysis
func (cd *ChordDetector) harmonicAnalysis(audioData []float64, chroma []float64) ([]ChordCandidate, []float64) {
	// This is a placeholder for harmonic analysis
	// In a full implementation, this would analyze the harmonic content
	// of the audio to detect chords

	// For now, fall back to template matching
	return cd.templateMatching(chroma, 0, 0.0)
}

// cqtBasedDetection performs chord detection using Constant-Q Transform
func (cd *ChordDetector) cqtBasedDetection(audioData []float64) ([]ChordCandidate, []float64) {
	// Extract CQT-based chroma
	windowSize := 2048
	chromaResult, err := cd.chromaCQT.ComputeChroma(audioData, windowSize)
	if err != nil {
		return []ChordCandidate{}, []float64{}
	}

	// Average over time frames to get single chroma vector
	if len(chromaResult) == 0 {
		return []ChordCandidate{}, []float64{}
	}

	avgChroma := make([]float64, len(chromaResult[0]))
	for _, frame := range chromaResult {
		for i, val := range frame {
			avgChroma[i] += val
		}
	}

	// Normalize by number of frames
	numFrames := float64(len(chromaResult))
	for i := range avgChroma {
		avgChroma[i] /= numFrames
	}

	// Use template matching on averaged CQT chroma
	return cd.templateMatching(avgChroma, 0, 0.0)
}

// statisticalAnalysis performs statistical chord detection
func (cd *ChordDetector) statisticalAnalysis(chroma []float64) ([]ChordCandidate, []float64) {
	// This could implement more sophisticated statistical methods
	// For now, use template matching
	return cd.templateMatching(chroma, 0, 0.0)
}

// hybridAnalysis combines multiple detection methods
func (cd *ChordDetector) hybridAnalysis(audioData []float64, chroma []float64, bassNote int, bassConfidence float64) ([]ChordCandidate, []float64) {
	// Combine template matching with other methods
	templateCandidates, templateScores := cd.templateMatching(chroma, bassNote, bassConfidence)

	// Could add other methods here and combine results
	// For now, just return template matching results
	return templateCandidates, templateScores
}

// Helper functions

func (cd *ChordDetector) normalizeChroma(chroma []float64) []float64 {
	if !cd.params.NormalizeTemplates {
		return chroma
	}

	normalizer := common.NewNormalizer(common.Energy)
	return normalizer.Normalize(chroma)
}

func (cd *ChordDetector) rotatePattern(pattern []float64, semitones int) []float64 {
	result := make([]float64, len(pattern))
	for i, val := range pattern {
		newIndex := (i + semitones) % len(pattern)
		result[newIndex] = val
	}
	return result
}

func (cd *ChordDetector) calculateTemplateScore(chroma, template []float64, weight float64) float64 {
	if len(chroma) != len(template) {
		return 0.0
	}

	// Calculate dot product (correlation)
	score := 0.0
	for i := range chroma {
		score += chroma[i] * template[i]
	}

	return score * weight
}

func (cd *ChordDetector) calculateBassBonus(root, bassNote int, bassConfidence float64, quality ChordQuality) float64 {
	template := cd.templates[quality]
	if template == nil {
		return 0.0
	}

	// Check if bass note matches any chord tone
	for _, interval := range template.Intervals {
		expectedBass := (root + interval) % 12
		if expectedBass == bassNote {
			return cd.params.BassWeight * bassConfidence
		}
	}

	return 0.0
}

func (cd *ChordDetector) detectInversion(chroma []float64, template *ChordTemplate, root, bassNote int) (ChordInversion, float64) {
	if len(template.Inversions) == 0 {
		return ChordRoot, 0.0
	}

	bestInversion := ChordRoot
	bestScore := 0.0

	// Test each inversion
	for inv := 0; inv < len(template.Inversions); inv++ {
		if inv >= len(template.Intervals) {
			break
		}

		// Check if bass note matches expected bass for this inversion
		expectedBass := (root + template.Intervals[inv]) % 12
		if expectedBass == bassNote {
			// Calculate score for this inversion pattern
			rotatedPattern := cd.rotatePattern(template.Inversions[inv], root)
			score := cd.calculateTemplateScore(chroma, rotatedPattern, template.Weight)

			if score > bestScore {
				bestScore = score
				bestInversion = ChordInversion(inv)
			}
		}
	}

	return bestInversion, bestScore
}

func (cd *ChordDetector) applyTemporalSmoothing(candidates []ChordCandidate) []ChordCandidate {
	if len(cd.chordHistory) == 0 {
		cd.chordHistory = candidates
		return candidates
	}

	// Simple temporal smoothing - boost confidence of chords that appear consistently
	for i := range candidates {
		for _, historical := range cd.chordHistory {
			if candidates[i].Root == historical.Root && candidates[i].Quality == historical.Quality {
				// Boost confidence for consistent chords
				candidates[i].Confidence = math.Min(candidates[i].Confidence*1.1, 1.0)
			}
		}
	}

	// Update history
	cd.chordHistory = candidates
	if len(cd.chordHistory) > cd.params.TemporalWindow {
		cd.chordHistory = cd.chordHistory[1:]
	}

	return candidates
}

func (cd *ChordDetector) calculateQualityMetrics(result *ChordDetectionResult, chroma []float64) {
	if len(result.Candidates) == 0 {
		return
	}

	// Calculate clarity (how much the best candidate stands out)
	if len(result.Candidates) > 1 {
		result.Clarity = result.Candidates[0].Confidence - result.Candidates[1].Confidence
	} else {
		result.Clarity = result.Candidates[0].Confidence
	}

	// Calculate ambiguity (inverse of clarity)
	result.Ambiguity = 1.0 - result.Clarity

	// Calculate consonance based on chord quality
	if template, exists := cd.templates[result.Quality]; exists {
		result.Consonance = template.Consonance
	}

	// Calculate stability from temporal tracking
	if len(cd.confidenceHistory) > 0 {
		variance := cd.calculateVariance(cd.confidenceHistory)
		result.Stability = math.Max(0.0, 1.0-variance)
	} else {
		result.Stability = result.Confidence
	}

	// Calculate tension (simplified measure based on dissonant intervals)
	result.Tension = cd.calculateHarmonicTension(chroma, result.Quality)
}

func (cd *ChordDetector) analyzeExtensions(result *ChordDetectionResult, chroma []float64) {
	if len(result.Candidates) == 0 {
		return
	}

	best := result.Candidates[0]
	template := cd.templates[best.Quality]
	if template == nil {
		return
	}

	// Check for 7th, 9th, 11th, 13th
	extensions := []int{}
	alterations := []int{}
	addedNotes := []int{}

	// Define extension intervals
	extensionIntervals := map[int]string{
		10: "b7",   // Minor 7th
		11: "maj7", // Major 7th
		2:  "9",    // 9th (octave reduced)
		5:  "11",   // 11th (octave reduced)
		9:  "13",   // 13th (octave reduced)
	}

	// TODO: Unused interval, name
	for interval, _ := range extensionIntervals {
		noteIndex := (best.Root + interval) % 12
		if chroma[noteIndex] > 0.3 { // Threshold for extension detection
			// Check if this is already a chord tone
			isChordTone := false
			for _, chordInterval := range template.Intervals {
				if interval == chordInterval {
					isChordTone = true
					break
				}
			}

			if !isChordTone {
				switch interval {
				case 10, 11:
					extensions = append(extensions, interval)
				case 2, 5, 9:
					if cd.params.MaxExtension >= cd.getExtensionNumber(interval) {
						extensions = append(extensions, interval)
					}
				}
			}
		}
	}

	result.Extensions = extensions
	result.Alterations = alterations
	result.AddedNotes = addedNotes
}

func (cd *ChordDetector) analyzeFunctionalRole(result *ChordDetectionResult) {
	// Simplified functional analysis
	// In a full implementation, this would consider key context

	if len(result.Candidates) == 0 {
		return
	}

	// Default functional roles based on chord quality
	switch result.Quality {
	case ChordMajor:
		result.FunctionalRole = "tonic"
	case ChordMinor:
		result.FunctionalRole = "subdominant"
	case ChordDom7:
		result.FunctionalRole = "dominant"
	case ChordDiminished:
		result.FunctionalRole = "leading_tone"
	default:
		result.FunctionalRole = "other"
	}
}

func (cd *ChordDetector) updateTemporalTracking(candidates []ChordCandidate) {
	if len(candidates) > 0 {
		cd.confidenceHistory = append(cd.confidenceHistory, candidates[0].Confidence)
		if len(cd.confidenceHistory) > cd.params.TemporalWindow {
			cd.confidenceHistory = cd.confidenceHistory[1:]
		}
	}
}

func (cd *ChordDetector) frequencyToChroma(frequency float64) int {
	// Convert frequency to MIDI note number, then to chroma class
	if frequency <= 0 {
		return 0
	}

	// A4 = 440 Hz = MIDI note 69
	midiNote := 69 + 12*math.Log2(frequency/440.0)
	chromaClass := int(midiNote) % 12
	if chromaClass < 0 {
		chromaClass += 12
	}

	return chromaClass
}

func (cd *ChordDetector) getNoteName(noteNumber int) string {
	noteNames := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	return noteNames[noteNumber%12]
}

func (cd *ChordDetector) getChordName(root int, quality ChordQuality, inversion ChordInversion) string {
	rootName := cd.getNoteName(root)

	var qualityName string
	if template, exists := cd.templates[quality]; exists {
		qualityName = template.Name
	} else {
		qualityName = "unknown"
	}

	chordName := rootName
	if qualityName != "major" {
		chordName += qualityName
	}

	// Add inversion notation
	if inversion != ChordRoot {
		inversionNames := []string{"", "/1st", "/2nd", "/3rd", "/4th"}
		if int(inversion) < len(inversionNames) {
			chordName += inversionNames[int(inversion)]
		}
	}

	return chordName
}

func (cd *ChordDetector) getMethodName() string {
	methods := map[ChordDetectionMethod]string{
		ChordTemplateMatching: "template_matching",
		ChordHarmonicAnalysis: "harmonic_analysis",
		ChordCQTBased:         "cqt_based",
		ChordStatistical:      "statistical",
		ChordMLBased:          "ml_based",
		ChordHybrid:           "hybrid",
	}

	if name, exists := methods[cd.params.Method]; exists {
		return name
	}
	return "unknown"
}

func (cd *ChordDetector) calculateVariance(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return variance
}

func (cd *ChordDetector) calculateHarmonicTension(chroma []float64, quality ChordQuality) float64 {
	// Simplified tension calculation based on dissonant intervals
	tension := 0.0

	// Check for dissonant intervals (minor 2nd, major 7th, tritone)
	dissonantIntervals := []int{1, 6, 11} // semitones

	for i := 0; i < 12; i++ {
		if chroma[i] > 0.2 {
			for j := i + 1; j < 12; j++ {
				if chroma[j] > 0.2 {
					interval := j - i
					for _, dissonant := range dissonantIntervals {
						if interval == dissonant {
							tension += chroma[i] * chroma[j]
						}
					}
				}
			}
		}
	}

	return math.Min(tension, 1.0)
}

func (cd *ChordDetector) getExtensionNumber(interval int) int {
	switch interval {
	case 2:
		return 9
	case 5:
		return 11
	case 9:
		return 13
	default:
		return 7
	}
}

// Public utility functions

// GetChordQualityName returns the human-readable name for a chord quality
func GetChordQualityName(quality ChordQuality) string {
	names := map[ChordQuality]string{
		ChordMajor:      "major",
		ChordMinor:      "minor",
		ChordDiminished: "diminished",
		ChordAugmented:  "augmented",
		ChordSus2:       "sus2",
		ChordSus4:       "sus4",
		ChordMaj7:       "major7",
		ChordMin7:       "minor7",
		ChordDom7:       "dominant7",
		ChordMinMaj7:    "minor-major7",
		ChordAug7:       "augmented7",
		ChordDim7:       "diminished7",
		ChordHalfDim7:   "half-diminished7",
		ChordAdd9:       "add9",
		ChordMaj9:       "major9",
		ChordMin9:       "minor9",
		ChordDom9:       "dominant9",
		ChordMaj11:      "major11",
		ChordMin11:      "minor11",
		ChordDom11:      "dominant11",
		ChordMaj13:      "major13",
		ChordMin13:      "minor13",
		ChordDom13:      "dominant13",
		ChordPowerChord: "power",
		ChordUnknown:    "unknown",
	}

	if name, exists := names[quality]; exists {
		return name
	}
	return "unknown"
}

// GetSupportedChordQualities returns a list of supported chord qualities
func GetSupportedChordQualities() []string {
	return []string{
		"major", "minor", "diminished", "augmented",
		"sus2", "sus4", "major7", "minor7", "dominant7",
		"minor-major7", "augmented7", "diminished7", "half-diminished7",
		"add9", "major9", "minor9", "dominant9",
		"major11", "minor11", "dominant11",
		"major13", "minor13", "dominant13", "power",
	}
}

// GetSupportedDetectionMethods returns a list of supported detection methods
func GetSupportedDetectionMethods() []string {
	return []string{
		"template_matching", "harmonic_analysis", "cqt_based",
		"statistical", "ml_based", "hybrid",
	}
}

// ChordProgressionAnalyzer analyzes sequences of chords
type ChordProgressionAnalyzer struct {
	detector *ChordDetector
	chords   []ChordDetectionResult
}

// NewChordProgressionAnalyzer creates a new chord progression analyzer
func NewChordProgressionAnalyzer(sampleRate int) *ChordProgressionAnalyzer {
	return &ChordProgressionAnalyzer{
		detector: NewChordDetector(sampleRate),
		chords:   make([]ChordDetectionResult, 0),
	}
}

// AddChord adds a chord detection result to the progression
func (cpa *ChordProgressionAnalyzer) AddChord(result ChordDetectionResult) {
	cpa.chords = append(cpa.chords, result)
}

// AnalyzeProgression analyzes the chord progression
func (cpa *ChordProgressionAnalyzer) AnalyzeProgression() map[string]interface{} {
	if len(cpa.chords) < 2 {
		return map[string]interface{}{
			"num_chords": len(cpa.chords),
			"analysis":   "insufficient_data",
		}
	}

	analysis := make(map[string]interface{})
	analysis["num_chords"] = len(cpa.chords)

	// Analyze chord transitions
	transitions := make([]string, 0)
	for i := 1; i < len(cpa.chords); i++ {
		from := cpa.chords[i-1].ChordName
		to := cpa.chords[i].ChordName
		transition := fmt.Sprintf("%s -> %s", from, to)
		transitions = append(transitions, transition)
	}
	analysis["transitions"] = transitions

	// Calculate average confidence
	totalConfidence := 0.0
	for _, chord := range cpa.chords {
		totalConfidence += chord.Confidence
	}
	analysis["average_confidence"] = totalConfidence / float64(len(cpa.chords))

	// Find most common chord quality
	qualityCounts := make(map[ChordQuality]int)
	for _, chord := range cpa.chords {
		qualityCounts[chord.Quality]++
	}

	var mostCommonQuality ChordQuality
	maxCount := 0
	for quality, count := range qualityCounts {
		if count > maxCount {
			maxCount = count
			mostCommonQuality = quality
		}
	}
	analysis["most_common_quality"] = GetChordQualityName(mostCommonQuality)

	return analysis
}
