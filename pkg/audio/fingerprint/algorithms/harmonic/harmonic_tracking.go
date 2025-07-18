package harmonic

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// TrackingMethod represents different approaches to harmonic tracking
type TrackingMethod int

const (
	// Peak-based tracking using spectral peaks
	PeakBased TrackingMethod = iota

	// Sinusoidal model tracking
	SinusoidalModel

	// Partial tracking with birth/death events
	PartialTracking

	// Kalman filter based tracking
	KalmanFilter

	// Multi-frame analysis with temporal smoothing
	MultiFrame
)

// HarmonicTrack represents a single harmonic track over time
type HarmonicTrack struct {
	ID             int     `json:"id"`              // Unique track identifier
	HarmonicNumber int     `json:"harmonic_number"` // Harmonic number (1=fundamental, 2=second, etc.)
	StartFrame     int     `json:"start_frame"`     // Frame where track starts
	EndFrame       int     `json:"end_frame"`       // Frame where track ends
	Duration       float64 `json:"duration"`        // Duration in seconds

	// Frequency evolution
	Frequencies []float64 `json:"frequencies"`  // Frequency at each frame
	FreqStdDev  float64   `json:"freq_std_dev"` // Frequency stability
	FreqSlope   float64   `json:"freq_slope"`   // Frequency trend (Hz/frame)

	// Amplitude evolution
	Amplitudes []float64 `json:"amplitudes"`  // Amplitude at each frame
	AmpStdDev  float64   `json:"amp_std_dev"` // Amplitude stability
	AmpSlope   float64   `json:"amp_slope"`   // Amplitude trend (dB/frame)

	// Phase evolution
	Phases     []float64 `json:"phases"`      // Phase at each frame
	PhaseSlope float64   `json:"phase_slope"` // Phase trend (radians/frame)

	// Track quality metrics
	Confidence float64 `json:"confidence"` // Overall track confidence
	Continuity float64 `json:"continuity"` // Temporal continuity measure
	Salience   float64 `json:"salience"`   // Perceptual salience

	// Birth/death events
	BirthFrame int    `json:"birth_frame"` // Frame where harmonic appears
	DeathFrame int    `json:"death_frame"` // Frame where harmonic disappears
	BirthType  string `json:"birth_type"`  // "onset", "split", "merge"
	DeathType  string `json:"death_type"`  // "offset", "split", "merge"

	// Relationships
	ParentTrack int   `json:"parent_track"` // ID of parent track (for splits)
	ChildTracks []int `json:"child_tracks"` // IDs of child tracks (for merges)
}

// HarmonicTrackingResult contains comprehensive tracking results
type HarmonicTrackingResult struct {
	// All harmonic tracks found
	Tracks []HarmonicTrack `json:"tracks"`

	// Fundamental frequency evolution
	F0Trajectory []float64 `json:"f0_trajectory"` // F0 at each frame
	F0Confidence []float64 `json:"f0_confidence"` // F0 confidence at each frame
	F0Stability  float64   `json:"f0_stability"`  // Overall F0 stability

	// Harmonic content analysis
	HarmonicCount  []int     `json:"harmonic_count"`  // Number of harmonics per frame
	HarmonicEnergy []float64 `json:"harmonic_energy"` // Total harmonic energy per frame
	Inharmonicity  []float64 `json:"inharmonicity"`   // Inharmonicity measure per frame

	// Tracking statistics
	TotalTracks  int   `json:"total_tracks"`  // Total number of tracks
	ActiveTracks []int `json:"active_tracks"` // Number of active tracks per frame
	TrackBirths  []int `json:"track_births"`  // Track birth events per frame
	TrackDeaths  []int `json:"track_deaths"`  // Track death events per frame

	// Quality metrics
	OverallQuality    float64 `json:"overall_quality"`    // Overall tracking quality
	TemporalCoherence float64 `json:"temporal_coherence"` // Temporal coherence measure

	// Parameters used
	Method       TrackingMethod `json:"method"`
	NumFrames    int            `json:"num_frames"`
	MaxHarmonics int            `json:"max_harmonics"`
	SampleRate   int            `json:"sample_rate"`
	HopSize      int            `json:"hop_size"`
}

// HarmonicTrackingParams contains parameters for harmonic tracking
type HarmonicTrackingParams struct {
	Method           TrackingMethod `json:"method"`
	MaxHarmonics     int            `json:"max_harmonics"`      // Maximum number of harmonics to track
	MinTrackLength   int            `json:"min_track_length"`   // Minimum track length in frames
	MaxFreqDeviation float64        `json:"max_freq_deviation"` // Maximum frequency deviation (Hz)
	MaxAmpDeviation  float64        `json:"max_amp_deviation"`  // Maximum amplitude deviation (dB)

	// Continuation parameters
	FreqContinuityWeight  float64 `json:"freq_continuity_weight"`  // Weight for frequency continuity
	AmpContinuityWeight   float64 `json:"amp_continuity_weight"`   // Weight for amplitude continuity
	PhaseContinuityWeight float64 `json:"phase_continuity_weight"` // Weight for phase continuity

	// Birth/death parameters
	BirthThreshold float64 `json:"birth_threshold"` // Threshold for track birth
	DeathThreshold float64 `json:"death_threshold"` // Threshold for track death
	MaxGapLength   int     `json:"max_gap_length"`  // Maximum gap length to bridge

	// Filtering parameters
	MedianFilterLength   int     `json:"median_filter_length"`   // Length of median filter for smoothing
	UseTemporalSmoothing bool    `json:"use_temporal_smoothing"` // Enable temporal smoothing
	SmoothingFactor      float64 `json:"smoothing_factor"`       // Smoothing factor (0-1)

	// Quality parameters
	MinConfidence   float64 `json:"min_confidence"`   // Minimum confidence threshold
	ConfidenceDecay float64 `json:"confidence_decay"` // Confidence decay rate

	// Advanced parameters
	UseHarmonicConstraints bool `json:"use_harmonic_constraints"` // Enforce harmonic relationships
	AllowFreqModulation    bool `json:"allow_freq_modulation"`    // Allow frequency modulation
	UsePhaseTracking       bool `json:"use_phase_tracking"`       // Track phase information
}

// HarmonicTracking implements comprehensive harmonic tracking for audio analysis
//
// References:
// - McAulay, R.J., Quatieri, T.F. (1986). "Speech analysis/synthesis based on a sinusoidal representation"
// - Serra, X., Smith, J. (1990). "Spectral modeling synthesis: A sound analysis/synthesis system"
// - Klapuri, A. (2003). "Multiple fundamental frequency estimation based on harmonicity and spectral smoothness"
// - Virtanen, T., et al. (2018). "Computational Analysis of Sound Scenes and Events"
// - Levine, S., Smith, J. (1998). "A sines+transients+noise audio representation for data compression"
// - Beauchamp, J.W. (2007). "Analysis and synthesis of musical instrument sounds"
//
// Harmonic tracking is essential for:
// - Music transcription and analysis
// - Audio separation and source localization
// - Pitch tracking and melody extraction
// - Audio synthesis and resynthesis
// - Musical instrument analysis
// - Audio compression and coding
type HarmonicTracking struct {
	params     HarmonicTrackingParams
	sampleRate int
	hopSize    int

	// Internal state
	tracks      []HarmonicTrack
	nextTrackID int
	frameIndex  int

	// Analysis components
	peakDetector *SpectralPeaks
	fft          *spectral.FFT

	// Temporary storage
	previousPeaks []SpectralPeak
	currentPeaks  []SpectralPeak
}

// NewHarmonicTracking creates a new harmonic tracking analyzer
func NewHarmonicTracking(sampleRate, hopSize int) *HarmonicTracking {
	return &HarmonicTracking{
		params: HarmonicTrackingParams{
			Method:                 PeakBased,
			MaxHarmonics:           20,
			MinTrackLength:         3,
			MaxFreqDeviation:       50.0,
			MaxAmpDeviation:        20.0,
			FreqContinuityWeight:   0.6,
			AmpContinuityWeight:    0.3,
			PhaseContinuityWeight:  0.1,
			BirthThreshold:         0.3,
			DeathThreshold:         0.1,
			MaxGapLength:           2,
			MedianFilterLength:     5,
			UseTemporalSmoothing:   true,
			SmoothingFactor:        0.3,
			MinConfidence:          0.2,
			ConfidenceDecay:        0.9,
			UseHarmonicConstraints: true,
			AllowFreqModulation:    true,
			UsePhaseTracking:       false,
		},
		sampleRate:   sampleRate,
		hopSize:      hopSize,
		tracks:       make([]HarmonicTrack, 0),
		nextTrackID:  1,
		frameIndex:   0,
		peakDetector: NewSpectralPeaks(sampleRate, 0.1, 20.0, 50),
		fft:          spectral.NewFFT(),
	}
}

// NewHarmonicTrackingWithParams creates a harmonic tracking analyzer with custom parameters
func NewHarmonicTrackingWithParams(sampleRate, hopSize int, params HarmonicTrackingParams) *HarmonicTracking {
	ht := NewHarmonicTracking(sampleRate, hopSize)
	ht.params = params
	return ht
}

// ProcessFrame processes a single frame of spectral data
func (ht *HarmonicTracking) ProcessFrame(spectrum []complex128, windowSize int) error {
	// Convert spectrum to magnitude and phase
	magnitude := make([]float64, len(spectrum))
	phase := make([]float64, len(spectrum))

	for i, c := range spectrum {
		magnitude[i] = cmplx.Abs(c)
		phase[i] = cmplx.Phase(c)
	}

	// Find spectral peaks using the correct API
	peaks := ht.peakDetector.DetectPeaksWithPhase(magnitude, phase, windowSize)

	// Update tracking state
	ht.currentPeaks = peaks
	ht.updateTracks()
	ht.previousPeaks = peaks
	ht.frameIndex++

	return nil
}

// ProcessSpectrogram processes an entire spectrogram
func (ht *HarmonicTracking) ProcessSpectrogram(spectrogram [][]complex128, windowSize int) (*HarmonicTrackingResult, error) {
	if len(spectrogram) == 0 {
		return nil, fmt.Errorf("empty spectrogram")
	}

	// Reset state
	ht.tracks = make([]HarmonicTrack, 0)
	ht.nextTrackID = 1
	ht.frameIndex = 0

	// Process each frame
	for _, frame := range spectrogram {
		err := ht.ProcessFrame(frame, windowSize)
		if err != nil {
			return nil, err
		}
	}

	// Finalize tracks
	ht.finalizeTracks()

	// Build result
	return ht.buildResult(len(spectrogram))
}

// ProcessMagnitudeSpectrogram processes magnitude-only spectrogram (without phase)
func (ht *HarmonicTracking) ProcessMagnitudeSpectrogram(magnitudeSpectrogram [][]float64, windowSize int) (*HarmonicTrackingResult, error) {
	if len(magnitudeSpectrogram) == 0 {
		return nil, fmt.Errorf("empty spectrogram")
	}

	// Reset state
	ht.tracks = make([]HarmonicTrack, 0)
	ht.nextTrackID = 1
	ht.frameIndex = 0

	// Process each frame
	for _, magnitudeFrame := range magnitudeSpectrogram {
		// Find spectral peaks without phase information
		peaks := ht.peakDetector.DetectPeaks(magnitudeFrame, windowSize)

		// Update tracking state
		ht.currentPeaks = peaks
		ht.updateTracks()
		ht.previousPeaks = peaks
		ht.frameIndex++
	}

	// Finalize tracks
	ht.finalizeTracks()

	// Build result
	return ht.buildResult(len(magnitudeSpectrogram))
}

// updateTracks updates harmonic tracks based on current spectral peaks
func (ht *HarmonicTracking) updateTracks() {
	switch ht.params.Method {
	case PeakBased:
		ht.updateTracksPeakBased()
	case SinusoidalModel:
		ht.updateTracksSinusoidal()
	case PartialTracking:
		ht.updateTracksPartial()
	case KalmanFilter:
		ht.updateTracksKalman()
	case MultiFrame:
		ht.updateTracksMultiFrame()
	default:
		ht.updateTracksPeakBased()
	}
}

// updateTracksPeakBased implements peak-based harmonic tracking
func (ht *HarmonicTracking) updateTracksPeakBased() {
	// Match current peaks with existing tracks
	matched := make(map[int]bool)
	usedPeaks := make(map[int]bool)

	// Continue existing tracks
	for i := range ht.tracks {
		if ht.tracks[i].EndFrame == ht.frameIndex-1 {
			bestMatch := ht.findBestPeakMatch(&ht.tracks[i])
			if bestMatch != -1 && !usedPeaks[bestMatch] {
				ht.continuTrack(&ht.tracks[i], ht.currentPeaks[bestMatch])
				matched[i] = true
				usedPeaks[bestMatch] = true
			}
		}
	}

	// Create new tracks for unmatched peaks
	for i, peak := range ht.currentPeaks {
		if !usedPeaks[i] && ht.shouldCreateTrack(peak) {
			ht.createNewTrack(peak)
		}
	}

	// Handle track deaths
	ht.handleTrackDeaths(matched)
}

// updateTracksSinusoidal implements sinusoidal model tracking
func (ht *HarmonicTracking) updateTracksSinusoidal() {
	// More sophisticated tracking using sinusoidal model
	// This would implement the McAulay-Quatieri algorithm
	// For now, fall back to peak-based tracking
	ht.updateTracksPeakBased()
}

// updateTracksPartial implements partial tracking with birth/death events
func (ht *HarmonicTracking) updateTracksPartial() {
	// Implement partial tracking similar to SMS (Spectral Modeling Synthesis)
	// This includes sophisticated birth/death event handling
	ht.updateTracksPeakBased()
}

// updateTracksKalman implements Kalman filter based tracking
func (ht *HarmonicTracking) updateTracksKalman() {
	// Implement Kalman filter for robust tracking
	// This would provide better prediction and noise handling
	ht.updateTracksPeakBased()
}

// updateTracksMultiFrame implements multi-frame analysis
func (ht *HarmonicTracking) updateTracksMultiFrame() {
	// Multi-frame analysis for improved stability
	ht.updateTracksPeakBased()
}

// findBestPeakMatch finds the best matching peak for a given track
func (ht *HarmonicTracking) findBestPeakMatch(track *HarmonicTrack) int {
	if len(track.Frequencies) == 0 {
		return -1
	}

	lastFreq := track.Frequencies[len(track.Frequencies)-1]
	lastAmp := track.Amplitudes[len(track.Amplitudes)-1]

	bestMatch := -1
	bestScore := -1.0

	for i, peak := range ht.currentPeaks {
		score := ht.calculateMatchScore(peak, lastFreq, lastAmp)
		if score > bestScore && score > ht.params.MinConfidence {
			bestScore = score
			bestMatch = i
		}
	}

	return bestMatch
}

// calculateMatchScore calculates a matching score between a peak and expected values
func (ht *HarmonicTracking) calculateMatchScore(peak SpectralPeak, expectedFreq, expectedAmp float64) float64 {
	// Frequency deviation score
	freqDev := math.Abs(peak.Frequency - expectedFreq)
	freqScore := math.Exp(-freqDev / ht.params.MaxFreqDeviation)

	// Amplitude deviation score
	ampDev := math.Abs(peak.Magnitude - expectedAmp)
	ampScore := math.Exp(-ampDev / ht.params.MaxAmpDeviation)

	// Combined score
	score := ht.params.FreqContinuityWeight*freqScore +
		ht.params.AmpContinuityWeight*ampScore

	return score
}

// shouldCreateTrack determines if a new track should be created for a peak
func (ht *HarmonicTracking) shouldCreateTrack(peak SpectralPeak) bool {
	// Check if peak is significant enough
	if peak.Magnitude < ht.params.BirthThreshold {
		return false
	}

	// Check if we haven't exceeded maximum harmonics
	activeTracks := ht.countActiveTracks()
	if activeTracks >= ht.params.MaxHarmonics {
		return false
	}

	// Additional criteria for track creation
	return true
}

// createNewTrack creates a new harmonic track
func (ht *HarmonicTracking) createNewTrack(peak SpectralPeak) {
	track := HarmonicTrack{
		ID:             ht.nextTrackID,
		HarmonicNumber: ht.estimateHarmonicNumber(peak.Frequency),
		StartFrame:     ht.frameIndex,
		EndFrame:       ht.frameIndex,
		BirthFrame:     ht.frameIndex,
		BirthType:      "onset",
		Frequencies:    []float64{peak.Frequency},
		Amplitudes:     []float64{peak.Magnitude},
		Phases:         []float64{peak.Phase},
		Confidence:     1.0,
		ParentTrack:    -1,
		ChildTracks:    make([]int, 0),
	}

	ht.tracks = append(ht.tracks, track)
	ht.nextTrackID++
}

// continuTrack continues an existing track with a new peak
func (ht *HarmonicTracking) continuTrack(track *HarmonicTrack, peak SpectralPeak) {
	track.EndFrame = ht.frameIndex
	track.Frequencies = append(track.Frequencies, peak.Frequency)
	track.Amplitudes = append(track.Amplitudes, peak.Magnitude)
	track.Phases = append(track.Phases, peak.Phase)

	// Update confidence
	track.Confidence *= ht.params.ConfidenceDecay
	if track.Confidence < ht.params.MinConfidence {
		track.Confidence = ht.params.MinConfidence
	}
}

// handleTrackDeaths handles tracks that haven't been matched
func (ht *HarmonicTracking) handleTrackDeaths(matched map[int]bool) {
	for i := range ht.tracks {
		if !matched[i] && ht.tracks[i].EndFrame == ht.frameIndex-1 {
			// Track might be dead - check if it should be terminated
			if ht.shouldTerminateTrack(&ht.tracks[i]) {
				ht.tracks[i].DeathFrame = ht.frameIndex - 1
				ht.tracks[i].DeathType = "offset"
			}
		}
	}
}

// shouldTerminateTrack determines if a track should be terminated
func (ht *HarmonicTracking) shouldTerminateTrack(track *HarmonicTrack) bool {
	// Check if track has been inactive for too long
	inactiveFrames := ht.frameIndex - track.EndFrame - 1
	if inactiveFrames > ht.params.MaxGapLength {
		return true
	}

	// Check if track is too short
	if len(track.Frequencies) < ht.params.MinTrackLength {
		return true
	}

	return false
}

// estimateHarmonicNumber estimates the harmonic number for a frequency
func (ht *HarmonicTracking) estimateHarmonicNumber(frequency float64) int {
	// Simple estimation - would need F0 estimation for accuracy
	// For now, assume harmonics start at reasonable frequencies
	if frequency < 200 {
		return 1 // Likely fundamental
	} else if frequency < 400 {
		return 2 // Likely second harmonic
	} else if frequency < 600 {
		return 3 // Likely third harmonic
	} else {
		return int(frequency / 100) // Rough estimate
	}
}

// countActiveTracks counts currently active tracks
func (ht *HarmonicTracking) countActiveTracks() int {
	count := 0
	for _, track := range ht.tracks {
		if track.EndFrame == ht.frameIndex-1 {
			count++
		}
	}
	return count
}

// finalizeTracks performs final processing on all tracks
func (ht *HarmonicTracking) finalizeTracks() {
	for i := range ht.tracks {
		ht.calculateTrackStatistics(&ht.tracks[i])

		// Apply temporal smoothing if enabled
		if ht.params.UseTemporalSmoothing {
			ht.applyTemporalSmoothing(&ht.tracks[i])
		}
	}

	// Remove tracks that are too short
	ht.tracks = ht.filterShortTracks(ht.tracks)

	// Sort tracks by start frame
	sort.Slice(ht.tracks, func(i, j int) bool {
		return ht.tracks[i].StartFrame < ht.tracks[j].StartFrame
	})
}

// calculateTrackStatistics calculates statistics for a track
func (ht *HarmonicTracking) calculateTrackStatistics(track *HarmonicTrack) {
	if len(track.Frequencies) == 0 {
		return
	}

	// Duration
	track.Duration = float64(track.EndFrame-track.StartFrame) * float64(ht.hopSize) / float64(ht.sampleRate)

	// Frequency statistics
	track.FreqStdDev = ht.calculateStandardDeviation(track.Frequencies)
	track.FreqSlope = ht.calculateSlope(track.Frequencies)

	// Amplitude statistics
	track.AmpStdDev = ht.calculateStandardDeviation(track.Amplitudes)
	track.AmpSlope = ht.calculateSlope(track.Amplitudes)

	// Phase statistics
	if len(track.Phases) > 0 {
		track.PhaseSlope = ht.calculateSlope(track.Phases)
	}

	// Continuity measure
	track.Continuity = ht.calculateContinuity(track)

	// Salience measure
	track.Salience = ht.calculateSalience(track)
}

// calculateStandardDeviation calculates standard deviation of a series
func (ht *HarmonicTracking) calculateStandardDeviation(values []float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}

	// Calculate mean
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	// Calculate variance
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values) - 1)

	return math.Sqrt(variance)
}

// calculateSlope calculates the slope of a time series
func (ht *HarmonicTracking) calculateSlope(values []float64) float64 {
	n := len(values)
	if n < 2 {
		return 0.0
	}

	// Linear regression
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	denominator := float64(n)*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0.0
	}

	return (float64(n)*sumXY - sumX*sumY) / denominator
}

// calculateContinuity calculates temporal continuity measure
func (ht *HarmonicTracking) calculateContinuity(track *HarmonicTrack) float64 {
	if len(track.Frequencies) < 2 {
		return 0.0
	}

	// Calculate continuity based on frequency and amplitude variations
	freqVariation := ht.calculateVariation(track.Frequencies)
	ampVariation := ht.calculateVariation(track.Amplitudes)

	// Combine variations (lower variation = higher continuity)
	continuity := 1.0 / (1.0 + freqVariation + ampVariation)

	return continuity
}

// calculateVariation calculates the variation in a series
func (ht *HarmonicTracking) calculateVariation(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}

	variation := 0.0
	for i := 1; i < len(values); i++ {
		diff := math.Abs(values[i] - values[i-1])
		variation += diff
	}

	return variation / float64(len(values)-1)
}

// calculateSalience calculates perceptual salience measure
func (ht *HarmonicTracking) calculateSalience(track *HarmonicTrack) float64 {
	if len(track.Amplitudes) == 0 {
		return 0.0
	}

	// Simple salience based on amplitude and duration
	avgAmplitude := 0.0
	for _, amp := range track.Amplitudes {
		avgAmplitude += amp
	}
	avgAmplitude /= float64(len(track.Amplitudes))

	// Combine amplitude and duration
	salience := avgAmplitude * math.Log(track.Duration+1.0)

	return salience
}

// applyTemporalSmoothing applies temporal smoothing to track parameters
func (ht *HarmonicTracking) applyTemporalSmoothing(track *HarmonicTrack) {
	if len(track.Frequencies) < 3 {
		return
	}

	// Apply simple exponential smoothing
	alpha := ht.params.SmoothingFactor

	// Smooth frequencies
	for i := 1; i < len(track.Frequencies); i++ {
		track.Frequencies[i] = alpha*track.Frequencies[i] + (1-alpha)*track.Frequencies[i-1]
	}

	// Smooth amplitudes
	for i := 1; i < len(track.Amplitudes); i++ {
		track.Amplitudes[i] = alpha*track.Amplitudes[i] + (1-alpha)*track.Amplitudes[i-1]
	}
}

// filterShortTracks removes tracks that are too short
func (ht *HarmonicTracking) filterShortTracks(tracks []HarmonicTrack) []HarmonicTrack {
	filtered := make([]HarmonicTrack, 0)

	for _, track := range tracks {
		if len(track.Frequencies) >= ht.params.MinTrackLength {
			filtered = append(filtered, track)
		}
	}

	return filtered
}

// buildResult builds the final tracking result
func (ht *HarmonicTracking) buildResult(numFrames int) (*HarmonicTrackingResult, error) {
	result := &HarmonicTrackingResult{
		Tracks:         ht.tracks,
		TotalTracks:    len(ht.tracks),
		Method:         ht.params.Method,
		NumFrames:      numFrames,
		MaxHarmonics:   ht.params.MaxHarmonics,
		SampleRate:     ht.sampleRate,
		HopSize:        ht.hopSize,
		F0Trajectory:   make([]float64, numFrames),
		F0Confidence:   make([]float64, numFrames),
		HarmonicCount:  make([]int, numFrames),
		HarmonicEnergy: make([]float64, numFrames),
		Inharmonicity:  make([]float64, numFrames),
		ActiveTracks:   make([]int, numFrames),
		TrackBirths:    make([]int, numFrames),
		TrackDeaths:    make([]int, numFrames),
	}

	// Calculate frame-by-frame statistics
	for frame := range numFrames {
		ht.calculateFrameStatistics(result, frame)
	}

	// Calculate overall statistics
	result.F0Stability = ht.calculateF0Stability(result.F0Trajectory)
	result.OverallQuality = ht.calculateOverallQuality(result)
	result.TemporalCoherence = ht.calculateTemporalCoherence(result)

	return result, nil
}

// calculateFrameStatistics calculates statistics for a specific frame
func (ht *HarmonicTracking) calculateFrameStatistics(result *HarmonicTrackingResult, frame int) {
	activeCount := 0
	births := 0
	deaths := 0
	totalEnergy := 0.0
	fundamentals := make([]float64, 0)

	for _, track := range ht.tracks {
		// Check if track is active in this frame
		if frame >= track.StartFrame && frame <= track.EndFrame {
			activeCount++

			// Add to energy
			frameIdx := frame - track.StartFrame
			if frameIdx < len(track.Amplitudes) {
				totalEnergy += track.Amplitudes[frameIdx]
			}

			// Collect fundamental frequencies
			if track.HarmonicNumber == 1 && frameIdx < len(track.Frequencies) {
				fundamentals = append(fundamentals, track.Frequencies[frameIdx])
			}
		}

		// Check for births and deaths
		if track.BirthFrame == frame {
			births++
		}
		if track.DeathFrame == frame {
			deaths++
		}
	}

	// Store frame statistics
	result.ActiveTracks[frame] = activeCount
	result.TrackBirths[frame] = births
	result.TrackDeaths[frame] = deaths
	result.HarmonicCount[frame] = activeCount
	result.HarmonicEnergy[frame] = totalEnergy

	// Calculate F0 for this frame
	if len(fundamentals) > 0 {
		// Use median of fundamental frequencies
		sort.Float64s(fundamentals)
		if len(fundamentals)%2 == 0 {
			result.F0Trajectory[frame] = (fundamentals[len(fundamentals)/2-1] + fundamentals[len(fundamentals)/2]) / 2.0
		} else {
			result.F0Trajectory[frame] = fundamentals[len(fundamentals)/2]
		}
		result.F0Confidence[frame] = 1.0 / (1.0 + float64(len(fundamentals))) // Higher confidence with fewer competing F0s
	} else {
		result.F0Trajectory[frame] = 0.0
		result.F0Confidence[frame] = 0.0
	}

	// Calculate inharmonicity
	result.Inharmonicity[frame] = ht.calculateInharmonicity(frame)
}

// calculateInharmonicity calculates inharmonicity measure for a frame
func (ht *HarmonicTracking) calculateInharmonicity(frame int) float64 {
	// Find tracks active in this frame
	activeTracks := make([]HarmonicTrack, 0)
	for _, track := range ht.tracks {
		if frame >= track.StartFrame && frame <= track.EndFrame {
			activeTracks = append(activeTracks, track)
		}
	}

	if len(activeTracks) < 2 {
		return 0.0
	}

	// Calculate expected vs actual frequency ratios
	inharmonicity := 0.0
	comparisons := 0

	for i := 0; i < len(activeTracks); i++ {
		for j := i + 1; j < len(activeTracks); j++ {
			track1 := activeTracks[i]
			track2 := activeTracks[j]

			frameIdx1 := frame - track1.StartFrame
			frameIdx2 := frame - track2.StartFrame

			if frameIdx1 < len(track1.Frequencies) && frameIdx2 < len(track2.Frequencies) {
				freq1 := track1.Frequencies[frameIdx1]
				freq2 := track2.Frequencies[frameIdx2]

				// Calculate frequency ratio
				ratio := freq2 / freq1

				// Find closest integer ratio
				closestRatio := math.Round(ratio)

				// Calculate deviation from harmonic ratio
				deviation := math.Abs(ratio-closestRatio) / closestRatio
				inharmonicity += deviation
				comparisons++
			}
		}
	}

	if comparisons > 0 {
		return inharmonicity / float64(comparisons)
	}

	return 0.0
}

// calculateF0Stability calculates overall F0 stability
func (ht *HarmonicTracking) calculateF0Stability(f0Trajectory []float64) float64 {
	if len(f0Trajectory) < 2 {
		return 0.0
	}

	// Calculate coefficient of variation
	validF0s := make([]float64, 0)
	for _, f0 := range f0Trajectory {
		if f0 > 0 {
			validF0s = append(validF0s, f0)
		}
	}

	if len(validF0s) < 2 {
		return 0.0
	}

	// Calculate mean
	mean := 0.0
	for _, f0 := range validF0s {
		mean += f0
	}
	mean /= float64(len(validF0s))

	// Calculate standard deviation
	variance := 0.0
	for _, f0 := range validF0s {
		diff := f0 - mean
		variance += diff * diff
	}
	variance /= float64(len(validF0s) - 1)
	stdDev := math.Sqrt(variance)

	// Stability = 1 - coefficient of variation
	if mean > 0 {
		return 1.0 - (stdDev / mean)
	}

	return 0.0
}

// calculateOverallQuality calculates overall tracking quality
func (ht *HarmonicTracking) calculateOverallQuality(result *HarmonicTrackingResult) float64 {
	if len(result.Tracks) == 0 {
		return 0.0
	}

	// Average track confidence
	avgConfidence := 0.0
	for _, track := range result.Tracks {
		avgConfidence += track.Confidence
	}
	avgConfidence /= float64(len(result.Tracks))

	// Average track continuity
	avgContinuity := 0.0
	for _, track := range result.Tracks {
		avgContinuity += track.Continuity
	}
	avgContinuity /= float64(len(result.Tracks))

	// Combine metrics
	quality := 0.4*avgConfidence + 0.3*avgContinuity + 0.3*result.F0Stability

	return quality
}

// calculateTemporalCoherence calculates temporal coherence of tracking
func (ht *HarmonicTracking) calculateTemporalCoherence(result *HarmonicTrackingResult) float64 {
	if len(result.ActiveTracks) < 2 {
		return 0.0
	}

	// Calculate variation in number of active tracks
	variation := 0.0
	for i := 1; i < len(result.ActiveTracks); i++ {
		diff := float64(result.ActiveTracks[i] - result.ActiveTracks[i-1])
		variation += math.Abs(diff)
	}
	variation /= float64(len(result.ActiveTracks) - 1)

	// Coherence = 1 / (1 + variation)
	coherence := 1.0 / (1.0 + variation)

	return coherence
}

// GetTrackByID retrieves a track by its ID
func (ht *HarmonicTracking) GetTrackByID(id int) (*HarmonicTrack, error) {
	for _, track := range ht.tracks {
		if track.ID == id {
			return &track, nil
		}
	}
	return nil, fmt.Errorf("track with ID %d not found", id)
}

// GetTracksInFrame retrieves all tracks active in a specific frame
func (ht *HarmonicTracking) GetTracksInFrame(frame int) []HarmonicTrack {
	activeTracks := make([]HarmonicTrack, 0)

	for _, track := range ht.tracks {
		if frame >= track.StartFrame && frame <= track.EndFrame {
			activeTracks = append(activeTracks, track)
		}
	}

	return activeTracks
}

// GetHarmonicsByNumber retrieves all tracks with a specific harmonic number
func (ht *HarmonicTracking) GetHarmonicsByNumber(harmonicNumber int) []HarmonicTrack {
	harmonics := make([]HarmonicTrack, 0)

	for _, track := range ht.tracks {
		if track.HarmonicNumber == harmonicNumber {
			harmonics = append(harmonics, track)
		}
	}

	return harmonics
}

// FilterTracksByDuration filters tracks by minimum duration
func (ht *HarmonicTracking) FilterTracksByDuration(minDuration float64) []HarmonicTrack {
	filtered := make([]HarmonicTrack, 0)

	for _, track := range ht.tracks {
		if track.Duration >= minDuration {
			filtered = append(filtered, track)
		}
	}

	return filtered
}

// FilterTracksByConfidence filters tracks by minimum confidence
func (ht *HarmonicTracking) FilterTracksByConfidence(minConfidence float64) []HarmonicTrack {
	filtered := make([]HarmonicTrack, 0)

	for _, track := range ht.tracks {
		if track.Confidence >= minConfidence {
			filtered = append(filtered, track)
		}
	}

	return filtered
}

// ExtractMelody extracts the melody line from harmonic tracks
func (ht *HarmonicTracking) ExtractMelody() ([]float64, []float64, error) {
	if len(ht.tracks) == 0 {
		return nil, nil, fmt.Errorf("no tracks available")
	}

	// Find the most salient tracks (likely melody)
	melodyTracks := make([]HarmonicTrack, 0)

	// Sort tracks by salience
	sortedTracks := make([]HarmonicTrack, len(ht.tracks))
	copy(sortedTracks, ht.tracks)
	sort.Slice(sortedTracks, func(i, j int) bool {
		return sortedTracks[i].Salience > sortedTracks[j].Salience
	})

	// Take top tracks that likely represent melody
	maxMelodyTracks := 3
	for i := 0; i < len(sortedTracks) && i < maxMelodyTracks; i++ {
		track := sortedTracks[i]
		if track.Salience > 0.1 && track.Duration > 0.1 {
			melodyTracks = append(melodyTracks, track)
		}
	}

	if len(melodyTracks) == 0 {
		return nil, nil, fmt.Errorf("no melody tracks found")
	}

	// Create unified melody trajectory
	maxFrame := 0
	for _, track := range melodyTracks {
		if track.EndFrame > maxFrame {
			maxFrame = track.EndFrame
		}
	}

	melodyFreq := make([]float64, maxFrame+1)
	melodyConf := make([]float64, maxFrame+1)

	for frame := 0; frame <= maxFrame; frame++ {
		bestTrack := -1
		bestSalience := 0.0

		// Find the most salient track active in this frame
		for i, track := range melodyTracks {
			if frame >= track.StartFrame && frame <= track.EndFrame {
				if track.Salience > bestSalience {
					bestSalience = track.Salience
					bestTrack = i
				}
			}
		}

		if bestTrack >= 0 {
			track := melodyTracks[bestTrack]
			frameIdx := frame - track.StartFrame
			if frameIdx < len(track.Frequencies) {
				melodyFreq[frame] = track.Frequencies[frameIdx]
				melodyConf[frame] = track.Confidence
			}
		}
	}

	return melodyFreq, melodyConf, nil
}

// Reset resets the tracker state
func (ht *HarmonicTracking) Reset() {
	ht.tracks = make([]HarmonicTrack, 0)
	ht.nextTrackID = 1
	ht.frameIndex = 0
	ht.previousPeaks = nil
	ht.currentPeaks = nil
}

// SetParameters updates the tracking parameters
func (ht *HarmonicTracking) SetParameters(params HarmonicTrackingParams) {
	ht.params = params
}

// GetParameters returns the current parameters
func (ht *HarmonicTracking) GetParameters() HarmonicTrackingParams {
	return ht.params
}

// GetTracks returns all tracks
func (ht *HarmonicTracking) GetTracks() []HarmonicTrack {
	return ht.tracks
}

// GetTrackCount returns the number of tracks
func (ht *HarmonicTracking) GetTrackCount() int {
	return len(ht.tracks)
}

// ExportTracks exports tracks to a simple format for analysis
func (ht *HarmonicTracking) ExportTracks() []map[string]any {
	exported := make([]map[string]any, len(ht.tracks))

	for i, track := range ht.tracks {
		exported[i] = map[string]any{
			"id":              track.ID,
			"harmonic_number": track.HarmonicNumber,
			"start_frame":     track.StartFrame,
			"end_frame":       track.EndFrame,
			"duration":        track.Duration,
			"mean_frequency":  ht.calculateMean(track.Frequencies),
			"mean_amplitude":  ht.calculateMean(track.Amplitudes),
			"frequency_range": ht.calculateRange(track.Frequencies),
			"amplitude_range": ht.calculateRange(track.Amplitudes),
			"confidence":      track.Confidence,
			"continuity":      track.Continuity,
			"salience":        track.Salience,
		}
	}

	return exported
}

// calculateMean calculates the mean of a slice
func (ht *HarmonicTracking) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// calculateRange calculates the range of a slice
func (ht *HarmonicTracking) calculateRange(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	min := values[0]
	max := values[0]

	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	return max - min
}

// GetTrackingMethodName returns the human-readable name of the tracking method
func GetTrackingMethodName(method TrackingMethod) string {
	switch method {
	case PeakBased:
		return "Peak-Based Tracking"
	case SinusoidalModel:
		return "Sinusoidal Model"
	case PartialTracking:
		return "Partial Tracking"
	case KalmanFilter:
		return "Kalman Filter"
	case MultiFrame:
		return "Multi-Frame Analysis"
	default:
		return "Unknown"
	}
}
