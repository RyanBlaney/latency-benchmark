package extractors

import (
	"math"
)

// ExtractedFeatures holds the core features needed for MVP
// Organized by content type requirements: speech (news/talk), music, sports, mixed, general
type ExtractedFeatures struct {
	// Core spectral features - needed by all content types
	SpectralFeatures *SpectralFeatures `json:"spectral_features,omitempty"`

	// Perceptual features - important for music and general content
	MFCC           [][]float64 `json:"mfcc,omitempty"`            // Critical for speech recognition and audio similarity
	ChromaFeatures [][]float64 `json:"chroma_features,omitempty"` // Harmonic content for music

	// Speech-specific features - for news/talk content
	SpeechFeatures *SpeechFeatures `json:"speech_features,omitempty"`

	// Temporal features - important for all content types
	TemporalFeatures *TemporalFeatures `json:"temporal_features,omitempty"`

	// Energy features - universal importance
	EnergyFeatures *EnergyFeatures `json:"energy_features,omitempty"`

	// Harmonic features - for music and tonal content
	HarmonicFeatures *HarmonicFeatures `json:"harmonic_features,omitempty"`

	// Extraction metadata
	ExtractionMetadata map[string]any `json:"extraction_metadata,omitempty"`
}

// SpectralFeatures contains frequency domain characteristics
// WHY: Spectral shape describes timbre, brightness, and tonal character
// Critical for distinguishing between different audio content types
type SpectralFeatures struct {
	SpectralCentroid  []float64   `json:"spectral_centroid"`  // Brightness - center of spectral mass
	SpectralRolloff   []float64   `json:"spectral_rolloff"`   // High frequency content indicator
	SpectralBandwidth []float64   `json:"spectral_bandwidth"` // Spectral spread around centroid
	SpectralFlatness  []float64   `json:"spectral_flatness"`  // Tonal vs noise content (Wiener entropy)
	SpectralCrest     []float64   `json:"spectral_crest"`     // Peak-to-average ratio
	SpectralSlope     []float64   `json:"spectral_slope"`     // Overall spectral tilt
	SpectralFlux      []float64   `json:"spectral_flux"`      // Rate of spectral change
	ZeroCrossingRate  []float64   `json:"zero_crossing_rate"` // Noisiness indicator
	SpectralContrast  [][]float64 `json:"spectral_contrast"`  // Contrast between frequency bands
}

// SpeechFeatures contains speech-specific characteristics
// WHY: Essential for news/talk content analysis, speaker identification,
// and distinguishing speech from other audio content
type SpeechFeatures struct {
	// Formant analysis - critical for vowel identification and speaker characteristics
	FormantFrequencies [][]float64 `json:"formant_frequencies"` // F1, F2, F3 formant tracks
	VocalTractLength   float64     `json:"vocal_tract_length"`  // Estimated vocal tract length (cm)

	// Voicing analysis - distinguishes voiced from unvoiced speech
	VoicingProbability []float64 `json:"voicing_probability"` // Frame-by-frame voicing strength

	// Spectral characteristics specific to speech
	SpectralTilt []float64 `json:"spectral_tilt"` // Spectral slope (voice quality)

	// Prosodic features - rhythm and timing of speech
	SpeechRate    float64   `json:"speech_rate"`    // Speaking rate (approximate)
	PauseDuration []float64 `json:"pause_duration"` // Duration of silent periods

	// Voice quality measures
	Jitter  float64 `json:"jitter"`  // Pitch period irregularity (%)
	Shimmer float64 `json:"shimmer"` // Amplitude irregularity (%)
}

// TemporalFeatures contains time domain characteristics
// WHY: Temporal structure reveals onset patterns, dynamics, and rhythmic content
// Important for all content types but especially music and speech prosody
type TemporalFeatures struct {
	// Energy and amplitude measures
	RMSEnergy        []float64 `json:"rms_energy"`        // Root-mean-square energy per frame
	PeakAmplitude    float64   `json:"peak_amplitude"`    // Maximum amplitude
	AverageAmplitude float64   `json:"average_amplitude"` // Mean amplitude

	// Dynamic characteristics
	DynamicRange float64   `json:"dynamic_range"` // Overall dynamic range (dB)
	CrestFactor  []float64 `json:"crest_factor"`  // Peak-to-RMS ratio per frame

	// Activity and silence detection
	SilenceRatio  float64   `json:"silence_ratio"`  // Proportion of silence
	ActivityLevel []float64 `json:"activity_level"` // Audio activity strength

	// Onset characteristics - important for rhythmic content
	OnsetDensity float64   `json:"onset_density"` // Onsets per second
	AttackTime   []float64 `json:"attack_time"`   // Attack time per onset

	// Envelope characteristics
	EnvelopeShape []float64 `json:"envelope_shape"` // Amplitude envelope
}

// EnergyFeatures contains energy distribution and dynamics
// WHY: Energy patterns reveal loudness structure, dynamic behavior,
// and can help distinguish between content types
type EnergyFeatures struct {
	// Frame-based energy analysis
	ShortTimeEnergy []float64 `json:"short_time_energy"` // Energy per analysis frame
	EnergyVariance  float64   `json:"energy_variance"`   // Variance in energy over time
	EnergyEntropy   []float64 `json:"energy_entropy"`    // Local energy distribution entropy

	// Loudness and dynamics
	LoudnessRange float64 `json:"loudness_range"` // Perceptual loudness range

	// Spectral energy distribution
	LowEnergyRatio  []float64 `json:"low_energy_ratio"`  // Low frequency energy proportion
	HighEnergyRatio []float64 `json:"high_energy_ratio"` // High frequency energy proportion
}

// HarmonicFeatures contains harmonic and pitch-related features
// WHY: Harmonic content is crucial for music analysis, pitch tracking,
// and distinguishing tonal from atonal content
type HarmonicFeatures struct {
	// Fundamental frequency tracking
	PitchEstimate   []float64 `json:"pitch_estimate"`   // F0 estimates per frame
	PitchConfidence []float64 `json:"pitch_confidence"` // Confidence in pitch detection
	VoicingStrength []float64 `json:"voicing_strength"` // Strength of harmonic content

	// Harmonic content measures
	HarmonicRatio      []float64 `json:"harmonic_ratio"`      // Harmonic-to-noise ratio
	InharmonicityRatio []float64 `json:"inharmonicity_ratio"` // Inharmonicity coefficients

	// Tonal characteristics
	TonalCentroid []float64 `json:"tonal_centroid"` // Harmonic centroid in pitch space
}

// Simple supporting structures for MVP

// FormantData represents a single formant measurement
type FormantData struct {
	Frequency  float64 `json:"frequency"`  // Formant frequency (Hz)
	Bandwidth  float64 `json:"bandwidth"`  // Formant bandwidth (Hz)
	Confidence float64 `json:"confidence"` // Detection confidence (0-1)
}

// OnsetEvent represents a detected onset
type OnsetEvent struct {
	Time     float64 `json:"time"`     // Onset time (seconds)
	Strength float64 `json:"strength"` // Onset strength
}

// Utility functions for basic calculations

// calculateZeroCrossingRate computes zero crossing rate for a signal frame
// WHY: ZCR is a simple measure of signal noisiness and frequency content
// High ZCR = noisy/high-freq content, Low ZCR = tonal content
func calculateZeroCrossingRate(pcm []float64) float64 {
	if len(pcm) <= 1 {
		return 0
	}
	crossings := 0
	for i := 1; i < len(pcm); i++ {
		if (pcm[i-1] >= 0 && pcm[i] < 0) || (pcm[i-1] < 0 && pcm[i] >= 0) {
			crossings++
		}
	}
	return float64(crossings) / float64(len(pcm)-1)
}

// calculateRMSEnergy computes root-mean-square energy
// WHY: RMS provides perceptually meaningful measure of signal strength
func calculateRMSEnergy(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}
	sum := 0.0
	for _, sample := range pcm {
		sum += sample * sample
	}
	return math.Sqrt(sum / float64(len(pcm)))
}

// calculateSpectralCentroid computes spectral centroid from magnitude spectrum
// WHY: Spectral centroid indicates brightness - center of mass of spectrum
func calculateSpectralCentroid(magnitude []float64, sampleRate int) float64 {
	if len(magnitude) == 0 {
		return 0
	}

	weightedSum := 0.0
	magnitudeSum := 0.0
	freqResolution := float64(sampleRate) / float64(2*(len(magnitude)-1))

	for i, mag := range magnitude {
		freq := float64(i) * freqResolution
		weightedSum += freq * mag
		magnitudeSum += mag
	}

	if magnitudeSum == 0 {
		return 0
	}
	return weightedSum / magnitudeSum
}
