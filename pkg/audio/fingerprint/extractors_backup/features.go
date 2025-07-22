package extractors

// ExtractedFeatures holds all the extractable features that we need
type ExtractedFeatures struct {
	SpectralFeatures   *SpectralFeatures `json:"spectral_features,omitempty"`
	MelSpectrum        [][]float64       `json:"mel_spectrum,omitempty"`
	MFCC               [][]float64       `json:"mfcc,omitempty"`
	ChromaFeatures     [][]float64       `json:"chroma_features,omitempty"`
	HarmonicFeatures   *HarmonicFeatures `json:"harmonic_features,omitempty"`
	SpeechFeatures     *SpeechFeatures   `json:"speech_features,omitempty"`
	TemporalFeatures   *TemporalFeatures `json:"temporal_features,omitempty"`
	EnergyFeatures     *EnergyFeatures   `json:"energy_features,omitempty"`
	ExtractionMetadata map[string]any    `json:"extraction_metadata,omitempty"`
}

// SpectralFeatures contains frequency domain characteristics
type SpectralFeatures struct {
	SpectralCentroid  []float64   `json:"spectral_centroid"`
	SpectralRolloff   []float64   `json:"spectral_rolloff"`
	SpectralBandwidth []float64   `json:"spectral_bandwidth"`
	SpectralFlatness  []float64   `json:"spectral_flatness"`
	SpectralCrest     []float64   `json:"spectral_crest"`
	SpectralSlope     []float64   `json:"spectral_slope"`
	SpectralFlux      []float64   `json:"spectral_flux"`
	ZeroCrossingRate  []float64   `json:"zero_crossing_rate"`
	SpectralContrast  [][]float64 `json:"spectral_contrast"`
}

// HarmonicFeatures contains harmonic and pitch-related features
type HarmonicFeatures struct {
	PitchEstimate      []float64 `json:"pitch_estimate"`
	HarmonicRatio      []float64 `json:"harmonic_ratio"`
	InharmonicityRatio []float64 `json:"inharmonicity_ratio"`
	TonalCentroid      []float64 `json:"tonal_centroid"`
}

// SpeechFeatures contains speech-specific characteristics
type SpeechFeatures struct {
	FormantFrequencies [][]float64 `json:"format_frequencies"`
	VoicingProbability []float64   `json:"voicing_probability"`
	SpectralTilt       []float64   `json:"spectral_tilt"`
	SpeechRate         float64     `json:"speech_rate"`
	PauseDuration      []float64   `json:"pause_duration"`
	VocalTractLength   float64     `json:"vocal_tract_length"`
}

// TemporalFeatures contains time domain characteristics
type TemporalFeatures struct {
	RMSEnergy        []float64 `json:"rms_energy"`
	DynamicRange     float64   `json:"dynamic_range"`
	SilenceRatio     float64   `json:"silence_ratio"`
	TempoVariation   float64   `json:"tempo_variation"`
	OnsetDensity     float64   `json:"onset_density"`
	PeakAmplitude    float64   `json:"peak_amplitude"`
	AverageAmplitude float64   `json:"average_amplitude"`
	AttackTime       []float64 `json:"attack_time"`
	DecayTime        []float64 `json:"decay_time"`
}

// EnergyFeatures contains energy and dynamics characteristics
type EnergyFeatures struct {
	ShortTimeEnergy []float64 `json:"short_time_energy"`
	EnergyEntropy   []float64 `json:"energy_entropy"`
	EnergyVariance  float64   `json:"energy_variance"`
	LoudnessRange   float64   `json:"loudness_range"`
	CrestFactor     []float64 `json:"crest_factor"`
}

// calculateZeroCrossingRate computes zero crossing rate
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

func minFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
