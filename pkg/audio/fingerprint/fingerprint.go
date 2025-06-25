package fingerprint

import (
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// AudioFingerprint represents a complete audio fingerprint
type AudioFingerprint struct {
	ID               string                 `json:"id"`
	StreamURL        string                 `json:"stream_url"`
	StreamType       common.StreamType      `json:"stream_type"`
	Timestamp        time.Time              `json:"timestamp"`
	Duration         time.Duration          `json:"duration"`
	SampleRate       int                    `json:"sample_rate"`
	Channels         int                    `json:"channels"`
	SpectralFeatures *SpectralFeatures      `json:"spectral_features"`
	MelSpectrum      [][]float64            `json:"mel_spectrum"`
	ChromaFeatures   [][]float64            `json:"chroma_features"`
	TemporalFeatures *TemporalFeatures      `json:"temporal_features"`
	Hash             string                 `json:"hash"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// SpectralFeatures contains frequency domain characteristics
type SpectralFeatures struct {
	SpectralCentroid  []float64   `json:"spectral_centroid"`
	SpectralRolloff   []float64   `json:"spectral_rolloff"`
	SpectralBandwidth []float64   `json:"spectral_bandwidth"`
	SpectralFlatness  []float64   `json:"spectral_flatness"`
	SpectralCrest     []float64   `json:"spectral_crest"`
	MFCC              [][]float64 `json:"mfcc"`
	ZeroCrossingRate  []float64   `json:"zero_crossing_rate"`
	SpectralContrast  [][]float64 `json:"spectral_contrast"`
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
}

