package fingerprint

import (
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// AudioFingerprint represents a complete audio fingerprint
type AudioFingerprint struct {
	ID               string                       `json:"id"`
	StreamURL        string                       `json:"stream_url"`
	StreamType       common.StreamType            `json:"stream_type"`
	Timestamp        time.Time                    `json:"timestamp"`
	Duration         time.Duration                `json:"duration"`
	SampleRate       int                          `json:"sample_rate"`
	Channels         int                          `json:"channels"`
	SpectralFeatures *extractors.SpectralFeatures `json:"spectral_features"`
	MelSpectrum      [][]float64                  `json:"mel_spectrum"`
	ChromaFeatures   [][]float64                  `json:"chroma_features"`
	TemporalFeatures *extractors.TemporalFeatures `json:"temporal_features"`
	Hash             string                       `json:"hash"`
	Metadata         map[string]any               `json:"metadata,omitempty"`
}
