package fingerprint

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/transcode"
)

func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateMFCCStats(mfcc [][]float64, statType string) []float64 {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return nil
	}

	numCoeffs := len(mfcc[0])

	// Handle delta calculation separately since it has different logic
	if statType == "delta" {
		if len(mfcc) <= 1 {
			return nil
		}

		deltaSum := make([]float64, numCoeffs)

		for t := 1; t < len(mfcc); t++ {
			for c := 0; c < numCoeffs && c < len(mfcc[t]) && c < len(mfcc[t-1]); c++ {
				delta := mfcc[t][c] - mfcc[t-1][c]
				deltaSum[c] += math.Abs(delta)
			}
		}

		// Average the deltas
		for c := range deltaSum {
			deltaSum[c] /= float64(len(mfcc) - 1)
		}

		return deltaSum
	}

	// Handle mean and std calculations
	stats := make([]float64, numCoeffs)

	for c := range numCoeffs {
		var values []float64
		for t := range len(mfcc) {
			if c < len(mfcc[t]) {
				values = append(values, mfcc[t][c])
			}
		}

		if len(values) == 0 {
			continue
		}

		switch statType {
		case "mean":
			stats[c] = calculateMean(values)
		case "std":
			mean := calculateMean(values)
			variance := 0.0
			for _, v := range values {
				variance += (v - mean) * (v - mean)
			}
			if len(values) > 1 {
				stats[c] = math.Sqrt(variance / float64(len(values)-1)) // Use sample std deviation
			} else {
				stats[c] = 0 // Single value has no variance
			}
		}
	}

	return stats
}

func calculateDuration(audioData *transcode.AudioData) time.Duration {
	if audioData.SampleRate <= 0 {
		return 0
	}
	seconds := float64(len(audioData.PCM)) / float64(audioData.SampleRate*audioData.Channels)
	return time.Duration(seconds * float64(time.Second))
}

func generateID(audioData *transcode.AudioData) string {
	hasher := sha256.New()
	fmt.Fprintf(hasher, "%d_%d_%d",
		time.Now().UnixNano(),
		len(audioData.PCM),
		audioData.SampleRate)
	return hex.EncodeToString(hasher.Sum(nil))[:16]
}

func addMetadata(fingerprint *AudioFingerprint, audioData *transcode.AudioData, extractor extractors.FeatureExtractor, config *FingerprintConfig) {
	fingerprint.Metadata["extractor_name"] = extractor.GetName()
	fingerprint.Metadata["feature_weights"] = extractor.GetFeatureWeights()
	fingerprint.Metadata["generation_time"] = time.Now()
	fingerprint.Metadata["config"] = config

	if audioData.Metadata != nil {
		fingerprint.Metadata["stream_metadata"] = audioData.Metadata
	}

	// Add feature statistics
	if fingerprint.Features != nil {
		stats := make(map[string]any)

		if fingerprint.Features.MFCC != nil {
			stats["mfcc_frames"] = len(fingerprint.Features.MFCC)
			if len(fingerprint.Features.MFCC) > 0 {
				stats["mfcc_coefficients"] = len(fingerprint.Features.MFCC[0])
			}
		}

		if fingerprint.Features.SpectralFeatures != nil {
			stats["spectral_frames"] = len(fingerprint.Features.SpectralFeatures.SpectralCentroid)
		}

		fingerprint.Metadata["feature_stats"] = stats
	}
}

func calculateChromaStats(chroma [][]float64, statType string) []float64 {
	if len(chroma) == 0 || len(chroma[0]) == 0 {
		return nil
	}

	numBins := len(chroma[0])
	stats := make([]float64, numBins)

	for b := range numBins {
		var values []float64
		for t := range len(chroma) {
			if b < len(chroma[t]) {
				values = append(values, chroma[t][b])
			}
		}

		if len(values) == 0 {
			continue
		}

		switch statType {
		case "mean":
			stats[b] = calculateMean(values)
		case "std":
			mean := calculateMean(values)
			variance := 0.0
			for _, v := range values {
				variance += (v - mean) * (v - mean)
			}
			stats[b] = math.Sqrt(variance / float64(len(values)))
		}
	}

	return stats
}

func quantizeFloat(val float64, decimals int) float64 {
	factor := math.Pow(10, float64(decimals))
	return math.Round(val*factor) / factor
}

func quantizeSlice(vals []float64, decimals int) []float64 {
	result := make([]float64, len(vals))
	for i, v := range vals {
		result[i] = quantizeFloat(v, decimals)
	}
	return result
}
