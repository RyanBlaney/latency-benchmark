package extractors

import (
	"fmt"
	"math"
	"testing"
)

func TestAlignmentWithGoDSP(t *testing.T) {
	fmt.Println("=== TESTING ALIGNMENT WITH GO-DSP ===")

	// Create a simple test with known offset
	test := NewSimpleSpeechTest()

	// Create original signal
	original := test.createSpeechLikeSignal()

	// Calculate feature extraction parameters FIRST
	spectrogram := test.createSpectrogram(original)
	hopSize := spectrogram.HopSize
	featureRate := float64(spectrogram.SampleRate) / float64(hopSize)

	fmt.Printf("Feature rate calculation: %d / %d = %.2f Hz\n",
		spectrogram.SampleRate, hopSize, featureRate)

	// Create delay that aligns with feature frames
	delaySeconds := 2.5
	delayFeatureFrames := int(delaySeconds * featureRate) // Round to nearest feature frame
	delaySamples := delayFeatureFrames * hopSize          // Align to hop size boundary
	actualDelaySeconds := float64(delaySamples) / float64(spectrogram.SampleRate)

	fmt.Printf("Delay alignment:\n")
	fmt.Printf("  Requested: %.3f seconds\n", delaySeconds)
	fmt.Printf("  Feature frames: %d\n", delayFeatureFrames)
	fmt.Printf("  Sample delay: %d\n", delaySamples)
	fmt.Printf("  Actual delay: %.3f seconds\n", actualDelaySeconds)

	// Create delayed version with frame-aligned offset
	delayed := make([]float64, len(original))
	for i := range delayed {
		if i >= delaySamples {
			delayed[i] = original[i-delaySamples]
		}
	}

	// Extract features from both
	originalFeatures, err1 := test.extractFeatures(original)
	delayedFeatures, err2 := test.extractFeatures(delayed)

	if err1 != nil || err2 != nil {
		t.Fatalf("Feature extraction failed: %v, %v", err1, err2)
	}

	// Test alignment
	result, err := AlignStreams(originalFeatures, delayedFeatures, 10.0, featureRate)
	if err != nil {
		t.Fatalf("Alignment failed: %v", err)
	}

	// The detected offset should match the actual delay
	expectedOffset := actualDelaySeconds
	error := math.Abs(result.OffsetSeconds - expectedOffset)

	fmt.Printf("ALIGNMENT RESULTS:\n")
	fmt.Printf("  Expected offset: %.3f seconds\n", expectedOffset)
	fmt.Printf("  Detected offset: %.3f seconds\n", result.OffsetSeconds)
	fmt.Printf("  Error: %.3f seconds\n", error)
	fmt.Printf("  Confidence: %.4f\n", result.Confidence)
	fmt.Printf("  Valid: %t\n", result.IsValid)

	// Check if alignment is good
	if error > 0.1 {
		t.Errorf("Offset error too large: %.3f seconds (expected < 0.1)", error)
	}

	if result.Confidence < 0.5 {
		t.Errorf("Confidence too low: %.4f (expected > 0.5)", result.Confidence)
	}

	if !result.IsValid {
		t.Error("Alignment marked as invalid")
	}

	fmt.Println("✅ Alignment with go-dsp test PASSED")
}

func TestEnergyOnlyAlignment(t *testing.T) {
	fmt.Println("=== TESTING ENERGY-ONLY ALIGNMENT ===")

	test := NewSimpleSpeechTest()

	// Create test signals with different delays
	delays := []float64{0.5, 1.0, 2.0, 3.0}

	for _, delaySeconds := range delays {
		t.Run(fmt.Sprintf("Delay_%.1fs", delaySeconds), func(t *testing.T) {
			// Create signals
			original := test.createSpeechLikeSignal()
			delaySamples := int(delaySeconds * float64(test.sampleRate))
			delayed := make([]float64, len(original))

			for i := range delayed {
				if i >= delaySamples {
					delayed[i] = original[i-delaySamples]
				}
			}

			// Extract features
			originalFeatures, _ := test.extractFeatures(original)
			delayedFeatures, _ := test.extractFeatures(delayed)

			// Calculate feature rate
			spectrogram := test.createSpectrogram(original)
			featureRate := float64(spectrogram.SampleRate) / float64(spectrogram.HopSize)

			// Test energy-only alignment
			result, err := AlignStreamsWithEnergyOnly(originalFeatures, delayedFeatures, 10.0, featureRate)
			if err != nil {
				t.Fatalf("Energy alignment failed: %v", err)
			}

			// FIXED: Expected offset should be negative
			expectedOffset := -delaySeconds
			error := math.Abs(result.OffsetSeconds - expectedOffset)

			fmt.Printf("Delay %.1fs: expected %.3fs, detected %.3fs, error %.3fs, confidence %.4f\n",
				delaySeconds, expectedOffset, result.OffsetSeconds, error, result.Confidence)

			if error > 0.3 {
				t.Errorf("Energy alignment error too large: %.3f", error)
			}
		})
	}

	fmt.Println("✅ Energy-only alignment test PASSED")
}

func TestSelfAlignment(t *testing.T) {
	fmt.Println("=== TESTING SELF-ALIGNMENT ===")

	test := NewSimpleSpeechTest()

	// Create signal
	signal := test.createSpeechLikeSignal()

	// Extract features (same signal twice)
	features1, err1 := test.extractFeatures(signal)
	features2, err2 := test.extractFeatures(signal)

	if err1 != nil || err2 != nil {
		t.Fatalf("Feature extraction failed: %v, %v", err1, err2)
	}

	// Calculate feature rate
	spectrogram := test.createSpectrogram(signal)
	featureRate := float64(spectrogram.SampleRate) / float64(spectrogram.HopSize)

	// Test self-alignment (should be 0 offset, high confidence)
	result, err := AlignStreams(features1, features2, 10.0, featureRate)
	if err != nil {
		t.Fatalf("Self-alignment failed: %v", err)
	}

	fmt.Printf("SELF-ALIGNMENT RESULTS:\n")
	fmt.Printf("  Detected offset: %.6f seconds\n", result.OffsetSeconds)
	fmt.Printf("  Confidence: %.4f\n", result.Confidence)
	fmt.Printf("  Valid: %t\n", result.IsValid)

	// Self-alignment should be near-perfect
	if math.Abs(result.OffsetSeconds) > 0.1 {
		t.Errorf("Self-alignment offset too large: %.6f seconds", result.OffsetSeconds)
	}

	if result.Confidence < 0.8 {
		t.Errorf("Self-alignment confidence too low: %.4f", result.Confidence)
	}

	fmt.Println("✅ Self-alignment test PASSED")
}
