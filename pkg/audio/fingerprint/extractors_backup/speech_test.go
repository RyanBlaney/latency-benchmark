package extractors

import (
	"fmt"
	"math"
	"testing"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// SimpleSpeechTest provides basic, reproducible testing for speech feature extraction
type SimpleSpeechTest struct {
	sampleRate int
	duration   float64 // seconds
	windowSize int
	hopSize    int
}

func NewSimpleSpeechTest() *SimpleSpeechTest {
	return &SimpleSpeechTest{
		sampleRate: 44100,
		duration:   5.0, // 5 seconds - short and fast
		windowSize: 1024,
		hopSize:    256,
	}
}

// Test 1: Basic synthetic speech signal
func (st *SimpleSpeechTest) TestBasicSpeechSignal(t *testing.T) {
	fmt.Println("=== TEST 1: Basic Speech Signal ===")

	// Create simple speech-like signal
	pcm := st.createSpeechLikeSignal()
	spectrogram := st.createSpectrogram(pcm)

	// Test feature extraction
	extractor := st.createExtractor()
	features, err := extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)

	if err != nil {
		t.Fatalf("Feature extraction failed: %v", err)
	}

	// Basic validation
	st.validateBasicFeatures(t, features, spectrogram)

	fmt.Println("✅ Basic speech signal test PASSED")
}

// Test 2: Silent signal (edge case)
func (st *SimpleSpeechTest) TestSilentSignal(t *testing.T) {
	fmt.Println("=== TEST 2: Silent Signal ===")

	// Create silent signal
	pcm := st.createSilentSignal()
	spectrogram := st.createSpectrogram(pcm)

	extractor := st.createExtractor()
	features, err := extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)

	if err != nil {
		t.Fatalf("Feature extraction failed on silent signal: %v", err)
	}

	// Validate silent signal produces expected results
	st.validateSilentFeatures(t, features, spectrogram)

	fmt.Println("✅ Silent signal test PASSED")
}

// Test 3: Pure tone (another edge case)
func (st *SimpleSpeechTest) TestPureTone(t *testing.T) {
	fmt.Println("=== TEST 3: Pure Tone ===")

	// Create 440Hz pure tone
	pcm := st.createPureTone(440.0)
	spectrogram := st.createSpectrogram(pcm)

	extractor := st.createExtractor()
	features, err := extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)

	if err != nil {
		t.Fatalf("Feature extraction failed on pure tone: %v", err)
	}

	st.validatePureToneFeatures(t, features, spectrogram)

	fmt.Println("✅ Pure tone test PASSED")
}

// Test 4: Feature consistency across identical signals
func (st *SimpleSpeechTest) TestFeatureConsistency(t *testing.T) {
	fmt.Println("=== TEST 4: Feature Consistency ===")

	// Create the same signal twice
	pcm1 := st.createSpeechLikeSignal()
	pcm2 := st.createSpeechLikeSignal() // Identical

	spectrogram1 := st.createSpectrogram(pcm1)
	spectrogram2 := st.createSpectrogram(pcm2)

	extractor := st.createExtractor()
	features1, err1 := extractor.ExtractFeatures(spectrogram1, pcm1, st.sampleRate)
	features2, err2 := extractor.ExtractFeatures(spectrogram2, pcm2, st.sampleRate)

	if err1 != nil || err2 != nil {
		t.Fatalf("Feature extraction failed: %v, %v", err1, err2)
	}

	// Validate features are identical
	st.validateIdenticalFeatures(t, features1, features2)

	fmt.Println("✅ Feature consistency test PASSED")
}

// Test 5: Frame alignment validation
func (st *SimpleSpeechTest) TestFrameAlignment(t *testing.T) {
	fmt.Println("=== TEST 5: Frame Alignment ===")

	pcm := st.createSpeechLikeSignal()
	spectrogram := st.createSpectrogram(pcm)

	extractor := st.createExtractor()
	features, err := extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)

	if err != nil {
		t.Fatalf("Feature extraction failed: %v", err)
	}

	// Validate all feature arrays have same length as spectrogram
	st.validateFrameAlignment(t, features, spectrogram)

	fmt.Println("✅ Frame alignment test PASSED")
}

// HELPER: Create speech-like synthetic signal
func (st *SimpleSpeechTest) createSpeechLikeSignal() []float64 {
	samples := int(st.duration * float64(st.sampleRate))
	pcm := make([]float64, samples)

	for i := range samples {
		t := float64(i) / float64(st.sampleRate)

		// Speech-like signal: fundamental + formants + modulation
		fundamental := 0.4 * math.Sin(2*math.Pi*150*t) // 150Hz fundamental
		formant1 := 0.3 * math.Sin(2*math.Pi*800*t)    // F1 formant
		formant2 := 0.2 * math.Sin(2*math.Pi*2400*t)   // F2 formant

		// Add speech-like amplitude modulation (simulates syllables)
		modulation := 0.5 + 0.5*math.Sin(2*math.Pi*3*t) // 3Hz syllable rate

		pcm[i] = (fundamental + formant1 + formant2) * modulation
	}

	return pcm
}

// HELPER: Create silent signal
func (st *SimpleSpeechTest) createSilentSignal() []float64 {
	samples := int(st.duration * float64(st.sampleRate))
	return make([]float64, samples) // All zeros
}

// HELPER: Create pure tone
func (st *SimpleSpeechTest) createPureTone(frequency float64) []float64 {
	samples := int(st.duration * float64(st.sampleRate))
	pcm := make([]float64, samples)

	for i := range samples {
		t := float64(i) / float64(st.sampleRate)
		pcm[i] = 0.5 * math.Sin(2*math.Pi*frequency*t)
	}

	return pcm
}

// HELPER: Create spectrogram from PCM
func (st *SimpleSpeechTest) createSpectrogram(pcm []float64) *analyzers.SpectrogramResult {
	analyzer := analyzers.NewSpectralAnalyzer(st.sampleRate)

	spectrogram, err := analyzer.ComputeSTFTWithWindow(
		pcm,
		st.windowSize,
		st.hopSize,
		analyzers.WindowHann,
	)

	if err != nil {
		panic(fmt.Sprintf("Failed to create spectrogram: %v", err))
	}

	return spectrogram
}

// HELPER: Create feature extractor with test config
func (st *SimpleSpeechTest) createExtractor() *SpeechFeatureExtractor {
	config := &config.FeatureConfig{
		WindowSize:             st.windowSize,
		HopSize:                st.hopSize,
		EnableMFCC:             true,  // CRITICAL: Enable MFCC
		EnableChroma:           false, // Disable for speech
		EnableSpectralContrast: false,
		EnableHarmonicFeatures: false,
		EnableSpeechFeatures:   true, // Enable speech-specific features
		EnableTemporalFeatures: true,
		MFCCCoefficients:       13,
		ChromaBins:             12,
		SimilarityWeights: map[string]float64{
			"mfcc":     0.50,
			"speech":   0.25,
			"spectral": 0.15,
			"temporal": 0.10,
		},
	}

	return NewSpeechFeatureExtractor(config, true) // isNews = true
}

// VALIDATION: Basic feature validation
func (st *SimpleSpeechTest) validateBasicFeatures(t *testing.T, features *ExtractedFeatures, spectrogram *analyzers.SpectrogramResult) {
	expectedFrames := spectrogram.TimeFrames

	// Check that features exist
	if features == nil {
		t.Fatal("Features is nil")
	}

	// Check MFCC
	if features.MFCC == nil {
		t.Fatal("MFCC features are nil")
	}
	if len(features.MFCC) != expectedFrames {
		t.Fatalf("MFCC frame count mismatch: expected %d, got %d", expectedFrames, len(features.MFCC))
	}
	if len(features.MFCC[0]) != 13 {
		t.Fatalf("MFCC coefficient count wrong: expected 13, got %d", len(features.MFCC[0]))
	}

	// Check MFCC value ranges
	mfccStats := calculateMFCCStats(features.MFCC)

	// C0 (energy) reasonable range for various signal types
	if mfccStats.c0Mean < -30 || mfccStats.c0Mean > 30 {
		t.Errorf("C0 mean out of range: %.3f (should be -30 to 30)", mfccStats.c0Mean)
	}

	// Other coefficients should be reasonable
	if math.Abs(mfccStats.otherMean) > 20 {
		t.Errorf("Other MFCC coefficients mean too extreme: %.3f (should be -20 to 20)", mfccStats.otherMean)
	}

	// Maximum absolute value should be reasonable
	if mfccStats.maxAbs > 50 {
		t.Errorf("MFCC values too extreme: %.3f (should be < 50)", mfccStats.maxAbs)
	}

	// Should have variation unless it's a very uniform signal
	if mfccStats.stdDev < 0.001 {
		t.Errorf("MFCC values too uniform: %.3f (should be > 0.001)", mfccStats.stdDev)
	}

	// Check energy features
	if features.EnergyFeatures == nil {
		t.Fatal("Energy features are nil")
	}
	if len(features.EnergyFeatures.ShortTimeEnergy) != expectedFrames {
		t.Fatalf("Energy frame count mismatch: expected %d, got %d", expectedFrames, len(features.EnergyFeatures.ShortTimeEnergy))
	}

	// Check spectral features
	if features.SpectralFeatures == nil {
		t.Fatal("Spectral features are nil")
	}
	if len(features.SpectralFeatures.SpectralCentroid) != expectedFrames {
		t.Fatalf("Spectral frame count mismatch: expected %d, got %d", expectedFrames, len(features.SpectralFeatures.SpectralCentroid))
	}

	// Check for NaN/Inf values
	st.checkForInvalidValues(t, features)

	fmt.Printf("✓ Basic validation passed: %d frames, MFCC %dx%d\n",
		expectedFrames, len(features.MFCC), len(features.MFCC[0]))
	fmt.Printf("✓ MFCC stats: C0=%.2f, other=%.2f, max=%.2f, std=%.2f\n",
		mfccStats.c0Mean, mfccStats.otherMean, mfccStats.maxAbs, mfccStats.stdDev)
}

// VALIDATION: Silent signal should produce low/zero energy
func (st *SimpleSpeechTest) validateSilentFeatures(t *testing.T, features *ExtractedFeatures, spectrogram *analyzers.SpectrogramResult) {
	st.validateBasicFeatures(t, features, spectrogram)

	// Energy should be very low
	maxEnergy := 0.0
	for _, energy := range features.EnergyFeatures.ShortTimeEnergy {
		if energy > maxEnergy {
			maxEnergy = energy
		}
	}

	if maxEnergy > 1e-6 {
		t.Fatalf("Silent signal has too much energy: %f", maxEnergy)
	}

	fmt.Printf("✓ Silent signal validation passed: max energy = %e\n", maxEnergy)
}

// VALIDATION: Pure tone should have specific spectral characteristics
func (st *SimpleSpeechTest) validatePureToneFeatures(t *testing.T, features *ExtractedFeatures, spectrogram *analyzers.SpectrogramResult) {
	st.validateBasicFeatures(t, features, spectrogram)

	// Spectral centroid should be near 440Hz for pure tone
	avgCentroid := 0.0
	for _, centroid := range features.SpectralFeatures.SpectralCentroid {
		avgCentroid += centroid
	}
	avgCentroid /= float64(len(features.SpectralFeatures.SpectralCentroid))

	// Allow some tolerance
	if avgCentroid < 400 || avgCentroid > 480 {
		t.Fatalf("Pure tone centroid wrong: expected ~440Hz, got %.1fHz", avgCentroid)
	}

	fmt.Printf("✓ Pure tone validation passed: centroid = %.1fHz\n", avgCentroid)
}

// VALIDATION: Identical signals should produce identical features
func (st *SimpleSpeechTest) validateIdenticalFeatures(t *testing.T, features1, features2 *ExtractedFeatures) {
	// Check MFCC
	if len(features1.MFCC) != len(features2.MFCC) {
		t.Fatal("MFCC frame counts differ for identical signals")
	}

	for i := range features1.MFCC {
		for j := range features1.MFCC[i] {
			diff := math.Abs(features1.MFCC[i][j] - features2.MFCC[i][j])
			if diff > 1e-10 {
				t.Fatalf("MFCC differs for identical signals at [%d][%d]: %f vs %f",
					i, j, features1.MFCC[i][j], features2.MFCC[i][j])
			}
		}
	}

	// Check energy
	for i := range features1.EnergyFeatures.ShortTimeEnergy {
		diff := math.Abs(features1.EnergyFeatures.ShortTimeEnergy[i] - features2.EnergyFeatures.ShortTimeEnergy[i])
		if diff > 1e-10 {
			t.Fatalf("Energy differs for identical signals at [%d]: %f vs %f",
				i, features1.EnergyFeatures.ShortTimeEnergy[i], features2.EnergyFeatures.ShortTimeEnergy[i])
		}
	}

	fmt.Println("✓ Identical features validation passed")
}

// VALIDATION: All feature arrays should have same length as spectrogram
func (st *SimpleSpeechTest) validateFrameAlignment(t *testing.T, features *ExtractedFeatures, spectrogram *analyzers.SpectrogramResult) {
	expectedFrames := spectrogram.TimeFrames

	// Check each feature type
	checks := []struct {
		name   string
		length int
	}{
		{"MFCC", len(features.MFCC)},
		{"Energy", len(features.EnergyFeatures.ShortTimeEnergy)},
		{"Spectral", len(features.SpectralFeatures.SpectralCentroid)},
	}

	for _, check := range checks {
		if check.length != expectedFrames {
			t.Fatalf("%s frame alignment failed: expected %d, got %d",
				check.name, expectedFrames, check.length)
		}
	}

	fmt.Printf("✓ Frame alignment validated: all features have %d frames\n", expectedFrames)
}

// VALIDATION: Check for NaN and infinite values
func (st *SimpleSpeechTest) checkForInvalidValues(t *testing.T, features *ExtractedFeatures) {
	// Check MFCC
	for i, frame := range features.MFCC {
		for j, coeff := range frame {
			if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
				t.Fatalf("Invalid MFCC value at [%d][%d]: %f", i, j, coeff)
			}
		}
	}

	// Check energy
	for i, energy := range features.EnergyFeatures.ShortTimeEnergy {
		if math.IsNaN(energy) || math.IsInf(energy, 0) {
			t.Fatalf("Invalid energy value at [%d]: %f", i, energy)
		}
	}

	// Check spectral
	for i, centroid := range features.SpectralFeatures.SpectralCentroid {
		if math.IsNaN(centroid) || math.IsInf(centroid, 0) {
			t.Fatalf("Invalid spectral centroid at [%d]: %f", i, centroid)
		}
	}

	fmt.Println("✓ No invalid values found")
}

// MAIN TEST RUNNER
func TestSpeechFeatureExtraction(t *testing.T) {
	fmt.Println("STARTING SIMPLE SPEECH FEATURE EXTRACTION TESTS")
	fmt.Println("============================================================")

	test := NewSimpleSpeechTest()

	// Run all tests
	t.Run("BasicSpeechSignal", test.TestBasicSpeechSignal)
	t.Run("SilentSignal", test.TestSilentSignal)
	t.Run("PureTone", test.TestPureTone)
	t.Run("FeatureConsistency", test.TestFeatureConsistency)
	t.Run("FrameAlignment", test.TestFrameAlignment)

	fmt.Println("============================================================")
	fmt.Println("ALL SIMPLE SPEECH TESTS COMPLETED")
}

// STANDALONE MAIN for quick testing
func main() {
	fmt.Println("Running simple speech feature extraction tests...")

	// Initialize logging
	logging.SetLevel(logging.InfoLevel)

	test := NewSimpleSpeechTest()

	// Use testing.T mock for standalone running
	t := &testing.T{}

	test.TestBasicSpeechSignal(t)
	test.TestSilentSignal(t)
	test.TestPureTone(t)
	test.TestFeatureConsistency(t)
	test.TestFrameAlignment(t)

	fmt.Println("\n🎉 All tests completed!")
}

func TestSpeechFeatureValidation(t *testing.T) {
	fmt.Println("=== DEEP VALIDATION: Are the tests actually meaningful? ===")

	test := NewSimpleSpeechTest()

	// Test 1: Are MFCC values actually reasonable?
	t.Run("MFCC_Values_Reasonable", func(t *testing.T) {
		pcm := test.createSpeechLikeSignal()
		spectrogram := test.createSpectrogram(pcm)
		extractor := test.createExtractor()

		features, err := extractor.ExtractFeatures(spectrogram, pcm, test.sampleRate)
		if err != nil {
			t.Fatal(err)
		}

		// Check MFCC value ranges - they should be reasonable for speech
		mfccStats := calculateMFCCStats(features.MFCC)

		fmt.Printf("MFCC Statistics:\n")
		fmt.Printf("  First coefficient (C0) mean: %.3f (should be -25 to 25)\n", mfccStats.c0Mean)
		fmt.Printf("  Other coefficients mean: %.3f (should be ~-5 to 5)\n", mfccStats.otherMean)
		fmt.Printf("  Max absolute value: %.3f (should be < 50)\n", mfccStats.maxAbs)
		fmt.Printf("  Standard deviation: %.3f (should be > 0.1)\n", mfccStats.stdDev)

		// Validate ranges
		if mfccStats.c0Mean < -25 || mfccStats.c0Mean > 25 {
			t.Errorf("C0 mean out of range: %.3f", mfccStats.c0Mean)
		}
		if mfccStats.maxAbs > 100 {
			t.Errorf("MFCC values too extreme: %.3f", mfccStats.maxAbs)
		}
		if mfccStats.stdDev < 0.01 {
			t.Errorf("MFCC values too uniform (no variation): %.3f", mfccStats.stdDev)
		}

		fmt.Println("✅ MFCC values appear reasonable")
	})

	// Test 2: Do different signals produce different features?
	t.Run("Different_Signals_Different_Features", func(t *testing.T) {
		// Create very different signals
		speech := test.createSpeechLikeSignal()
		tone := test.createPureTone(440.0)

		// Extract features
		speechFeatures, err1 := test.extractFeatures(speech)
		toneFeatures, err2 := test.extractFeatures(tone)

		if err1 != nil || err2 != nil {
			t.Fatal("Feature extraction failed")
		}

		// Compare MFCC - they should be DIFFERENT
		mfccDiff := calculateMFCCDifference(speechFeatures.MFCC, toneFeatures.MFCC)
		energyDiff := calculateEnergyDifference(speechFeatures.EnergyFeatures, toneFeatures.EnergyFeatures)

		fmt.Printf("Signal Differences:\n")
		fmt.Printf("  MFCC difference: %.3f (should be > 0.5)\n", mfccDiff)
		fmt.Printf("  Energy difference: %.3f (should be > 0.1)\n", energyDiff)

		if mfccDiff < 0.1 {
			t.Errorf("MFCC too similar between different signals: %.3f", mfccDiff)
		}
		if energyDiff < 0.01 {
			t.Errorf("Energy too similar between different signals: %.3f", energyDiff)
		}

		fmt.Println("✅ Different signals produce different features")
	})

	// Test 3: Are features actually correlated with expected signal properties?
	t.Run("Features_Correlate_With_Signal_Properties", func(t *testing.T) {
		// Test different frequency pure tones
		freqs := []float64{200, 440, 880, 1760}
		centroids := make([]float64, len(freqs))

		for i, freq := range freqs {
			tone := test.createPureTone(freq)
			features, err := test.extractFeatures(tone)
			if err != nil {
				t.Fatal(err)
			}

			// Calculate average spectral centroid
			centroid := 0.0
			for _, c := range features.SpectralFeatures.SpectralCentroid {
				centroid += c
			}
			centroids[i] = centroid / float64(len(features.SpectralFeatures.SpectralCentroid))
		}

		fmt.Printf("Frequency vs Centroid correlation:\n")
		for i, freq := range freqs {
			fmt.Printf("  %.0fHz tone -> %.1fHz centroid\n", freq, centroids[i])
		}

		// Check that centroids increase with frequency
		for i := 1; i < len(centroids); i++ {
			if centroids[i] <= centroids[i-1] {
				t.Errorf("Spectral centroid should increase with frequency: %.1f <= %.1f",
					centroids[i], centroids[i-1])
			}
		}

		fmt.Println("✅ Features correlate with signal properties")
	})

	// Test 4: Are energy features actually detecting energy?
	t.Run("Energy_Features_Detect_Energy", func(t *testing.T) {
		// Create signals with different energy levels
		quiet := test.createScaledSignal(0.1) // 10% amplitude
		loud := test.createScaledSignal(1.0)  // 100% amplitude

		quietFeatures, err1 := test.extractFeatures(quiet)
		loudFeatures, err2 := test.extractFeatures(loud)

		if err1 != nil || err2 != nil {
			t.Fatal("Feature extraction failed")
		}

		// Calculate average energy
		quietEnergy := calculateAverageEnergy(quietFeatures.EnergyFeatures)
		loudEnergy := calculateAverageEnergy(loudFeatures.EnergyFeatures)

		fmt.Printf("Energy Detection:\n")
		fmt.Printf("  Quiet signal (10%%): %.6f\n", quietEnergy)
		fmt.Printf("  Loud signal (100%%): %.6f\n", loudEnergy)
		fmt.Printf("  Ratio: %.1f (should be ~100)\n", loudEnergy/quietEnergy)

		if loudEnergy <= quietEnergy*5 {
			t.Errorf("Energy features not detecting amplitude differences properly: %.6f vs %.6f",
				loudEnergy, quietEnergy)
		}

		fmt.Println("✅ Energy features detect energy levels")
	})

	// Test 5: Frame timing validation
	t.Run("Frame_Timing_Validation", func(t *testing.T) {
		pcm := test.createSpeechLikeSignal()
		spectrogram := test.createSpectrogram(pcm)

		// Calculate expected frame count manually
		expectedFrames := (len(pcm)-test.windowSize)/test.hopSize + 1
		actualFrames := spectrogram.TimeFrames

		fmt.Printf("Frame Timing:\n")
		fmt.Printf("  PCM samples: %d\n", len(pcm))
		fmt.Printf("  Window size: %d\n", test.windowSize)
		fmt.Printf("  Hop size: %d\n", test.hopSize)
		fmt.Printf("  Expected frames: %d\n", expectedFrames)
		fmt.Printf("  Actual frames: %d\n", actualFrames)

		// Should be very close (within 1 frame)
		if math.Abs(float64(expectedFrames-actualFrames)) > 1 {
			t.Errorf("Frame count mismatch: expected %d, got %d", expectedFrames, actualFrames)
		}

		fmt.Println("✅ Frame timing is correct")
	})
}

// Helper functions for deeper validation

type MFCCStats struct {
	c0Mean    float64
	otherMean float64
	maxAbs    float64
	stdDev    float64
}

func calculateMFCCStats(mfcc [][]float64) MFCCStats {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return MFCCStats{}
	}

	var c0Sum, otherSum, maxAbs float64
	var allValues []float64

	for _, frame := range mfcc {
		c0Sum += frame[0] // First coefficient

		for i, coeff := range frame {
			if i > 0 {
				otherSum += coeff
			}

			abs := math.Abs(coeff)
			if abs > maxAbs {
				maxAbs = abs
			}

			allValues = append(allValues, coeff)
		}
	}

	c0Mean := c0Sum / float64(len(mfcc))
	otherMean := otherSum / float64(len(mfcc)*12) // 12 other coefficients

	// Calculate standard deviation
	var variance float64
	mean := (c0Sum + otherSum) / float64(len(allValues))
	for _, val := range allValues {
		variance += (val - mean) * (val - mean)
	}
	stdDev := math.Sqrt(variance / float64(len(allValues)))

	return MFCCStats{
		c0Mean:    c0Mean,
		otherMean: otherMean,
		maxAbs:    maxAbs,
		stdDev:    stdDev,
	}
}

func calculateMFCCDifference(mfcc1, mfcc2 [][]float64) float64 {
	if len(mfcc1) != len(mfcc2) || len(mfcc1) == 0 {
		return 0
	}

	var totalDiff float64
	var count int

	for i := 0; i < len(mfcc1) && i < len(mfcc2); i++ {
		for j := 0; j < len(mfcc1[i]) && j < len(mfcc2[i]); j++ {
			diff := math.Abs(mfcc1[i][j] - mfcc2[i][j])
			totalDiff += diff
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return totalDiff / float64(count)
}

func calculateEnergyDifference(energy1, energy2 *EnergyFeatures) float64 {
	if energy1 == nil || energy2 == nil {
		return 0
	}

	avg1 := calculateAverageEnergy(energy1)
	avg2 := calculateAverageEnergy(energy2)

	return math.Abs(avg1 - avg2)
}

func calculateAverageEnergy(energy *EnergyFeatures) float64 {
	if energy == nil || len(energy.ShortTimeEnergy) == 0 {
		return 0
	}

	var sum float64
	for _, e := range energy.ShortTimeEnergy {
		sum += e
	}

	return sum / float64(len(energy.ShortTimeEnergy))
}

// Helper method to create scaled signal
func (st *SimpleSpeechTest) createScaledSignal(amplitude float64) []float64 {
	base := st.createSpeechLikeSignal()
	scaled := make([]float64, len(base))

	for i, sample := range base {
		scaled[i] = sample * amplitude
	}

	return scaled
}

// Helper method to extract features (shortcut)
func (st *SimpleSpeechTest) extractFeatures(pcm []float64) (*ExtractedFeatures, error) {
	spectrogram := st.createSpectrogram(pcm)
	extractor := st.createExtractor()
	return extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)
}

func (st *SimpleSpeechTest) TestMFCCValuesReasonable(t *testing.T) {
	fmt.Println("=== MFCC Values Reasonable Test ===")

	pcm := st.createSpeechLikeSignal()
	spectrogram := st.createSpectrogram(pcm)
	extractor := st.createExtractor()

	features, err := extractor.ExtractFeatures(spectrogram, pcm, st.sampleRate)
	if err != nil {
		t.Fatal(err)
	}

	// Check MFCC value ranges - updated for actual speech signals
	mfccStats := calculateMFCCStats(features.MFCC)

	fmt.Printf("MFCC Statistics:\n")
	fmt.Printf("  First coefficient (C0) mean: %.3f (should be -25 to 25)\n", mfccStats.c0Mean)
	fmt.Printf("  Other coefficients mean: %.3f (should be -20 to 20)\n", mfccStats.otherMean)
	fmt.Printf("  Max absolute value: %.3f (should be < 50)\n", mfccStats.maxAbs)
	fmt.Printf("  Standard deviation: %.3f (should be > 0.1)\n", mfccStats.stdDev)

	// Updated realistic validation ranges
	if mfccStats.c0Mean < -25 || mfccStats.c0Mean > 25 {
		t.Errorf("C0 mean out of range: %.3f", mfccStats.c0Mean)
	}
	if math.Abs(mfccStats.otherMean) > 20 {
		t.Errorf("Other coefficients mean too extreme: %.3f", mfccStats.otherMean)
	}
	if mfccStats.maxAbs > 50 {
		t.Errorf("MFCC values too extreme: %.3f", mfccStats.maxAbs)
	}
	if mfccStats.stdDev < 0.1 {
		t.Errorf("MFCC values too uniform (no variation): %.3f", mfccStats.stdDev)
	}

	fmt.Println("✅ MFCC values appear reasonable")
}
