package chroma

import (
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/harmonic"
)

// HPCPParams holds parameters for HPCP computation
type HPCPParams struct {
	Size              int     `json:"size"`               // Size of output HPCP vector (12, 24, 36)
	ReferenceFreq     float64 `json:"reference_freq"`     // Reference frequency for A4 (440 Hz)
	HarmonicsRemoval  bool    `json:"harmonics_removal"`  // Remove harmonics
	Normalized        bool    `json:"normalized"`         // Normalize output
	WeightType        string  `json:"weight_type"`        // "none", "cosine", "squared_cosine"
	WindowSize        float64 `json:"window_size"`        // Window size in semitones (default 1.0)
	MaxShifted        bool    `json:"max_shifted"`        // Use maximum shifted correlation
	NonLinear         bool    `json:"non_linear"`         // Apply non-linear transformation
	BandPreset        bool    `json:"band_preset"`        // Use preset frequency bands
	MinFreq           float64 `json:"min_freq"`           // Minimum frequency to consider
	MaxFreq           float64 `json:"max_freq"`           // Maximum frequency to consider
	SplitFreq         float64 `json:"split_freq"`         // Split frequency for band preset
	HarmonicsStrength float64 `json:"harmonics_strength"` // Strength factor for harmonics
	MaxHarmonics      int     `json:"max_harmonics"`      // Maximum number of harmonics to consider
}

// HPCPResult contains the result of HPCP computation
type HPCPResult struct {
	HPCP       []float64 `json:"hpcp"`       // Harmonic pitch class profile
	Size       int       `json:"size"`       // Size of HPCP vector
	Resolution float64   `json:"resolution"` // Frequency resolution per bin
	RefFreq    float64   `json:"ref_freq"`   // Reference frequency used
	Energy     float64   `json:"energy"`     // Total energy in HPCP
	Entropy    float64   `json:"entropy"`    // Entropy of distribution
}

// HPCP computes Harmonic Pitch Class Profile from spectral peaks
type HPCP struct {
	params      HPCPParams
	sampleRate  int
	initialized bool

	// Internal state
	weights         []float64 // Weighting coefficients for each bin
	freqBins        []float64 // Frequency bins for HPCP
	harmonicWeights []float64 // Harmonic weighting factors

	// Reuse existing components
	normalizer *common.Normalizer
}

// NewHPCP creates a new HPCP analyzer with default parameters
func NewHPCP(sampleRate int) *HPCP {
	return &HPCP{
		params: HPCPParams{
			Size:              12,
			ReferenceFreq:     440.0,
			HarmonicsRemoval:  false,
			Normalized:        true,
			WeightType:        "cosine",
			WindowSize:        1.0,
			MaxShifted:        false,
			NonLinear:         false,
			BandPreset:        true,
			MinFreq:           40.0,
			MaxFreq:           5000.0,
			SplitFreq:         500.0,
			HarmonicsStrength: 1.0,
			MaxHarmonics:      0,
		},
		sampleRate: sampleRate,
		normalizer: common.NewNormalizer(common.Energy),
	}
}

// NewHPCPWithParams creates a new HPCP analyzer with custom parameters
func NewHPCPWithParams(sampleRate int, params HPCPParams) *HPCP {
	return &HPCP{
		params:     params,
		sampleRate: sampleRate,
		normalizer: common.NewNormalizer(common.Energy),
	}
}

// Initialize sets up internal state for HPCP computation
func (h *HPCP) Initialize() {
	if h.initialized {
		return
	}

	// Pre-compute frequency bins for HPCP
	h.freqBins = make([]float64, h.params.Size)
	for i := 0; i < h.params.Size; i++ {
		// Each bin represents a semitone
		semitone := float64(i) * 12.0 / float64(h.params.Size)
		// Convert semitone to frequency (A4 = 440 Hz at semitone 9)
		h.freqBins[i] = h.params.ReferenceFreq * math.Pow(2.0, (semitone-9.0)/12.0)
	}

	// Pre-compute weights
	h.computeWeights()

	// Pre-compute harmonic weights if needed
	if h.params.MaxHarmonics > 0 {
		h.computeHarmonicWeights()
	}

	h.initialized = true
}

// computeWeights pre-computes weighting coefficients
func (h *HPCP) computeWeights() {
	h.weights = make([]float64, h.params.Size)

	switch h.params.WeightType {
	case "cosine":
		for i := 0; i < h.params.Size; i++ {
			// Cosine weighting within window
			angle := math.Pi * float64(i) / float64(h.params.Size-1)
			h.weights[i] = math.Cos(angle)
		}
	case "squared_cosine":
		for i := 0; i < h.params.Size; i++ {
			angle := math.Pi * float64(i) / float64(h.params.Size-1)
			weight := math.Cos(angle)
			h.weights[i] = weight * weight
		}
	default: // "none"
		for i := 0; i < h.params.Size; i++ {
			h.weights[i] = 1.0
		}
	}
}

// computeHarmonicWeights pre-computes harmonic weighting factors
func (h *HPCP) computeHarmonicWeights() {
	h.harmonicWeights = make([]float64, h.params.MaxHarmonics+1)

	for i := 1; i <= h.params.MaxHarmonics; i++ {
		// Weight decreases with harmonic number
		h.harmonicWeights[i] = h.params.HarmonicsStrength / float64(i)
	}
}

// ComputeFromSpectralPeaks computes HPCP from spectral peaks
func (h *HPCP) ComputeFromSpectralPeaks(peaks []harmonic.SpectralPeak) HPCPResult {
	if !h.initialized {
		h.Initialize()
	}

	hpcp := make([]float64, h.params.Size)

	for _, peak := range peaks {
		// Skip peaks outside frequency range
		if peak.Frequency < h.params.MinFreq || peak.Frequency > h.params.MaxFreq {
			continue
		}

		// Apply band preset filtering if enabled
		if h.params.BandPreset && h.shouldFilterPeak(peak.Frequency) {
			continue
		}

		// Convert frequency to pitch class
		pitchClass := h.frequencyToPitchClass(peak.Frequency)

		// Compute weight for this peak
		weight := h.computePeakWeight(peak)

		// Add contribution to HPCP bins
		h.addPeakContribution(hpcp, pitchClass, weight, peak.Frequency)

		// Add harmonic contributions if enabled
		if h.params.MaxHarmonics > 0 && !h.params.HarmonicsRemoval {
			h.addHarmonicContributions(hpcp, peak)
		}
	}

	// Apply post-processing
	if h.params.NonLinear {
		h.applyNonLinearTransform(hpcp)
	}

	if h.params.Normalized {
		hpcp = h.normalizer.Normalize(hpcp)
	}

	// Apply maximum shifted correlation if enabled
	if h.params.MaxShifted {
		hpcp = h.applyMaxShifted(hpcp)
	}

	return HPCPResult{
		HPCP:       hpcp,
		Size:       h.params.Size,
		Resolution: 12.0 / float64(h.params.Size), // semitones per bin
		RefFreq:    h.params.ReferenceFreq,
		Energy:     h.computeEnergy(hpcp),
		Entropy:    h.computeEntropy(hpcp),
	}
}

// ComputeFromSpectrum computes HPCP directly from magnitude spectrum
func (h *HPCP) ComputeFromSpectrum(magnitude []float64, windowSize int) HPCPResult {
	if !h.initialized {
		h.Initialize()
	}

	// Extract spectral peaks using the correct interface
	peakDetector := harmonic.NewSpectralPeaks(
		h.sampleRate,
		0.00001, // minPeakHeight
		20.0,    // minPeakDistance
		60,      // maxPeaks
	)

	peaks := peakDetector.DetectPeaks(magnitude, windowSize)

	return h.ComputeFromSpectralPeaks(peaks)
}

// frequencyToPitchClass converts frequency to pitch class (0-11 or 0-size)
func (h *HPCP) frequencyToPitchClass(freq float64) float64 {
	// Convert frequency to MIDI note number
	midiNote := 69 + 12*math.Log2(freq/h.params.ReferenceFreq)

	// Extract pitch class and scale to HPCP size
	pitchClass := math.Mod(midiNote, 12)
	if pitchClass < 0 {
		pitchClass += 12
	}

	// Scale to HPCP size
	return pitchClass * float64(h.params.Size) / 12.0
}

// computePeakWeight computes the weight for a spectral peak
func (h *HPCP) computePeakWeight(peak harmonic.SpectralPeak) float64 {
	weight := peak.Magnitude

	// Apply frequency-dependent weighting if band preset is used
	if h.params.BandPreset {
		if peak.Frequency < h.params.SplitFreq {
			// Boost lower frequencies
			weight *= 2.0
		}
	}

	return weight
}

// addPeakContribution adds a peak's contribution to the HPCP vector
func (h *HPCP) addPeakContribution(hpcp []float64, pitchClass, weight, freq float64) {
	// Determine which bins to contribute to based on window size
	windowSizeBins := h.params.WindowSize * float64(h.params.Size) / 12.0

	startBin := int(math.Floor(pitchClass - windowSizeBins/2))
	endBin := int(math.Ceil(pitchClass + windowSizeBins/2))

	for bin := startBin; bin <= endBin; bin++ {
		// Handle circular wrapping
		wrappedBin := bin
		for wrappedBin < 0 {
			wrappedBin += h.params.Size
		}
		wrappedBin = wrappedBin % h.params.Size

		// Compute distance from peak center
		distance := math.Abs(float64(bin) - pitchClass)
		if distance > float64(h.params.Size)/2 {
			distance = float64(h.params.Size) - distance
		}

		// Apply window function
		if distance <= windowSizeBins/2 {
			windowWeight := h.computeWindowWeight(distance, windowSizeBins)
			hpcp[wrappedBin] += weight * windowWeight
		}
	}
}

// computeWindowWeight computes the window weight for a given distance
func (h *HPCP) computeWindowWeight(distance, windowSize float64) float64 {
	if windowSize == 0 {
		return 1.0
	}

	switch h.params.WeightType {
	case "cosine":
		angle := math.Pi * distance / windowSize
		return math.Max(0, math.Cos(angle))
	case "squared_cosine":
		angle := math.Pi * distance / windowSize
		cosVal := math.Max(0, math.Cos(angle))
		return cosVal * cosVal
	default:
		return 1.0
	}
}

// addHarmonicContributions adds harmonic contributions to HPCP
func (h *HPCP) addHarmonicContributions(hpcp []float64, peak harmonic.SpectralPeak) {
	for harmonic := 2; harmonic <= h.params.MaxHarmonics; harmonic++ {
		harmonicFreq := peak.Frequency * float64(harmonic)

		// Skip if harmonic is outside range
		if harmonicFreq > h.params.MaxFreq {
			break
		}

		// Compute harmonic pitch class
		harmonicPitchClass := h.frequencyToPitchClass(harmonicFreq)

		// Compute harmonic weight
		harmonicWeight := peak.Magnitude * h.harmonicWeights[harmonic]

		// Add contribution
		h.addPeakContribution(hpcp, harmonicPitchClass, harmonicWeight, harmonicFreq)
	}
}

// shouldFilterPeak determines if a peak should be filtered based on band preset
func (h *HPCP) shouldFilterPeak(freq float64) bool {
	// This is a simplified version - you might want more sophisticated filtering
	return false
}

// applyNonLinearTransform applies non-linear transformation to HPCP
func (h *HPCP) applyNonLinearTransform(hpcp []float64) {
	for i := range hpcp {
		if hpcp[i] > 0 {
			hpcp[i] = math.Log(1 + hpcp[i])
		}
	}
}

// applyMaxShifted applies maximum shifted correlation
func (h *HPCP) applyMaxShifted(hpcp []float64) []float64 {
	maxCorr := 0.0
	bestShift := 0

	// Try all possible shifts
	for shift := 0; shift < h.params.Size; shift++ {
		corr := h.computeShiftedCorrelation(hpcp, shift)
		if corr > maxCorr {
			maxCorr = corr
			bestShift = shift
		}
	}

	// Apply best shift
	return h.shiftHPCP(hpcp, bestShift)
}

// computeShiftedCorrelation computes correlation for a given shift
func (h *HPCP) computeShiftedCorrelation(hpcp []float64, shift int) float64 {
	corr := 0.0
	for i := 0; i < len(hpcp); i++ {
		shiftedIdx := (i + shift) % len(hpcp)
		corr += hpcp[i] * hpcp[shiftedIdx]
	}
	return corr
}

// shiftHPCP shifts HPCP vector by given amount
func (h *HPCP) shiftHPCP(hpcp []float64, shift int) []float64 {
	shifted := make([]float64, len(hpcp))
	for i := 0; i < len(hpcp); i++ {
		shifted[i] = hpcp[(i+shift)%len(hpcp)]
	}
	return shifted
}

// computeEnergy computes total energy in HPCP
func (h *HPCP) computeEnergy(hpcp []float64) float64 {
	energy := 0.0
	for _, val := range hpcp {
		energy += val * val
	}
	return math.Sqrt(energy)
}

// computeEntropy computes entropy of HPCP distribution
func (h *HPCP) computeEntropy(hpcp []float64) float64 {
	// Normalize to probability distribution
	sum := 0.0
	for _, val := range hpcp {
		sum += val
	}

	if sum == 0 {
		return 0
	}

	entropy := 0.0
	for _, val := range hpcp {
		if val > 0 {
			prob := val / sum
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

// GetParams returns the current parameters
func (h *HPCP) GetParams() HPCPParams {
	return h.params
}

// SetParams updates the parameters
func (h *HPCP) SetParams(params HPCPParams) {
	h.params = params
	h.initialized = false // Force re-initialization
}
