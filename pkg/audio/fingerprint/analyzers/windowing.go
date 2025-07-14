package analyzers

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// WindowType represents different window function types
type WindowType string

const (
	WindowHann           WindowType = "hann"
	WindowHamming        WindowType = "hamming"
	WindowBlackman       WindowType = "blackman"
	WindowBlackmanHarris WindowType = "blackman_harris"
	WindowKaiser         WindowType = "kaiser"
	WindowTukey          WindowType = "tukey"
	WindowRectangular    WindowType = "rectangular"
	WindowBartlett       WindowType = "bartlett"
	WindowWelch          WindowType = "welch"
)

// WindowConfig holds window configuration parameters
type WindowConfig struct {
	Type      WindowType `json:"type"`
	Size      int        `json:"size"`
	Beta      float64    `json:"beta"`      // Kaiser window parameter
	Alpha     float64    `json:"alpha"`     // Tukey window parameter (0.0 to 1.0)
	Normalize bool       `json:"normalize"` // Whether to normalize window energy
	Symmetric bool       `json:"symmetric"` // Symmetric vs periodic window
}

// Window represents a window function with its coefficients and metadata
type Window struct {
	Type         WindowType `json:"type"`
	Size         int        `json:"size"`
	Coefficients []float64  `json:"coefficients"`
	Energy       float64    `json:"energy"`       // Total energy of window
	PowerGain    float64    `json:"power_gain"`   // Power gain correction factor
	NoiseGain    float64    `json:"noise_gain"`   // Noise gain correction factor
	ENBW         float64    `json:"enbw"`         // Equivalent Noise Bandwidth
	ScallopLoss  float64    `json:"scallop_loss"` // Scallop loss in dB
	Coherent     bool       `json:"coherent"`     // Whether suitable for coherent averaging
}

// WindowGenerator generates and manages window functions
type WindowGenerator struct {
	logger logging.Logger
	cache  map[string]*Window // Cache for generated windows
}

// NewWindowGenerator creates a new window generator
func NewWindowGenerator() *WindowGenerator {
	return &WindowGenerator{
		logger: logging.WithFields(logging.Fields{
			"component": "window_generator",
		}),
		cache: make(map[string]*Window),
	}
}

// DefaultWindowConfig returns a default window configuration
func DefaultWindowConfig() *WindowConfig {
	return &WindowConfig{
		Type:      WindowHann,
		Size:      2048,
		Beta:      8.6, // Kaiser window parameter
		Alpha:     0.5, // Tukey window parameter
		Normalize: true,
		Symmetric: true,
	}
}

// Generate creates a window with the specified configuration
func (wg *WindowGenerator) Generate(config *WindowConfig) (*Window, error) {
	if config == nil {
		config = DefaultWindowConfig()
	}

	logger := wg.logger.WithFields(logging.Fields{
		"function":    "Generate",
		"window_type": config.Type,
		"window_size": config.Size,
	})

	// Validate configuration
	if err := wg.validateConfig(config); err != nil {
		logger.Error(err, "Invalid window configuration")
		return nil, err
	}

	// Check cache first
	cacheKey := wg.getCacheKey(config)
	if cached, exists := wg.cache[cacheKey]; exists {
		logger.Debug("Returning cached window")
		return cached, nil
	}

	logger.Debug("Generating new window")

	// Generate window coefficients
	coefficients, err := wg.generateCoefficients(config)
	if err != nil {
		logger.Error(err, "Failed to generate window coefficients")
		return nil, err
	}

	// Calculate window properties
	window := &Window{
		Type:         config.Type,
		Size:         config.Size,
		Coefficients: coefficients,
	}

	wg.calculateWindowProperties(window)

	// Apply normalization if requested
	if config.Normalize {
		wg.normalizeWindow(window)
	}

	// Cache the window
	wg.cache[cacheKey] = window

	logger.Info("Window generated successfully", logging.Fields{
		"energy":       window.Energy,
		"power_gain":   window.PowerGain,
		"noise_gain":   window.NoiseGain,
		"enbw":         window.ENBW,
		"scallop_loss": window.ScallopLoss,
	})

	return window, nil
}

// Apply applies the window to a signal
func (w *Window) Apply(signal []float64) ([]float64, error) {
	if len(signal) != w.Size {
		return nil, fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), w.Size)
	}

	windowed := make([]float64, w.Size)
	for i := 0; i < w.Size; i++ {
		windowed[i] = signal[i] * w.Coefficients[i]
	}

	return windowed, nil
}

// ApplyInPlace applies the window to a signal in-place
func (w *Window) ApplyInPlace(signal []float64) error {
	if len(signal) != w.Size {
		return fmt.Errorf("signal length (%d) doesn't match window size (%d)", len(signal), w.Size)
	}

	for i := 0; i < w.Size; i++ {
		signal[i] *= w.Coefficients[i]
	}

	return nil
}

// GetCoherentGain returns the coherent gain of the window
func (w *Window) GetCoherentGain() float64 {
	sum := 0.0
	for _, coeff := range w.Coefficients {
		sum += coeff
	}
	return sum / float64(w.Size)
}

// GetProcessingGain returns the processing gain in dB
func (w *Window) GetProcessingGain() float64 {
	return 10 * math.Log10(w.PowerGain)
}

// validateConfig validates window configuration
func (wg *WindowGenerator) validateConfig(config *WindowConfig) error {
	if config.Size <= 0 {
		return fmt.Errorf("window size must be positive: %d", config.Size)
	}

	if config.Size > 1048576 { // 1M samples max
		return fmt.Errorf("window size too large: %d", config.Size)
	}

	// Validate type-specific parameters
	switch config.Type {
	case WindowKaiser:
		if config.Beta < 0 {
			return fmt.Errorf("Kaiser beta parameter must be non-negative: %f", config.Beta)
		}
	case WindowTukey:
		if config.Alpha < 0 || config.Alpha > 1 {
			return fmt.Errorf("Tukey alpha parameter must be between 0 and 1: %f", config.Alpha)
		}
	}

	return nil
}

// generateCoefficients generates window coefficients based on type
func (wg *WindowGenerator) generateCoefficients(config *WindowConfig) ([]float64, error) {
	coefficients := make([]float64, config.Size)
	// TODO: unused
	//N := float64(config.Size)

	switch config.Type {
	case WindowHann:
		wg.generateHann(coefficients, config.Symmetric)

	case WindowHamming:
		wg.generateHamming(coefficients, config.Symmetric)

	case WindowBlackman:
		wg.generateBlackman(coefficients, config.Symmetric)

	case WindowBlackmanHarris:
		wg.generateBlackmanHarris(coefficients, config.Symmetric)

	case WindowKaiser:
		wg.generateKaiser(coefficients, config.Beta, config.Symmetric)

	case WindowTukey:
		wg.generateTukey(coefficients, config.Alpha, config.Symmetric)

	case WindowRectangular:
		wg.generateRectangular(coefficients)

	case WindowBartlett:
		wg.generateBartlett(coefficients, config.Symmetric)

	case WindowWelch:
		wg.generateWelch(coefficients)

	default:
		return nil, fmt.Errorf("unsupported window type: %s", config.Type)
	}

	return coefficients, nil
}

// generateHann generates Hann window coefficients
func (wg *WindowGenerator) generateHann(coefficients []float64, symmetric bool) {
	N := len(coefficients)
	denominator := float64(N)
	if !symmetric {
		denominator = float64(N)
	} else {
		denominator = float64(N - 1)
	}

	for i := range N {
		coefficients[i] = 0.5 * (1.0 - math.Cos(2*math.Pi*float64(i)/denominator))
	}
}

// generateHamming generates Hamming window coefficients
func (wg *WindowGenerator) generateHamming(coefficients []float64, symmetric bool) {
	N := len(coefficients)
	denominator := float64(N)
	if !symmetric {
		denominator = float64(N)
	} else {
		denominator = float64(N - 1)
	}

	for i := range N {
		coefficients[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/denominator)
	}
}

// generateBlackman generates Blackman window coefficients
func (wg *WindowGenerator) generateBlackman(coefficients []float64, symmetric bool) {
	N := len(coefficients)
	denominator := float64(N)
	if !symmetric {
		denominator = float64(N)
	} else {
		denominator = float64(N - 1)
	}

	a0, a1, a2 := 0.42, 0.5, 0.08

	for i := range N {
		arg := 2 * math.Pi * float64(i) / denominator
		coefficients[i] = a0 - a1*math.Cos(arg) + a2*math.Cos(2*arg)
	}
}

// generateBlackmanHarris generates Blackman-Harris window coefficients
func (wg *WindowGenerator) generateBlackmanHarris(coefficients []float64, symmetric bool) {
	N := len(coefficients)
	denominator := float64(N)
	if !symmetric {
		denominator = float64(N)
	} else {
		denominator = float64(N - 1)
	}

	a0, a1, a2, a3 := 0.35875, 0.48829, 0.14128, 0.01168

	for i := range N {
		arg := 2 * math.Pi * float64(i) / denominator
		coefficients[i] = a0 - a1*math.Cos(arg) + a2*math.Cos(2*arg) - a3*math.Cos(3*arg)
	}
}

// generateKaiser generates Kaiser window coefficients
func (wg *WindowGenerator) generateKaiser(coefficients []float64, beta float64, symmetric bool) {
	N := len(coefficients)
	denominator := float64(N)
	if !symmetric {
		denominator = float64(N)
	} else {
		denominator = float64(N - 1)
	}

	// Calculate I0(beta) for normalization
	i0Beta := wg.besselI0(beta)

	for i := range N {
		arg := 2.0*float64(i)/denominator - 1.0
		coefficients[i] = wg.besselI0(beta*math.Sqrt(1-arg*arg)) / i0Beta
	}
}

// generateTukey generates Tukey window coefficients
func (wg *WindowGenerator) generateTukey(coefficients []float64, alpha float64, symmetric bool) {
	N := len(coefficients)

	// Tukey window is rectangular in the middle with cosine tapers on the sides
	taperLength := int(alpha * float64(N) / 2.0)

	for i := range N {
		if i < taperLength {
			// Rising cosine taper
			arg := math.Pi * float64(i) / float64(taperLength)
			coefficients[i] = 0.5 * (1 + math.Cos(arg-math.Pi))
		} else if i >= N-taperLength {
			// Falling cosine taper
			arg := math.Pi * float64(i-(N-taperLength)) / float64(taperLength)
			coefficients[i] = 0.5 * (1 + math.Cos(arg))
		} else {
			// Rectangular middle section
			coefficients[i] = 1.0
		}
	}
}

// generateRectangular generates rectangular window coefficients
func (wg *WindowGenerator) generateRectangular(coefficients []float64) {
	for i := range coefficients {
		coefficients[i] = 1.0
	}
}

// generateBartlett generates Bartlett (triangular) window coefficients
func (wg *WindowGenerator) generateBartlett(coefficients []float64, symmetric bool) {
	N := len(coefficients)

	for i := range N {
		if i <= N/2 {
			coefficients[i] = 2.0 * float64(i) / float64(N-1)
		} else {
			coefficients[i] = 2.0 - 2.0*float64(i)/float64(N-1)
		}
	}
}

// generateWelch generates Welch window coefficients
func (wg *WindowGenerator) generateWelch(coefficients []float64) {
	N := len(coefficients)

	for i := range N {
		arg := (float64(i) - float64(N-1)/2.0) / (float64(N-1) / 2.0)
		coefficients[i] = 1.0 - arg*arg
	}
}

// besselI0 computes the zero-order modified Bessel function of the first kind
func (wg *WindowGenerator) besselI0(x float64) float64 {
	// Series expansion approximation
	sum := 1.0
	term := 1.0

	for k := 1; k < 50; k++ {
		term *= (x / (2.0 * float64(k))) * (x / (2.0 * float64(k)))
		sum += term

		// Check for convergence
		if term < 1e-12 {
			break
		}
	}

	return sum
}

// calculateWindowProperties calculates various window properties
func (wg *WindowGenerator) calculateWindowProperties(window *Window) {
	N := float64(window.Size)

	// Calculate energy (sum of squares)
	energy := 0.0
	for _, coeff := range window.Coefficients {
		energy += coeff * coeff
	}
	window.Energy = energy

	// Calculate coherent gain (sum of coefficients)
	coherentSum := 0.0
	for _, coeff := range window.Coefficients {
		coherentSum += coeff
	}

	// Power gain (for incoherent averaging)
	window.PowerGain = energy / N

	// Noise gain (for coherent averaging)
	window.NoiseGain = coherentSum / N

	// Equivalent Noise Bandwidth
	window.ENBW = N * energy / (coherentSum * coherentSum)

	// Scallop loss (worst-case loss between bins)
	// For most windows, this is approximately the coherent gain
	window.ScallopLoss = -20 * math.Log10(math.Abs(window.NoiseGain))

	// Determine if suitable for coherent averaging
	window.Coherent = window.NoiseGain > 0.5 // Threshold for coherent suitability
}

// normalizeWindow normalizes the window for unity gain
func (wg *WindowGenerator) normalizeWindow(window *Window) {
	// Normalize for unity power gain
	normFactor := 1.0 / math.Sqrt(window.PowerGain)

	for i := range window.Coefficients {
		window.Coefficients[i] *= normFactor
	}

	// Recalculate properties after normalization
	wg.calculateWindowProperties(window)
}

// getCacheKey generates a cache key for the window configuration
func (wg *WindowGenerator) getCacheKey(config *WindowConfig) string {
	return fmt.Sprintf("%s_%d_%f_%f_%v_%v",
		config.Type, config.Size, config.Beta, config.Alpha,
		config.Normalize, config.Symmetric)
}

// GetRecommendedWindow returns a recommended window for a specific use case
func GetRecommendedWindow(useCase string, size int) *WindowConfig {
	config := &WindowConfig{
		Size:      size,
		Normalize: true,
		Symmetric: true,
	}

	switch useCase {
	case "general_analysis":
		config.Type = WindowHann
	case "speech_analysis":
		config.Type = WindowHamming
	case "music_analysis":
		config.Type = WindowBlackman
	case "high_resolution":
		config.Type = WindowBlackmanHarris
	case "low_leakage":
		config.Type = WindowKaiser
		config.Beta = 8.6
	case "transient_analysis":
		config.Type = WindowTukey
		config.Alpha = 0.25
	case "maximum_resolution":
		config.Type = WindowRectangular
	default:
		config.Type = WindowHann // Default fallback
	}

	return config
}

// GetWindowInfo returns information about available window types
func GetWindowInfo() map[WindowType]map[string]any {
	return map[WindowType]map[string]any{
		WindowHann: {
			"description":  "Good general-purpose window with low spectral leakage",
			"use_cases":    []string{"general spectral analysis", "STFT", "filtering"},
			"main_lobe":    "4 bins",
			"side_lobe":    "-31.5 dB",
			"scallop_loss": "1.42 dB",
		},
		WindowHamming: {
			"description":  "Good for speech analysis, better side-lobe suppression than Hann",
			"use_cases":    []string{"speech processing", "communication systems"},
			"main_lobe":    "4 bins",
			"side_lobe":    "-42.7 dB",
			"scallop_loss": "1.75 dB",
		},
		WindowBlackman: {
			"description":  "Excellent side-lobe suppression, good for music analysis",
			"use_cases":    []string{"music analysis", "high dynamic range signals"},
			"main_lobe":    "6 bins",
			"side_lobe":    "-58.1 dB",
			"scallop_loss": "1.1 dB",
		},
		WindowBlackmanHarris: {
			"description":  "Very low side-lobes, best for high-precision analysis",
			"use_cases":    []string{"precision measurements", "scientific analysis"},
			"main_lobe":    "8 bins",
			"side_lobe":    "-92.0 dB",
			"scallop_loss": "0.83 dB",
		},
		WindowKaiser: {
			"description":  "Adjustable trade-off between main-lobe width and side-lobe level",
			"use_cases":    []string{"flexible analysis", "filter design"},
			"main_lobe":    "variable",
			"side_lobe":    "variable (β dependent)",
			"scallop_loss": "variable",
		},
		WindowTukey: {
			"description":  "Rectangular center with cosine tapers, good for transients",
			"use_cases":    []string{"transient analysis", "burst signals"},
			"main_lobe":    "variable",
			"side_lobe":    "variable (α dependent)",
			"scallop_loss": "variable",
		},
		WindowRectangular: {
			"description":  "Maximum frequency resolution, high spectral leakage",
			"use_cases":    []string{"maximum resolution", "sinusoidal signals"},
			"main_lobe":    "2 bins",
			"side_lobe":    "-13.3 dB",
			"scallop_loss": "3.92 dB",
		},
	}
}
