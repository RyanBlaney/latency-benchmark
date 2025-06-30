package fingerprint

// SpectralAnalyzer provides FFT and spectral analysis functions
type SpectralAnalyzer struct {
	windowSize int
	sampleRate int
}

// NewSpectralAnalyzer creates a new spectral analyzer
func NewSpectralAnalyzer(windowSize, sampleRate int) *SpectralAnalyzer {
	return &SpectralAnalyzer{
		windowSize: windowSize,
		sampleRate: sampleRate,
	}
}

// Complex represents a complex number
type Complex struct {
	Real, Imag float64
}

// FFTResult holds FFT aanlysis results
type FFTResult struct {
	Magnitude []float64 `json:"magnitude"`
	Phase     []float64 `json:"phase"`
	Frequency []float64 `json:"frequency"`
	PowerdB   []float64 `json:"power_db"`
}

// WindowType represents different window functions
type WindowType int

const (
	WindowHann WindowType = iota
	WindowHamming
	WindowBlackman
	WindowRectangular
)
