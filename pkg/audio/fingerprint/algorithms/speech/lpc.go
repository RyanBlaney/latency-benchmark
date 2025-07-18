package speech

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/stats"
)

// LPCAnalyzer performs Linear Predictive Coding analysis
// WHY: LPC models the vocal tract as an all-pole filter, essential for
// formant extraction, speech compression, and vocal tract modeling
type LPCAnalyzer struct {
	sampleRate int
	order      int // LPC order (typically 12 + fs/1000)
	autocorr   *stats.AutoCorrelation
}

// LPCResult contains LPC analysis results
type LPCResult struct {
	Coefficients    []float64 `json:"coefficients"`     // LPC coefficients (a1, a2, ..., ap)
	ReflectionCoeff []float64 `json:"reflection_coeff"` // Reflection coefficients (k1, k2, ..., kp)
	Gain            float64   `json:"gain"`             // LPC gain
	ResidualEnergy  float64   `json:"residual_energy"`  // Prediction error energy
	PredictionError []float64 `json:"prediction_error"` // LPC residual signal
	Order           int       `json:"order"`            // LPC order used
	StabilityCheck  bool      `json:"stability_check"`  // Whether filter is stable
}

// NewLPCAnalyzer creates a new LPC analyzer
func NewLPCAnalyzer(sampleRate int, order int) *LPCAnalyzer {
	if order <= 0 {
		order = 12 + sampleRate/1000 // Rule of thumb for speech
	}

	return &LPCAnalyzer{
		sampleRate: sampleRate,
		order:      order,
		autocorr:   stats.NewAutoCorrelation(1024),
	}
}

// Analyze performs LPC analysis on the input signal
func (lpc *LPCAnalyzer) Analyze(signal []float64) (*LPCResult, error) {
	if len(signal) < lpc.order*2 {
		return nil, fmt.Errorf("signal too short for LPC analysis of order %d", lpc.order)
	}

	// Compute autocorrelation sequence
	autocorrResult, err := lpc.autocorr.Compute(signal)
	if err != nil {
		return nil, fmt.Errorf("autocorrelation computation failed: %w", err)
	}

	// Extract autocorrelation values needed for LPC
	R := make([]float64, lpc.order+1)
	maxLen := minInt(len(autocorrResult.Correlations), lpc.order+1)
	copy(R, autocorrResult.Correlations[:maxLen])

	// Perform Levinson-Durbin recursion
	coeffs, reflectionCoeffs, gain, residualEnergy, err := lpc.levinsonDurbin(R)
	if err != nil {
		return nil, fmt.Errorf("Levinson-Durbin algorithm failed: %w", err)
	}

	// Compute prediction error (LPC residual)
	predictionError := lpc.computePredictionError(signal, coeffs)

	// Check filter stability
	stable := lpc.checkStability(coeffs)

	return &LPCResult{
		Coefficients:    coeffs,
		ReflectionCoeff: reflectionCoeffs,
		Gain:            gain,
		ResidualEnergy:  residualEnergy,
		PredictionError: predictionError,
		Order:           lpc.order,
		StabilityCheck:  stable,
	}, nil
}

// levinsonDurbin performs the Levinson-Durbin recursion algorithm
// WHY: Efficient method to solve the Yule-Walker equations for LPC coefficients
func (lpc *LPCAnalyzer) levinsonDurbin(R []float64) ([]float64, []float64, float64, float64, error) {
	p := lpc.order

	if len(R) < p+1 {
		return nil, nil, 0, 0, fmt.Errorf("insufficient autocorrelation values")
	}

	if R[0] == 0 {
		return nil, nil, 0, 0, fmt.Errorf("zero energy signal")
	}

	// Initialize arrays
	a := make([]float64, p+1) // LPC coefficients
	k := make([]float64, p)   // Reflection coefficients
	E := R[0]                 // Prediction error energy

	a[0] = 1.0 // Convention: a[0] = 1

	// Levinson-Durbin recursion
	for i := 1; i <= p; i++ {
		// Calculate reflection coefficient k[i]
		numerator := R[i]
		for j := 1; j < i; j++ {
			numerator -= a[j] * R[i-j]
		}

		if E == 0 {
			return nil, nil, 0, 0, fmt.Errorf("prediction error energy became zero")
		}

		k[i-1] = numerator / E

		// Update LPC coefficients
		a[i] = k[i-1]
		for j := 1; j < i; j++ {
			a[j] = a[j] - k[i-1]*a[i-j]
		}

		// Update prediction error energy
		E *= (1 - k[i-1]*k[i-1])

		if E <= 0 {
			break
		}
	}

	// Calculate gain
	gain := math.Sqrt(E)

	return a, k, gain, E, nil
}

// computePredictionError computes LPC residual signal
func (lpc *LPCAnalyzer) computePredictionError(signal []float64, coeffs []float64) []float64 {
	residual := make([]float64, len(signal))

	for n := range len(signal) {
		prediction := 0.0

		// Apply LPC prediction filter
		for k := 1; k < len(coeffs) && k <= n; k++ {
			prediction += coeffs[k] * signal[n-k]
		}

		residual[n] = signal[n] - prediction
	}

	return residual
}

// checkStability checks if LPC filter is stable (all poles inside unit circle)
func (lpc *LPCAnalyzer) checkStability(coeffs []float64) bool {
	// For stability, all reflection coefficients must satisfy |k_i| < 1
	// We can check this by converting back to reflection coefficients

	// Simple stability check: ensure no coefficient has magnitude >= 1
	for i := 1; i < len(coeffs); i++ {
		if math.Abs(coeffs[i]) >= 1.0 {
			return false
		}
	}

	return true
}

// ConvertToReflectionCoeffs converts LPC coefficients to reflection coefficients
func (lpc *LPCAnalyzer) ConvertToReflectionCoeffs(lpcCoeffs []float64) []float64 {
	p := len(lpcCoeffs) - 1
	if p <= 0 {
		return []float64{}
	}

	k := make([]float64, p)
	a := make([][]float64, p+1)

	// Initialize
	for i := 0; i <= p; i++ {
		a[i] = make([]float64, i+1)
	}

	// Copy input coefficients
	for i := 0; i <= p; i++ {
		a[p][i] = lpcCoeffs[i]
	}

	// Step-down recursion
	for i := p; i >= 1; i-- {
		k[i-1] = a[i][i]

		if math.Abs(k[i-1]) >= 1.0 {
			// Unstable - clamp reflection coefficient
			if k[i-1] >= 1.0 {
				k[i-1] = 0.99
			} else {
				k[i-1] = -0.99
			}
		}

		denominator := 1 - k[i-1]*k[i-1]
		if denominator == 0 {
			break
		}

		for j := 1; j < i; j++ {
			a[i-1][j] = (a[i][j] - k[i-1]*a[i][i-j]) / denominator
		}
	}

	return k
}

// EstimateFormantBandwidths estimates formant bandwidths from reflection coefficients
func (lpc *LPCAnalyzer) EstimateFormantBandwidths(reflectionCoeffs []float64) []float64 {
	bandwidths := make([]float64, len(reflectionCoeffs))

	for i, k := range reflectionCoeffs {
		// Bandwidth estimation from reflection coefficient
		// BW ≈ -ln(|k|) * fs / π
		if math.Abs(k) > 0 && math.Abs(k) < 1 {
			bandwidths[i] = -math.Log(math.Abs(k)) * float64(lpc.sampleRate) / math.Pi
		} else {
			bandwidths[i] = 100.0 // Default bandwidth
		}
	}

	return bandwidths
}

// GetSpectralEnvelope computes LPC spectral envelope
func (lpc *LPCAnalyzer) GetSpectralEnvelope(coeffs []float64, nfft int) ([]float64, error) {
	if nfft <= 0 {
		nfft = 512
	}

	envelope := make([]float64, nfft/2+1)

	for k := range len(envelope) {
		// Frequency in radians
		omega := 2 * math.Pi * float64(k) / float64(nfft)

		// Evaluate H(e^jω) = 1 / A(e^jω)
		// where A(z) = 1 + a1*z^-1 + a2*z^-2 + ... + ap*z^-p

		realPart := 1.0
		imagPart := 0.0

		for i := 1; i < len(coeffs); i++ {
			angle := -float64(i) * omega
			realPart += coeffs[i] * math.Cos(angle)
			imagPart += coeffs[i] * math.Sin(angle)
		}

		magnitude := math.Sqrt(realPart*realPart + imagPart*imagPart)
		if magnitude > 0 {
			envelope[k] = 1.0 / magnitude
		} else {
			envelope[k] = 0.0
		}
	}

	return envelope, nil
}

// Helper function
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
