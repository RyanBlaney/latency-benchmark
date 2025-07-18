package spectral

import (
	"fmt"
	"math"
)

// MFCC computes Mel-Frequency Cepstral Coefficients
// Essential for speech and audio feature extraction
type MFCC struct {
	numCoefficients int
	numMelFilters   int
	sampleRate      int
	lowFreq         float64
	highFreq        float64
	useLiftering    bool
	lifterCoeff     float64

	// Internal components
	melScale    *MelScale
	filterBank  [][]float64
	dctMatrix   [][]float64
	initialized bool
}

// MFCCParams contains parameters for MFCC computation
type MFCCParams struct {
	NumCoefficients int     `json:"num_coefficients"` // Number of MFCC coefficients (default: 13)
	NumMelFilters   int     `json:"num_mel_filters"`  // Number of mel filter bank filters (default: 26)
	LowFreq         float64 `json:"low_freq"`         // Low frequency bound (default: 0)
	HighFreq        float64 `json:"high_freq"`        // High frequency bound (default: sampleRate/2)
	UseLiftering    bool    `json:"use_liftering"`    // Apply liftering (default: true)
	LifterCoeff     float64 `json:"lifter_coeff"`     // Liftering coefficient (default: 22)
}

// MFCCResult contains MFCC computation results
type MFCCResult struct {
	MFCC        []float64 `json:"mfcc"`         // MFCC coefficients
	MelSpectrum []float64 `json:"mel_spectrum"` // Mel spectrum used
	LogEnergy   float64   `json:"log_energy"`   // Log energy (C0 before liftering)
}

// NewMFCC creates a new MFCC computer with default parameters
func NewMFCC(sampleRate, numCoefficients int) *MFCC {
	params := MFCCParams{
		NumCoefficients: numCoefficients,
		NumMelFilters:   26,
		LowFreq:         0.0,
		HighFreq:        float64(sampleRate) / 2.0,
		UseLiftering:    true,
		LifterCoeff:     22.0,
	}
	return NewMFCCWithParams(sampleRate, params)
}

// NewMFCCWithParams creates a new MFCC computer with custom parameters
func NewMFCCWithParams(sampleRate int, params MFCCParams) *MFCC {
	// Set defaults
	if params.NumCoefficients <= 0 {
		params.NumCoefficients = 13
	}
	if params.NumMelFilters <= 0 {
		params.NumMelFilters = 26
	}
	if params.HighFreq <= 0 {
		params.HighFreq = float64(sampleRate) / 2.0
	}
	if params.LifterCoeff <= 0 {
		params.LifterCoeff = 22.0
	}

	mfcc := &MFCC{
		numCoefficients: params.NumCoefficients,
		numMelFilters:   params.NumMelFilters,
		sampleRate:      sampleRate,
		lowFreq:         params.LowFreq,
		highFreq:        params.HighFreq,
		useLiftering:    params.UseLiftering,
		lifterCoeff:     params.LifterCoeff,
		melScale:        NewMelScale(),
	}

	return mfcc
}

// Initialize prepares the MFCC computer for the given FFT size
func (mfcc *MFCC) Initialize(fftSize int) error {
	if fftSize <= 0 {
		return fmt.Errorf("invalid FFT size: %d", fftSize)
	}

	// Create mel filter bank
	mfcc.filterBank = mfcc.melScale.CreateMelFilterBank(
		mfcc.numMelFilters,
		fftSize,
		mfcc.sampleRate,
		mfcc.lowFreq,
		mfcc.highFreq,
	)

	if len(mfcc.filterBank) == 0 {
		return fmt.Errorf("failed to create mel filter bank")
	}

	// Create DCT matrix
	mfcc.createDCTMatrix()

	mfcc.initialized = true
	return nil
}

// Compute calculates MFCC coefficients from magnitude spectrum
func (mfcc *MFCC) Compute(magnitudeSpectrum []float64) (*MFCCResult, error) {
	if !mfcc.initialized {
		// Auto-initialize based on spectrum size
		fftSize := (len(magnitudeSpectrum) - 1) * 2
		if err := mfcc.Initialize(fftSize); err != nil {
			return nil, fmt.Errorf("failed to initialize MFCC: %w", err)
		}
	}

	if len(magnitudeSpectrum) == 0 {
		return nil, fmt.Errorf("empty magnitude spectrum")
	}

	// Convert to power spectrum
	powerSpectrum := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		powerSpectrum[i] = mag * mag
	}

	// Apply mel filter bank
	melSpectrum := mfcc.melScale.ApplyFilterBank(powerSpectrum, mfcc.filterBank)

	// Apply logarithm with floor to avoid log(0)
	logMelSpectrum := make([]float64, len(melSpectrum))
	for i, mel := range melSpectrum {
		if mel > 0 {
			logMelSpectrum[i] = math.Log(mel)
		} else {
			logMelSpectrum[i] = math.Log(1e-10) // Small positive value
		}
	}

	// Apply DCT
	mfccCoeffs := mfcc.applyDCT(logMelSpectrum)

	// Store C0 (log energy) before liftering
	logEnergy := 0.0
	if len(mfccCoeffs) > 0 {
		logEnergy = mfccCoeffs[0]
	}

	// Apply liftering if enabled
	if mfcc.useLiftering {
		mfccCoeffs = mfcc.applyLiftering(mfccCoeffs)
	}

	return &MFCCResult{
		MFCC:        mfccCoeffs,
		MelSpectrum: melSpectrum,
		LogEnergy:   logEnergy,
	}, nil
}

// ComputeFrames processes multiple frames of magnitude spectra
func (mfcc *MFCC) ComputeFrames(spectrogram [][]float64) ([][]float64, error) {
	if len(spectrogram) == 0 {
		return [][]float64{}, nil
	}

	// Initialize with first frame
	if !mfcc.initialized {
		fftSize := (len(spectrogram[0]) - 1) * 2
		if err := mfcc.Initialize(fftSize); err != nil {
			return nil, fmt.Errorf("failed to initialize MFCC: %w", err)
		}
	}

	mfccFrames := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		result, err := mfcc.Compute(magnitudeSpectrum)
		if err != nil {
			return nil, fmt.Errorf("failed to compute MFCC for frame %d: %w", t, err)
		}
		mfccFrames[t] = result.MFCC
	}

	return mfccFrames, nil
}

// createDCTMatrix creates the Discrete Cosine Transform matrix
func (mfcc *MFCC) createDCTMatrix() {
	mfcc.dctMatrix = make([][]float64, mfcc.numCoefficients)

	for k := 0; k < mfcc.numCoefficients; k++ {
		mfcc.dctMatrix[k] = make([]float64, mfcc.numMelFilters)

		for n := 0; n < mfcc.numMelFilters; n++ {
			// DCT-II formula
			mfcc.dctMatrix[k][n] = math.Cos(math.Pi * float64(k) * (float64(n) + 0.5) / float64(mfcc.numMelFilters))

			// Normalization
			if k == 0 {
				mfcc.dctMatrix[k][n] *= math.Sqrt(1.0 / float64(mfcc.numMelFilters))
			} else {
				mfcc.dctMatrix[k][n] *= math.Sqrt(2.0 / float64(mfcc.numMelFilters))
			}
		}
	}
}

// applyDCT applies the Discrete Cosine Transform
func (mfcc *MFCC) applyDCT(logMelSpectrum []float64) []float64 {
	mfccCoeffs := make([]float64, mfcc.numCoefficients)

	for k := 0; k < mfcc.numCoefficients; k++ {
		sum := 0.0
		for n := 0; n < len(logMelSpectrum) && n < len(mfcc.dctMatrix[k]); n++ {
			sum += logMelSpectrum[n] * mfcc.dctMatrix[k][n]
		}
		mfccCoeffs[k] = sum
	}

	return mfccCoeffs
}

// applyLiftering applies liftering to enhance higher-order coefficients
func (mfcc *MFCC) applyLiftering(mfccCoeffs []float64) []float64 {
	liftered := make([]float64, len(mfccCoeffs))

	for i, coeff := range mfccCoeffs {
		if i == 0 {
			// Don't lifter C0
			liftered[i] = coeff
		} else {
			// Apply sinusoidal liftering
			lifter := 1.0 + (mfcc.lifterCoeff/2.0)*math.Sin(math.Pi*float64(i)/mfcc.lifterCoeff)
			liftered[i] = coeff * lifter
		}
	}

	return liftered
}

// GetFilterBank returns the mel filter bank (for debugging/visualization)
func (mfcc *MFCC) GetFilterBank() [][]float64 {
	return mfcc.filterBank
}

// GetDCTMatrix returns the DCT matrix (for debugging/visualization)
func (mfcc *MFCC) GetDCTMatrix() [][]float64 {
	return mfcc.dctMatrix
}

// GetParams returns the current MFCC parameters
func (mfcc *MFCC) GetParams() MFCCParams {
	return MFCCParams{
		NumCoefficients: mfcc.numCoefficients,
		NumMelFilters:   mfcc.numMelFilters,
		LowFreq:         mfcc.lowFreq,
		HighFreq:        mfcc.highFreq,
		UseLiftering:    mfcc.useLiftering,
		LifterCoeff:     mfcc.lifterCoeff,
	}
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
