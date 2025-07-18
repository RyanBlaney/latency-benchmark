package spectral

import (
	"math"
)

// MelScale provides mel frequency conversion utilities
// Essential for MFCC computation and speech analysis
type MelScale struct {
	// No state needed - TODO: why
}

// NewMelScale creates a new mel scale converter
func NewMelScale() *MelScale {
	return &MelScale{}
}

// HzToMel converts frequency in Hz to mel scale
func (ms *MelScale) HzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// MelToHz converts mel scale to frequency in Hz
func (ms *MelScale) MelToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// CreateMelFilterBank creates mel-scale filter bank
func (ms *MelScale) CreateMelFilterBank(numFilters int, fftSize int, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	if numFilters <= 0 || fftSize <= 0 {
		return nil
	}

	// Convert frequency limits to mel scale
	lowMel := ms.HzToMel(lowFreq)
	highMel := ms.HzToMel(highFreq)

	// Create equally spaced mel points
	melPoints := make([]float64, numFilters+2)
	melStep := (highMel - lowMel) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = lowMel + float64(i)*melStep
	}

	// Convert mel points back to Hz
	hzPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		hzPoints[i] = ms.MelToHz(mel)
	}

	// Convert Hz to FFT bin indices
	binPoints := make([]int, len(hzPoints))
	for i, hz := range hzPoints {
		binPoints[i] = int(math.Floor((float64(fftSize)+1.0)*hz/float64(sampleRate) + 0.5))
		binPoints[i] = min(binPoints[i], fftSize/2)
	}

	// Create filter bank
	filterBank := make([][]float64, numFilters)
	for i := range filterBank {
		filterBank[i] = make([]float64, fftSize/2+1)
	}

	// Build triangular filters
	for m := 1; m <= numFilters; m++ {
		leftBin := binPoints[m-1]
		centerBin := binPoints[m]
		rightBin := binPoints[m+1]

		// Rising edge
		for k := leftBin; k < centerBin && k < len(filterBank[m-1]); k++ {
			if centerBin != leftBin {
				filterBank[m-1][k] = float64(k-leftBin) / float64(centerBin-leftBin)
			}
		}

		// Falling edge
		for k := centerBin; k < rightBin && k < len(filterBank[m-1]); k++ {
			if rightBin != centerBin {
				filterBank[m-1][k] = float64(rightBin-k) / float64(rightBin-centerBin)
			}
		}
	}

	return filterBank
}

// ApplyFilterBank applies mel filter bank to power spectrum
func (ms *MelScale) ApplyFilterBank(powerSpectrum []float64, filterBank [][]float64) []float64 {
	if len(filterBank) == 0 || len(powerSpectrum) == 0 {
		return []float64{}
	}

	melSpectrum := make([]float64, len(filterBank))

	for i, filter := range filterBank {
		sum := 0.0
		for j := 0; j < len(filter) && j < len(powerSpectrum); j++ {
			sum += powerSpectrum[j] * filter[j]
		}
		melSpectrum[i] = sum
	}

	return melSpectrum
}

// ComputeMelSpectrum computes mel-scale spectrum from magnitude spectrum
func (ms *MelScale) ComputeMelSpectrum(magnitudeSpectrum []float64, numFilters int, sampleRate int, lowFreq, highFreq float64) []float64 {
	// Convert to power spectrum
	powerSpectrum := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		powerSpectrum[i] = mag * mag
	}

	// Create filter bank
	fftSize := (len(magnitudeSpectrum) - 1) * 2
	filterBank := ms.CreateMelFilterBank(numFilters, fftSize, sampleRate, lowFreq, highFreq)

	// Apply filter bank
	return ms.ApplyFilterBank(powerSpectrum, filterBank)
}

// ComputeMelSpectrogramFrames processes multiple frames
func (ms *MelScale) ComputeMelSpectrogramFrames(spectrogram [][]float64, numFilters int, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	if len(spectrogram) == 0 {
		return [][]float64{}
	}

	melSpectrogram := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		melSpectrogram[t] = ms.ComputeMelSpectrum(magnitudeSpectrum, numFilters, sampleRate, lowFreq, highFreq)
	}

	return melSpectrogram
}
