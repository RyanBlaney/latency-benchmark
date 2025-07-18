package spectral

import (
	"math"
)

// PowerSpectrum provides power spectral density computation
// Extracted from your existing ComputePowerSpectrum implementation
type PowerSpectrum struct {
	// No state needed - stateless calculation
}

// NewPowerSpectrum creates a new power spectrum calculator
func NewPowerSpectrum() *PowerSpectrum {
	return &PowerSpectrum{}
}

// Compute computes power spectral density from magnitude spectrum
func (ps *PowerSpectrum) Compute(magnitudeSpectrum []float64) []float64 {
	if len(magnitudeSpectrum) == 0 {
		return []float64{}
	}

	power := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		power[i] = mag * mag
	}

	return power
}

// ComputeFromSTFT computes power spectrum from STFT result
// This is your existing working implementation
func (ps *PowerSpectrum) ComputeFromSTFT(stftResult *STFTResult) [][]float64 {
	power := make([][]float64, stftResult.TimeFrames)

	for t := 0; t < stftResult.TimeFrames; t++ {
		power[t] = make([]float64, stftResult.FreqBins)
		for f := 0; f < stftResult.FreqBins; f++ {
			mag := stftResult.Magnitude[t][f]
			power[t][f] = mag * mag
		}
	}

	return power
}

// ComputeFrames processes multiple magnitude spectrum frames
func (ps *PowerSpectrum) ComputeFrames(spectrogram [][]float64) [][]float64 {
	if len(spectrogram) == 0 {
		return [][]float64{}
	}

	power := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		power[t] = ps.Compute(magnitudeSpectrum)
	}

	return power
}

// ComputeLog computes log power spectrum in dB with floor
// This is your existing ComputeLogPowerSpectrum implementation
func (ps *PowerSpectrum) ComputeLog(magnitudeSpectrum []float64, floorDB float64) []float64 {
	if len(magnitudeSpectrum) == 0 {
		return []float64{}
	}

	floor := math.Pow(10, floorDB/10.0)
	logPower := make([]float64, len(magnitudeSpectrum))

	for i, mag := range magnitudeSpectrum {
		power := mag * mag
		if power < floor {
			power = floor
		}
		logPower[i] = 10 * math.Log10(power)
	}

	return logPower
}

// ComputeLogFromSTFT computes log power spectrum from STFT result
// This is your existing working implementation
func (ps *PowerSpectrum) ComputeLogFromSTFT(stftResult *STFTResult, floorDB float64) [][]float64 {
	logPower := make([][]float64, stftResult.TimeFrames)
	floor := math.Pow(10, floorDB/10.0)

	for t := 0; t < stftResult.TimeFrames; t++ {
		logPower[t] = make([]float64, stftResult.FreqBins)
		for f := 0; f < stftResult.FreqBins; f++ {
			mag := stftResult.Magnitude[t][f]
			power := mag * mag
			if power < floor {
				power = floor
			}
			logPower[t][f] = 10 * math.Log10(power)
		}
	}

	return logPower
}

// ComputeLogFrames processes multiple frames with log conversion
func (ps *PowerSpectrum) ComputeLogFrames(spectrogram [][]float64, floorDB float64) [][]float64 {
	if len(spectrogram) == 0 {
		return [][]float64{}
	}

	logPower := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		logPower[t] = ps.ComputeLog(magnitudeSpectrum, floorDB)
	}

	return logPower
}
