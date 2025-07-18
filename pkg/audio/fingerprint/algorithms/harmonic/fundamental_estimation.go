package harmonic

import (
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// FundamentalEstimation provides fundamental frequency (F0) estimation
type FundamentalEstimation struct {
	sampleRate    int
	minF0         float64
	maxF0         float64
	fft           *spectral.FFT
	powerSpectrum *spectral.PowerSpectrum
}

// NewFundamentalEstimation creates a new F0 estimator
func NewFundamentalEstimation(sampleRate int, minF0, maxF0 float64) *FundamentalEstimation {
	return &FundamentalEstimation{
		sampleRate:    sampleRate,
		minF0:         minF0,
		maxF0:         maxF0,
		fft:           spectral.NewFFT(),
		powerSpectrum: spectral.NewPowerSpectrum(),
	}
}

// EstimateAutocorrelation estimates F0 using autocorrelation method
func (fe *FundamentalEstimation) EstimateAutocorrelation(signal []float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate lag bounds from F0 range
	minLag := int(float64(fe.sampleRate) / fe.maxF0)
	maxLag := int(float64(fe.sampleRate) / fe.minF0)

	if maxLag >= len(signal) {
		maxLag = len(signal) - 1
	}
	if minLag < 1 {
		minLag = 1
	}

	// Calculate autocorrelation
	autocorr := fe.calculateAutocorrelation(signal, maxLag)

	// Find peak in valid F0 range
	bestLag := fe.findBestPeak(autocorr, minLag, maxLag)

	if bestLag == 0 {
		return 0.0 // No valid F0 found
	}

	return float64(fe.sampleRate) / float64(bestLag)
}

// EstimateYIN estimates F0 using YIN algorithm (simplified version)
func (fe *FundamentalEstimation) EstimateYIN(signal []float64, threshold float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Calculate lag bounds
	minLag := int(float64(fe.sampleRate) / fe.maxF0)
	maxLag := int(float64(fe.sampleRate) / fe.minF0)

	if maxLag >= len(signal)/2 {
		maxLag = len(signal)/2 - 1
	}
	if minLag < 1 {
		minLag = 1
	}

	// Calculate difference function
	diffFunction := fe.calculateDifferenceFunction(signal, maxLag)

	// Calculate cumulative mean normalized difference function
	cmndf := fe.calculateCMNDF(diffFunction)

	// Find first minimum below threshold
	for lag := minLag; lag <= maxLag; lag++ {
		if lag < len(cmndf) && cmndf[lag] < threshold {
			// Parabolic interpolation for sub-sample accuracy
			f0 := fe.parabolicInterpolation(cmndf, lag)
			if f0 > 0 {
				return float64(fe.sampleRate) / f0
			}
		}
	}

	return 0.0 // No valid F0 found
}

// EstimateCepstrum estimates F0 using cepstral analysis
func (fe *FundamentalEstimation) EstimateCepstrum(signal []float64) float64 {
	if len(signal) == 0 {
		return 0.0
	}

	// Apply window to signal
	windowedSignal := fe.applyWindow(signal)

	// Compute FFT
	fftResult := fe.fft.Compute(windowedSignal)

	// Convert to power spectrum
	powerSpec := make([]float64, len(fftResult))
	for i, c := range fftResult {
		powerSpec[i] = real(c)*real(c) + imag(c)*imag(c)
	}

	// Take log of power spectrum
	logPowerSpec := make([]float64, len(powerSpec))
	for i, power := range powerSpec {
		if power > 1e-10 {
			logPowerSpec[i] = math.Log(power)
		} else {
			logPowerSpec[i] = math.Log(1e-10)
		}
	}

	// IFFT to get cepstrum
	cepstrum := fe.fft.ComputeInverseReal(fe.complexFromReal(logPowerSpec))

	// Find peak in quefrency domain
	minQuefrency := int(float64(fe.sampleRate) / fe.maxF0)
	maxQuefrency := int(float64(fe.sampleRate) / fe.minF0)

	if maxQuefrency >= len(cepstrum) {
		maxQuefrency = len(cepstrum) - 1
	}
	if minQuefrency < 1 {
		minQuefrency = 1
	}

	bestQuefrency := fe.findCepstralPeak(cepstrum, minQuefrency, maxQuefrency)

	if bestQuefrency == 0 {
		return 0.0
	}

	return float64(fe.sampleRate) / float64(bestQuefrency)
}

// calculateAutocorrelation computes autocorrelation function
func (fe *FundamentalEstimation) calculateAutocorrelation(signal []float64, maxLag int) []float64 {
	autocorr := make([]float64, maxLag+1)

	for lag := 0; lag <= maxLag; lag++ {
		sum := 0.0
		count := 0

		for i := 0; i < len(signal)-lag; i++ {
			sum += signal[i] * signal[i+lag]
			count++
		}

		if count > 0 {
			autocorr[lag] = sum / float64(count)
		}
	}

	// Normalize by autocorr[0]
	if autocorr[0] > 0 {
		for i := range autocorr {
			autocorr[i] /= autocorr[0]
		}
	}

	return autocorr
}

// calculateDifferenceFunction computes YIN difference function
func (fe *FundamentalEstimation) calculateDifferenceFunction(signal []float64, maxLag int) []float64 {
	diff := make([]float64, maxLag+1)

	for lag := 0; lag <= maxLag; lag++ {
		sum := 0.0
		count := 0

		for i := 0; i < len(signal)-lag; i++ {
			d := signal[i] - signal[i+lag]
			sum += d * d
			count++
		}

		if count > 0 {
			diff[lag] = sum / float64(count)
		}
	}

	return diff
}

// calculateCMNDF computes cumulative mean normalized difference function
func (fe *FundamentalEstimation) calculateCMNDF(diff []float64) []float64 {
	cmndf := make([]float64, len(diff))
	cmndf[0] = 1.0

	for lag := 1; lag < len(diff); lag++ {
		sum := 0.0
		for j := 1; j <= lag; j++ {
			sum += diff[j]
		}

		if sum > 0 {
			cmndf[lag] = diff[lag] / (sum / float64(lag))
		} else {
			cmndf[lag] = 1.0
		}
	}

	return cmndf
}

// findBestPeak finds the best peak in autocorrelation
func (fe *FundamentalEstimation) findBestPeak(autocorr []float64, minLag, maxLag int) int {
	bestLag := 0
	bestValue := -1.0

	for lag := minLag; lag <= maxLag && lag < len(autocorr); lag++ {
		// Check if it's a local maximum
		if lag > 0 && lag < len(autocorr)-1 {
			if autocorr[lag] > autocorr[lag-1] &&
				autocorr[lag] > autocorr[lag+1] &&
				autocorr[lag] > bestValue {
				bestValue = autocorr[lag]
				bestLag = lag
			}
		}
	}

	return bestLag
}

// findCepstralPeak finds peak in cepstrum
func (fe *FundamentalEstimation) findCepstralPeak(cepstrum []float64, minQuefrency, maxQuefrency int) int {
	bestQuefrency := 0
	bestValue := -1.0

	for q := minQuefrency; q <= maxQuefrency && q < len(cepstrum); q++ {
		if cepstrum[q] > bestValue {
			bestValue = cepstrum[q]
			bestQuefrency = q
		}
	}

	return bestQuefrency
}

// parabolicInterpolation provides sub-sample accuracy using parabolic interpolation
func (fe *FundamentalEstimation) parabolicInterpolation(data []float64, peakIndex int) float64 {
	if peakIndex <= 0 || peakIndex >= len(data)-1 {
		return float64(peakIndex)
	}

	y1 := data[peakIndex-1]
	y2 := data[peakIndex]
	y3 := data[peakIndex+1]

	// Parabolic interpolation formula
	denom := 2.0 * (2.0*y2 - y1 - y3)
	if math.Abs(denom) < 1e-10 {
		return float64(peakIndex)
	}

	offset := (y3 - y1) / denom
	return float64(peakIndex) + offset
}

// applyWindow applies Hann window to signal
func (fe *FundamentalEstimation) applyWindow(signal []float64) []float64 {
	windowed := make([]float64, len(signal))

	for i, sample := range signal {
		window := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(len(signal)-1)))
		windowed[i] = sample * window
	}

	return windowed
}

// complexFromReal converts real array to complex array
func (fe *FundamentalEstimation) complexFromReal(real []float64) []complex128 {
	cmplx := make([]complex128, len(real))
	for i, r := range real {
		cmplx[i] = complex128(complex(r, 0))
	}
	return cmplx
}

// EstimateMultipleF0 estimates multiple F0 candidates with confidence
func (fe *FundamentalEstimation) EstimateMultipleF0(signal []float64) []struct {
	F0         float64
	Confidence float64
} {
	var candidates []struct {
		F0         float64
		Confidence float64
	}

	// Get estimates from different methods
	autocorrF0 := fe.EstimateAutocorrelation(signal)
	yinF0 := fe.EstimateYIN(signal, 0.1)
	cepstrumF0 := fe.EstimateCepstrum(signal)

	// Add non-zero estimates
	if autocorrF0 > 0 {
		candidates = append(candidates, struct {
			F0         float64
			Confidence float64
		}{autocorrF0, 0.8})
	}

	if yinF0 > 0 {
		candidates = append(candidates, struct {
			F0         float64
			Confidence float64
		}{yinF0, 0.9})
	}

	if cepstrumF0 > 0 {
		candidates = append(candidates, struct {
			F0         float64
			Confidence float64
		}{cepstrumF0, 0.7})
	}

	return candidates
}
