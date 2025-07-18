package spectral

import (
	"math"
)

// BarkScale provides bark frequency conversion utilities
// Based on critical bands of human auditory perception
type BarkScale struct {
	// No state needed - stateless conversion functions
}

// NewBarkScale creates a new bark scale converter
func NewBarkScale() *BarkScale {
	return &BarkScale{}
}

// HzToBark converts frequency in Hz to bark scale
// Using Traunmüller (1990) formula
func (bs *BarkScale) HzToBark(hz float64) float64 {
	return (26.81 * hz / (1960.0 + hz)) - 0.53
}

// BarkToHz converts bark scale to frequency in Hz
// Inverse of Traunmüller formula
func (bs *BarkScale) BarkToHz(bark float64) float64 {
	return 1960.0 * (bark + 0.53) / (26.28 - bark)
}

// HzToBarkZwicker converts frequency in Hz to bark scale using Zwicker & Terhardt (1980)
func (bs *BarkScale) HzToBarkZwicker(hz float64) float64 {
	return 13.0*math.Atan(0.00076*hz) + 3.5*math.Atan((hz/7500.0*hz/7500.0))
}

// CreateBarkFilterBank creates bark-scale filter bank
func (bs *BarkScale) CreateBarkFilterBank(numFilters int, fftSize int, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	if numFilters <= 0 || fftSize <= 0 {
		return nil
	}

	// Convert frequency limits to bark scale
	lowBark := bs.HzToBark(lowFreq)
	highBark := bs.HzToBark(highFreq)

	// Create equally spaced bark points
	barkPoints := make([]float64, numFilters+2)
	barkStep := (highBark - lowBark) / float64(numFilters+1)
	for i := range barkPoints {
		barkPoints[i] = lowBark + float64(i)*barkStep
	}

	// Convert bark points back to Hz
	hzPoints := make([]float64, len(barkPoints))
	for i, bark := range barkPoints {
		hzPoints[i] = bs.BarkToHz(bark)
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

// ApplyFilterBank applies bark filter bank to power spectrum
func (bs *BarkScale) ApplyFilterBank(powerSpectrum []float64, filterBank [][]float64) []float64 {
	if len(filterBank) == 0 || len(powerSpectrum) == 0 {
		return []float64{}
	}

	barkSpectrum := make([]float64, len(filterBank))

	for i, filter := range filterBank {
		sum := 0.0
		for j := 0; j < len(filter) && j < len(powerSpectrum); j++ {
			sum += powerSpectrum[j] * filter[j]
		}
		barkSpectrum[i] = sum
	}

	return barkSpectrum
}

// ComputeBarkSpectrum computes bark-scale spectrum from magnitude spectrum
func (bs *BarkScale) ComputeBarkSpectrum(magnitudeSpectrum []float64, numFilters int, sampleRate int, lowFreq, highFreq float64) []float64 {
	// Convert to power spectrum
	powerSpectrum := make([]float64, len(magnitudeSpectrum))
	for i, mag := range magnitudeSpectrum {
		powerSpectrum[i] = mag * mag
	}

	// Create filter bank
	fftSize := (len(magnitudeSpectrum) - 1) * 2
	filterBank := bs.CreateBarkFilterBank(numFilters, fftSize, sampleRate, lowFreq, highFreq)

	// Apply filter bank
	return bs.ApplyFilterBank(powerSpectrum, filterBank)
}

// ComputeBarkSpectrogramFrames processes multiple frames
func (bs *BarkScale) ComputeBarkSpectrogramFrames(spectrogram [][]float64, numFilters int, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	if len(spectrogram) == 0 {
		return [][]float64{}
	}

	barkSpectrogram := make([][]float64, len(spectrogram))

	for t, magnitudeSpectrum := range spectrogram {
		barkSpectrogram[t] = bs.ComputeBarkSpectrum(magnitudeSpectrum, numFilters, sampleRate, lowFreq, highFreq)
	}

	return barkSpectrogram
}

// GetCriticalBandEdges returns the 24 critical band edge frequencies in Hz
func (bs *BarkScale) GetCriticalBandEdges() []float64 {
	return []float64{
		0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
		1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
		5300, 6400, 7700, 9500, 12000, 15500,
	}
}

// GetBarkBandCenters returns the center frequencies of bark bands
func (bs *BarkScale) GetBarkBandCenters() []float64 {
	return []float64{
		50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170,
		1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800,
		5800, 7000, 8500, 10500, 13500,
	}
}
