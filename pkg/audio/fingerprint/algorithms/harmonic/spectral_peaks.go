package harmonic

import (
	"math"
	"sort"
)

// SpectralPeak represents a detected spectral peak
type SpectralPeak struct {
	Frequency float64 // Peak frequency in Hz
	Magnitude float64 // Peak magnitude
	Phase     float64 // Peak phase
	BinIndex  int     // Original FFT bin index
	Harmonic  int     // Harmonic number (0 = fundamental, 1 = 2nd harmonic, etc.)
}

// SpectralPeaks provides harmonic-aware peak detection and analysis
type SpectralPeaks struct {
	sampleRate      int
	minPeakHeight   float64
	minPeakDistance float64 // Minimum distance between peaks in Hz
	maxPeaks        int
}

// NewSpectralPeaks creates a new spectral peaks analyzer
func NewSpectralPeaks(sampleRate int, minPeakHeight, minPeakDistance float64, maxPeaks int) *SpectralPeaks {
	return &SpectralPeaks{
		sampleRate:      sampleRate,
		minPeakHeight:   minPeakHeight,
		minPeakDistance: minPeakDistance,
		maxPeaks:        maxPeaks,
	}
}

// DetectPeaks detects spectral peaks in magnitude spectrum
func (sp *SpectralPeaks) DetectPeaks(magnitudeSpectrum []float64, windowSize int) []SpectralPeak {
	if len(magnitudeSpectrum) == 0 {
		return []SpectralPeak{}
	}

	freqResolution := float64(sp.sampleRate) / float64(windowSize)
	minDistanceBins := int(sp.minPeakDistance / freqResolution)
	minDistanceBins = max(minDistanceBins, 1)

	var peaks []SpectralPeak

	// Find local maxima
	for i := 1; i < len(magnitudeSpectrum)-1; i++ {
		// Check if it's a local maximum above threshold
		if magnitudeSpectrum[i] > magnitudeSpectrum[i-1] &&
			magnitudeSpectrum[i] > magnitudeSpectrum[i+1] &&
			magnitudeSpectrum[i] >= sp.minPeakHeight {

			// Check minimum distance from existing peaks using bin indices
			validPeak := true
			for _, existingPeak := range peaks {
				binDistance := int(math.Abs(float64(i - existingPeak.BinIndex)))
				if binDistance < minDistanceBins {
					// Keep the higher peak
					if magnitudeSpectrum[i] > existingPeak.Magnitude {
						// Remove existing peak
						for j := 0; j < len(peaks); j++ {
							if peaks[j].BinIndex == existingPeak.BinIndex {
								peaks = append(peaks[:j], peaks[j+1:]...)
								break
							}
						}
					} else {
						validPeak = false
					}
					break
				}
			}

			if validPeak {
				frequency := float64(i) * freqResolution
				peak := SpectralPeak{
					Frequency: frequency,
					Magnitude: magnitudeSpectrum[i],
					Phase:     0.0, // Set separately if phase spectrum available
					BinIndex:  i,
					Harmonic:  -1, // Unassigned initially
				}
				peaks = append(peaks, peak)
			}
		}
	}

	// Sort peaks by magnitude (descending)
	sort.Slice(peaks, func(i, j int) bool {
		return peaks[i].Magnitude > peaks[j].Magnitude
	})

	// Limit number of peaks
	if len(peaks) > sp.maxPeaks {
		peaks = peaks[:sp.maxPeaks]
	}

	return peaks
}

// DetectPeaksWithPhase detects peaks including phase information
func (sp *SpectralPeaks) DetectPeaksWithPhase(magnitudeSpectrum, phaseSpectrum []float64, windowSize int) []SpectralPeak {
	peaks := sp.DetectPeaks(magnitudeSpectrum, windowSize)

	// Add phase information
	for i := range peaks {
		if peaks[i].BinIndex < len(phaseSpectrum) {
			peaks[i].Phase = phaseSpectrum[peaks[i].BinIndex]
		}
	}

	return peaks
}

// RefineWithInterpolation refines peak locations using parabolic interpolation
func (sp *SpectralPeaks) RefineWithInterpolation(magnitudeSpectrum []float64, peaks []SpectralPeak, windowSize int) []SpectralPeak {
	freqResolution := float64(sp.sampleRate) / float64(windowSize)
	refinedPeaks := make([]SpectralPeak, len(peaks))

	for i, peak := range peaks {
		refinedPeak := peak
		binIdx := peak.BinIndex

		// Parabolic interpolation for sub-bin accuracy
		if binIdx > 0 && binIdx < len(magnitudeSpectrum)-1 {
			y1 := magnitudeSpectrum[binIdx-1]
			y2 := magnitudeSpectrum[binIdx]
			y3 := magnitudeSpectrum[binIdx+1]

			// Parabolic interpolation
			denom := 2.0 * (2.0*y2 - y1 - y3)
			if math.Abs(denom) > 1e-10 {
				offset := (y3 - y1) / denom
				refinedFreq := (float64(binIdx) + offset) * freqResolution

				// Interpolated magnitude
				a := 0.5 * (y1 - 2.0*y2 + y3)
				b := 0.5 * (y3 - y1)
				interpolatedMag := y2 + a*offset*offset + b*offset

				refinedPeak.Frequency = refinedFreq
				refinedPeak.Magnitude = interpolatedMag
			}
		}

		refinedPeaks[i] = refinedPeak
	}

	return refinedPeaks
}

// AssignHarmonics assigns harmonic numbers to peaks based on fundamental frequency
func (sp *SpectralPeaks) AssignHarmonics(peaks []SpectralPeak, f0 float64, tolerance float64) []SpectralPeak {
	assignedPeaks := make([]SpectralPeak, len(peaks))
	copy(assignedPeaks, peaks)

	for i := range assignedPeaks {
		// Find closest harmonic
		bestHarmonic := -1
		bestError := math.Inf(1)

		// Check harmonics 1-20 (reasonable range)
		for harmonic := 1; harmonic <= 20; harmonic++ {
			expectedFreq := f0 * float64(harmonic)
			error := math.Abs(assignedPeaks[i].Frequency - expectedFreq)
			relativeError := error / expectedFreq

			if relativeError < tolerance && error < bestError {
				bestError = error
				bestHarmonic = harmonic
			}
		}

		if bestHarmonic > 0 {
			assignedPeaks[i].Harmonic = bestHarmonic - 1 // 0-indexed
		}
	}

	return assignedPeaks
}

// FilterHarmonicPeaks filters peaks to keep only those assigned to harmonics
func (sp *SpectralPeaks) FilterHarmonicPeaks(peaks []SpectralPeak) []SpectralPeak {
	var harmonicPeaks []SpectralPeak

	for _, peak := range peaks {
		if peak.Harmonic >= 0 {
			harmonicPeaks = append(harmonicPeaks, peak)
		}
	}

	// Sort by harmonic number
	sort.Slice(harmonicPeaks, func(i, j int) bool {
		return harmonicPeaks[i].Harmonic < harmonicPeaks[j].Harmonic
	})

	return harmonicPeaks
}

// AnalyzeHarmonicSeries analyzes the harmonic series structure
func (sp *SpectralPeaks) AnalyzeHarmonicSeries(peaks []SpectralPeak, f0 float64) map[string]float64 {
	analysis := make(map[string]float64)

	harmonicPeaks := sp.FilterHarmonicPeaks(peaks)
	if len(harmonicPeaks) == 0 {
		return analysis
	}

	// Count detected harmonics
	analysis["num_harmonics"] = float64(len(harmonicPeaks))

	// Calculate harmonic strength (magnitude of fundamental)
	fundamentalMag := 0.0
	for _, peak := range harmonicPeaks {
		if peak.Harmonic == 0 { // Fundamental
			fundamentalMag = peak.Magnitude
			break
		}
	}
	analysis["fundamental_magnitude"] = fundamentalMag

	// Calculate total harmonic energy
	totalEnergy := 0.0
	for _, peak := range harmonicPeaks {
		totalEnergy += peak.Magnitude * peak.Magnitude
	}
	analysis["total_harmonic_energy"] = totalEnergy

	// Calculate odd/even harmonic ratio
	oddEnergy := 0.0
	evenEnergy := 0.0
	for _, peak := range harmonicPeaks {
		energy := peak.Magnitude * peak.Magnitude
		if (peak.Harmonic+1)%2 == 1 { // Odd harmonic (1st, 3rd, 5th...)
			oddEnergy += energy
		} else { // Even harmonic (2nd, 4th, 6th...)
			evenEnergy += energy
		}
	}

	if evenEnergy > 0 {
		analysis["odd_even_ratio"] = oddEnergy / evenEnergy
	} else {
		analysis["odd_even_ratio"] = math.Inf(1)
	}

	// Calculate spectral slope (harmonic decay rate)
	if len(harmonicPeaks) >= 2 {
		// Linear regression on log(magnitude) vs harmonic number
		var sumX, sumY, sumXY, sumXX float64
		n := float64(len(harmonicPeaks))

		for _, peak := range harmonicPeaks {
			x := float64(peak.Harmonic + 1)       // 1-indexed harmonic number
			y := math.Log(peak.Magnitude + 1e-10) // Avoid log(0)

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
		}

		slope := (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
		analysis["harmonic_decay_slope"] = slope
	}

	return analysis
}

// DetectSubharmonics detects subharmonic components (f0/2, f0/3, etc.)
func (sp *SpectralPeaks) DetectSubharmonics(peaks []SpectralPeak, f0 float64, tolerance float64) []SpectralPeak {
	var subharmonics []SpectralPeak

	for _, peak := range peaks {
		// Check if peak frequency matches a subharmonic
		for divisor := 2; divisor <= 5; divisor++ {
			expectedFreq := f0 / float64(divisor)
			error := math.Abs(peak.Frequency - expectedFreq)
			relativeError := error / expectedFreq

			if relativeError < tolerance {
				subharmonic := peak
				subharmonic.Harmonic = -divisor // Negative for subharmonics
				subharmonics = append(subharmonics, subharmonic)
				break
			}
		}
	}

	return subharmonics
}

// CalculatePeakStats calculates statistics for detected peaks
func (sp *SpectralPeaks) CalculatePeakStats(peaks []SpectralPeak) map[string]float64 {
	stats := make(map[string]float64)

	if len(peaks) == 0 {
		return stats
	}

	// Basic statistics
	stats["num_peaks"] = float64(len(peaks))

	// Magnitude statistics
	var magnitudes []float64
	for _, peak := range peaks {
		magnitudes = append(magnitudes, peak.Magnitude)
	}

	stats["max_magnitude"] = magnitudes[0] // Already sorted
	stats["min_magnitude"] = magnitudes[len(magnitudes)-1]

	// Mean magnitude
	sum := 0.0
	for _, mag := range magnitudes {
		sum += mag
	}
	stats["mean_magnitude"] = sum / float64(len(magnitudes))

	// Frequency range
	minFreq := peaks[0].Frequency
	maxFreq := peaks[0].Frequency
	for _, peak := range peaks {
		if peak.Frequency < minFreq {
			minFreq = peak.Frequency
		}
		if peak.Frequency > maxFreq {
			maxFreq = peak.Frequency
		}
	}
	stats["frequency_range"] = maxFreq - minFreq
	stats["min_frequency"] = minFreq
	stats["max_frequency"] = maxFreq

	return stats
}
