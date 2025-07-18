package common

import (
	"math"
)

// InterpolationType defines interpolation method
type InterpolationType int

const (
	Linear InterpolationType = iota
	Cubic
	Hermite
	Lanczos
)

// Interpolator provides various interpolation methods
type Interpolator struct {
	method InterpolationType
}

// NewInterpolator creates a new interpolator
func NewInterpolator(method InterpolationType) *Interpolator {
	return &Interpolator{
		method: method,
	}
}

// Interpolate performs interpolation at fractional index
func (interp *Interpolator) Interpolate(data []float64, index float64) float64 {
	switch interp.method {
	case Linear:
		return interp.linearInterpolate(data, index)
	case Cubic:
		return interp.cubicInterpolate(data, index)
	case Hermite:
		return interp.hermiteInterpolate(data, index)
	case Lanczos:
		return interp.lanczosInterpolate(data, index)
	default:
		return interp.linearInterpolate(data, index)
	}
}

// linearInterpolate performs linear interpolation
func (interp *Interpolator) linearInterpolate(data []float64, index float64) float64 {
	if len(data) == 0 {
		return 0.0
	}

	if index <= 0 {
		return data[0]
	}
	if index >= float64(len(data)-1) {
		return data[len(data)-1]
	}

	i := int(index)
	frac := index - float64(i)

	if i >= len(data)-1 {
		return data[len(data)-1]
	}

	return data[i] + frac*(data[i+1]-data[i])
}

// cubicInterpolate performs cubic interpolation
func (interp *Interpolator) cubicInterpolate(data []float64, index float64) float64 {
	if len(data) < 4 {
		return interp.linearInterpolate(data, index)
	}

	if index <= 1 {
		return data[int(math.Max(0, index))]
	}
	if index >= float64(len(data)-2) {
		return data[len(data)-1]
	}

	i := int(index)
	frac := index - float64(i)

	// Ensure we have 4 points for cubic interpolation
	if i < 1 {
		i = 1
	}
	if i >= len(data)-2 {
		i = len(data) - 3
	}

	y0 := data[i-1]
	y1 := data[i]
	y2 := data[i+1]
	y3 := data[i+2]

	// Cubic interpolation using Catmull-Rom spline
	a0 := -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
	a1 := y0 - 2.5*y1 + 2*y2 - 0.5*y3
	a2 := -0.5*y0 + 0.5*y2
	a3 := y1

	return a0*frac*frac*frac + a1*frac*frac + a2*frac + a3
}

// hermiteInterpolate performs Hermite interpolation
func (interp *Interpolator) hermiteInterpolate(data []float64, index float64) float64 {
	if len(data) < 4 {
		return interp.linearInterpolate(data, index)
	}

	if index <= 1 {
		return data[int(math.Max(0, index))]
	}
	if index >= float64(len(data)-2) {
		return data[len(data)-1]
	}

	i := int(index)
	frac := index - float64(i)

	if i < 1 {
		i = 1
	}
	if i >= len(data)-2 {
		i = len(data) - 3
	}

	y0 := data[i-1]
	y1 := data[i]
	y2 := data[i+1]
	y3 := data[i+2]

	// Calculate derivatives
	m0 := 0.5 * (y2 - y0)
	m1 := 0.5 * (y3 - y1)

	// Hermite basis functions
	t := frac
	t2 := t * t
	t3 := t2 * t

	h00 := 2*t3 - 3*t2 + 1
	h10 := t3 - 2*t2 + t
	h01 := -2*t3 + 3*t2
	h11 := t3 - t2

	return h00*y1 + h10*m0 + h01*y2 + h11*m1
}

// lanczosInterpolate performs Lanczos interpolation
func (interp *Interpolator) lanczosInterpolate(data []float64, index float64) float64 {
	if len(data) < 6 {
		return interp.cubicInterpolate(data, index)
	}

	a := 3.0 // Lanczos parameter
	i := int(index)
	frac := index - float64(i)

	if i < int(a) {
		return data[0]
	}
	if i >= len(data)-int(a) {
		return data[len(data)-1]
	}

	sum := 0.0
	for j := i - int(a) + 1; j <= i+int(a); j++ {
		if j >= 0 && j < len(data) {
			x := index - float64(j)
			weight := interp.lanczosKernel(x, a)
			sum += data[j] * weight
		}
	}

	return sum
}

// lanczosKernel computes Lanczos kernel function
func (interp *Interpolator) lanczosKernel(x, a float64) float64 {
	if math.Abs(x) < 1e-10 {
		return 1.0
	}
	if math.Abs(x) >= a {
		return 0.0
	}

	px := math.Pi * x
	return (a * math.Sin(px) * math.Sin(px/a)) / (px * px)
}

// ResampleSignal resamples a signal to a new sample rate
func (interp *Interpolator) ResampleSignal(signal []float64, originalRate, targetRate int) []float64 {
	if len(signal) == 0 || originalRate <= 0 || targetRate <= 0 {
		return signal
	}

	ratio := float64(originalRate) / float64(targetRate)
	newLength := int(float64(len(signal)) / ratio)

	if newLength <= 0 {
		return []float64{}
	}

	resampled := make([]float64, newLength)

	for i := range resampled {
		sourceIndex := float64(i) * ratio
		resampled[i] = interp.Interpolate(signal, sourceIndex)
	}

	return resampled
}

// UpsampleSignal upsamples by integer factor using interpolation
func (interp *Interpolator) UpsampleSignal(signal []float64, factor int) []float64 {
	if len(signal) == 0 || factor <= 1 {
		return signal
	}

	upsampled := make([]float64, len(signal)*factor)

	// Insert zeros
	for i, sample := range signal {
		upsampled[i*factor] = sample
	}

	// Interpolate missing samples
	for i := 1; i < len(upsampled); i++ {
		if upsampled[i] == 0.0 { // Zero-stuffed sample
			// Find surrounding non-zero samples for interpolation
			prevIdx := (i / factor) * factor
			nextIdx := prevIdx + factor

			if nextIdx < len(upsampled) {
				frac := float64(i-prevIdx) / float64(factor)
				upsampled[i] = upsampled[prevIdx] + frac*(upsampled[nextIdx]-upsampled[prevIdx])
			}
		}
	}

	return upsampled
}

// DownsampleSignal downsamples by integer factor with anti-aliasing
func (interp *Interpolator) DownsampleSignal(signal []float64, factor int) []float64 {
	if len(signal) == 0 || factor <= 1 {
		return signal
	}

	// Simple decimation (should apply low-pass filter first in practice)
	newLength := len(signal) / factor
	downsampled := make([]float64, newLength)

	for i := range downsampled {
		sourceIdx := i * factor
		if sourceIdx < len(signal) {
			downsampled[i] = signal[sourceIdx]
		}
	}

	return downsampled
}

// InterpolateArray interpolates an entire array to a new length
func (interp *Interpolator) InterpolateArray(data []float64, newLength int) []float64 {
	if len(data) == 0 || newLength <= 0 {
		return []float64{}
	}

	if newLength == len(data) {
		result := make([]float64, len(data))
		copy(result, data)
		return result
	}

	result := make([]float64, newLength)
	ratio := float64(len(data)-1) / float64(newLength-1)

	for i := range result {
		sourceIndex := float64(i) * ratio
		result[i] = interp.Interpolate(data, sourceIndex)
	}

	return result
}

// BilinearInterpolate performs 2D bilinear interpolation
func BilinearInterpolate(data [][]float64, x, y float64) float64 {
	if len(data) == 0 || len(data[0]) == 0 {
		return 0.0
	}

	rows := len(data)
	cols := len(data[0])

	// Clamp coordinates
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x >= float64(cols-1) {
		x = float64(cols - 1)
	}
	if y >= float64(rows-1) {
		y = float64(rows - 1)
	}

	x1 := int(x)
	y1 := int(y)
	x2 := x1 + 1
	y2 := y1 + 1

	// Clamp indices
	if x2 >= cols {
		x2 = cols - 1
	}
	if y2 >= rows {
		y2 = rows - 1
	}

	// Fractional parts
	fx := x - float64(x1)
	fy := y - float64(y1)

	// Bilinear interpolation
	q11 := data[y1][x1]
	q12 := data[y2][x1]
	q21 := data[y1][x2]
	q22 := data[y2][x2]

	r1 := q11 + fx*(q21-q11)
	r2 := q12 + fx*(q22-q12)

	return r1 + fy*(r2-r1)
}
