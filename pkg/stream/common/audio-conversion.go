package common

import (
	"math"
	"time"
	"unsafe"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
	"github.com/tunein/go-transcoding/v10/transcode"
)

// extractAudioFromFrame extracts PCM data from a decoded frame using GetPlaneBuffer
func ExtractAudioFromFrame(frame *transcode.Frame, expectedSampleRate, expectedChannels int) ([]float64, time.Duration, error) {
	// Get the raw audio data from the frame
	// For most audio formats, plane 0 contains the audio data
	audioBuffer := frame.GetPlaneBuffer(0)
	if len(audioBuffer) == 0 {
		return nil, 0, NewStreamError(StreamTypeHLS, "",
			ErrCodeDecoding, "empty audio buffer in frame", nil)
	}

	// We need to determine the actual audio format and parameters
	// Since we don't have direct access to format info, we'll make educated guesses
	// and provide fallback logic

	// TODO: eliminate guesswork

	// Try to detect the sample format and convert accordingly
	samples, actualSampleRate, actualChannels, err := convertRawAudioToFloat64(
		audioBuffer,
		expectedSampleRate,
		expectedChannels,
	)
	if err != nil {
		return nil, 0, NewStreamError(StreamTypeHLS, "",
			ErrCodeDecoding, "failed to convert audio data", err)
	}

	// Calculate the actual duration based on sample count
	sampleCount := len(samples)
	if actualChannels > 0 {
		duration := time.Duration(float64(sampleCount/actualChannels) / float64(actualSampleRate) * float64(time.Second))
		return samples, duration, nil
	}

	return samples, 0, nil
}

// convertRawAudioToFloat64 converts raw audio buffer to float64 PCM
func convertRawAudioToFloat64(buffer []byte, expectedSampleRate, expectedChannels int) ([]float64, int, int, error) {
	if len(buffer) == 0 {
		return nil, 0, 0, NewStreamError(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "empty audio buffer", nil)
	}

	// Try different audio formats in order of likelihood for HLS streams

	// Try 16-bit signed PCM first (most common for HLS)
	if samples, err := tryConvertS16(buffer, expectedSampleRate, expectedChannels); err == nil {
		return samples, expectedSampleRate, expectedChannels, nil
	}

	// Try 32-bit float PCM
	if samples, err := tryConvertFloat32(buffer, expectedSampleRate, expectedChannels); err == nil {
		return samples, expectedSampleRate, expectedChannels, nil
	}

	// Try 32-bit signed PCM
	if samples, err := tryConvertS32(buffer, expectedSampleRate, expectedChannels); err == nil {
		return samples, expectedSampleRate, expectedChannels, nil
	}

	// Try 8-bit unsigned PCM (less common)
	if samples, err := tryConvertU8(buffer, expectedSampleRate, expectedChannels); err == nil {
		return samples, expectedSampleRate, expectedChannels, nil
	}

	// If all conversions fail, try to make a reasonable guess based on buffer size
	return convertWithSizeHeuristic(buffer, expectedSampleRate, expectedChannels)
}

// tryConvertS16 attempts to convert buffer as 16-bit signed PCM
func tryConvertS16(buffer []byte, sampleRate, channels int) ([]float64, error) {
	// Check if buffer size makes sense for 16-bit audio
	if len(buffer)%2 != 0 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "buffer size not aligned for 16-bit samples", nil,
			logging.Fields{"buffer_size": len(buffer)})
	}

	sampleCount := len(buffer) / 2
	samples := make([]float64, sampleCount)

	for i := range sampleCount {
		// Read 16-bit little-endian signed integer
		sample := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		// Convert to float64 [-1.0, 1.0]
		samples[i] = float64(sample) / 32768.0
	}

	// Sanity check: reasonable sample count for expected parameters
	expectedSamples := estimateExpectedSamples(sampleRate, channels)
	if sampleCount < expectedSamples/4 || sampleCount > expectedSamples*4 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "sample count doesn't match expected range for S16", nil,
			logging.Fields{
				"sample_count": sampleCount,
				"expected_min": estimateExpectedSamples(sampleRate, channels) / 4,
				"expected_max": estimateExpectedSamples(sampleRate, channels) * 4,
			})

	}

	return samples, nil
}

// tryConvertFloat32 attempts to convert buffer as 32-bit float PCM
func tryConvertFloat32(buffer []byte, sampleRate, channels int) ([]float64, error) {
	// Check if buffer size makes sense for 32-bit float audio
	if len(buffer)%4 != 0 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "buffer size not aligned for 32-bit samples", nil,
			logging.Fields{"buffer_size": len(buffer)})
	}

	sampleCount := len(buffer) / 4
	samples := make([]float64, sampleCount)

	for i := range sampleCount {
		// Read 32-bit little-endian float
		bits := uint32(buffer[i*4]) | uint32(buffer[i*4+1])<<8 | uint32(buffer[i*4+2])<<16 | uint32(buffer[i*4+3])<<24
		float32Val := *(*float32)(unsafe.Pointer(&bits))
		samples[i] = float64(float32Val)
	}

	// Sanity check for float values (should be roughly in [-1, 1] range for audio)
	validFloats := 0
	for _, sample := range samples {
		if sample >= -2.0 && sample <= 2.0 && !math.IsNaN(sample) && !math.IsInf(sample, 0) {
			validFloats++
		}
	}

	if float64(validFloats)/float64(len(samples)) < 0.8 { // At least 80% should be valid
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "too many invalid float values, probably not float32 format", nil,
			logging.Fields{
				"valid_floats":  validFloats,
				"total_samples": len(samples),
				"valid_ratio":   float64(validFloats) / float64(len(samples)),
			})
	}

	return samples, nil
}

// tryConvertS32 attempts to convert buffer as 32-bit signed PCM
func tryConvertS32(buffer []byte, sampleRate, channels int) ([]float64, error) {
	// Check if buffer size makes sense for 32-bit audio
	if len(buffer)%4 != 0 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "buffer size not aligned for 32-bit samples", nil,
			logging.Fields{"buffer_size": len(buffer)})
	}

	sampleCount := len(buffer) / 4
	samples := make([]float64, sampleCount)

	for i := range sampleCount {
		// Read 32-bit little-endian signed integer
		sample := int32(buffer[i*4]) | int32(buffer[i*4+1])<<8 | int32(buffer[i*4+2])<<16 | int32(buffer[i*4+3])<<24
		// Convert to float64 [-1.0, 1.0]
		samples[i] = float64(sample) / 2147483648.0
	}

	// Sanity check: reasonable sample count for expected parameters
	expectedSamples := estimateExpectedSamples(sampleRate, channels)
	if sampleCount < expectedSamples/4 || sampleCount > expectedSamples*4 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "sample count doesn't match expected range for U8", nil,
			logging.Fields{
				"sample_count": sampleCount,
				"expected_min": estimateExpectedSamples(sampleRate, channels) / 4,
				"expected_max": estimateExpectedSamples(sampleRate, channels) * 4,
			})
	}

	return samples, nil
}

// tryConvertU8 attempts to convert buffer as 8-bit unsigned PCM
func tryConvertU8(buffer []byte, sampleRate, channels int) ([]float64, error) {
	sampleCount := len(buffer)
	samples := make([]float64, sampleCount)

	for i := range sampleCount {
		// Convert 8-bit unsigned to float64 [-1.0, 1.0]
		// 8-bit unsigned: 0-255, center at 128
		samples[i] = (float64(buffer[i]) - 128.0) / 128.0
	}

	// Sanity check: reasonable sample count for expected parameters
	expectedSamples := estimateExpectedSamples(sampleRate, channels)
	if sampleCount < expectedSamples/4 || sampleCount > expectedSamples*4 {
		return nil, NewStreamErrorWithFields(StreamTypeHLS, "",
			ErrCodeInvalidFormat, "sample count doesn't match expected range for U8", nil,
			logging.Fields{
				"sample_count": sampleCount,
				"expected_min": estimateExpectedSamples(sampleRate, channels) / 4,
				"expected_max": estimateExpectedSamples(sampleRate, channels) * 4,
			})

	}

	return samples, nil
}

// convertWithSizeHeuristic makes a best guess based on buffer size
func convertWithSizeHeuristic(buffer []byte, expectedSampleRate, expectedChannels int) ([]float64, int, int, error) {
	bufferSize := len(buffer)

	// Estimate what format might make sense based on buffer size
	// Typical HLS segment is 2-10 seconds, let's assume 5 seconds average
	estimatedDuration := 5.0 // seconds
	expectedBytes := int(estimatedDuration * float64(expectedSampleRate) * float64(expectedChannels))

	// Try different byte-per-sample ratios
	for _, bytesPerSample := range []int{2, 4, 1, 3} { // 16-bit, 32-bit, 8-bit, 24-bit
		if abs(expectedBytes*bytesPerSample-bufferSize) < bufferSize/4 { // Within 25%
			switch bytesPerSample {
			case 1:
				if samples, err := tryConvertU8(buffer, expectedSampleRate, expectedChannels); err == nil {
					return samples, expectedSampleRate, expectedChannels, nil
				}
			case 2:
				if samples, err := tryConvertS16(buffer, expectedSampleRate, expectedChannels); err == nil {
					return samples, expectedSampleRate, expectedChannels, nil
				}
			case 4:
				// Try both S32 and Float32
				if samples, err := tryConvertS32(buffer, expectedSampleRate, expectedChannels); err == nil {
					return samples, expectedSampleRate, expectedChannels, nil
				}
				if samples, err := tryConvertFloat32(buffer, expectedSampleRate, expectedChannels); err == nil {
					return samples, expectedSampleRate, expectedChannels, nil
				}
			}
		}
	}

	return nil, 0, 0, NewStreamErrorWithFields(StreamTypeHLS, "",
		ErrCodeInvalidFormat, "unable to determine audio format", nil,
		logging.Fields{
			"buffer_size":          bufferSize,
			"expected_sample_rate": expectedSampleRate,
			"expected_channels":    expectedChannels,
		})
}

// estimateExpectedSamples estimates how many samples we expect for typical HLS segments
func estimateExpectedSamples(sampleRate, channels int) int {
	// Typical HLS segment duration is 2-10 seconds, use 5 seconds as estimate
	estimatedDuration := 5.0 // seconds
	return int(estimatedDuration * float64(sampleRate) * float64(channels))
}

// Enhanced version that can handle planar audio formats
func extractAudioFromFramePlanar(frame *transcode.Frame, expectedSampleRate, expectedChannels int) ([]float64, time.Duration, error) {
	// For planar formats, each channel is in a separate plane
	// Try to detect if this is planar by checking multiple planes

	var actualSampleRate = expectedSampleRate

	// Check if we have multiple planes (planar format)
	planesWithData := 0
	planeSize := 0

	for plane := range 8 { // Check up to 8 planes (more than enough for audio)
		buffer := frame.GetPlaneBuffer(plane)
		if len(buffer) > 0 {
			planesWithData++
			if planeSize == 0 {
				planeSize = len(buffer)
			} else if len(buffer) != planeSize {
				// Inconsistent plane sizes, probably not planar
				break
			}
		} else {
			break
		}
	}

	if planesWithData > 1 && planesWithData <= expectedChannels {
		// Likely planar format
		return extractPlanarAudio(frame, planesWithData, actualSampleRate)
	} else {
		// Likely interleaved format
		return ExtractAudioFromFrame(frame, expectedSampleRate, expectedChannels)
	}
}

// extractPlanarAudio handles planar audio formats where each channel is in a separate plane
func extractPlanarAudio(frame *transcode.Frame, numChannels, sampleRate int) ([]float64, time.Duration, error) {
	var channelData [][]float64

	// Extract data from each plane
	for plane := range numChannels {
		buffer := frame.GetPlaneBuffer(plane)
		if len(buffer) == 0 {
			return nil, 0, NewStreamErrorWithFields(StreamTypeHLS, "",
				ErrCodeDecoding, "empty buffer in plane", nil,
				logging.Fields{"plane": plane})
		}

		// Convert this plane's data
		samples, _, _, err := convertRawAudioToFloat64(buffer, sampleRate, 1) // 1 channel per plane
		if err != nil {
			return nil, 0, NewStreamErrorWithFields(StreamTypeHLS, "",
				ErrCodeDecoding, "failed to convert plane", err,
				logging.Fields{"plane": plane})
		}

		channelData = append(channelData, samples)
	}

	// Verify all channels have the same number of samples
	sampleCount := len(channelData[0])
	for i, channel := range channelData {
		if len(channel) != sampleCount {
			return nil, 0, NewStreamErrorWithFields(StreamTypeHLS, "",
				ErrCodeInvalidFormat, "channel sample count mismatch", nil,
				logging.Fields{
					"channel":          i,
					"actual_samples":   len(channel),
					"expected_samples": sampleCount,
				})
		}
	}

	// Interleave the planar data
	interleavedSamples := make([]float64, sampleCount*numChannels)
	for sample := range sampleCount {
		for channel := range numChannels {
			interleavedSamples[sample*numChannels+channel] = channelData[channel][sample]
		}
	}

	duration := time.Duration(float64(sampleCount) / float64(sampleRate) * float64(time.Second))

	return interleavedSamples, duration, nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}

	return x
}
