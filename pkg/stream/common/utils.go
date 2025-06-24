package common

import (
	"strconv"
	"strings"
	"time"
)

// ParseBitrateFromString extracts bitrate from string (e.g., "128", "96k")
func ParseBitrateFromString(s string) int {
	// Remove 'k' suffix if present
	s = strings.TrimSuffix(strings.ToLower(s), "k")
	s = strings.TrimSpace(s)

	if bitrate, err := strconv.Atoi(s); err == nil {
		return bitrate
	}
	return 0
}

// ParseSampleRateFromString extracts sample rate from string
func ParseSampleRateFromString(s string) int {
	s = strings.TrimSpace(s)

	if sampleRate, err := strconv.Atoi(s); err == nil {
		return sampleRate
	}
	return 0
}

// ParseChannelsFromString extracts channel count from string
func ParseChannelsFromString(s string) int {
	s = strings.ToLower(strings.TrimSpace(s))

	switch s {
	case "mono", "1":
		return 1
	case "stereo", "2":
		return 2
	default:
		if channels, err := strconv.Atoi(s); err == nil && channels > 0 && channels <= 8 {
			return channels
		}
	}
	return 0
}

// NormalizeCodecName normalizes codec names to standard values
func NormalizeCodecName(codec string) string {
	codec = strings.ToLower(strings.TrimSpace(codec))

	switch {
	case strings.Contains(codec, "mp4a") || codec == "m4a":
		return "aac"
	case codec == "mp3" || strings.Contains(codec, "mpeg"):
		return "mp3"
	case strings.Contains(codec, "aac"):
		return "aac"
	case strings.Contains(codec, "ogg") || strings.Contains(codec, "vorbis"):
		return "ogg"
	case strings.Contains(codec, "flac"):
		return "flac"
	default:
		return codec
	}
}

// SafeStringCopy creates a safe copy of a string, handling empty values
func SafeStringCopy(s string) string {
	return strings.TrimSpace(s)
}

// IsValidURL performs basic URL validation
func IsValidURL(url string) bool {
	url = strings.TrimSpace(url)
	return strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://")
}

// FormatDuration formats duration for display
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return d.String()
	}

	seconds := int(d.Seconds())
	if seconds < 60 {
		return strconv.Itoa(seconds) + "s"
	}

	minutes := seconds / 60
	remainingSeconds := seconds % 60

	if remainingSeconds == 0 {
		return strconv.Itoa(minutes) + "m"
	}

	return strconv.Itoa(minutes) + "m" + strconv.Itoa(remainingSeconds) + "s"
}

// CleanHeaderValue cleans and normalizes header values
func CleanHeaderValue(value string) string {
	// Remove quotes and trim whitespace
	value = strings.Trim(value, "\"'")
	return strings.TrimSpace(value)
}

// ExtractContentType extracts main content type without parameters
func ExtractContentType(contentType string) string {
	contentType = strings.ToLower(strings.TrimSpace(contentType))

	// Remove charset and other parameters
	if idx := strings.Index(contentType, ";"); idx != -1 {
		contentType = contentType[:idx]
	}

	return strings.TrimSpace(contentType)
}

