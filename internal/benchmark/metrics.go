package benchmark

import (
	"math"

	"github.com/RyanBlaney/latency-benchmark/internal/latency"
	"github.com/RyanBlaney/latency-benchmark/pkg/logging"
)

// MetricsCalculator handles calculation of various performance metrics
type MetricsCalculator struct {
	logger logging.Logger
}

// NewMetricsCalculator creates a new metrics calculator
func NewMetricsCalculator(logger logging.Logger) *MetricsCalculator {
	if logger == nil {
		logger = logging.NewDefaultLogger()
	}

	return &MetricsCalculator{
		logger: logger,
	}
}

// LatencyStats represents statistical measures of latency
type LatencyStats struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P95    float64 `json:"p95"`
	P99    float64 `json:"p99"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	StdDev float64 `json:"std_dev"`
	Count  int     `json:"count"`
}

// PerformanceMetrics represents comprehensive performance analysis
type PerformanceMetrics struct {
	LatencySource                      *LatencyStats `json:"latency_source"`
	LatencyBackup                      *LatencyStats `json:"latency_backup"`
	LatencyCloudfrontHLS               *LatencyStats `json:"latency_cloudfront_hls"`
	LatencyCloudfrontICEcast           *LatencyStats `json:"latency_cloudfront_icecast"`
	LatencyAISHLS                      *LatencyStats `json:"latency_ais_hls"`
	LatencyAISICEcast                  *LatencyStats `json:"latency_ais_icecast"`
	LatencyCloudfrontHLSFromBackup     *LatencyStats `json:"latency_cloudfront_hls_from_backup"`
	LatencyCloudfrontICEcastFromBackup *LatencyStats `json:"latency_cloudfront_icecast_from_backup"`
	LatencyAISHLSFromBackup            *LatencyStats `json:"latency_ais_hls_from_backup"`
	LatencyAISICEcastFromBackup        *LatencyStats `json:"latency_ais_icecast_from_backup"`
	TimeToFirstByte                    *LatencyStats `json:"time_to_first_byte"`
	ProcessingTime                     *LatencyStats `json:"processing_time"`
	AlignmentConfidence                *LatencyStats `json:"alignment_confidence"`
	FingerprintSimilarity              *LatencyStats `json:"fingerprint_similarity"`
}

// QualityMetrics represents stream quality analysis
type QualityMetrics struct {
	StreamValidityRate      float64        `json:"stream_validity_rate"`
	AlignmentSuccessRate    float64        `json:"alignment_success_rate"`
	FingerprintMatchRate    float64        `json:"fingerprint_match_rate"`
	AverageHealthScore      float64        `json:"average_health_score"`
	HealthScoreDistribution map[string]int `json:"health_score_distribution"`
}

// ReliabilityMetrics represents reliability analysis
type ReliabilityMetrics struct {
	OverallSuccessRate    float64            `json:"overall_success_rate"`
	StreamTypeReliability map[string]float64 `json:"stream_type_reliability"`
	ErrorDistribution     map[string]int     `json:"error_distribution"`
	RetryRate             float64            `json:"retry_rate"`
}

// CalculatePerformanceMetrics calculates detailed performance metrics
func (mc *MetricsCalculator) CalculatePerformanceMetrics(summary *latency.BenchmarkSummary) *PerformanceMetrics {
	var latencySource []float64
	var latencyCloudfrontHLS []float64
	var latencyCloudfrontICEcast []float64
	var latencyAISHLS []float64
	var latencyAISICEcast []float64
	var latencyCloudfrontHLSFromBackup []float64
	var latencyCloudfrontICEcastFromBackup []float64
	var latencyAISHLSFromBackup []float64
	var latencyAISICEcastFromBackup []float64
	var ttfbValues []float64
	var processingTimes []float64
	var alignmentConfidences []float64
	var fingerprintSimilarities []float64

	// Collect data from all measurements
	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			continue
		}

		metrics := broadcast.LivenessMetrics

		// Collect latency data
		if metrics != nil {
			if metrics.SourceLag != nil {
				liveness := metrics.SourceLag
				if liveness.Status == "MATCH" {
					latencySource = append(latencySource, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.HLSCloudfrontCDNLag != nil {
				liveness := metrics.HLSCloudfrontCDNLag
				if liveness.Status == "MATCH" {
					latencyCloudfrontHLS = append(latencyCloudfrontHLS, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.ICEcastCloudfrontCDNLag != nil {
				liveness := metrics.ICEcastCloudfrontCDNLag
				if liveness.Status == "MATCH" {
					latencyCloudfrontICEcast = append(latencyCloudfrontICEcast, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.HLSAISCDNLag != nil {
				liveness := metrics.HLSAISCDNLag
				if liveness.Status == "MATCH" {
					latencyAISHLS = append(latencyAISHLS, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.ICEcastAISCDNLag != nil {
				liveness := metrics.ICEcastAISCDNLag
				if liveness.Status == "MATCH" {
					latencyAISICEcast = append(latencyAISICEcast, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.HLSCloudfrontCDNLagFromBackup != nil {
				liveness := metrics.HLSCloudfrontCDNLagFromBackup
				if liveness.Status == "MATCH" {
					latencyCloudfrontHLSFromBackup = append(latencyCloudfrontHLSFromBackup, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.ICEcastCloudfrontCDNLagFromBackup != nil {
				liveness := metrics.ICEcastCloudfrontCDNLagFromBackup
				if liveness.Status == "MATCH" {
					latencyCloudfrontICEcastFromBackup = append(latencyCloudfrontICEcastFromBackup, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.HLSAISCDNLagFromBackup != nil {
				liveness := metrics.HLSAISCDNLagFromBackup
				if liveness.Status == "MATCH" {
					latencyAISHLSFromBackup = append(latencyAISHLSFromBackup, math.Abs(liveness.LagSeconds))
				}
			}
			if metrics.ICEcastAISCDNLagFromBackup != nil {
				liveness := metrics.ICEcastAISCDNLagFromBackup
				if liveness.Status == "MATCH" {
					latencyAISICEcastFromBackup = append(latencyAISICEcastFromBackup, math.Abs(liveness.LagSeconds))
				}
			}
		}

		// Collect stream performance data
		for _, stream := range broadcast.StreamMeasurements {
			if stream.Error == nil {
				ttfbValues = append(ttfbValues, float64(stream.TimeToFirstByte.Milliseconds()))
				processingTimes = append(processingTimes, float64(stream.TotalProcessingTime.Milliseconds()))
			}
		}

		// Collect alignment confidence data
		for _, alignment := range broadcast.AlignmentMeasurements {
			if alignment.Error == nil && alignment.AlignmentResult != nil {
				alignmentConfidences = append(alignmentConfidences, alignment.AlignmentResult.OffsetConfidence)
			}
		}

		// Collect fingerprint similarity data
		for _, comparison := range broadcast.FingerprintComparisons {
			if comparison.Error == nil && comparison.SimilarityResult != nil {
				fingerprintSimilarities = append(fingerprintSimilarities, comparison.SimilarityResult.OverallSimilarity)
			}
		}
	}

	if len(latencySource) != 0 {
		return &PerformanceMetrics{
			LatencySource:                      mc.calculateStats(latencySource),
			LatencyCloudfrontHLS:               mc.calculateStats(latencyCloudfrontHLS),
			LatencyCloudfrontICEcast:           mc.calculateStats(latencyCloudfrontICEcast),
			LatencyAISHLS:                      mc.calculateStats(latencyAISHLS),
			LatencyAISICEcast:                  mc.calculateStats(latencyAISICEcast),
			LatencyCloudfrontHLSFromBackup:     mc.calculateStats(latencyCloudfrontHLSFromBackup),
			LatencyCloudfrontICEcastFromBackup: mc.calculateStats(latencyCloudfrontICEcastFromBackup),
			LatencyAISHLSFromBackup:            mc.calculateStats(latencyAISHLSFromBackup),
			LatencyAISICEcastFromBackup:        mc.calculateStats(latencyAISICEcastFromBackup),
			TimeToFirstByte:                    mc.calculateStats(ttfbValues),
			ProcessingTime:                     mc.calculateStats(processingTimes),
			AlignmentConfidence:                mc.calculateStats(alignmentConfidences),
			FingerprintSimilarity:              mc.calculateStats(fingerprintSimilarities),
		}
	} else {
		return &PerformanceMetrics{
			LatencyCloudfrontHLS:     mc.calculateStats(latencyCloudfrontHLS),
			LatencyCloudfrontICEcast: mc.calculateStats(latencyCloudfrontICEcast),
			LatencyAISHLS:            mc.calculateStats(latencyAISHLS),
			LatencyAISICEcast:        mc.calculateStats(latencyAISICEcast),
			TimeToFirstByte:          mc.calculateStats(ttfbValues),
			ProcessingTime:           mc.calculateStats(processingTimes),
			AlignmentConfidence:      mc.calculateStats(alignmentConfidences),
			FingerprintSimilarity:    mc.calculateStats(fingerprintSimilarities),
		}
	}
}

// CalculateQualityMetrics calculates stream quality metrics
func (mc *MetricsCalculator) CalculateQualityMetrics(summary *latency.BenchmarkSummary) *QualityMetrics {
	totalStreams := 0
	validStreams := 0
	totalAlignments := 0
	successfulAlignments := 0
	totalComparisons := 0
	successfulComparisons := 0
	totalHealthScore := 0.0
	healthScoreDistribution := map[string]int{
		"excellent": 0, // 0.9-1.0
		"good":      0, // 0.7-0.9
		"fair":      0, // 0.5-0.7
		"poor":      0, // 0.0-0.5
	}

	validBroadcasts := 0

	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			continue
		}

		validBroadcasts++

		// Stream validity analysis
		for _, stream := range broadcast.StreamMeasurements {
			totalStreams++
			if stream.Error == nil && stream.StreamValidation.IsValid {
				validStreams++
			}
		}

		// Alignment success analysis
		for _, alignment := range broadcast.AlignmentMeasurements {
			totalAlignments++
			if alignment.Error == nil && alignment.IsValidAlignment {
				successfulAlignments++
			}
		}

		// Fingerprint comparison analysis
		for _, comparison := range broadcast.FingerprintComparisons {
			totalComparisons++
			if comparison.Error == nil && comparison.IsValidMatch {
				successfulComparisons++
			}
		}

		// Health score analysis
		if broadcast.OverallValidation != nil {
			healthScore := broadcast.OverallValidation.OverallHealthScore
			totalHealthScore += healthScore

			// Categorize health score
			switch {
			case healthScore >= 0.9:
				healthScoreDistribution["excellent"]++
			case healthScore >= 0.7:
				healthScoreDistribution["good"]++
			case healthScore >= 0.5:
				healthScoreDistribution["fair"]++
			default:
				healthScoreDistribution["poor"]++
			}
		}
	}

	metrics := &QualityMetrics{
		HealthScoreDistribution: healthScoreDistribution,
	}

	if totalStreams > 0 {
		metrics.StreamValidityRate = float64(validStreams) / float64(totalStreams)
	}

	if totalAlignments > 0 {
		metrics.AlignmentSuccessRate = float64(successfulAlignments) / float64(totalAlignments)
	}

	if totalComparisons > 0 {
		metrics.FingerprintMatchRate = float64(successfulComparisons) / float64(totalComparisons)
	}

	if validBroadcasts > 0 {
		metrics.AverageHealthScore = totalHealthScore / float64(validBroadcasts)
	}

	return metrics
}

// CalculateReliabilityMetrics calculates reliability metrics
func (mc *MetricsCalculator) CalculateReliabilityMetrics(summary *latency.BenchmarkSummary) *ReliabilityMetrics {
	totalBroadcasts := len(summary.BroadcastMeasurements)
	successfulBroadcasts := summary.SuccessfulBroadcasts

	streamTypeStats := map[string]struct {
		total      int
		successful int
	}{
		"hls":     {0, 0},
		"icecast": {0, 0},
	}

	errorDistribution := make(map[string]int)

	// Analyze broadcast-level reliability
	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			errorType := mc.categorizeError(broadcast.Error)
			errorDistribution[errorType]++
			continue
		}

		// Analyze stream-level reliability by type
		for _, stream := range broadcast.StreamMeasurements {
			streamTypeKey := string(stream.Endpoint.Type)
			stats := streamTypeStats[streamTypeKey]
			stats.total++

			if stream.Error == nil && stream.StreamValidation.IsValid {
				stats.successful++
			} else if stream.Error != nil {
				errorType := mc.categorizeError(stream.Error)
				errorDistribution[errorType]++
			}

			streamTypeStats[streamTypeKey] = stats
		}
	}

	// Calculate stream type reliability
	streamTypeReliability := make(map[string]float64)
	for streamType, stats := range streamTypeStats {
		if stats.total > 0 {
			streamTypeReliability[streamType] = float64(stats.successful) / float64(stats.total)
		}
	}

	metrics := &ReliabilityMetrics{
		StreamTypeReliability: streamTypeReliability,
		ErrorDistribution:     errorDistribution,
		RetryRate:             0.0, // Would need retry tracking in the orchestrator
	}

	if totalBroadcasts > 0 {
		metrics.OverallSuccessRate = float64(successfulBroadcasts) / float64(totalBroadcasts)
	}

	return metrics
}

// calculateStats calculates statistical measures for a dataset
func (mc *MetricsCalculator) calculateStats(data []float64) *LatencyStats {
	if len(data) == 0 {
		return &LatencyStats{Count: 0}
	}

	// Sort data for percentile calculations
	sortedData := make([]float64, len(data))
	copy(sortedData, data)
	mc.quickSort(sortedData, 0, len(sortedData)-1)

	stats := &LatencyStats{
		Count:  len(data),
		Min:    sortedData[0],
		Max:    sortedData[len(sortedData)-1],
		Median: mc.percentile(sortedData, 50),
		P95:    mc.percentile(sortedData, 95),
		P99:    mc.percentile(sortedData, 99),
	}

	// Calculate mean
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	stats.Mean = sum / float64(len(data))

	// Calculate standard deviation
	sumSquaredDiffs := 0.0
	for _, value := range data {
		diff := value - stats.Mean
		sumSquaredDiffs += diff * diff
	}
	stats.StdDev = math.Sqrt(sumSquaredDiffs / float64(len(data)))

	// Clean up any infinite or NaN values for JSON serialization
	stats = mc.sanitizeStats(stats)

	return stats
}

// sanitizeStats removes infinite and NaN values to prevent JSON serialization errors
func (mc *MetricsCalculator) sanitizeStats(stats *LatencyStats) *LatencyStats {
	// Replace any infinite or NaN values with safe defaults
	if math.IsInf(stats.Mean, 0) || math.IsNaN(stats.Mean) {
		stats.Mean = 0
	}
	if math.IsInf(stats.Median, 0) || math.IsNaN(stats.Median) {
		stats.Median = 0
	}
	if math.IsInf(stats.P95, 0) || math.IsNaN(stats.P95) {
		stats.P95 = 0
	}
	if math.IsInf(stats.P99, 0) || math.IsNaN(stats.P99) {
		stats.P99 = 0
	}
	if math.IsInf(stats.Min, 0) || math.IsNaN(stats.Min) {
		stats.Min = 0
	}
	if math.IsInf(stats.Max, 0) || math.IsNaN(stats.Max) {
		stats.Max = 0
	}
	if math.IsInf(stats.StdDev, 0) || math.IsNaN(stats.StdDev) {
		stats.StdDev = 0
	}

	return stats
}

// percentile calculates the specified percentile of sorted data
func (mc *MetricsCalculator) percentile(sortedData []float64, p float64) float64 {
	if len(sortedData) == 0 {
		return 0
	}

	if len(sortedData) == 1 {
		return sortedData[0]
	}

	// Calculate index for percentile
	index := (p / 100.0) * float64(len(sortedData)-1)

	// If index is not an integer, interpolate
	if index != float64(int(index)) {
		lower := int(math.Floor(index))
		upper := int(math.Ceil(index))

		if upper >= len(sortedData) {
			return sortedData[len(sortedData)-1]
		}

		weight := index - float64(lower)
		return sortedData[lower]*(1-weight) + sortedData[upper]*weight
	}

	return sortedData[int(index)]
}

// quickSort implements quicksort algorithm for sorting float64 slices
func (mc *MetricsCalculator) quickSort(arr []float64, low, high int) {
	if low < high {
		pi := mc.partition(arr, low, high)
		mc.quickSort(arr, low, pi-1)
		mc.quickSort(arr, pi+1, high)
	}
}

// partition is a helper function for quicksort
func (mc *MetricsCalculator) partition(arr []float64, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j <= high-1; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// categorizeError categorizes errors into meaningful categories
func (mc *MetricsCalculator) categorizeError(err error) string {
	if err == nil {
		return "none"
	}

	errStr := err.Error()

	// Network-related errors
	if mc.containsAny(errStr, []string{"timeout", "connection", "network", "dns"}) {
		return "network"
	}

	// Stream format errors
	if mc.containsAny(errStr, []string{"decode", "format", "codec", "audio"}) {
		return "format"
	}

	// HTTP errors
	if mc.containsAny(errStr, []string{"404", "403", "500", "http"}) {
		return "http"
	}

	// Processing errors
	if mc.containsAny(errStr, []string{"fingerprint", "alignment", "processing"}) {
		return "processing"
	}

	// Configuration errors
	if mc.containsAny(errStr, []string{"config", "validation", "invalid"}) {
		return "configuration"
	}

	return "other"
}

// containsAny checks if a string contains any of the specified substrings
func (mc *MetricsCalculator) containsAny(str string, substrings []string) bool {
	for _, substr := range substrings {
		if len(str) >= len(substr) {
			for i := 0; i <= len(str)-len(substr); i++ {
				if str[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}
