package benchmark

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/tunein/cdn-benchmark-cli/internal/latency"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// Orchestrator coordinates the entire CDN benchmarking process
type Orchestrator struct {
	benchmarkConfig *latency.BenchmarkConfig
	broadcastConfig *latency.BroadcastConfig
	engine          *latency.MeasurementEngine
	logger          logging.Logger
	metrics         *MetricsCalculator
}

// NewOrchestrator creates a new benchmark orchestrator
func NewOrchestrator(benchmarkCfg *latency.BenchmarkConfig, broadcastCfg *latency.BroadcastConfig, logger logging.Logger) (*Orchestrator, error) {
	if logger == nil {
		logger = logging.NewDefaultLogger()
	}

	// Create measurement engine
	engineConfig := &latency.EngineConfig{
		OperationTimeout:       benchmarkCfg.Benchmark.OperationTimeout,
		AudioSegmentDuration:   benchmarkCfg.Benchmark.AudioSegmentDuration,
		MinAlignmentConfidence: benchmarkCfg.Benchmark.MinAlignmentConfidence,
		MaxAlignmentOffset:     benchmarkCfg.Benchmark.MaxAlignmentOffset,
		MinSimilarity:          benchmarkCfg.Benchmark.MinFingerprintSimilarity,
		EnableDetailedAnalysis: benchmarkCfg.Benchmark.EnableDetailedAnalysis,
		UserAgent:              benchmarkCfg.Stream.UserAgent,
		Logger:                 logger,
	}

	engine := latency.NewMeasurementEngine(engineConfig)
	metrics := NewMetricsCalculator(logger)

	return &Orchestrator{
		benchmarkConfig: benchmarkCfg,
		broadcastConfig: broadcastCfg,
		engine:          engine,
		logger:          logger,
		metrics:         metrics,
	}, nil
}

// RunBenchmark executes the complete CDN benchmark
func (o *Orchestrator) RunBenchmark(ctx context.Context) (*latency.BenchmarkSummary, error) {
	startTime := time.Now()

	o.logger.Info("Starting CDN benchmark", map[string]any{
		"enabled_groups":    len(o.broadcastConfig.GetEnabledBroadcastGroups()),
		"segment_duration":  o.benchmarkConfig.Benchmark.AudioSegmentDuration.Seconds(),
		"operation_timeout": o.benchmarkConfig.Benchmark.OperationTimeout.Seconds(),
	})

	// Create benchmark context with timeout
	benchmarkCtx, cancel := context.WithTimeout(ctx, o.benchmarkConfig.Benchmark.BenchmarkTimeout)
	defer cancel()

	// Get enabled broadcast groups
	enabledGroups := o.broadcastConfig.GetEnabledBroadcastGroups()
	if len(enabledGroups) == 0 {
		return nil, fmt.Errorf("no enabled broadcast groups found")
	}

	// Execute measurements with concurrency control
	broadcastMeasurements, err := o.measureBroadcastGroups(benchmarkCtx, enabledGroups)
	if err != nil {
		return nil, fmt.Errorf("broadcast measurements failed: %w", err)
	}

	endTime := time.Now()

	// Calculate summary metrics
	summary := &latency.BenchmarkSummary{
		BroadcastMeasurements: broadcastMeasurements,
		StartTime:             startTime,
		EndTime:               endTime,
		TotalDuration:         endTime.Sub(startTime),
	}

	// Calculate aggregate metrics
	o.calculateSummaryMetrics(summary)

	o.logger.Info("CDN benchmark completed", map[string]any{
		"total_duration_s":      summary.TotalDuration.Seconds(),
		"successful_broadcasts": summary.SuccessfulBroadcasts,
		"failed_broadcasts":     summary.FailedBroadcasts,
		"overall_health_score":  summary.OverallHealthScore,
	})

	return summary, nil
}

// measureBroadcastGroups measures all broadcast groups with concurrency control
func (o *Orchestrator) measureBroadcastGroups(ctx context.Context, groups map[string]*latency.BroadcastGroup) (map[string]*latency.BroadcastMeasurement, error) {
	results := make(map[string]*latency.BroadcastMeasurement)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create semaphore for concurrency control
	semaphore := make(chan struct{}, o.benchmarkConfig.Benchmark.MaxConcurrentBroadcasts)

	for groupName, group := range groups {
		wg.Add(1)
		go func(name string, grp *latency.BroadcastGroup) {
			defer wg.Done()

			// Acquire semaphore
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				return
			}

			measurement := o.measureBroadcastGroup(ctx, name, grp)

			mu.Lock()
			results[name] = measurement
			mu.Unlock()
		}(groupName, group)
	}

	wg.Wait()
	return results, nil
}

// measureBroadcastGroup measures a single broadcast group
func (o *Orchestrator) measureBroadcastGroup(ctx context.Context, groupName string, group *latency.BroadcastGroup) *latency.BroadcastMeasurement {
	measurement := &latency.BroadcastMeasurement{
		Group:                  group,
		StreamMeasurements:     make(map[string]*latency.StreamMeasurement),
		AlignmentMeasurements:  make(map[string]*latency.AlignmentMeasurement),
		FingerprintComparisons: make(map[string]*latency.FingerprintComparison),
		Timestamp:              time.Now(),
	}

	benchmarkStart := time.Now()

	o.logger.Info("Starting broadcast group measurement", map[string]any{
		"group_name":   groupName,
		"stream_count": len(group.Streams),
		"content_type": group.ContentType,
	})

	// Step 1: Measure all streams concurrently
	streamMeasurements := o.measureAllStreams(ctx, group)
	measurement.StreamMeasurements = streamMeasurements

	// Check if we have enough valid streams to continue
	validStreams := o.countValidStreams(streamMeasurements)
	if validStreams < 2 {
		measurement.Error = fmt.Errorf("insufficient valid streams (%d) for meaningful comparison", validStreams)
		measurement.TotalBenchmarkTime = time.Since(benchmarkStart)
		return measurement
	}

	// Step 2: Perform alignment measurements
	measurement.AlignmentMeasurements = o.measureAlignments(ctx, group, streamMeasurements)

	// Step 3: Perform fingerprint comparisons (if enabled)
	if !o.benchmarkConfig.Benchmark.SkipFingerprintComparison {
		measurement.FingerprintComparisons = o.measureFingerprintComparisons(ctx, group, streamMeasurements, measurement.AlignmentMeasurements)
	}

	// Step 4: Calculate liveness metrics
	measurement.LivenessMetrics = o.calculateLivenessMetrics(measurement.AlignmentMeasurements)

	// Step 5: Perform overall validation
	measurement.OverallValidation = o.validateOverallHealth(measurement)

	measurement.TotalBenchmarkTime = time.Since(benchmarkStart)

	o.logger.Info("Broadcast group measurement completed", map[string]any{
		"group_name":           groupName,
		"total_time_s":         measurement.TotalBenchmarkTime.Seconds(),
		"valid_streams":        validStreams,
		"alignment_count":      len(measurement.AlignmentMeasurements),
		"fingerprint_count":    len(measurement.FingerprintComparisons),
		"overall_health_score": measurement.OverallValidation.OverallHealthScore,
	})

	return measurement
}

// measureAllStreams measures all streams in a broadcast group concurrently
func (o *Orchestrator) measureAllStreams(ctx context.Context, group *latency.BroadcastGroup) map[string]*latency.StreamMeasurement {
	results := make(map[string]*latency.StreamMeasurement)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create semaphore for stream concurrency control
	semaphore := make(chan struct{}, o.benchmarkConfig.Benchmark.MaxConcurrentStreams)

	for streamName, stream := range group.Streams {
		if !stream.Enabled {
			continue
		}

		wg.Add(1)
		go func(name string, endpoint *latency.StreamEndpoint) {
			defer wg.Done()

			// Acquire semaphore
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				return
			}

			measurement := o.engine.MeasureStream(ctx, endpoint)

			mu.Lock()
			results[name] = measurement
			mu.Unlock()
		}(streamName, stream)
	}

	wg.Wait()
	return results
}

// measureAlignments performs alignment measurements between relevant stream pairs
func (o *Orchestrator) measureAlignments(ctx context.Context, group *latency.BroadcastGroup, streamMeasurements map[string]*latency.StreamMeasurement) map[string]*latency.AlignmentMeasurement {
	alignments := make(map[string]*latency.AlignmentMeasurement)

	// Define the alignment pairs we want to measure
	alignmentPairs := []struct {
		name  string
		type1 latency.StreamType
		role1 latency.StreamRole
		type2 latency.StreamType
		role2 latency.StreamRole
	}{
		{"hls_source_to_cdn", latency.StreamTypeHLS, latency.StreamRoleSource, latency.StreamTypeHLS, latency.StreamRoleCDN},
		{"icecast_source_to_cdn", latency.StreamTypeICEcast, latency.StreamRoleSource, latency.StreamTypeICEcast, latency.StreamRoleCDN},
		{"hls_to_icecast_source", latency.StreamTypeHLS, latency.StreamRoleSource, latency.StreamTypeICEcast, latency.StreamRoleSource},
	}

	for _, pair := range alignmentPairs {
		stream1 := o.findStreamMeasurement(streamMeasurements, pair.type1, pair.role1)
		stream2 := o.findStreamMeasurement(streamMeasurements, pair.type2, pair.role2)

		if stream1 != nil && stream2 != nil && stream1.Error == nil && stream2.Error == nil {
			alignment := o.engine.MeasureAlignment(ctx, stream1, stream2)
			alignments[pair.name] = alignment
		}
	}

	return alignments
}

// measureFingerprintComparisons performs fingerprint comparisons with alignment
func (o *Orchestrator) measureFingerprintComparisons(ctx context.Context, group *latency.BroadcastGroup, streamMeasurements map[string]*latency.StreamMeasurement, alignmentMeasurements map[string]*latency.AlignmentMeasurement) map[string]*latency.FingerprintComparison {
	comparisons := make(map[string]*latency.FingerprintComparison)

	// Perform fingerprint comparisons using alignment data where available
	comparisonPairs := []struct {
		name      string
		alignment string // corresponding alignment measurement name
	}{
		{"hls_source_to_cdn_fingerprint", "hls_source_to_cdn"},
		{"icecast_source_to_cdn_fingerprint", "icecast_source_to_cdn"},
		{"hls_to_icecast_source_fingerprint", "hls_to_icecast_source"},
	}

	for _, pair := range comparisonPairs {
		alignment, hasAlignment := alignmentMeasurements[pair.alignment]
		if !hasAlignment || alignment.Error != nil {
			continue
		}

		var alignmentFeatures = alignment.AlignmentResult
		if !alignment.IsValidAlignment {
			alignmentFeatures = nil // Don't use invalid alignment
		}

		comparison := o.engine.CompareFingerprintSimilarity(ctx, alignment.Stream1, alignment.Stream2, alignmentFeatures)
		comparisons[pair.name] = comparison
	}

	return comparisons
}

// calculateLivenessMetrics calculates how far behind live each stream is
func (o *Orchestrator) calculateLivenessMetrics(alignmentMeasurements map[string]*latency.AlignmentMeasurement) *latency.LivenessMetrics {
	metrics := &latency.LivenessMetrics{}

	// HLS Source to CDN latency
	if hlsAlignment, exists := alignmentMeasurements["hls_source_to_cdn"]; exists && hlsAlignment.IsValidAlignment {
		metrics.CDNLatencyHLS = hlsAlignment.LatencySeconds
	}

	// ICEcast Source to CDN latency
	if icecastAlignment, exists := alignmentMeasurements["icecast_source_to_cdn"]; exists && icecastAlignment.IsValidAlignment {
		metrics.CDNLatencyICEcast = icecastAlignment.LatencySeconds
	}

	// Cross-protocol comparison (HLS vs ICEcast at source)
	if crossAlignment, exists := alignmentMeasurements["hls_to_icecast_source"]; exists && crossAlignment.IsValidAlignment {
		metrics.CrossProtocolLag = crossAlignment.LatencySeconds
	}

	// For absolute liveness, we'd need a reference time source
	// For now, we calculate relative liveness based on the fastest stream
	// This is a simplified implementation - in practice, you might have a live reference

	return metrics
}

// validateOverallHealth validates the overall health of all streams
func (o *Orchestrator) validateOverallHealth(measurement *latency.BroadcastMeasurement) *latency.OverallValidation {
	validation := &latency.OverallValidation{
		AllStreamsValid:         true,
		ValidStreamCount:        0,
		InvalidStreamCount:      0,
		StreamValidityIssues:    []string{},
		FingerprintMatchesValid: true,
		AlignmentQualityGood:    true,
	}

	// Validate individual streams
	for streamName, stream := range measurement.StreamMeasurements {
		if stream.Error != nil {
			validation.AllStreamsValid = false
			validation.InvalidStreamCount++
			validation.StreamValidityIssues = append(validation.StreamValidityIssues,
				fmt.Sprintf("stream %s: %v", streamName, stream.Error))
			continue
		}

		if stream.StreamValidation.IsValid {
			validation.ValidStreamCount++
		} else {
			validation.AllStreamsValid = false
			validation.InvalidStreamCount++
			for _, err := range stream.StreamValidation.ValidationErrors {
				validation.StreamValidityIssues = append(validation.StreamValidityIssues,
					fmt.Sprintf("stream %s: %s", streamName, err))
			}
		}
	}

	// Validate fingerprint matches
	for _, comparison := range measurement.FingerprintComparisons {
		if comparison.Error == nil && !comparison.IsValidMatch {
			validation.FingerprintMatchesValid = false
			break
		}
	}

	// Validate alignment quality
	for _, alignment := range measurement.AlignmentMeasurements {
		if alignment.Error == nil && alignment.IsValidAlignment {
			if alignment.AlignmentResult.AlignmentQuality < 0.5 { // Configurable threshold
				validation.AlignmentQualityGood = false
				break
			}
		}
	}

	// Calculate overall health score
	validation.OverallHealthScore = o.calculateHealthScore(validation, measurement)

	return validation
}

// Helper functions

func (o *Orchestrator) countValidStreams(measurements map[string]*latency.StreamMeasurement) int {
	count := 0
	for _, measurement := range measurements {
		if measurement.Error == nil && measurement.StreamValidation.IsValid {
			count++
		}
	}
	return count
}

func (o *Orchestrator) findStreamMeasurement(measurements map[string]*latency.StreamMeasurement, streamType latency.StreamType, role latency.StreamRole) *latency.StreamMeasurement {
	for _, measurement := range measurements {
		if measurement.Endpoint.Type == streamType && measurement.Endpoint.Role == role {
			return measurement
		}
	}
	return nil
}

func (o *Orchestrator) calculateHealthScore(validation *latency.OverallValidation, measurement *latency.BroadcastMeasurement) float64 {
	if validation.ValidStreamCount == 0 {
		return 0.0
	}

	// Base score from stream validity
	streamScore := float64(validation.ValidStreamCount) / float64(validation.ValidStreamCount+validation.InvalidStreamCount)

	// Bonus for fingerprint matches
	fingerprintBonus := 0.0
	if validation.FingerprintMatchesValid {
		fingerprintBonus = 0.1
	}

	// Bonus for good alignment quality
	alignmentBonus := 0.0
	if validation.AlignmentQualityGood {
		alignmentBonus = 0.1
	}

	totalScore := streamScore + fingerprintBonus + alignmentBonus
	if totalScore > 1.0 {
		totalScore = 1.0
	}

	return totalScore
}

func (o *Orchestrator) calculateSummaryMetrics(summary *latency.BenchmarkSummary) {
	totalBroadcasts := len(summary.BroadcastMeasurements)
	successCount := 0
	failureCount := 0

	var avgMetrics latency.AverageLatencyMetrics
	var totalHealthScore float64

	validMeasurements := 0

	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			failureCount++
			continue
		}

		successCount++
		totalHealthScore += broadcast.OverallValidation.OverallHealthScore

		if broadcast.LivenessMetrics != nil {
			avgMetrics.AvgCDNLatencyHLS += broadcast.LivenessMetrics.CDNLatencyHLS
			avgMetrics.AvgCDNLatencyICEcast += broadcast.LivenessMetrics.CDNLatencyICEcast
			avgMetrics.AvgCrossProtocolLag += broadcast.LivenessMetrics.CrossProtocolLag
			validMeasurements++
		}

		// Aggregate stream performance metrics
		for _, stream := range broadcast.StreamMeasurements {
			if stream.Error == nil {
				avgMetrics.AvgTimeToFirstByte += float64(stream.TimeToFirstByte.Milliseconds())
				avgMetrics.AvgAudioExtractionTime += float64(stream.TotalProcessingTime.Milliseconds())
			}
		}
	}

	// Calculate averages
	if validMeasurements > 0 {
		avgMetrics.AvgCDNLatencyHLS /= float64(validMeasurements)
		avgMetrics.AvgCDNLatencyICEcast /= float64(validMeasurements)
		avgMetrics.AvgCrossProtocolLag /= float64(validMeasurements)
	}

	streamCount := 0
	for _, broadcast := range summary.BroadcastMeasurements {
		streamCount += len(broadcast.StreamMeasurements)
	}

	if streamCount > 0 {
		avgMetrics.AvgTimeToFirstByte /= float64(streamCount)
		avgMetrics.AvgAudioExtractionTime /= float64(streamCount)
	}

	summary.SuccessfulBroadcasts = successCount
	summary.FailedBroadcasts = failureCount
	summary.AverageLatencyMetrics = &avgMetrics

	if totalBroadcasts > 0 {
		summary.OverallHealthScore = totalHealthScore / float64(totalBroadcasts)
	}
}
