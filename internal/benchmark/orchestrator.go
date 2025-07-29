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

	o.logger.Info("Starting CDN benchmark", logging.Fields{
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

	o.logger.Info("CDN benchmark completed", logging.Fields{
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

	o.logger.Info("Starting broadcast group measurement", logging.Fields{
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

	// Step 2: Perform the 3 required alignment measurements
	measurement.AlignmentMeasurements = o.measureRequiredAlignments(ctx, group, streamMeasurements)

	// Step 3: Perform fingerprint comparisons (if enabled)
	if !o.benchmarkConfig.Benchmark.SkipFingerprintComparison {
		measurement.FingerprintComparisons = o.measureFingerprintComparisons(ctx, group, streamMeasurements, measurement.AlignmentMeasurements)
	}

	// Step 4: Calculate liveness metrics using the correct algorithm
	measurement.LivenessMetrics = o.calculateCorrectLivenessMetrics(measurement.AlignmentMeasurements, streamMeasurements)

	// Step 5: Perform overall validation
	measurement.OverallValidation = o.validateOverallHealth(measurement)

	measurement.TotalBenchmarkTime = time.Since(benchmarkStart)

	o.logger.Info("Broadcast group measurement completed", logging.Fields{
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

// measureRequiredAlignments performs the 3 specific alignment measurements required
func (o *Orchestrator) measureRequiredAlignments(ctx context.Context, group *latency.BroadcastGroup, streamMeasurements map[string]*latency.StreamMeasurement) map[string]*latency.AlignmentMeasurement {
	alignments := make(map[string]*latency.AlignmentMeasurement)

	// Define the 3 required alignment pairs
	alignmentPairs := []struct {
		name  string
		type1 latency.StreamType
		role1 latency.StreamRole
		type2 latency.StreamType
		role2 latency.StreamRole
	}{
		{"hls_source_to_cdn", latency.StreamTypeHLS, latency.StreamRoleSource, latency.StreamTypeHLS, latency.StreamRoleCDN},
		{"icecast_source_to_cdn", latency.StreamTypeICEcast, latency.StreamRoleSource, latency.StreamTypeICEcast, latency.StreamRoleCDN},
		{"hls_cdn_to_icecast_cdn", latency.StreamTypeHLS, latency.StreamRoleCDN, latency.StreamTypeICEcast, latency.StreamRoleCDN},
	}

	for _, pair := range alignmentPairs {
		stream1 := o.findStreamMeasurement(streamMeasurements, pair.type1, pair.role1)
		stream2 := o.findStreamMeasurement(streamMeasurements, pair.type2, pair.role2)

		if stream1 != nil && stream2 != nil && stream1.Error == nil && stream2.Error == nil {
			o.logger.Info("Performing temporal alignment", logging.Fields{
				"alignment_name": pair.name,
				"stream1":        fmt.Sprintf("%s_%s", pair.type1, pair.role1),
				"stream2":        fmt.Sprintf("%s_%s", pair.type2, pair.role2),
			})
			alignment := o.engine.MeasureAlignment(ctx, stream1, stream2)
			alignments[pair.name] = alignment
		} else {
			o.logger.Info("Skipping alignment - missing streams", logging.Fields{
				"alignment_name": pair.name,
				"stream1_found":  stream1 != nil && stream1.Error == nil,
				"stream2_found":  stream2 != nil && stream2.Error == nil,
			})
		}
	}

	return alignments
}

// calculateCorrectLivenessMetrics calculates liveness using the correct algorithm
func (o *Orchestrator) calculateCorrectLivenessMetrics(alignmentMeasurements map[string]*latency.AlignmentMeasurement, streamMeasurements map[string]*latency.StreamMeasurement) *latency.LivenessMetrics {
	metrics := &latency.LivenessMetrics{}

	o.logger.Info("Calculating liveness metrics using temporal alignment", logging.Fields{
		"alignment_count": len(alignmentMeasurements),
		"stream_count":    len(streamMeasurements),
	})

	// Step 1: Build a graph of temporal offsets between streams
	streamOffsets := make(map[string]float64) // stream_key -> offset from reference
	streamTTFBs := make(map[string]time.Duration)

	// Collect all streams and their TTFBs
	for _, stream := range streamMeasurements {
		if stream.Error != nil {
			continue
		}
		streamKey := fmt.Sprintf("%s_%s", stream.Endpoint.Type, stream.Endpoint.Role)
		streamTTFBs[streamKey] = stream.TimeToFirstByte
	}

	// Step 2: Process valid alignments to build offset relationships
	alignmentData := make(map[string]struct {
		stream1Key    string
		stream2Key    string
		offsetSeconds float64 // stream2 is this many seconds behind stream1
		isValid       bool
	})

	for alignmentName, alignment := range alignmentMeasurements {
		if alignment.Error != nil || !alignment.IsValidAlignment {
			o.logger.Info("Invalid alignment, skipping", logging.Fields{
				"alignment_name": alignmentName,
				"error":          alignment.Error,
				"is_valid":       alignment.IsValidAlignment,
			})
			continue
		}

		stream1Key := fmt.Sprintf("%s_%s", alignment.Stream1.Endpoint.Type, alignment.Stream1.Endpoint.Role)
		stream2Key := fmt.Sprintf("%s_%s", alignment.Stream2.Endpoint.Type, alignment.Stream2.Endpoint.Role)

		alignmentData[alignmentName] = struct {
			stream1Key    string
			stream2Key    string
			offsetSeconds float64
			isValid       bool
		}{
			stream1Key:    stream1Key,
			stream2Key:    stream2Key,
			offsetSeconds: alignment.LatencySeconds,
			isValid:       true,
		}

		o.logger.Info("Valid alignment found", logging.Fields{
			"alignment_name": alignmentName,
			"stream1":        stream1Key,
			"stream2":        stream2Key,
			"offset_seconds": alignment.LatencySeconds,
			"confidence":     alignment.AlignmentResult.OffsetConfidence,
		})
	}

	// Step 3: Find the reference stream (the one that appears to be most live)
	// We'll use a simple approach: pick the first stream and calculate all others relative to it
	var referenceStream string
	var referenceTTFB time.Duration

	// Try to find HLS source as reference first, then others
	preferredOrder := []string{"hls_source", "icecast_source", "hls_cdn", "icecast_cdn"}
	for _, preferred := range preferredOrder {
		if ttfb, exists := streamTTFBs[preferred]; exists {
			referenceStream = preferred
			referenceTTFB = ttfb
			break
		}
	}

	if referenceStream == "" {
		// Just pick the first available stream
		for streamKey, ttfb := range streamTTFBs {
			referenceStream = streamKey
			referenceTTFB = ttfb
			break
		}
	}

	if referenceStream == "" {
		o.logger.Info("No valid streams found for liveness calculation")
		return metrics
	}

	o.logger.Info("Selected reference stream", logging.Fields{
		"reference_stream": referenceStream,
		"reference_ttfb":   referenceTTFB.Milliseconds(),
	})

	// Step 4: Calculate offsets for all streams relative to reference
	streamOffsets[referenceStream] = 0.0 // Reference is at 0 offset

	// Build offset graph using alignment data
	for _, data := range alignmentData {
		if !data.isValid {
			continue
		}

		// If we know the offset of stream1 and we have alignment to stream2
		if stream1Offset, hasStream1 := streamOffsets[data.stream1Key]; hasStream1 {
			if _, hasStream2 := streamOffsets[data.stream2Key]; !hasStream2 {
				// stream2 is data.offsetSeconds behind stream1
				streamOffsets[data.stream2Key] = stream1Offset + data.offsetSeconds
				o.logger.Info("Calculated stream offset", logging.Fields{
					"stream":        data.stream2Key,
					"offset":        streamOffsets[data.stream2Key],
					"via_stream":    data.stream1Key,
					"alignment_gap": data.offsetSeconds,
				})
			}
		}

		// Also try the reverse direction
		if stream2Offset, hasStream2 := streamOffsets[data.stream2Key]; hasStream2 {
			if _, hasStream1 := streamOffsets[data.stream1Key]; !hasStream1 {
				// stream1 is data.offsetSeconds ahead of stream2
				streamOffsets[data.stream1Key] = stream2Offset - data.offsetSeconds
				o.logger.Info("Calculated stream offset (reverse)", logging.Fields{
					"stream":        data.stream1Key,
					"offset":        streamOffsets[data.stream1Key],
					"via_stream":    data.stream2Key,
					"alignment_gap": -data.offsetSeconds,
				})
			}
		}
	}

	// Step 5: Calculate final end-to-end latency for each stream
	finalLatencies := make(map[string]float64)

	for streamKey, ttfb := range streamTTFBs {
		offset, hasOffset := streamOffsets[streamKey]
		if !hasOffset {
			o.logger.Info("No offset available for stream, skipping", logging.Fields{
				"stream": streamKey,
			})
			continue
		}

		// Base latency = reference TTFB + temporal offset
		baseLatency := referenceTTFB.Seconds() + offset

		// Add TTFB penalty if this stream's TTFB > reference TTFB
		ttfbPenalty := 0.0
		if ttfb > referenceTTFB {
			ttfbPenalty = (ttfb - referenceTTFB).Seconds()
		}

		finalLatency := baseLatency + ttfbPenalty
		finalLatencies[streamKey] = finalLatency

		o.logger.Info("Final stream latency calculated", logging.Fields{
			"stream":            streamKey,
			"is_reference":      streamKey == referenceStream,
			"reference_ttfb_ms": referenceTTFB.Milliseconds(),
			"stream_ttfb_ms":    ttfb.Milliseconds(),
			"temporal_offset_s": offset,
			"ttfb_penalty_s":    ttfbPenalty,
			"final_latency_s":   finalLatency,
		})
	}

	// Step 6: Map to metrics structure
	if val, exists := finalLatencies["hls_source"]; exists {
		metrics.HLSSourceLag = val
	}
	if val, exists := finalLatencies["hls_cdn"]; exists {
		metrics.HLSCDNLag = val
	}
	if val, exists := finalLatencies["icecast_source"]; exists {
		metrics.ICEcastSourceLag = val
	}
	if val, exists := finalLatencies["icecast_cdn"]; exists {
		metrics.ICEcastCDNLag = val
	}

	// Calculate relative CDN latencies (CDN - Source for same protocol)
	if metrics.HLSCDNLag > 0 && metrics.HLSSourceLag > 0 {
		metrics.CDNLatencyHLS = metrics.HLSCDNLag - metrics.HLSSourceLag
	}
	if metrics.ICEcastCDNLag > 0 && metrics.ICEcastSourceLag > 0 {
		metrics.CDNLatencyICEcast = metrics.ICEcastCDNLag - metrics.ICEcastSourceLag
	}

	// Cross-protocol comparison (HLS vs ICEcast sources)
	if metrics.HLSSourceLag > 0 && metrics.ICEcastSourceLag > 0 {
		metrics.CrossProtocolLag = metrics.HLSSourceLag - metrics.ICEcastSourceLag
	}

	o.logger.Info("Liveness metrics completed", logging.Fields{
		"hls_source_lag":      metrics.HLSSourceLag,
		"hls_cdn_lag":         metrics.HLSCDNLag,
		"icecast_source_lag":  metrics.ICEcastSourceLag,
		"icecast_cdn_lag":     metrics.ICEcastCDNLag,
		"cdn_latency_hls":     metrics.CDNLatencyHLS,
		"cdn_latency_icecast": metrics.CDNLatencyICEcast,
		"cross_protocol_lag":  metrics.CrossProtocolLag,
	})

	return metrics
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
		{"hls_cdn_to_icecast_cdn_fingerprint", "hls_cdn_to_icecast_cdn"},
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
