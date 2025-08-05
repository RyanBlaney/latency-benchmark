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
		AdBypassRules:          benchmarkCfg.Stream.AdBypassRules,
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

	benchmarkCtx, cancel := context.WithTimeout(ctx, o.benchmarkConfig.Benchmark.BenchmarkTimeout)
	defer cancel()

	// For now, we only process the first broadcast group and first broadcast
	// Later this will be extended to support index-based selection for parallel execution
	broadcast, broadcastKey, err := o.broadcastConfig.SelectBroadcastByIndex(0)
	if err != nil {
		return nil, fmt.Errorf("failed to select broadcast by index 0: %w", err)
	}

	o.logger.Info("Selected broadcast for benchmarking", logging.Fields{
		"broadcast_key": broadcastKey,
		"stream_count":  len(broadcast.Streams),
		"content_type":  broadcast.ContentType,
	})

	// Process the selected broadcast
	broadcastMeasurement := o.measureSingleBroadcast(benchmarkCtx, broadcastKey, broadcast)

	// Create summary with single broadcast result
	broadcastMeasurements := map[string]*latency.BroadcastMeasurement{
		broadcastKey: broadcastMeasurement,
	}

	endTime := time.Now()

	summary := &latency.BenchmarkSummary{
		BroadcastMeasurements: broadcastMeasurements,
		StartTime:             startTime,
		EndTime:               endTime,
		TotalDuration:         endTime.Sub(startTime),
	}

	o.calculateSummaryMetrics(summary)

	o.logger.Info("CDN benchmark completed", logging.Fields{
		"total_duration_s":      summary.TotalDuration.Seconds(),
		"successful_broadcasts": summary.SuccessfulBroadcasts,
		"failed_broadcasts":     summary.FailedBroadcasts,
		"overall_health_score":  summary.OverallHealthScore,
	})

	return summary, nil
}

// RunBenchmarkByIndex processes a specific broadcast by index (for parallel execution)
func (o *Orchestrator) RunBenchmarkByIndex(ctx context.Context, index int) (*latency.BenchmarkSummary, error) {
	startTime := time.Now()

	o.logger.Info("Starting CDN benchmark by index", logging.Fields{
		"index":             index,
		"segment_duration":  o.benchmarkConfig.Benchmark.AudioSegmentDuration.Seconds(),
		"operation_timeout": o.benchmarkConfig.Benchmark.OperationTimeout.Seconds(),
	})

	benchmarkCtx, cancel := context.WithTimeout(ctx, o.benchmarkConfig.Benchmark.BenchmarkTimeout)
	defer cancel()

	// Select broadcast by index
	broadcast, broadcastKey, err := o.broadcastConfig.SelectBroadcastByIndex(index)
	if err != nil {
		return nil, fmt.Errorf("failed to select broadcast by index %d: %w", index, err)
	}

	o.logger.Info("Selected broadcast for benchmarking", logging.Fields{
		"index":         index,
		"broadcast_key": broadcastKey,
		"stream_count":  len(broadcast.Streams),
		"content_type":  broadcast.ContentType,
	})

	// Process the selected broadcast
	broadcastMeasurement := o.measureSingleBroadcast(benchmarkCtx, broadcastKey, broadcast)

	// Create summary with single broadcast result
	broadcastMeasurements := map[string]*latency.BroadcastMeasurement{
		broadcastKey: broadcastMeasurement,
	}

	endTime := time.Now()

	summary := &latency.BenchmarkSummary{
		BroadcastMeasurements: broadcastMeasurements,
		StartTime:             startTime,
		EndTime:               endTime,
		TotalDuration:         endTime.Sub(startTime),
	}

	o.calculateSummaryMetrics(summary)

	o.logger.Info("CDN benchmark completed", logging.Fields{
		"index":                 index,
		"total_duration_s":      summary.TotalDuration.Seconds(),
		"successful_broadcasts": summary.SuccessfulBroadcasts,
		"failed_broadcasts":     summary.FailedBroadcasts,
		"overall_health_score":  summary.OverallHealthScore,
	})

	return summary, nil
}

// measureSingleBroadcast measures a single broadcast (the 5-6 streams)
func (o *Orchestrator) measureSingleBroadcast(ctx context.Context, broadcastKey string, broadcast *latency.Broadcast) *latency.BroadcastMeasurement {
	measurement := &latency.BroadcastMeasurement{
		Group:                  broadcast, // Fixed: point to the actual broadcast
		StreamMeasurements:     make(map[string]*latency.StreamMeasurement),
		AlignmentMeasurements:  make(map[string]*latency.AlignmentMeasurement),
		FingerprintComparisons: make(map[string]*latency.FingerprintComparison),
		Timestamp:              time.Now(),
	}

	benchmarkStart := time.Now()

	o.logger.Info("Starting broadcast measurement", logging.Fields{
		"broadcast_key": broadcastKey,
		"stream_count":  len(broadcast.Streams),
		"content_type":  broadcast.ContentType,
	})

	// Step 1: Measure all streams in this broadcast
	streamMeasurements := o.measureAllStreamsInBroadcast(ctx, broadcast)
	measurement.StreamMeasurements = streamMeasurements

	// Check if we have enough valid streams
	validStreams := o.countValidStreams(streamMeasurements)
	if validStreams < 2 {
		measurement.Error = fmt.Errorf("insufficient valid streams (%d) for meaningful comparison", validStreams)
		measurement.TotalBenchmarkTime = time.Since(benchmarkStart)
		return measurement
	}

	// Step 2: Perform source-to-CDN alignments only
	measurement.AlignmentMeasurements = o.measureSourceToCDNAlignments(ctx, streamMeasurements)

	// Step 3: Perform fingerprint comparisons (if enabled)
	if !o.benchmarkConfig.Benchmark.SkipFingerprintComparison {
		measurement.FingerprintComparisons = o.measureFingerprintComparisons(ctx, measurement.AlignmentMeasurements)
	}

	// Step 4: Calculate liveness metrics
	measurement.LivenessMetrics = o.calculateLivenessMetrics(measurement.AlignmentMeasurements)

	// Step 5: Perform overall validation
	measurement.OverallValidation = o.validateOverallHealth(measurement)

	measurement.TotalBenchmarkTime = time.Since(benchmarkStart)

	o.logger.Info("Broadcast measurement completed", logging.Fields{
		"broadcast_key":        broadcastKey,
		"total_time_s":         measurement.TotalBenchmarkTime.Seconds(),
		"valid_streams":        validStreams,
		"alignment_count":      len(measurement.AlignmentMeasurements),
		"fingerprint_count":    len(measurement.FingerprintComparisons),
		"overall_health_score": measurement.OverallValidation.OverallHealthScore,
	})

	return measurement
}

// measureAllStreamsInBroadcast measures all streams within a single broadcast
func (o *Orchestrator) measureAllStreamsInBroadcast(ctx context.Context, broadcast *latency.Broadcast) map[string]*latency.StreamMeasurement {
	results := make(map[string]*latency.StreamMeasurement)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create a start signal to synchronize all downloads
	startSignal := make(chan struct{})

	// Start all goroutines but have them wait for the signal
	for streamName, stream := range broadcast.Streams {
		if !stream.Enabled {
			o.logger.Info("Skipping disabled stream", logging.Fields{
				"stream_name": streamName,
				"url":         stream.URL,
			})
			continue
		}

		wg.Add(1)
		go func(name string, endpoint *latency.StreamEndpoint) {
			defer wg.Done()

			// Wait for the start signal before proceeding
			<-startSignal

			measurement := o.engine.MeasureStream(ctx, endpoint)

			mu.Lock()
			results[name] = measurement
			mu.Unlock()
		}(streamName, stream)
	}

	// Small delay to ensure all goroutines are ready and waiting
	time.Sleep(100 * time.Millisecond)

	o.logger.Info("Starting synchronized stream measurements", logging.Fields{
		"stream_count": len(results) + 1, // +1 for the streams being started
	})

	// Signal all streams to start simultaneously
	close(startSignal)

	wg.Wait()
	return results
}

// measureSourceToCDNAlignments creates alignments only between source and CDN streams
func (o *Orchestrator) measureSourceToCDNAlignments(ctx context.Context, streamMeasurements map[string]*latency.StreamMeasurement) map[string]*latency.AlignmentMeasurement {
	alignments := make(map[string]*latency.AlignmentMeasurement)

	// Separate streams by role
	var sourceStreams []string
	var cdnStreams []string

	for key, stream := range streamMeasurements {
		if stream.Error != nil {
			continue // Skip failed streams
		}

		switch stream.Endpoint.Role {
		case latency.StreamRoleSource:
			sourceStreams = append(sourceStreams, key)
		case latency.StreamRoleCDN:
			cdnStreams = append(cdnStreams, key)
		}
	}

	o.logger.Info("Found streams for alignment", logging.Fields{
		"source_streams":    sourceStreams,
		"cdn_streams":       cdnStreams,
		"total_comparisons": len(sourceStreams) * len(cdnStreams),
	})

	// Create alignments: each source vs each CDN
	for _, sourceKey := range sourceStreams {
		for _, cdnKey := range cdnStreams {
			alignmentName := fmt.Sprintf("%s_to_%s", sourceKey, cdnKey)

			o.logger.Info("Performing temporal alignment", logging.Fields{
				"alignment_name": alignmentName,
				"source_stream":  sourceKey,
				"cdn_stream":     cdnKey,
				"source_url":     streamMeasurements[sourceKey].Endpoint.URL,
				"cdn_url":        streamMeasurements[cdnKey].Endpoint.URL,
			})

			alignment := o.engine.MeasureAlignment(ctx, streamMeasurements[sourceKey], streamMeasurements[cdnKey])
			alignments[alignmentName] = alignment
		}
	}

	return alignments
}

// calculateCorrectLivenessMetrics calculates liveness using the hardcoded config. Deprecated for automated approach.
func (o *Orchestrator) calculateCorrectLivenessMetrics(alignmentMeasurements map[string]*latency.AlignmentMeasurement, streamMeasurements map[string]*latency.StreamMeasurement) *latency.LivenessMetrics {
	metrics := &latency.LivenessMetrics{}

	o.logger.Info("Calculating liveness metrics using direct alignment measurements", logging.Fields{
		"alignment_count": len(alignmentMeasurements),
		"stream_count":    len(streamMeasurements),
	})

	// Step 1: Extract CDN latencies directly from alignment measurements
	for alignmentName, alignment := range alignmentMeasurements {
		if alignment.Error != nil || !alignment.IsValidAlignment {
			o.logger.Info("Invalid alignment, skipping", logging.Fields{
				"alignment_name": alignmentName,
				"error":          alignment.Error,
				"is_valid":       alignment.IsValidAlignment,
			})
			continue
		}

		switch alignmentName {
		case "primary_source_to_hls_cloudfront_cdn":
			metrics.HLSCloudfrontCDNLag = alignment.LatencySeconds
			o.logger.Info("HLS Cloudfront latency from direct alignment", logging.Fields{
				"latency_seconds": metrics.HLSCloudfrontCDNLag,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "primary_source_to_icecast_cloudfront_cdn":
			metrics.ICEcastCloudfrontCDNLag = alignment.LatencySeconds
			o.logger.Info("ICEcast Cloudfront latency from direct alignment", logging.Fields{
				"latency_seconds": metrics.ICEcastCloudfrontCDNLag,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "primary_source_to_hls_ais_cdn":
			metrics.HLSAISCDNLag = alignment.LatencySeconds
			o.logger.Info("HLS Soundstack lag from direct alignment", logging.Fields{
				"latency_seconds": metrics.HLSAISCDNLag,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "primary_source_to_icecast_ais_cdn":
			metrics.ICEcastAISCDNLag = alignment.LatencySeconds
			o.logger.Info("ICEcast Soundstack lag from direct alignment", logging.Fields{
				"latency_seconds": metrics.HLSAISCDNLag,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "backup_source_to_hls_cloudfront_cdn":
			metrics.HLSCloudfrontCDNLagFromBackup = alignment.LatencySeconds
			o.logger.Info("HLS Cloudfront latency from direct alignment (via backup stream)", logging.Fields{
				"latency_seconds": metrics.HLSCloudfrontCDNLagFromBackup,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "backup_source_to_icecast_cloudfront_cdn":
			metrics.ICEcastCloudfrontCDNLagFromBackup = alignment.LatencySeconds
			o.logger.Info("ICEcast Cloudfront latency from direct alignment (via backup stream)", logging.Fields{
				"latency_seconds": metrics.ICEcastCloudfrontCDNLagFromBackup,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "backup_source_to_hls_ais_cdn":
			metrics.HLSAISCDNLagFromBackup = alignment.LatencySeconds
			o.logger.Info("HLS Soundstack lag from direct alignment (via backup stream)", logging.Fields{
				"latency_seconds": metrics.HLSAISCDNLagFromBackup,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})

		case "backup_source_to_icecast_ais_cdn":
			metrics.ICEcastAISCDNLagFromBackup = alignment.LatencySeconds
			o.logger.Info("ICEcast Soundstack lag from direct alignment (via backup stream)", logging.Fields{
				"latency_seconds": metrics.HLSAISCDNLagFromBackup,
				"confidence":      alignment.AlignmentResult.OffsetConfidence,
			})
		}
	}

	// Step 2: Calculate individual stream lags (for completeness)
	// These are derived by adding TTFB to the CDN latencies
	streamTTFBs := make(map[string]time.Duration)
	for _, stream := range streamMeasurements {
		if stream.Error != nil {
			continue
		}
		streamKey := fmt.Sprintf("%s_%s", stream.Endpoint.Type, stream.Endpoint.Role)
		streamTTFBs[streamKey] = stream.TimeToFirstByte
	}

	// Set source lags to their TTFB (they are the reference points)
	if ttfb, exists := streamTTFBs["primary_source_to_hls_cloudfront_cdn"]; exists {
		metrics.HLSCloudfrontCDNLag += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["primary_source_to_icecast_cloudfront_cdn"]; exists {
		metrics.ICEcastCloudfrontCDNLag += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["primary_source_to_hls_ais_cdn"]; exists {
		metrics.HLSAISCDNLag += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["primary_source_to_icecast_ais_cdn"]; exists {
		metrics.ICEcastAISCDNLag += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["backup_source_to_hls_cloudfront_cdn"]; exists {
		metrics.HLSCloudfrontCDNLagFromBackup += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["backup_source_to_icecast_cloudfront_cdn"]; exists {
		metrics.ICEcastCloudfrontCDNLagFromBackup += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["backup_source_to_hls_ais_cdn"]; exists {
		metrics.HLSAISCDNLagFromBackup += ttfb.Seconds()
	}
	if ttfb, exists := streamTTFBs["backup_source_to_icecast_ais_cdn"]; exists {
		metrics.ICEcastAISCDNLagFromBackup += ttfb.Seconds()
	}

	o.logger.Info("Liveness metrics completed using direct measurements", logging.Fields{
		"primary_source_to_hls_cloudfront_cdn":     metrics.HLSCloudfrontCDNLag,
		"primary_source_to_icecast_cloudfront_cdn": metrics.ICEcastCloudfrontCDNLag,
		"primary_source_to_hls_ais_cdn":            metrics.HLSAISCDNLag,
		"primary_source_to_icecast_ais_cdn":        metrics.ICEcastAISCDNLag,
		"backup_source_to_hls_cloudfront_cdn":      metrics.HLSCloudfrontCDNLagFromBackup,
		"backup_source_to_icecast_cloudfront_cdn":  metrics.ICEcastCloudfrontCDNLagFromBackup,
		"backup_source_to_hls_ais_cdn":             metrics.HLSAISCDNLagFromBackup,
		"backup_source_to_icecast_ais_cdn":         metrics.ICEcastAISCDNLagFromBackup,
	})

	return metrics
}

// measureFingerprintComparisons performs fingerprint comparisons using alignment data
func (o *Orchestrator) measureFingerprintComparisons(ctx context.Context, alignmentMeasurements map[string]*latency.AlignmentMeasurement) map[string]*latency.FingerprintComparison {
	comparisons := make(map[string]*latency.FingerprintComparison)

	// For each alignment, perform a fingerprint comparison
	for alignmentName, alignment := range alignmentMeasurements {
		if alignment.Error != nil || !alignment.IsValidAlignment {
			continue
		}

		comparisonName := fmt.Sprintf("%s_fingerprint", alignmentName)

		var alignmentFeatures = alignment.AlignmentResult
		if !alignment.IsValidAlignment {
			alignmentFeatures = nil // Don't use invalid alignment
		}

		comparison := o.engine.CompareFingerprintSimilarity(ctx, alignment.Stream1, alignment.Stream2, alignmentFeatures)
		comparisons[comparisonName] = comparison
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
	validation.OverallHealthScore = o.calculateHealthScore(validation)

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

// calculateLivenessMetrics calculates liveness metrics from alignment data
func (o *Orchestrator) calculateLivenessMetrics(alignmentMeasurements map[string]*latency.AlignmentMeasurement) *latency.LivenessMetrics {
	metrics := &latency.LivenessMetrics{}

	o.logger.Info("Calculating liveness metrics from alignments", logging.Fields{
		"alignment_count": len(alignmentMeasurements),
	})

	// Extract latencies from source-to-CDN alignments
	for alignmentName, alignment := range alignmentMeasurements {
		if alignment.Error != nil || !alignment.IsValidAlignment {
			continue
		}

		latency := alignment.LatencySeconds

		// Map based on stream names in the alignment
		// Pattern: {source_name}_to_{cdn_name}
		switch {
		case contains(alignmentName, "primary_source") && contains(alignmentName, "hls") && contains(alignmentName, "cloudfront"):
			metrics.HLSCloudfrontCDNLag = latency
		case contains(alignmentName, "primary_source") && contains(alignmentName, "hls") && contains(alignmentName, "ais"):
			metrics.HLSAISCDNLag = latency
		case contains(alignmentName, "primary_source") && contains(alignmentName, "icecast") && contains(alignmentName, "cloudfront"):
			metrics.ICEcastCloudfrontCDNLag = latency
		case contains(alignmentName, "primary_source") && contains(alignmentName, "icecast") && contains(alignmentName, "ais"):
			metrics.ICEcastAISCDNLag = latency
		case contains(alignmentName, "backup_source") && contains(alignmentName, "hls") && contains(alignmentName, "cloudfront"):
			metrics.HLSCloudfrontCDNLagFromBackup = latency
		case contains(alignmentName, "backup_source") && contains(alignmentName, "hls") && contains(alignmentName, "ais"):
			metrics.HLSAISCDNLagFromBackup = latency
		case contains(alignmentName, "backup_source") && contains(alignmentName, "icecast") && contains(alignmentName, "cloudfront"):
			metrics.ICEcastCloudfrontCDNLagFromBackup = latency
		case contains(alignmentName, "backup_source") && contains(alignmentName, "icecast") && contains(alignmentName, "ais"):
			metrics.ICEcastAISCDNLagFromBackup = latency
		}

		o.logger.Debug("Mapped alignment to liveness metric", logging.Fields{
			"alignment_name": alignmentName,
			"latency":        latency,
		})
	}

	return metrics
}

func contains(str, substr string) bool {
	return len(str) >= len(substr) && func() bool {
		for i := 0; i <= len(str)-len(substr); i++ {
			if str[i:i+len(substr)] == substr {
				return true
			}
		}
		return false
	}()
}

func (o *Orchestrator) calculateHealthScore(validation *latency.OverallValidation) float64 {
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

	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			failureCount++
			continue
		}

		successCount++
		totalHealthScore += broadcast.OverallValidation.OverallHealthScore

		// TODO: ensure that only valid measurements are used
		if broadcast.LivenessMetrics != nil {
			avgMetrics.AvgSourceLag = broadcast.LivenessMetrics.PrimarySourceLag

			backupHLS := 0.0
			backupICEcast := 0.0
			div := 2.0
			if broadcast.LivenessMetrics.BackupSourceLag != 0.0 {
				backupHLS = broadcast.LivenessMetrics.HLSCloudfrontCDNLagFromBackup + broadcast.LivenessMetrics.HLSAISCDNLagFromBackup
				backupICEcast = broadcast.LivenessMetrics.ICEcastCloudfrontCDNLagFromBackup + broadcast.LivenessMetrics.ICEcastAISCDNLagFromBackup
				div *= 2

				avgMetrics.AvgSourceLag = (avgMetrics.AvgSourceLag + broadcast.LivenessMetrics.BackupSourceLag) / 2.0
			}
			avgMetrics.AvgHLSCDNLag =
				(broadcast.LivenessMetrics.HLSCloudfrontCDNLag +
					broadcast.LivenessMetrics.HLSAISCDNLag +
					backupHLS) / div
			avgMetrics.AvgICEcastCDNLag =
				(broadcast.LivenessMetrics.ICEcastCloudfrontCDNLag +
					broadcast.LivenessMetrics.ICEcastAISCDNLag +
					backupICEcast) / div
			avgMetrics.AvgCDNLag = (avgMetrics.AvgHLSCDNLag + avgMetrics.AvgICEcastCDNLag/2.0)
		}

		// Aggregate stream performance metrics
		for _, stream := range broadcast.StreamMeasurements {
			if stream.Error == nil {
				avgMetrics.AvgTimeToFirstByte += float64(stream.TimeToFirstByte.Milliseconds())
				avgMetrics.AvgAudioExtractionTime += float64(stream.TotalProcessingTime.Milliseconds())
			}
		}
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
