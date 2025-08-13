package benchmark

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/RyanBlaney/latency-benchmark-common/logging"
	"github.com/RyanBlaney/latency-benchmark/internal/latency"
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

	o.logger.Debug("Starting CDN benchmark", logging.Fields{
		"enabled_groups":    len(o.broadcastConfig.GetEnabledBroadcastGroups()),
		"segment_duration":  o.benchmarkConfig.Benchmark.AudioSegmentDuration.Seconds(),
		"operation_timeout": o.benchmarkConfig.Benchmark.OperationTimeout.Seconds(),
	})

	benchmarkCtx, cancel := context.WithTimeout(ctx, o.benchmarkConfig.Benchmark.BenchmarkTimeout)
	defer cancel()

	broadcast, broadcastKey, err := o.broadcastConfig.SelectBroadcastByIndex(0)
	if err != nil {
		return nil, fmt.Errorf("failed to select broadcast by index 0: %w", err)
	}

	o.logger.Debug("Selected broadcast for benchmarking", logging.Fields{
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

	return summary, nil
}

// RunBenchmarkByIndex processes a specific broadcast by index (for parallel execution)
func (o *Orchestrator) RunBenchmarkByIndex(ctx context.Context, index int) (*latency.BenchmarkSummary, error) {
	startTime := time.Now()

	o.logger.Debug("Starting CDN benchmark by index", logging.Fields{
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

	o.logger.Debug("Selected broadcast for benchmarking", logging.Fields{
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

	o.logger.Debug("CDN benchmark completed", logging.Fields{
		"index":                 index,
		"total_duration_s":      summary.TotalDuration.Seconds(),
		"successful_broadcasts": summary.SuccessfulBroadcasts,
		"failed_broadcasts":     summary.FailedBroadcasts,
		// "overall_health_score":  summary.OverallHealthScore,
	})

	return summary, nil
}

// measureSingleBroadcast measures a single broadcast (the 5-6 streams)
func (o *Orchestrator) measureSingleBroadcast(ctx context.Context, broadcastKey string, broadcast *latency.Broadcast) *latency.BroadcastMeasurement {
	measurement := &latency.BroadcastMeasurement{
		Broadcast:              broadcast,
		Group:                  o.broadcastConfig.GetBroadcastGroup(broadcastKey),
		StreamMeasurements:     make(map[string]*latency.StreamMeasurement),
		AlignmentMeasurements:  make(map[string]*latency.AlignmentMeasurement),
		FingerprintComparisons: make(map[string]*latency.FingerprintComparison),
		Timestamp:              time.Now(),
	}

	benchmarkStart := time.Now()

	o.logger.Debug("Starting broadcast measurement", logging.Fields{
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
	measurement.LivenessMetrics = o.calculateLivenessMetrics(measurement.AlignmentMeasurements, measurement.FingerprintComparisons)

	// Step 5: Perform overall validation
	// measurement.OverallValidation = o.validateOverallHealth(measurement)

	measurement.TotalBenchmarkTime = time.Since(benchmarkStart)

	o.logger.Debug("Broadcast measurement completed", logging.Fields{
		"broadcast_key":     broadcastKey,
		"total_time_s":      measurement.TotalBenchmarkTime.Seconds(),
		"valid_streams":     validStreams,
		"alignment_count":   len(measurement.AlignmentMeasurements),
		"fingerprint_count": len(measurement.FingerprintComparisons),
		// "overall_health_score": measurement.OverallValidation.OverallHealthScore,
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
		if stream.Enabled != nil {
			if !*stream.Enabled {
				o.logger.Debug("Skipping disabled stream", logging.Fields{
					"stream_name": streamName,
					"url":         stream.URL,
				})
				continue
			}
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

	o.logger.Debug("Starting synchronized stream measurements", logging.Fields{
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

	o.logger.Debug("Found streams for alignment", logging.Fields{
		"source_streams":    sourceStreams,
		"cdn_streams":       cdnStreams,
		"total_comparisons": len(sourceStreams) * len(cdnStreams),
	})

	// Create alignments: each source vs each CDN (backup stream comparisons not needed)
	for _, sourceKey := range sourceStreams {
		if contains(sourceKey, "pri") {
			for _, cdnKey := range cdnStreams {
				alignmentName := fmt.Sprintf("%s_vs_%s", sourceKey, cdnKey)

				o.logger.Debug("Performing temporal alignment", logging.Fields{
					"alignment_name": alignmentName,
					"source_stream":  sourceKey,
					"cdn_stream":     cdnKey,
					"source_url":     streamMeasurements[sourceKey].Endpoint.URL,
					"cdn_url":        streamMeasurements[cdnKey].Endpoint.URL,
				})

				alignment := o.engine.MeasureAlignment(ctx, streamMeasurements[sourceKey], streamMeasurements[cdnKey])
				alignments[alignmentName] = alignment
			}

			// Measure source to source latency
			for _, source2Key := range sourceStreams {
				if !contains(source2Key, "pri") {
					alignmentName := fmt.Sprintf("%s_vs_%s", sourceKey, source2Key)

					o.logger.Debug("Performing temporal alignment", logging.Fields{
						"alignment_name":    alignmentName,
						"pri_source_stream": sourceKey,
						"bk_source_stream":  source2Key,
						"pri_source_url":    streamMeasurements[sourceKey].Endpoint.URL,
						"bk_source_url":     streamMeasurements[source2Key].Endpoint.URL,
					})

					alignment := o.engine.MeasureAlignment(ctx, streamMeasurements[sourceKey], streamMeasurements[source2Key])
					alignments[alignmentName] = alignment
				}
			}
		}
	}

	return alignments
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
func (o *Orchestrator) calculateLivenessMetrics(
	alignments map[string]*latency.AlignmentMeasurement,
	fingerprints map[string]*latency.FingerprintComparison) *latency.LivenessMetrics {
	metrics := &latency.LivenessMetrics{}

	o.logger.Debug("Calculating liveness metrics from alignments", logging.Fields{
		"alignment_count": len(alignments),
	})

	// Extract latencies from source-to-CDN alignments
	for alignmentName, alignment := range alignments {
		if alignment.Error != nil || !alignment.IsValidAlignment {
			continue
		}

		fingerprint := fingerprints[fmt.Sprintf("%s_fingerprint", alignmentName)]
		if fingerprint == nil {
			continue
		}

		// Map based on stream names in the alignment
		// Pattern: {source_name}_to_{cdn_name}
		switch {
		case contains(alignmentName, "pri_source") &&
			contains(alignmentName, "hls") &&
			contains(alignmentName, "ti"):
			metrics.HLSCloudfrontCDNLag = o.calculateLiveness(alignment, fingerprint)

		case contains(alignmentName, "pri_source") &&
			contains(alignmentName, "hls") &&
			contains(alignmentName, "ss_ais"):
			metrics.HLSAISCDNLag = o.calculateLiveness(alignment, fingerprint)

		case contains(alignmentName, "pri_source") &&
			contains(alignmentName, "mp3") &&
			contains(alignmentName, "ti"):
			metrics.ICEcastCloudfrontCDNLag = o.calculateLiveness(alignment, fingerprint)

		case contains(alignmentName, "pri_source") &&
			contains(alignmentName, "mp3") &&
			contains(alignmentName, "ss_ais"):
			metrics.ICEcastAISCDNLag = o.calculateLiveness(alignment, fingerprint)

		case contains(alignmentName, "pri_source") &&
			contains(alignmentName, "bk_source"):
			metrics.SourceLag = o.calculateSourceLiveness(alignment, fingerprint)
		}
	}

	return metrics
}

func (o *Orchestrator) calculateLiveness(alignment *latency.AlignmentMeasurement, fingerprint *latency.FingerprintComparison) *latency.Liveness {
	var status string
	lagSeconds := roundToDecimalPlaces(alignment.LatencySeconds, o.benchmarkConfig.Output.Precision)

	if lagSeconds < 0 || !alignment.IsValidAlignment {
		status = "NO_ALIGNMENT"
	}

	if fingerprint.SimilarityResult.OverallSimilarity < o.benchmarkConfig.Benchmark.MinFingerprintSimilarity {
		status = "NO_MATCH"
		fmt.Printf("OVERALL SIMILARITY TOO LOW: %f\n", alignment.AlignmentResult.OverallSimilarity)
	}

	if status != "NO_ALIGNMENT" && status != "NO_MATCH" {
		status = "MATCH"
	}

	return &latency.Liveness{
		LagSeconds: lagSeconds,
		Status:     status,
	}
}

func (o *Orchestrator) calculateSourceLiveness(alignment *latency.AlignmentMeasurement, fingerprint *latency.FingerprintComparison) *latency.Liveness {
	var primaryLiveness *latency.Liveness
	lagSeconds := roundToDecimalPlaces(alignment.LatencySeconds, o.benchmarkConfig.Output.Precision)

	if !alignment.IsValidAlignment {
		primaryLiveness = &latency.Liveness{
			LagSeconds: 0,
			Status:     "NO_ALIGNMENT",
		}
	} else {
		primaryLiveness = &latency.Liveness{
			LagSeconds: lagSeconds,
		}
	}

	if fingerprint.SimilarityResult.OverallSimilarity < o.benchmarkConfig.Benchmark.MinFingerprintSimilarity {
		fmt.Printf("OVERALL SIMILARITY TOO LOW: %f\n", alignment.AlignmentResult.OverallSimilarity)
		primaryLiveness.Status = "NO_MATCH"
	}

	if primaryLiveness.Status != "NO_ALIGNMENT" && primaryLiveness.Status != "NO_MATCH" {
		primaryLiveness.Status = "MATCH"
	}
	return primaryLiveness
}

func roundToDecimalPlaces(f float64, decimals int) float64 {
	multiplier := math.Pow10(decimals)
	return math.Round(f*multiplier) / multiplier
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

func (o *Orchestrator) calculateSummaryMetrics(summary *latency.BenchmarkSummary) {
	successCount := 0
	failureCount := 0

	for _, broadcast := range summary.BroadcastMeasurements {
		if broadcast.Error != nil {
			failureCount++
			continue
		}

		successCount++
	}

	streamCount := 0
	for _, broadcast := range summary.BroadcastMeasurements {
		streamCount += len(broadcast.StreamMeasurements)
	}
}
