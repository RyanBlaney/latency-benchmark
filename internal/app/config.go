package app

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/internal/latency"
	"gopkg.in/yaml.v3"
)

type BenchmarkConfig = latency.BenchmarkConfig
type BroadcastConfig = latency.BroadcastConfig
type BenchmarkSettings = latency.BenchmarkSettings

// loadBenchmarkConfigFromFile loads benchmark configuration from a file (app settings only)
func loadBenchmarkConfigFromFile(filePath string) (*BenchmarkConfig, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("configuration file does not exist: %s", filePath)
	}

	// Determine file format
	ext := filepath.Ext(filePath)
	switch ext {
	case ".yaml", ".yml":
		return loadBenchmarkConfigFromYAML(filePath)
	case ".json":
		return loadBenchmarkConfigFromJSON(filePath)
	default:
		// Try YAML first, then JSON
		if cfg, err := loadBenchmarkConfigFromYAML(filePath); err == nil {
			return cfg, nil
		}
		return loadBenchmarkConfigFromJSON(filePath)
	}
}

// loadBroadcastConfigFromFile loads broadcast groups configuration from a separate file
func loadBroadcastConfigFromFile(filePath string) (*BroadcastConfig, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("broadcast configuration file does not exist: %s", filePath)
	}

	// Determine file format
	ext := filepath.Ext(filePath)
	switch ext {
	case ".yaml", ".yml":
		return loadBroadcastConfigFromYAML(filePath)
	case ".json":
		return loadBroadcastConfigFromJSON(filePath)
	default:
		// Try YAML first, then JSON
		if cfg, err := loadBroadcastConfigFromYAML(filePath); err == nil {
			return cfg, nil
		}
		return loadBroadcastConfigFromJSON(filePath)
	}
}

// loadBroadcastConfigFromYAML loads broadcast config from YAML file
func loadBroadcastConfigFromYAML(filePath string) (*BroadcastConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open YAML broadcast config file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read YAML broadcast config file: %w", err)
	}

	var config BroadcastConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse YAML broadcast config: %w", err)
	}

	config.ApplyInheritance()

	return &config, nil
}

// loadBroadcastConfigFromJSON loads broadcast config from JSON file
func loadBroadcastConfigFromJSON(filePath string) (*BroadcastConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open JSON broadcast config file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON broadcast config file: %w", err)
	}

	var config BroadcastConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse JSON broadcast config: %w", err)
	}

	config.ApplyInheritance()

	return &config, nil
}

// loadBenchmarkConfigFromYAML loads from YAML file
func loadBenchmarkConfigFromYAML(filePath string) (*BenchmarkConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open YAML config file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read YAML config file: %w", err)
	}

	var config BenchmarkConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	return &config, nil
}

// loadBenchmarkConfigFromJSON loads from JSON file
func loadBenchmarkConfigFromJSON(filePath string) (*BenchmarkConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open JSON config file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON config file: %w", err)
	}

	var config BenchmarkConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse JSON config: %w", err)
	}

	return &config, nil
}

// mergeBenchmarkConfig merges base config, benchmark config, and CLI flags
func mergeBenchmarkConfig(baseConfig *configs.Config, benchmarkConfig *BenchmarkConfig, ctx *Context) *BenchmarkConfig {
	// Start with benchmark config if it exists, otherwise create new one
	if benchmarkConfig == nil {
		benchmarkConfig = &BenchmarkConfig{
			Config: baseConfig,
		}
	} else {
		// Use base config as foundation
		benchmarkConfig.Config = baseConfig
	}

	// Apply defaults
	applyBenchmarkDefaults(&benchmarkConfig.Benchmark)

	// Override with CLI flags
	if ctx.Timeout > 0 {
		benchmarkConfig.Benchmark.BenchmarkTimeout = ctx.Timeout
	}
	if ctx.SegmentDuration > 0 {
		benchmarkConfig.Benchmark.AudioSegmentDuration = ctx.SegmentDuration
	}
	if ctx.MaxConcurrent > 0 {
		benchmarkConfig.Benchmark.MaxConcurrentBroadcasts = ctx.MaxConcurrent
	}
	if ctx.OutputFormat != "" {
		benchmarkConfig.Benchmark.OutputFormat = ctx.OutputFormat
	}

	benchmarkConfig.Benchmark.EnableDetailedAnalysis = ctx.DetailedAnalysis
	benchmarkConfig.Benchmark.SkipFingerprintComparison = ctx.SkipFingerprint
	benchmarkConfig.Benchmark.IncludeFeatureBreakdown = ctx.DetailedAnalysis
	benchmarkConfig.Benchmark.IncludeRawData = ctx.DetailedAnalysis

	return benchmarkConfig
}

// applyBenchmarkDefaults sets default values for benchmark settings
func applyBenchmarkDefaults(settings *BenchmarkSettings) {
	if settings.BenchmarkTimeout == 0 {
		settings.BenchmarkTimeout = 30 * time.Minute
	}
	if settings.OperationTimeout == 0 {
		settings.OperationTimeout = 300 * time.Second
	}
	if settings.MaxConcurrentBroadcasts == 0 {
		settings.MaxConcurrentBroadcasts = 3
	}
	if settings.MaxConcurrentStreams == 0 {
		settings.MaxConcurrentStreams = 4
	}
	if settings.AudioSegmentDuration == 0 {
		settings.AudioSegmentDuration = 90 * time.Second
	}
	if settings.MinAlignmentConfidence == 0 {
		settings.MinAlignmentConfidence = 0.4
	}
	if settings.MaxAlignmentOffset == 0 {
		settings.MaxAlignmentOffset = 15.0
	}
	if settings.MinFingerprintSimilarity == 0 {
		settings.MinFingerprintSimilarity = 0.6
	}
	if settings.MaxRetries == 0 {
		settings.MaxRetries = 2
	}
	if settings.RetryDelay == 0 {
		settings.RetryDelay = 5 * time.Second
	}
	if settings.MinHealthScore == 0 {
		settings.MinHealthScore = 0.7
	}
	if settings.OutputFormat == "" {
		settings.OutputFormat = "json"
	}

	// Default validation settings
	if !settings.ValidateHLSStructure && !settings.ValidateICEcastMetadata && !settings.ValidateAudioFormat && !settings.ValidateBitrateConsistency {
		settings.ValidateHLSStructure = true
		settings.ValidateICEcastMetadata = true
		settings.ValidateAudioFormat = true
		settings.ValidateBitrateConsistency = true
	}

	// Default output settings
	settings.PrettyPrint = true
	settings.GenerateSummary = true
}

// GenerateExampleConfig generates an example benchmark configuration file
func GenerateExampleConfig(outputFile string) error {
	exampleConfig := &BenchmarkConfig{
		Config: configs.GetDefaultConfig(),
		Benchmark: BenchmarkSettings{
			BenchmarkTimeout:           30 * time.Minute,
			OperationTimeout:           120 * time.Second,
			MaxConcurrentBroadcasts:    3,
			MaxConcurrentStreams:       4,
			AudioSegmentDuration:       90 * time.Second,
			MinAlignmentConfidence:     0.4,
			MaxAlignmentOffset:         15.0,
			MinFingerprintSimilarity:   0.6,
			EnableDetailedAnalysis:     false,
			SkipFingerprintComparison:  false,
			MaxRetries:                 2,
			RetryDelay:                 5 * time.Second,
			ValidateHLSStructure:       true,
			ValidateICEcastMetadata:    true,
			ValidateAudioFormat:        true,
			ValidateBitrateConsistency: true,
			MinHealthScore:             0.7,
			FailFast:                   false,
			OutputFormat:               "json",
			IncludeRawData:             false,
			IncludeFeatureBreakdown:    false,
			PrettyPrint:                true,
			GenerateSummary:            true,
		},
	}

	// Write to YAML file
	data, err := yaml.Marshal(exampleConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal example config: %w", err)
	}

	// Ensure directory exists
	dir := filepath.Dir(outputFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	if err := os.WriteFile(outputFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	fmt.Printf("✅ Example application configuration written to: %s\n", outputFile)
	return nil
}

// GenerateExampleBroadcastConfig generates an example broadcast configuration file
func GenerateExampleBroadcastConfig(outputFile string) error {
	exampleConfig := &BroadcastConfig{
		Version:     "1.0",
		Environment: "example",
		UpdatedAt:   time.Now(),
		Description: "Example broadcast groups configuration",
		BroadcastGroups: map[string]*latency.BroadcastGroup{
			"news": {
				Name:        "News",
				Description: "Live News Broadcasts",
				ContentType: "news",
				Broadcasts: map[string]*latency.Broadcast{
					"msnbc_news": {
						Name:        "MSNBC News",
						Description: "MSNBC Live News Broadcast",
						Streams: map[string]*latency.StreamEndpoint{
							"hls_source": {
								URL:         "https://source.example.com/msnbc/live.m3u8",
								Type:        latency.StreamTypeHLS,
								Role:        latency.StreamRoleSource,
								ContentType: "news",
							},
							"hls_cdn": {
								URL:         "https://cdn.example.com/msnbc/live.m3u8",
								Type:        latency.StreamTypeHLS,
								Role:        latency.StreamRoleCDN,
								ContentType: "news",
							},
							"icecast_source": {
								URL:         "https://source.example.com/msnbc/live.mp3",
								Type:        latency.StreamTypeICEcast,
								Role:        latency.StreamRoleSource,
								ContentType: "news",
							},
							"icecast_cdn": {
								URL:         "https://cdn.example.com/msnbc/live.mp3",
								Type:        latency.StreamTypeICEcast,
								Role:        latency.StreamRoleCDN,
								ContentType: "news",
							},
						},
					},
				},
			},
		},
	}

	// Write to YAML file
	data, err := yaml.Marshal(exampleConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal example broadcast config: %w", err)
	}

	// Ensure directory exists
	dir := filepath.Dir(outputFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	if err := os.WriteFile(outputFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write broadcast config file: %w", err)
	}

	fmt.Printf("✅ Example broadcast configuration written to: %s\n", outputFile)
	return nil
}

// ValidateConfig validates a configuration file
func ValidateConfig(configFile string) error {
	config, err := loadBenchmarkConfigFromFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Use default base config for validation
	baseConfig := configs.GetDefaultConfig()
	mergedConfig := mergeBenchmarkConfig(baseConfig, config, &Context{})

	if err := mergedConfig.Validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	fmt.Printf("✅ Application configuration is valid: %s\n", configFile)
	fmt.Printf("   - Audio segment duration: %.1fs\n", config.Benchmark.AudioSegmentDuration.Seconds())
	fmt.Printf("   - Max concurrent broadcasts: %d\n", config.Benchmark.MaxConcurrentBroadcasts)

	return nil
}

// ValidateBroadcastConfig validates a broadcast configuration file
func ValidateBroadcastConfig(configFile string) error {
	config, err := loadBroadcastConfigFromFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to load broadcast config: %w", err)
	}

	if err := config.Validate(); err != nil {
		return fmt.Errorf("broadcast configuration validation failed: %w", err)
	}

	fmt.Printf("✅ Broadcast configuration is valid: %s\n", configFile)
	fmt.Printf("   - %d broadcast groups found\n", len(config.GetEnabledBroadcastGroups()))
	fmt.Printf("   - Environment: %s\n", config.Environment)
	fmt.Printf("   - Version: %s\n", config.Version)

	return nil
}
