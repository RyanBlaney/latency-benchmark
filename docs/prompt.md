I'm working on my TuneIn CDN Benchmark CLI project (10-week capstone). This is a Go microservice for automated CDN performance testing with HLS/ICEcast stream support, audio quality analysis, and DataDog integration.

Current status: [Brief 1-2 sentence update on what you're working on]

Key architecture decisions:

- Interface-first design for stream handlers (HLS/ICEcast)
- Cobra CLI with Viper configuration management
- Factory pattern for stream detection/creation
- Centralized config in configs/ package
- Audio analysis pipeline with FFT/mel-scale processing

I need help with: [Specific task/question]

### To Attach

- cmd/root.go - Root command with Viper/Cobra setup
- configs/config.go - Configuration structures
- configs/defaults.go - Default configuration values
- pkg/stream/common/interfaces.go - Core interfaces
- pkg/stream/factory.go - Stream factory implementation
- pkg/stream/detector.go - Stream type detection
