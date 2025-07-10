package fingerprint

import "time"

type similarityMethod string

const (
	methodWeighted  similarityMethod = "weighted"
	methodEuclidean similarityMethod = "euclidean"
	methodCosine    similarityMethod = "cosine"
	methodPearson   similarityMethod = "pearson"
	methodAdaptive  similarityMethod = "adaptive"
)

// SimilarityResult holds the result of fingerprint comparison
type SimilarityResult struct {
	OverallSimilarity     float64                   `json:"overall_similarity"` // 0.0-1.0
	HashSimilarity        float64                   `json:"hash_similarity"`    // Hash-based similarity
	FeatureSimilarity     float64                   `json:"feature_similarity"` // Feature-based similarity
	ContentTypeMatch      bool                      `json:"content_type_match"`
	PerceptualHashMatches map[string]float64        `json:"perceptual_hash_matches"` // Per hash type similarity
	FeatureDistances      map[string]float64        `json:"feature_distances"`       // Per feature distances
	QualityMetrics        *ComparisonQualityMetrics `json:"qualtiy_metrics"`         // Comparison quality
	ProcessingTime        time.Duration             `json:"processing_time"`
	Confidence            float64                   `json:"confidence"`
	Metadata              map[string]any            `json:"metadata"`
}

// ComparisonQualityMetrics holds quality metrics for the comparison
type ComparisonQualityMetrics struct {
	DataAvailability float64 `json:"data_availability"`
}
