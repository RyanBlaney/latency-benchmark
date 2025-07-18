package stats

import (
	"fmt"
	"math"
	"math/rand"
)

// ClusteringAlgorithm represents different clustering methods
type ClusteringAlgorithm int

const (
	KMeans ClusteringAlgorithm = iota
	KMedoids
	HierarchicalAgglomerative
	DBSCAN
	GaussianMixture
)

// DistanceMetric represents different distance/similarity measures
type DistanceMetric int

const (
	EuclideanDistance DistanceMetric = iota
	ManhattanDistance
	CosineDistance
	PearsonDistance
	MahalanobisDistance
)

// LinkageCriterion for hierarchical clustering
type LinkageCriterion int

const (
	SingleLinkage LinkageCriterion = iota
	CompleteLinkage
	AverageLinkage
	WardLinkage
)

// Cluster represents a cluster of data points
type Cluster struct {
	ID       int         `json:"id"`
	Center   []float64   `json:"center"`   // Cluster centroid
	Points   [][]float64 `json:"points"`   // Points in this cluster
	Indices  []int       `json:"indices"`  // Original indices of points
	Size     int         `json:"size"`     // Number of points in cluster
	Variance float64     `json:"variance"` // Within-cluster variance
	Radius   float64     `json:"radius"`   // Maximum distance from center
}

// ClusteringResult contains the results of clustering analysis
type ClusteringResult struct {
	Clusters           []Cluster   `json:"clusters"`
	Labels             []int       `json:"labels"`            // Cluster assignment for each point
	Centers            [][]float64 `json:"centers"`           // Cluster centers
	Inertia            float64     `json:"inertia"`           // Total within-cluster sum of squares
	SilhouetteScore    float64     `json:"silhouette_score"`  // Overall silhouette coefficient
	DaviesBouldinIndex float64     `json:"davies_bouldin"`    // Davies-Bouldin index
	CalinskiHarabasz   float64     `json:"calinski_harabasz"` // Calinski-Harabasz index
	NumClusters        int         `json:"num_clusters"`
	Converged          bool        `json:"converged"`
	Iterations         int         `json:"iterations"`
}

// ClusteringParams contains parameters for clustering algorithms
type ClusteringParams struct {
	NumClusters   int                 `json:"num_clusters"`
	MaxIterations int                 `json:"max_iterations"`
	Tolerance     float64             `json:"tolerance"`
	Algorithm     ClusteringAlgorithm `json:"algorithm"`
	Distance      DistanceMetric      `json:"distance"`
	Linkage       LinkageCriterion    `json:"linkage"`

	// DBSCAN specific parameters
	Epsilon   float64 `json:"epsilon"`    // Maximum distance between points
	MinPoints int     `json:"min_points"` // Minimum points to form cluster

	// Initialization parameters
	InitMethod string `json:"init_method"` // "random", "kmeans++", "manual"
	RandomSeed int64  `json:"random_seed"`

	// Convergence parameters
	RelativeTolerance bool `json:"relative_tolerance"`
}

// Clustering implements various clustering algorithms for audio feature analysis
//
// References:
//   - Jain, A. K., & Dubes, R. C. (1988). "Algorithms for clustering data"
//   - MacQueen, J. (1967). "Some methods for classification and analysis of
//     multivariate observations"
//   - Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of
//     careful seeding"
//   - Ester, M., et al. (1996). "A density-based algorithm for discovering
//     clusters in large spatial databases with noise"
//   - Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the
//     interpretation and validation of cluster analysis"
type Clustering struct {
	params ClusteringParams
	rng    *rand.Rand
}

// NewClustering creates a new clustering analyzer with default parameters
func NewClustering() *Clustering {
	return &Clustering{
		params: ClusteringParams{
			NumClusters:       3,
			MaxIterations:     100,
			Tolerance:         1e-4,
			Algorithm:         KMeans,
			Distance:          EuclideanDistance,
			Linkage:           WardLinkage,
			Epsilon:           0.5,
			MinPoints:         5,
			InitMethod:        "kmeans++",
			RandomSeed:        42,
			RelativeTolerance: true,
		},
		rng: rand.New(rand.NewSource(42)),
	}
}

// NewClusteringWithParams creates a clustering analyzer with custom parameters
func NewClusteringWithParams(params ClusteringParams) *Clustering {
	return &Clustering{
		params: params,
		rng:    rand.New(rand.NewSource(params.RandomSeed)),
	}
}

// Fit performs clustering on the input data
func (c *Clustering) Fit(data [][]float64) (*ClusteringResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}

	switch c.params.Algorithm {
	case KMeans:
		return c.kmeans(data)
	case KMedoids:
		return c.kmedoids(data)
	case HierarchicalAgglomerative:
		return c.hierarchical(data)
	case DBSCAN:
		return c.dbscan(data)
	case GaussianMixture:
		return c.gaussianMixture(data)
	default:
		return nil, fmt.Errorf("unsupported clustering algorithm")
	}
}

// kmeans implements the K-means clustering algorithm
// Algorithm: Lloyd's algorithm with k-means++ initialization
func (c *Clustering) kmeans(data [][]float64) (*ClusteringResult, error) {
	n := len(data)
	k := c.params.NumClusters
	dim := len(data[0])

	if k > n {
		return nil, fmt.Errorf("number of clusters (%d) cannot exceed number of data points (%d)", k, n)
	}

	// Initialize centers using k-means++ or random initialization
	centers := c.initializeCenters(data, k)
	labels := make([]int, n)
	prevLabels := make([]int, n)

	converged := false
	iterations := 0

	for iterations < c.params.MaxIterations && !converged {
		// Assignment step: assign each point to closest center
		totalMovement := 0.0
		for i, point := range data {
			minDist := math.Inf(1)
			bestCluster := 0

			for j, center := range centers {
				dist := c.distance(point, center)
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}

			if labels[i] != bestCluster {
				totalMovement += 1.0
			}
			labels[i] = bestCluster
		}

		// Update step: recalculate centers
		newCenters := make([][]float64, k)
		clusterSizes := make([]int, k)

		for i := range newCenters {
			newCenters[i] = make([]float64, dim)
		}

		for i, point := range data {
			cluster := labels[i]
			clusterSizes[cluster]++
			for j := range point {
				newCenters[cluster][j] += point[j]
			}
		}

		// Average to get centroids
		centerMovement := 0.0
		for i := range newCenters {
			if clusterSizes[i] > 0 {
				for j := range newCenters[i] {
					newCenters[i][j] /= float64(clusterSizes[i])
				}
				// Calculate center movement
				centerMovement += c.distance(centers[i], newCenters[i])
			}
		}

		centers = newCenters

		// Check convergence
		if c.params.RelativeTolerance {
			converged = (totalMovement / float64(n)) < c.params.Tolerance
		} else {
			converged = centerMovement < c.params.Tolerance
		}

		copy(prevLabels, labels)
		iterations++
	}

	// Build result
	result := &ClusteringResult{
		Centers:     centers,
		Labels:      labels,
		NumClusters: k,
		Converged:   converged,
		Iterations:  iterations,
	}

	// Build clusters and calculate metrics
	result.Clusters = c.buildClusters(data, labels, centers)
	result.Inertia = c.calculateInertia(data, labels, centers)
	result.SilhouetteScore = c.calculateSilhouetteScore(data, labels)
	result.DaviesBouldinIndex = c.calculateDaviesBouldinIndex(data, labels, centers)
	result.CalinskiHarabasz = c.calculateCalinskiHarabasz(data, labels, centers)

	return result, nil
}

// initializeCenters initializes cluster centers using k-means++ algorithm
// Reference: Arthur, D., & Vassilvitskii, S. (2007)
func (c *Clustering) initializeCenters(data [][]float64, k int) [][]float64 {
	n := len(data)
	dim := len(data[0])
	centers := make([][]float64, k)

	if c.params.InitMethod == "random" {
		// Random initialization
		for i := range k {
			centers[i] = make([]float64, dim)
			copy(centers[i], data[c.rng.Intn(n)])
		}
		return centers
	}

	// k-means++ initialization
	// Choose first center randomly
	centers[0] = make([]float64, dim)
	copy(centers[0], data[c.rng.Intn(n)])

	// Choose remaining centers
	for i := 1; i < k; i++ {
		distances := make([]float64, n)
		totalDist := 0.0

		// Calculate squared distances to nearest center
		for j, point := range data {
			minDist := math.Inf(1)
			for l := range i {
				dist := c.distance(point, centers[l])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[j] = minDist * minDist
			totalDist += distances[j]
		}

		// Choose next center with probability proportional to squared distance
		if totalDist > 0 {
			r := c.rng.Float64() * totalDist
			cumSum := 0.0
			for j, dist := range distances {
				cumSum += dist
				if cumSum >= r {
					centers[i] = make([]float64, dim)
					copy(centers[i], data[j])
					break
				}
			}
		} else {
			// Fallback to random selection
			centers[i] = make([]float64, dim)
			copy(centers[i], data[c.rng.Intn(n)])
		}
	}

	return centers
}

// distance calculates distance between two points based on selected metric
func (c *Clustering) distance(a, b []float64) float64 {
	switch c.params.Distance {
	case EuclideanDistance:
		return euclideanDistance(a, b)
	case ManhattanDistance:
		return manhattanDistance(a, b)
	case CosineDistance:
		return cosineDistance(a, b)
	case PearsonDistance:
		return pearsonDistance(a, b)
	default:
		return euclideanDistance(a, b)
	}
}

// euclideanDistance calculates Euclidean distance between two points
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// manhattanDistance calculates Manhattan (L1) distance between two points
func manhattanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

// cosineDistance calculates cosine distance (1 - cosine similarity)
func cosineDistance(a, b []float64) float64 {
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - similarity
}

// pearsonDistance calculates Pearson correlation distance (1 - |correlation|)
func pearsonDistance(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 1.0
	}

	// Calculate means
	meanA := 0.0
	meanB := 0.0
	for i := range a {
		meanA += a[i]
		meanB += b[i]
	}
	meanA /= float64(n)
	meanB /= float64(n)

	// Calculate correlation coefficient
	numerator := 0.0
	sumSqA := 0.0
	sumSqB := 0.0

	for i := range a {
		diffA := a[i] - meanA
		diffB := b[i] - meanB
		numerator += diffA * diffB
		sumSqA += diffA * diffA
		sumSqB += diffB * diffB
	}

	if sumSqA == 0 || sumSqB == 0 {
		return 1.0
	}

	correlation := numerator / math.Sqrt(sumSqA*sumSqB)
	return 1.0 - math.Abs(correlation)
}

// buildClusters constructs cluster objects from labels and centers
func (c *Clustering) buildClusters(data [][]float64, labels []int, centers [][]float64) []Cluster {
	k := len(centers)
	clusters := make([]Cluster, k)

	for i := range clusters {
		clusters[i].ID = i
		clusters[i].Center = make([]float64, len(centers[i]))
		copy(clusters[i].Center, centers[i])
		clusters[i].Points = [][]float64{}
		clusters[i].Indices = []int{}
	}

	// Assign points to clusters
	for i, point := range data {
		cluster := labels[i]
		clusters[cluster].Points = append(clusters[cluster].Points, point)
		clusters[cluster].Indices = append(clusters[cluster].Indices, i)
		clusters[cluster].Size++
	}

	// Calculate cluster statistics
	for i := range clusters {
		if clusters[i].Size > 0 {
			clusters[i].Variance = c.calculateClusterVariance(clusters[i].Points, clusters[i].Center)
			clusters[i].Radius = c.calculateClusterRadius(clusters[i].Points, clusters[i].Center)
		}
	}

	return clusters
}

// calculateClusterVariance computes within-cluster variance
func (c *Clustering) calculateClusterVariance(points [][]float64, center []float64) float64 {
	if len(points) == 0 {
		return 0.0
	}

	variance := 0.0
	for _, point := range points {
		dist := c.distance(point, center)
		variance += dist * dist
	}

	return variance / float64(len(points))
}

// calculateClusterRadius computes maximum distance from center
func (c *Clustering) calculateClusterRadius(points [][]float64, center []float64) float64 {
	maxDist := 0.0
	for _, point := range points {
		dist := c.distance(point, center)
		if dist > maxDist {
			maxDist = dist
		}
	}
	return maxDist
}

// calculateInertia computes total within-cluster sum of squares
func (c *Clustering) calculateInertia(data [][]float64, labels []int, centers [][]float64) float64 {
	inertia := 0.0
	for i, point := range data {
		cluster := labels[i]
		dist := c.distance(point, centers[cluster])
		inertia += dist * dist
	}
	return inertia
}

// calculateSilhouetteScore computes the average silhouette coefficient
// Reference: Rousseeuw, P. J. (1987)
func (c *Clustering) calculateSilhouetteScore(data [][]float64, labels []int) float64 {
	n := len(data)
	if n < 2 {
		return 0.0
	}

	silhouettes := make([]float64, n)

	for i := range n {
		a := c.calculateIntraClusterDistance(data, labels, i)   // Average distance within cluster
		b := c.calculateNearestClusterDistance(data, labels, i) // Average distance to nearest cluster

		if a < b {
			silhouettes[i] = (b - a) / b
		} else if a > b {
			silhouettes[i] = (b - a) / a
		} else {
			silhouettes[i] = 0.0
		}
	}

	// Calculate average silhouette
	sum := 0.0
	for _, s := range silhouettes {
		sum += s
	}

	return sum / float64(n)
}

// calculateIntraClusterDistance computes average distance within the same cluster
func (c *Clustering) calculateIntraClusterDistance(data [][]float64, labels []int, pointIdx int) float64 {
	cluster := labels[pointIdx]
	sum := 0.0
	count := 0

	for i, point := range data {
		if i != pointIdx && labels[i] == cluster {
			sum += c.distance(data[pointIdx], point)
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return sum / float64(count)
}

// calculateNearestClusterDistance computes average distance to nearest cluster
func (c *Clustering) calculateNearestClusterDistance(data [][]float64, labels []int, pointIdx int) float64 {
	currentCluster := labels[pointIdx]

	// Find all other clusters
	clusterDistances := make(map[int][]float64)
	for i, point := range data {
		if labels[i] != currentCluster {
			cluster := labels[i]
			if _, exists := clusterDistances[cluster]; !exists {
				clusterDistances[cluster] = []float64{}
			}
			clusterDistances[cluster] = append(clusterDistances[cluster], c.distance(data[pointIdx], point))
		}
	}

	// Find minimum average distance to any other cluster
	minAvgDist := math.Inf(1)
	for _, distances := range clusterDistances {
		sum := 0.0
		for _, dist := range distances {
			sum += dist
		}
		avgDist := sum / float64(len(distances))
		if avgDist < minAvgDist {
			minAvgDist = avgDist
		}
	}

	return minAvgDist
}

// calculateDaviesBouldinIndex computes Davies-Bouldin index (lower is better)
func (c *Clustering) calculateDaviesBouldinIndex(data [][]float64, labels []int, centers [][]float64) float64 {
	k := len(centers)
	if k < 2 {
		return 0.0
	}

	// Calculate within-cluster scatter for each cluster
	scatters := make([]float64, k)
	for i := range k {
		count := 0
		sum := 0.0
		for j, point := range data {
			if labels[j] == i {
				sum += c.distance(point, centers[i])
				count++
			}
		}
		if count > 0 {
			scatters[i] = sum / float64(count)
		}
	}

	// Calculate Davies-Bouldin index
	db := 0.0
	for i := range k {
		maxRatio := 0.0
		for j := range k {
			if i != j {
				centerDist := c.distance(centers[i], centers[j])
				if centerDist > 0 {
					ratio := (scatters[i] + scatters[j]) / centerDist
					if ratio > maxRatio {
						maxRatio = ratio
					}
				}
			}
		}
		db += maxRatio
	}

	return db / float64(k)
}

// calculateCalinskiHarabasz computes Calinski-Harabasz index (higher is better)
func (c *Clustering) calculateCalinskiHarabasz(data [][]float64, labels []int, centers [][]float64) float64 {
	n := len(data)
	k := len(centers)

	if k < 2 || n == k {
		return 0.0
	}

	// Calculate overall centroid
	dim := len(data[0])
	overallCenter := make([]float64, dim)
	for _, point := range data {
		for j := range point {
			overallCenter[j] += point[j]
		}
	}
	for j := range overallCenter {
		overallCenter[j] /= float64(n)
	}

	// Calculate between-cluster sum of squares
	bgss := 0.0
	for i, center := range centers {
		count := 0
		for _, label := range labels {
			if label == i {
				count++
			}
		}
		if count > 0 {
			dist := c.distance(center, overallCenter)
			bgss += float64(count) * dist * dist
		}
	}

	// Calculate within-cluster sum of squares
	wgss := 0.0
	for i, point := range data {
		cluster := labels[i]
		dist := c.distance(point, centers[cluster])
		wgss += dist * dist
	}

	if wgss == 0 {
		return 0.0
	}

	return (bgss / float64(k-1)) / (wgss / float64(n-k))
}

// kmedoids implements the k-medoids clustering algorithm (PAM)
// Reference: Kaufman, L., & Rousseeuw, P. J. (1990). "Finding Groups in Data"
func (c *Clustering) kmedoids(data [][]float64) (*ClusteringResult, error) {
	n := len(data)
	k := c.params.NumClusters

	if k > n {
		return nil, fmt.Errorf("number of clusters (%d) cannot exceed number of data points (%d)", k, n)
	}

	// Initialize medoids randomly
	medoidIndices := make([]int, k)
	used := make(map[int]bool)
	for i := range k {
		for {
			idx := c.rng.Intn(n)
			if !used[idx] {
				medoidIndices[i] = idx
				used[idx] = true
				break
			}
		}
	}

	labels := make([]int, n)
	prevLabels := make([]int, n)

	converged := false
	iterations := 0

	for iterations < c.params.MaxIterations && !converged {
		// Assignment step: assign each point to closest medoid
		for i := range n {
			minDist := math.Inf(1)
			bestCluster := 0

			for j, medoidIdx := range medoidIndices {
				dist := c.distance(data[i], data[medoidIdx])
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}
			labels[i] = bestCluster
		}

		// Update step: find new medoids
		totalCost := 0.0
		for clusterIdx := range k {
			// Find all points in this cluster
			clusterPoints := []int{}
			for i := range n {
				if labels[i] == clusterIdx {
					clusterPoints = append(clusterPoints, i)
				}
			}

			if len(clusterPoints) == 0 {
				continue
			}

			// Find the point that minimizes total distance to all other points in cluster
			bestMedoid := medoidIndices[clusterIdx]
			bestCost := math.Inf(1)

			for _, candidateIdx := range clusterPoints {
				cost := 0.0
				for _, pointIdx := range clusterPoints {
					cost += c.distance(data[candidateIdx], data[pointIdx])
				}

				if cost < bestCost {
					bestCost = cost
					bestMedoid = candidateIdx
				}
			}

			medoidIndices[clusterIdx] = bestMedoid
			totalCost += bestCost
		}

		// Check convergence
		changed := false
		for i := range n {
			if labels[i] != prevLabels[i] {
				changed = true
				break
			}
		}

		converged = !changed
		copy(prevLabels, labels)
		iterations++
	}

	// Build centers from medoids
	centers := make([][]float64, k)
	for i, medoidIdx := range medoidIndices {
		centers[i] = make([]float64, len(data[medoidIdx]))
		copy(centers[i], data[medoidIdx])
	}

	// Build result
	result := &ClusteringResult{
		Centers:     centers,
		Labels:      labels,
		NumClusters: k,
		Converged:   converged,
		Iterations:  iterations,
	}

	result.Clusters = c.buildClusters(data, labels, centers)
	result.Inertia = c.calculateInertia(data, labels, centers)
	result.SilhouetteScore = c.calculateSilhouetteScore(data, labels)
	result.DaviesBouldinIndex = c.calculateDaviesBouldinIndex(data, labels, centers)
	result.CalinskiHarabasz = c.calculateCalinskiHarabasz(data, labels, centers)

	return result, nil
}

// hierarchical implements agglomerative hierarchical clustering
// Reference: Hastie, T., et al. (2009). "The Elements of Statistical Learning"
func (c *Clustering) hierarchical(data [][]float64) (*ClusteringResult, error) {
	n := len(data)
	k := c.params.NumClusters

	if k > n {
		return nil, fmt.Errorf("number of clusters (%d) cannot exceed number of data points (%d)", k, n)
	}

	// Initialize each point as its own cluster
	clusters := make([][]int, n)
	for i := range n {
		clusters[i] = []int{i}
	}

	// Compute initial distance matrix
	distMatrix := make([][]float64, n)
	for i := range n {
		distMatrix[i] = make([]float64, n)
		for j := range n {
			if i != j {
				distMatrix[i][j] = c.distance(data[i], data[j])
			}
		}
	}

	// Merge clusters until we have k clusters
	for len(clusters) > k {
		// Find the two closest clusters
		minDist := math.Inf(1)
		mergeI, mergeJ := -1, -1

		for i := 0; i < len(clusters); i++ {
			for j := i + 1; j < len(clusters); j++ {
				dist := c.calculateClusterDistance(clusters[i], clusters[j], distMatrix)
				if dist < minDist {
					minDist = dist
					mergeI, mergeJ = i, j
				}
			}
		}

		// Merge the two closest clusters
		if mergeI != -1 && mergeJ != -1 {
			// Merge cluster j into cluster i
			clusters[mergeI] = append(clusters[mergeI], clusters[mergeJ]...)

			// Remove cluster j
			clusters = append(clusters[:mergeJ], clusters[mergeJ+1:]...)
		}
	}

	// Build labels and centers
	labels := make([]int, n)
	centers := make([][]float64, len(clusters))

	for clusterIdx, cluster := range clusters {
		// Calculate centroid
		dim := len(data[0])
		center := make([]float64, dim)

		for _, pointIdx := range cluster {
			labels[pointIdx] = clusterIdx
			for j := range dim {
				center[j] += data[pointIdx][j]
			}
		}

		// Average to get centroid
		for j := range dim {
			center[j] /= float64(len(cluster))
		}
		centers[clusterIdx] = center
	}

	// Build result
	result := &ClusteringResult{
		Centers:     centers,
		Labels:      labels,
		NumClusters: len(clusters),
		Converged:   true,
		Iterations:  n - k,
	}

	result.Clusters = c.buildClusters(data, labels, centers)
	result.Inertia = c.calculateInertia(data, labels, centers)
	result.SilhouetteScore = c.calculateSilhouetteScore(data, labels)
	result.DaviesBouldinIndex = c.calculateDaviesBouldinIndex(data, labels, centers)
	result.CalinskiHarabasz = c.calculateCalinskiHarabasz(data, labels, centers)

	return result, nil
}

// calculateClusterDistance computes distance between two clusters based on linkage criterion
func (c *Clustering) calculateClusterDistance(cluster1, cluster2 []int, distMatrix [][]float64) float64 {
	switch c.params.Linkage {
	case SingleLinkage:
		// Minimum distance between any two points
		minDist := math.Inf(1)
		for _, i := range cluster1 {
			for _, j := range cluster2 {
				if distMatrix[i][j] < minDist {
					minDist = distMatrix[i][j]
				}
			}
		}
		return minDist

	case CompleteLinkage:
		// Maximum distance between any two points
		maxDist := 0.0
		for _, i := range cluster1 {
			for _, j := range cluster2 {
				if distMatrix[i][j] > maxDist {
					maxDist = distMatrix[i][j]
				}
			}
		}
		return maxDist

	case AverageLinkage:
		// Average distance between all pairs
		sumDist := 0.0
		count := 0
		for _, i := range cluster1 {
			for _, j := range cluster2 {
				sumDist += distMatrix[i][j]
				count++
			}
		}
		return sumDist / float64(count)

	case WardLinkage:
		// Ward's minimum variance method
		// Simplified implementation - use average linkage as approximation
		sumDist := 0.0
		count := 0
		for _, i := range cluster1 {
			for _, j := range cluster2 {
				sumDist += distMatrix[i][j] * distMatrix[i][j]
				count++
			}
		}
		return math.Sqrt(sumDist / float64(count))

	default:
		return c.calculateClusterDistance(cluster1, cluster2, distMatrix)
	}
}

// dbscan implements DBSCAN density-based clustering
// Reference: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
func (c *Clustering) dbscan(data [][]float64) (*ClusteringResult, error) {
	n := len(data)
	labels := make([]int, n)
	visited := make([]bool, n)

	// Initialize all points as noise (-1)
	for i := range n {
		labels[i] = -1
	}

	clusterID := 0

	for i := range n {
		if visited[i] {
			continue
		}

		visited[i] = true

		// Find all neighbors within epsilon
		neighbors := c.findNeighbors(data, i, c.params.Epsilon)

		// If not enough neighbors, mark as noise
		if len(neighbors) < c.params.MinPoints {
			labels[i] = -1 // Noise
			continue
		}

		// Start a new cluster
		labels[i] = clusterID

		// Expand cluster
		seedSet := make([]int, len(neighbors))
		copy(seedSet, neighbors)

		for j := 0; j < len(seedSet); j++ {
			q := seedSet[j]

			if !visited[q] {
				visited[q] = true

				// Find neighbors of q
				qNeighbors := c.findNeighbors(data, q, c.params.Epsilon)

				// If q has enough neighbors, add them to seed set
				if len(qNeighbors) >= c.params.MinPoints {
					seedSet = append(seedSet, qNeighbors...)
				}
			}

			// If q is not yet member of any cluster, add it to current cluster
			if labels[q] == -1 {
				labels[q] = clusterID
			}
		}

		clusterID++
	}

	// Count actual clusters (excluding noise)
	maxCluster := -1
	for _, label := range labels {
		if label > maxCluster {
			maxCluster = label
		}
	}

	numClusters := maxCluster + 1
	if numClusters <= 0 {
		return nil, fmt.Errorf("no clusters found with current parameters")
	}

	// Build centers by calculating centroids
	centers := make([][]float64, numClusters)
	clusterSizes := make([]int, numClusters)
	dim := len(data[0])

	for i := range numClusters {
		centers[i] = make([]float64, dim)
	}

	for i, point := range data {
		if labels[i] >= 0 { // Skip noise points
			cluster := labels[i]
			clusterSizes[cluster]++
			for j := range point {
				centers[cluster][j] += point[j]
			}
		}
	}

	// Average to get centroids
	for i := range numClusters {
		if clusterSizes[i] > 0 {
			for j := range dim {
				centers[i][j] /= float64(clusterSizes[i])
			}
		}
	}

	// Build result
	result := &ClusteringResult{
		Centers:     centers,
		Labels:      labels,
		NumClusters: numClusters,
		Converged:   true,
		Iterations:  1,
	}

	result.Clusters = c.buildClusters(data, labels, centers)
	result.Inertia = c.calculateInertia(data, labels, centers)
	result.SilhouetteScore = c.calculateSilhouetteScore(data, labels)
	result.DaviesBouldinIndex = c.calculateDaviesBouldinIndex(data, labels, centers)
	result.CalinskiHarabasz = c.calculateCalinskiHarabasz(data, labels, centers)

	return result, nil
}

// findNeighbors finds all points within epsilon distance of the given point
func (c *Clustering) findNeighbors(data [][]float64, pointIdx int, epsilon float64) []int {
	neighbors := []int{}

	for i, point := range data {
		if i != pointIdx {
			dist := c.distance(data[pointIdx], point)
			if dist <= epsilon {
				neighbors = append(neighbors, i)
			}
		}
	}

	return neighbors
}

// gaussianMixture implements Gaussian Mixture Model clustering using EM algorithm
// Reference: Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
func (c *Clustering) gaussianMixture(data [][]float64) (*ClusteringResult, error) {
	n := len(data)
	k := c.params.NumClusters
	dim := len(data[0])

	if k > n {
		return nil, fmt.Errorf("number of clusters (%d) cannot exceed number of data points (%d)", k, n)
	}

	// Initialize parameters
	means := c.initializeCenters(data, k)

	// Initialize covariances as identity matrices
	covariances := make([][][]float64, k)
	for i := range k {
		covariances[i] = make([][]float64, dim)
		for j := range dim {
			covariances[i][j] = make([]float64, dim)
			covariances[i][j][j] = 1.0 // Identity matrix
		}
	}

	// Initialize mixing coefficients
	mixingCoeffs := make([]float64, k)
	for i := range k {
		mixingCoeffs[i] = 1.0 / float64(k)
	}

	// Responsibilities matrix
	responsibilities := make([][]float64, n)
	for i := range n {
		responsibilities[i] = make([]float64, k)
	}

	// EM algorithm
	prevLogLikelihood := math.Inf(-1)
	iterations := 0

	for iterations < c.params.MaxIterations {
		// E-step: calculate responsibilities
		logLikelihood := 0.0

		for i := range n {
			sum := 0.0
			for j := range k {
				responsibilities[i][j] = mixingCoeffs[j] * c.gaussianPDF(data[i], means[j], covariances[j])
				sum += responsibilities[i][j]
			}

			// Normalize responsibilities
			if sum > 0 {
				for j := range k {
					responsibilities[i][j] /= sum
				}
				logLikelihood += math.Log(sum)
			}
		}

		// Check convergence
		if math.Abs(logLikelihood-prevLogLikelihood) < c.params.Tolerance {
			break
		}
		prevLogLikelihood = logLikelihood

		// M-step: update parameters
		for j := range k {
			// Calculate effective number of points assigned to cluster j
			nj := 0.0
			for i := range n {
				nj += responsibilities[i][j]
			}

			if nj > 0 {
				// Update mean
				for d := range dim {
					means[j][d] = 0.0
					for i := range n {
						means[j][d] += responsibilities[i][j] * data[i][d]
					}
					means[j][d] /= nj
				}

				// Update covariance (simplified: diagonal covariance)
				for d := range dim {
					covariances[j][d][d] = 0.0
					for i := range n {
						diff := data[i][d] - means[j][d]
						covariances[j][d][d] += responsibilities[i][j] * diff * diff
					}
					covariances[j][d][d] /= nj

					// Add small regularization to prevent singularity
					covariances[j][d][d] += 1e-6
				}

				// Update mixing coefficient
				mixingCoeffs[j] = nj / float64(n)
			}
		}

		iterations++
	}

	// Assign points to clusters based on maximum responsibility
	labels := make([]int, n)
	for i := range n {
		maxResp := 0.0
		bestCluster := 0
		for j := range k {
			if responsibilities[i][j] > maxResp {
				maxResp = responsibilities[i][j]
				bestCluster = j
			}
		}
		labels[i] = bestCluster
	}

	// Build result
	result := &ClusteringResult{
		Centers:     means,
		Labels:      labels,
		NumClusters: k,
		Converged:   iterations < c.params.MaxIterations,
		Iterations:  iterations,
	}

	result.Clusters = c.buildClusters(data, labels, means)
	result.Inertia = c.calculateInertia(data, labels, means)
	result.SilhouetteScore = c.calculateSilhouetteScore(data, labels)
	result.DaviesBouldinIndex = c.calculateDaviesBouldinIndex(data, labels, means)
	result.CalinskiHarabasz = c.calculateCalinskiHarabasz(data, labels, means)

	return result, nil
}

// gaussianPDF computes the probability density function of a multivariate Gaussian
func (c *Clustering) gaussianPDF(x, mean []float64, covariance [][]float64) float64 {
	dim := len(x)

	// Calculate (x - mean)
	diff := make([]float64, dim)
	for i := range dim {
		diff[i] = x[i] - mean[i]
	}

	// Calculate determinant of covariance matrix (simplified for diagonal)
	det := 1.0
	for i := range dim {
		det *= covariance[i][i]
	}

	// Calculate (x - mean)^T * Σ^(-1) * (x - mean) (simplified for diagonal)
	quadratic := 0.0
	for i := range dim {
		if covariance[i][i] > 0 {
			quadratic += diff[i] * diff[i] / covariance[i][i]
		}
	}

	// Calculate PDF
	normalization := 1.0 / math.Sqrt(math.Pow(2*math.Pi, float64(dim))*det)
	return normalization * math.Exp(-0.5*quadratic)
}
