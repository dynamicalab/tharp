#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ios>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>


#define USE_CACHE

typedef Spectra::SparseGenRealShiftSolve<double> MatOp;
struct Edgelist {
    std::vector<int> rows;
    std::vector<int> cols;
};

using CacheKey = std::pair<double, double>; // Cache key type for memoization (only kappa_i and kappa_j)

const double SQRT_PI = 1.7724538509055160272981674833411;
const double PI = 3.1415926535897932384626433832795;

const double ZERO_THRESHOLD = 1e-12;
const double SOLVER_SIGMA = 1e-9;

// Pre-computed Gauss-Legendre nodes and weights (50 points)
const int NB_POINTS = 50;

// Global pre-computed theta values and weights transformed to [0, π]
std::vector<double> THETA;
std::vector<double> WEIGHTS;

// Initialize global quadrature data
void initialize_quadrature() {
    THETA.resize(NB_POINTS);
    WEIGHTS.resize(NB_POINTS);
    const int n = NB_POINTS;
    for (int i = 0; i < (n + 1) / 2; i++) {
        double x = std::cos(PI * (i + 0.75) / (n + 0.5)), pp = 0;
        for (int it = 0; it < 100; it++) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; j++) {
                double p3 = p2; p2 = p1;
                p1 = ((2.0 * j + 1.0) * x * p2 - j * p3) / (j + 1);
            }
            pp = n * (x * p1 - p2) / (x * x - 1.0);
            double dx = p1 / pp;
            x -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        const double w = 2.0 / ((1.0 - x * x) * pp * pp);
        THETA[i]       = PI / 2.0 * (1.0 - x);   // maps -x to [0, pi]
        THETA[n-1-i]   = PI / 2.0 * (1.0 + x);
        WEIGHTS[i] = WEIGHTS[n-1-i] = w * PI / 2.0;
    }
}

// LRU cache using std::map (simplified version)
std::map<CacheKey, std::pair<double, double>> integral_cache;
const size_t MAX_CACHE_SIZE = 50000;
size_t NB_CACHE_SUCCESSFUL_LOOKUPS = 0;

// Global parameters (set once, used for all computations)
double GLOBAL_MU = 0.0;
double GLOBAL_R = 0.0;
double GLOBAL_BETA = 0.0;
int GLOBAL_D = 0;

// Set global parameters
void set_global_parameters(double mu, double R, double beta, int D) {
    // Clear cache when parameters change
    if (GLOBAL_MU != mu || GLOBAL_R != R || GLOBAL_BETA != beta || GLOBAL_D != D) {
        integral_cache.clear();
    }
    GLOBAL_MU = mu;
    GLOBAL_R = R;
    GLOBAL_BETA = beta;
    GLOBAL_D = D;
}

// Core integration function (equivalent to JIT-compiled Python version)
inline std::pair<double, double> compute_integrals_core(
    double kappa_i, double kappa_j, double mu, double R, double beta, int D,
    const std::vector<double>& theta, const std::vector<double>& weights) {

    double I_den = 0.0;
    double I_num = 0.0;

    // Pre-compute constant terms
    double kappa_prod = std::pow(mu * kappa_i * kappa_j, 1.0 / D);

    for (int i = 0; i < NB_POINTS; i++) {
        double t = theta[i];
        double w = weights[i];

        // Compute integrand
        double term = std::pow(R * t / kappa_prod, beta);
        double sin_term = std::pow(std::sin(t), D - 1);
        double integrand_val = sin_term / (1.0 + term);

        I_den += w * integrand_val;
        I_num += w * t * integrand_val;
    }

    return {I_num, I_den};
}

// Main function with caching (only caches based on kappa_i and kappa_j)
std::pair<double, double> compute_integrals(double kappa_i, double kappa_j) {

    #ifdef USE_CACHE
    //// Create cache key (only kappa values)
    CacheKey key = std::make_pair(kappa_i, kappa_j);

    // Check cache
    auto it = integral_cache.find(key);
    if (it != integral_cache.end()) {
        NB_CACHE_SUCCESSFUL_LOOKUPS++;
        return it->second;
    }
    #endif

    // Compute if not in cache (use global parameters)
    auto result = compute_integrals_core(
        kappa_i, kappa_j, GLOBAL_MU, GLOBAL_R, GLOBAL_BETA, GLOBAL_D, THETA, WEIGHTS);

    #ifdef USE_CACHE
    // Add to cache (simple size limit)
    if (integral_cache.size() >= MAX_CACHE_SIZE) {
        // Simple eviction: clear oldest entries (first 10%)
        auto erase_count = MAX_CACHE_SIZE / 10;
        auto erase_it = integral_cache.begin();
        for (size_t i = 0; i < erase_count && erase_it != integral_cache.end(); i++) {
            erase_it = integral_cache.erase(erase_it);
        }
    }

    integral_cache[key] = result;
    #endif
    return result;
}

// Compute expected angular distance
double expected_angular_distance(double kappa_i, double kappa_j) {

    auto [I_num, I_den] = compute_integrals(kappa_i, kappa_j);

    if (I_den == 0.0) {
        std::cerr << "Warning: denominator is zero\n";
        return 0.0;
    }

    return I_num / I_den;
}

// Clear cache (useful for testing or memory management)
void clear_cache() {
    integral_cache.clear();
}



// Custom implmentation of the Gamma function
double gamma_half_integer(int n) {

    // Case 1: n is even, so n/2 is an integer
    // Γ(k) = (k-1)! where k = n/2
    if (n % 2 == 0) {
        int k = n / 2;
        double result = 1.0;
        for (int i = 1; i < k; ++i) {
            result *= i;
        }
        return result;
    }
    // Case 2: n is odd, so n/2 is a half-integer
    // Γ((2m+1)/2) = (2m-1)!! / 2^m * sqrt(π)
    // where (2m-1)!! = 1 * 3 * 5 * ... * (2m-1)
    else {
        int m = (n - 1) / 2;
        double result = SQRT_PI;
        for (int i = 0; i < m; ++i) {
            result *= (2 * i + 1) / 2.0;
        }
        return result;
    }
}



// Reads edgelist from file
std::pair<int, Edgelist> load_edgelist(const std::string& filename) {
    Edgelist edgelist;
    edgelist.rows.reserve(100000);
    edgelist.cols.reserve(100000);

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int maxVertex = -1;
    std::string line;

    // First pass: read all edges and find matrix dimension
    while (std::getline(file, line)) {
        // Skip empty lines and comments (lines starting with #)
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int vertex1, vertex2;

        if (!(iss >> vertex1 >> vertex2)) {
            std::cerr << "Error: Invalid line format: " << line << std::endl;
            continue;
        }

        edgelist.rows.push_back(vertex1);
        edgelist.cols.push_back(vertex2);

        // Track maximum vertex ID
        maxVertex = std::max(maxVertex, std::max(vertex1, vertex2));
    }

    file.close();


    if (edgelist.rows.empty()) {
        std::cerr << "Error: No valid edges found in file" << std::endl;
        std::exit(-1);
    }

    // Matrix dimension is maxVertex + 1 (assuming 0-indexed)
    int nb_vertices = maxVertex + 1;

    return {nb_vertices, edgelist};
}



// Computes the degrees
std::vector<int> compute_degrees(const Edgelist &edgelist, int nb_vertices) {
    std::vector<int> degrees(nb_vertices, 0);
    for(int i = 0, ii(edgelist.rows.size()); i<ii; i++) {
        degrees[edgelist.rows[i]] += 1;
        degrees[edgelist.cols[i]] += 1;
    }
    return degrees;
}


bool fill_the_matrix(Eigen::SparseMatrix<double> &A,
                     const Edgelist &edgelist,
                     const std::vector<int> &degree) {
    initialize_quadrature();

    const int nb_vertices = A.innerSize();
    const int nb_edges = edgelist.rows.size();

    // First, chord length from the expected angular distance of each edge.
    double M2(0), average(0);
    for (int i = 0; i < nb_edges; i++) {
        const int v1 = edgelist.rows[i];
        const int v2 = edgelist.cols[i];
        int k1 = degree[v1], k2 = degree[v2];
        if (k1 > k2) std::swap(k1, k2);

        const double value = 2. * std::sin(expected_angular_distance(k1, k2) / 2);
        A.insert(v1, v2) = value;
        A.insert(v2, v1) = value;

        // Welford update (same as the Python np.var over the 2M entries,
        // since each edge appears twice with the same value)
        const double delta = value - average;
        average += delta / (i + 1);
        M2 += delta * (value - average);
    }
    double variance = M2 / nb_edges;
    if (variance < 1e-300) variance = 1.0;   // guard against zero variance

    // Second, store the kernel exponent e = d^2 / var and track the per-row minimum.
    std::vector<double> row_min(nb_vertices, std::numeric_limits<double>::infinity());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            const double e = it.value() * it.value() / variance;
            it.valueRef() = e;
            row_min[it.row()] = std::min(row_min[it.row()], e);
        }
    }

    // Third, shifted exponential. The shift cancels in the normalization,
    // but guarantees every row sum is >= 1.
    std::vector<double> row_sum(nb_vertices, 0.0);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            const double w = std::exp(-(it.value() - row_min[it.row()]));
            it.valueRef() = w;
            row_sum[it.row()] += w;
        }
    }

    // Fourth, normalize by ROW and negate: off-diagonals of I - P.
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            it.valueRef() = -it.value() / row_sum[it.row()];
        }
    }

    // Fifth, unit diagonal.
    for (int i = 0; i < nb_vertices; i++) A.insert(i, i) = 1.;

    A.makeCompressed();
    return true;
}



// Finds the smallest eigenvalues and their eigenvectors.
//
typedef Spectra::SparseGenRealShiftSolve<double> MatOp;

Spectra::GenEigsRealShiftSolver<MatOp> compute_smallest_eigenvalues(Eigen::SparseMatrix<double> &A,
                                                                                       const int nb_evalues,
                                                                                       const int D) {
    const int nb_vertices = A.innerSize();

    MatOp op(A);

    const int ncv = std::min(3*nb_evalues, nb_vertices-2);  // Must be < n
    Spectra::GenEigsRealShiftSolver<MatOp> eigs(op, nb_evalues, ncv, SOLVER_SIGMA);
    eigs.init();

    const int nconv = eigs.compute(Spectra::SortRule::LargestMagn, 5000, ZERO_THRESHOLD);
    if (eigs.info() == Spectra::CompInfo::Successful) {
        return eigs;
    }
    else {
        std::cerr << "Eigenvalue computation failed!" << std::endl;
        std::cerr << "Error code: " << static_cast<int>(eigs.info()) << std::endl;
        std::cerr << "Error code meanings:" << std::endl;
        std::cerr << "  0 = Successful" << std::endl;
        std::cerr << "  1 = NotComputed" << std::endl;
        std::cerr << "  2 = NotConverging" << std::endl;
        std::cerr << "  3 = NumericalIssue" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}



// Extracts the coordinates.
std::vector<std::vector<double>> compute_position_on_the_hypersphere(const Spectra::GenEigsRealShiftSolver<MatOp> &eigs,
                                        const int nb_evalues,
                                        const int D) {
    Eigen::VectorXcd evalues = eigs.eigenvalues();
    Eigen::MatrixXd evectors = eigs.eigenvectors().real();

    const int nb_vertices = evectors.innerSize();
    std::vector<std::vector<double>> coordinates(nb_vertices, std::vector<double>(D+1));

    // Create pairs of (eigenvalue, index) for sorting
    std::vector<std::pair<double, int>> eval_pairs;

    for (int i = 0; i < nb_evalues; ++i)
        if (std::abs(evalues(i)) > ZERO_THRESHOLD)
            eval_pairs.push_back({std::abs(evalues(i)), i});
    if (eval_pairs.size() < D+1) {
        std::cerr << "Not enough eigenvalues computed to embed the graph." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Sort by eigenvalue (ascending order)
    std::sort(eval_pairs.begin(), eval_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int i = 0; i < nb_vertices; ++i) {
        for (int j = 0; j < D+1; ++j) {
            int idx = eval_pairs[j].second;
            coordinates[i][j] = evectors(i, idx);
        }
    }

    double norm;
    for (int i = 0; i < nb_vertices; ++i) {
        norm = 0;
        for (int j(0), jj(D+1); j < jj; ++j) {
            norm += coordinates[i][j] * coordinates[i][j];
        }
        norm = std::sqrt(norm);
        for (int j(0), jj(D+1); j < jj; ++j) {
            coordinates[i][j] /= norm;
        }
    }

    return coordinates;
}



// Rescales the positions radially.
bool rescale_positions(std::vector<std::vector<double>> &coordinates,
                           const std::vector<int> degree,
                           const int D,
                           const double ball_radius,
                           const double kappa0) {

    const int nb_vertices = coordinates.size();

    const double r_max = 4.0 / static_cast<double>(D) * std::log(static_cast<double>(nb_vertices));
    for(int i = 0; i < nb_vertices; ++i) {
        double r_i;
        if (degree[i] > 0) {
            r_i = ball_radius - (2. / D) * std::log(static_cast<double>(degree[i]) / kappa0);
        }
        else {
            r_i = ball_radius;
        }
        r_i = std::max(0., r_i);
        // Radial coordinates are clipped to avoid large coordinates
        r_i = std::min(r_i, r_max);

        double scale = std::sinh(r_i);

        for(int j(0), jj(D+1); j<jj; ++j)
            coordinates[i][j] *= scale;
    }
    return true;
}



int main(int argc, char** argv) {
    if (argc == 1) {
        fprintf(stderr, "Error: no file provided.");
        return -1;
    } else if (argc == 2) {
        fprintf(stderr, "Error: no dimension provided.");
        return -1;
    } else if (argc == 3) {
        fprintf(stderr, "Error: no number of connected components provided.");
        return -1;
    }
    const std::string edgelist_fname = argv[1];
    const int D = std::stoi(argv[2])-1;
    const int num_connected_components = std::stoi(argv[3]);

    // Loads the edgelist.
    const auto [nb_vertices, edgelist] = load_edgelist(edgelist_fname);
    const int nb_edges = edgelist.rows.size();

    // Computes the degrees.
    std::vector<int> degree = compute_degrees(edgelist, nb_vertices);

    // Sets the parameters.
    const int nb_evalues = std::min(num_connected_components+D+100, nb_vertices-4);
    const double beta = D + 1.5;
    const double mu = (beta * gamma_half_integer(D) * std::sin(D * PI / beta) * nb_vertices) / (4. * std::pow(PI, 1. + D/2.) * nb_edges);
    const double R_SD = std::pow( (nb_vertices * gamma_half_integer(D + 1)) / (2. * std::pow(PI, (D + 1.) / 2.)), 1. / D );
    set_global_parameters(mu, R_SD, beta, D);

    // Fills the matrix.
    Eigen::SparseMatrix<double> A(nb_vertices, nb_vertices);
    A.reserve(Eigen::VectorXi::Constant(nb_vertices, nb_evalues));
    fill_the_matrix(A, edgelist, degree);

    // Finds the smallest eigenvalues.
    auto eigs = compute_smallest_eigenvalues(A, nb_evalues, D);

    // Finds the positions on the unit hypersphere.
    auto coordinates = compute_position_on_the_hypersphere(eigs, nb_evalues, D);

    // Rescale the positions radially.
    double kappa0 = *std::min_element(degree.begin(), degree.end());
    kappa0 = std::max(kappa0, 1.);

    double ball_radius = 2. * std::log( (2. * R_SD) / std::pow(mu * kappa0 * kappa0, 1. / D ));
    ball_radius = std::max(1., ball_radius);
    rescale_positions(coordinates, degree, D, ball_radius, kappa0);

    int pos = edgelist_fname.length();
    int i=0;
    for (auto c: edgelist_fname) {
        if (c=='.') {
            pos = i;
        }
        i++;
    }
    const std::string embedding_fname = edgelist_fname.substr(0, pos) + "-le" + std::to_string(D+1) + "D.txt";
    std::ofstream embedding_file(embedding_fname);
    if (!embedding_file.is_open()) {
        std::cerr << "Cannot open file: " << embedding_fname;
        return -1;
    }

    embedding_file << "# " << beta << " " << ball_radius << "\n";
    for (int i=0; i<nb_vertices; ++i) {
        for (int j=0; j<D+1; ++j) {
            embedding_file << std::fixed << coordinates[i][j] << "  ";
        }
        embedding_file << std::endl;
    }
    embedding_file.close();


    return 0;
}


// Compilation instructions:
// clang++ -std=c++23 -O3 -Ilib/json/single_include -Ilib/eigen -Ilib/spectra/include laplacian-eigenmap.cpp -o laplacian-eigenmap
