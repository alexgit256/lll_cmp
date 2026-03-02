/*
Compile as:
g++ -O3 -std=c++17 test_prec.cpp \
  -I"$CONDA_PREFIX/include" \
  -L"$CONDA_PREFIX/lib" \
  -Wl,-rpath,"$CONDA_PREFIX/lib" \
  -lfplll -lgmp -lmpfr \
  -o lll_demo

OR (for debug):
g++ -O0 -g -std=c++17 test_prec.cpp \
  -I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib" \
  -Wl,-rpath,"$CONDA_PREFIX/lib" \
  -lfplll -lgmp -lmpfr -o lll_demo_dbg

gdb --args ./lll_demo_dbg
run
... wait a bit and ctrl-C ...
bt
*/
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

#include <cassert>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>

#include <fplll/fplll.h>
#include <fplll/hlll.h>         // HLLLReduction
#include <fplll/householder.h>  // MatHouseholder

//For thread-safe IO
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <iomanip>

// multiprocessing
#include <thread>
#include <atomic>
#include <algorithm>

// -------------------- Timing helper --------------------
struct Timer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double stop_s() const {
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  }
};

// -------------------- One-place configuration --------------------
struct Config {
  // float mode: "d", "dd", or "mpfr"
  enum class FloatMode { D, DD, MPFR };

  FloatMode mode = FloatMode::D;
  // FloatMode mode = FloatMode::DD;
  // Only used if mode==MPFR: precision in *bits*
  int mpfr_prec_bits = 53;

  // Reduction parameters
  double delta = 0.99;
  double eta   = 0.51;

  // Standard LLL flags (e.g., LLL_DEFAULT, LLL_SIEGEL, ...)
  int lll_flags = fplll::LLL_DEFAULT;

  // HLLL flags (uses same flag space in fplll)
  int hlll_flags = fplll::LLL_DEFAULT;

  // HLLL params (defaults from fplll)
  double theta = fplll::HLLL_DEF_THETA;
  double c     = fplll::HLLL_DEF_C;

  // If you want to force the same “style” of standard LLL:
  // LM_FAST / LM_HEURISTIC / LM_PROVED / LM_WRAPPER exist for wrapper calls,
  // but here we're directly using LLLReduction<...>.
};

// -------------------- Profile utilities --------------------
template <class FT>
static inline double ft_to_double(const FT &x) {
  // get_d() exists for FP_NR<...>
  return x.get_d();
}

template <class ZT, class FT>
std::vector<double> profile_from_gso(fplll::MatGSO<ZT, FT> &gso, int n) {
  std::vector<double> prof;
  prof.reserve(n);
  for (int i = 0; i < n; i++) {
    FT rii;
    gso.get_r(rii, i, i);
    double v = ft_to_double(rii);
    prof.push_back(0.5 * std::log(v));
  }
  return prof;
}

template <class ZT, class FT>
std::vector<double> profile_from_householder(fplll::MatHouseholder<ZT, FT> &hh, int n) {
  std::vector<double> prof;
  prof.reserve(n);
  for (int i = 0; i < n; i++) {
    FT rii;
    long expo = 0;
    hh.get_R(rii, i, i, expo); // rii * 2^expo (if row-expo enabled; otherwise expo=0)
    double mant = std::abs(ft_to_double(rii));
    // log(|rii|*2^expo) = log(|rii|) + expo*log(2)
    double logabs = std::log(mant) + double(expo) * std::log(2.0);
    prof.push_back(logabs);
  }
  return prof;
}

static void print_profile(const std::string &label, const std::vector<double> &p) {
  std::cout << label << ":\n";
  for (size_t i = 0; i < p.size(); i++) {
    std::cout << p[i] << (i + 1 == p.size() ? "\n" : " ");
  }
}

static std::vector<double> dummy_profile(int n) {
  // Python-readable and easy to spot in analysis
  return std::vector<double>(n, std::numeric_limits<double>::quiet_NaN());
}

namespace logging {

static std::mutex g_log_mutex;

static std::string mpz_to_dec_string(const mpz_t z) {
  // mpz_get_str allocates with malloc; must free().
  char *s = mpz_get_str(nullptr, 10, z);
  if (!s) return "0";
  std::string out(s);
  std::free(s);
  return out;
}

static std::string float_mode_to_string(const Config &cfg) {
  switch (cfg.mode) {
    case Config::FloatMode::D:    return "d";
    case Config::FloatMode::DD:   return "dd";
    case Config::FloatMode::MPFR: return "mpfr";
    default: return "unknown";
  }
}

static int precision_bits(const Config &cfg) {
  // For "d" and many "dd" implementations, define something sensible.
  // You can adjust these if you want the log to reflect mantissa bits vs MPFR precision.
  switch (cfg.mode) {
    case Config::FloatMode::D:    return 53;
    case Config::FloatMode::DD:   return 106;   // typical double-double mantissa-ish
    case Config::FloatMode::MPFR: return cfg.mpfr_prec_bits;
    default: return -1;
  }
}

static std::string py_list_of_doubles(const std::vector<double> &v) {
  // Python-readable list; use high precision so round-trip is decent.
  std::ostringstream oss;
  oss << "[";
  oss << std::setprecision(17);
  for (size_t i = 0; i < v.size(); i++) {
    oss << v[i];
    if (i + 1 != v.size()) oss << ", ";
  }
  oss << "]";
  return oss.str();
}

/*
Parsed by python as:
import ast
rows = []
with open("lats/8380417_256_128.txt") as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        rows.append(ast.literal_eval(ln))
*/

static std::string ensure_lat_file_path(const mpz_t q_mpz, int n, int m) {
  namespace fs = std::filesystem;
  fs::path dir("lats");
  fs::create_directories(dir);

  const std::string q_str = mpz_to_dec_string(q_mpz);
  fs::path file = dir / (q_str + "_" + std::to_string(n) + "_" + std::to_string(m) + ".txt");

  // Create file if missing (empty file is fine)
  if (!fs::exists(file)) {
    std::ofstream ofs(file, std::ios::out);
    // Optional: header comment for humans; still safe to ignore in Python by skipping lines starting with '#'
    ofs << "# One python dict per line. Parse with ast.literal_eval.\n";
    ofs.close();
  }
  return file.string();
}

// Thread-safe append: one dict per line
static void append_experiment_result(
    const mpz_t q_mpz,
    int n,
    int m,
    uint64_t seed,
    const std::string &algorithm_name,
    const Config &cfg,
    bool success,
    double t_gso,
    double t_alg,
    double t_extra_stage,
    const std::string &extra_stage_name,
    const std::vector<double> &profile
) {
  const std::string path = ensure_lat_file_path(q_mpz, n, m);

  std::ostringstream line;
  line << "{";
  line << "'seed': " << seed << ", ";
  line << "'algorithm': " << "'" << algorithm_name << "', ";
  line << "'float': " << "'" << float_mode_to_string(cfg) << "', ";
  line << "'prec_bits': " << precision_bits(cfg) << ", ";
  line << "'delta': " << std::setprecision(17) << cfg.delta << ", ";
  line << "'eta': " << std::setprecision(17) << cfg.eta << ", ";
  line << "'lll_flags': " << cfg.lll_flags << ", ";
  line << "'hlll_flags': " << cfg.hlll_flags << ", ";
  line << "'success': " << (success ? "True" : "False") << ", ";

  // timings as dict
  line << "'timings_s': {";
  line << "'gso': " << std::setprecision(17) << t_gso << ", ";
  if (!extra_stage_name.empty() && t_extra_stage >= 0.0) {
    line << "'" << extra_stage_name << "': " << std::setprecision(17) << t_extra_stage << ", ";
  }
  line << "'alg': " << std::setprecision(17) << t_alg;
  line << "}, ";

  // profile
  line << "'profile': " << py_list_of_doubles(profile);

  line << "}";

  // Thread-safe append (in-process)
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::ofstream ofs(path, std::ios::app);
  ofs << line.str() << "\n";
  ofs.flush();
}

} // namespace logging

// -------------------- Precision setup for MPFR --------------------
static void set_mpfr_prec_bits(int bits) {
  // This controls FP_NR<mpfr_t> precision. Must be set before constructing objects using it. :contentReference[oaicite:2]{index=2}
  fplll::FP_NR<mpfr_t>::set_prec(bits);
}

// -------------------- Core runner (templated by FT) --------------------
template <class FT>
int run_with_float(const Config &cfg,
                   const fplll::Z_NR<mpz_t> &q,
                   uint64_t seed,
                   fplll::ZZ_mat<mpz_t> &B0,
                   fplll::ZZ_mat<mpz_t> &B1){
  using namespace fplll;

  using ZT = Z_NR<mpz_t>;
  const int n = B0.get_rows();
  const int m = n / 2;

  // -------------------- Standard LLL path: GSO + LLL --------------------
  Timer t;
  double t_gso = 0.0, t_lll = 0.0;
  bool ok_lll = false;
  std::vector<double> prof_lll;

  try {
    // No transforms: pass empty U / UinvT (no tracking).
    ZZ_mat<mpz_t> U0, U0invT;

    MatGSO<ZT, FT> gso0(B0, U0, U0invT, GSO_DEFAULT);

    t.start();
    gso0.update_gso();
    t_gso = t.stop_s();

    t.start();
    LLLReduction<ZT, FT> lll(gso0, cfg.delta, cfg.eta, cfg.lll_flags);
    ok_lll = lll.lll();
    t_lll = t.stop_s();

    if (ok_lll) {
      prof_lll = profile_from_gso<ZT, FT>(gso0, n);
    } else {
      prof_lll = dummy_profile(n);
      std::cerr << "Standard LLL returned failure status.\n";
    }
  } catch (const std::exception &e) {
    ok_lll = false;
    prof_lll = dummy_profile(n);
    std::cerr << "Standard LLL threw exception: " << e.what() << "\n";
  }

  // Log standard LLL result (always)
  logging::append_experiment_result(
      q.get_data(), n, m, seed,
      "L2-Cholesky",
      cfg,
      ok_lll,
      /*t_gso=*/ t_gso,
      /*t_alg=*/ t_lll,
      /*t_extra_stage=*/ -1.0,
      /*extra_stage_name=*/ "",
      /*profile=*/ prof_lll);

  // -------------------- Householder path: build HH + HLLL --------------------
  Timer th;
  double t_hh_build = 0.0, t_hlll = 0.0;
  bool ok_hlll = false;
  std::vector<double> prof_hlll;

  try {
    // No transforms tracked: pass empty U / UT.
    ZZ_mat<mpz_t> Uh, UTh;

    th.start();
    MatHouseholder<ZT, FT> hh(B1, Uh, UTh, /*flags=*/0);
    hh.update_R();  // full R computation
    t_hh_build = th.stop_s();

    th.start();
    HLLLReduction<ZT, FT> hlll(hh, cfg.delta, cfg.eta, cfg.theta, cfg.c, cfg.hlll_flags);
    ok_hlll = hlll.hlll();
    t_hlll = th.stop_s();

    if (ok_hlll) {
      prof_hlll = profile_from_householder<ZT, FT>(hh, n);
    } else {
      prof_hlll = dummy_profile(n);
      std::cerr << "HLLL returned failure, status=" << hlll.get_status() << "\n";
    }
  } catch (const std::exception &e) {
    ok_hlll = false;
    prof_hlll = dummy_profile(n);
    std::cerr << "HLLL threw exception: " << e.what() << "\n";
  }

  // Log HLLL result (always)
  logging::append_experiment_result(
      q.get_data(), n, m, seed,
      "HLLL",
      cfg,
      ok_hlll,
      /*t_gso=*/ 0.0,                  // no MatGSO stage here; keep 0 or set to t_gso if you want
      /*t_alg=*/ t_hlll,
      /*t_extra_stage=*/ t_hh_build,
      /*extra_stage_name=*/ "householder_R",
      /*profile=*/ prof_hlll);

  // -------------------- Optional sanity check (only if both succeeded) --------------------
  int sanity_check = 1;
  if (sanity_check && ok_lll && ok_hlll) {
    std::cout << "checking sanity...\n";
    auto sum_profile = [](const std::vector<double> &p) {
      return std::accumulate(p.begin(), p.end(), 0.0);
    };

    double s_std  = sum_profile(prof_lll);
    double s_hlll = sum_profile(prof_hlll);

    const double abs_tol = 1e-6 * n;
    const double rel_tol = 1e-9;

    double diff   = std::abs(s_std - s_hlll);
    double scale  = std::max({1.0, std::abs(s_std), std::abs(s_hlll)});
    double thresh = abs_tol + rel_tol * scale;

    if (diff > thresh) {
      std::cerr
          << "Profile-sum mismatch!\n"
          << "  sum(profile_standard) = " << s_std << "\n"
          << "  sum(profile_hlll)     = " << s_hlll << "\n"
          << "  diff                  = " << diff << "\n"
          << "  thresh                = " << thresh << "\n";
    }
    assert(diff <= thresh);
  }

  // -------------------- Report --------------------
  std::cout << "Timings (seconds)\n";
  std::cout << "  GSO.update_gso():            " << t_gso << "\n";
  std::cout << "  Standard LLL (MatGSO):       " << t_lll << "\n";
  std::cout << "  Householder R computation:   " << t_hh_build << "\n";
  std::cout << "  HLLL (Householder inside):   " << t_hlll << "\n";

  print_profile("profile_standard (0.5*log(r_ii))", prof_lll);
  print_profile("profile_hlll      (log(|R_ii|))", prof_hlll);

  if (!ok_lll && !ok_hlll) return 2;
  if (!ok_lll || !ok_hlll) return 1;
  return 0;
}

// loop for parallelism
template <class FT>
void worker_loop(const Config &cfg,
                 const fplll::Z_NR<mpz_t> &q,
                 int n, int m,
                 std::atomic<uint64_t> &next_seed,
                 uint64_t end_seed_exclusive)
{
  using namespace fplll;
  const int nm = n - m;

  while (true) {
    uint64_t seed = next_seed.fetch_add(1, std::memory_order_relaxed);
    if (seed >= end_seed_exclusive) break;

    // Build lattice for this seed
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, (unsigned long)seed);

    ZZ_mat<mpz_t> B0;
    B0.gen_zero(n, n);

    // q*I block
    for (int i = 0; i < m; i++) B0[i][i] = q;

    // A block
    Z_NR<mpz_t> tmp;
    for (int i = 0; i < nm; i++)
      for (int j = 0; j < m; j++) {
        mpz_urandomm(tmp.get_data(), state, q.get_data());
        B0[m + i][j] = tmp;
      }

    // bottom-right identity
    for (int i = 0; i < nm; i++) B0[m + i][m + i] = 1;

    gmp_randclear(state);

    ZZ_mat<mpz_t> B1 = B0;

    // Run both algorithms + logging happens inside run_with_float
    (void)run_with_float<FT>(cfg, q, seed, B0, B1);
  }
}

// -------------------- Float dispatch --------------------
int main() {
  using namespace fplll;

  Config cfg;
  cfg.mode = Config::FloatMode::DD;   // float type
  // cfg.mode = Config::FloatMode::MPFR;
  cfg.mpfr_prec_bits = 106;

  cfg.delta = 0.99;
  cfg.eta   = 0.51;
  cfg.lll_flags  = LLL_DEFAULT;
  cfg.hlll_flags = cfg.lll_flags;

  // Modulus
  Z_NR<mpz_t> q;
  mpz_set_str(q.get_data(), "8380417", 10);

  // Dimensions
  const int n  = 192;
  const int m  = n / 2;

  // Parallel schedule: seeds [0, n_trials)
  const uint64_t seed_begin = 0;
  const uint64_t n_trials   = 20;
  const uint64_t seed_end_exclusive = seed_begin + n_trials;

  // Number of workers
  unsigned n_workers = std::thread::hardware_concurrency();
  if (n_workers == 0) n_workers = 4;
  // Optional cap:
  n_workers = std::min<unsigned>(n_workers, 8);
  std::cout << "Using " << n_workers <<" workers.\n";

  // MPFR must be set ONCE before threads start.
  if (cfg.mode == Config::FloatMode::MPFR) {
    set_mpfr_prec_bits(cfg.mpfr_prec_bits);
  }

  std::atomic<uint64_t> next_seed(seed_begin);
  std::vector<std::thread> threads;
  threads.reserve(n_workers);

  try {
    if (cfg.mode == Config::FloatMode::D) {
      using FT = FP_NR<double>;
      for (unsigned t = 0; t < n_workers; t++) {
        threads.emplace_back(worker_loop<FT>,
                             std::cref(cfg), std::cref(q),
                             n, m,
                             std::ref(next_seed),
                             seed_end_exclusive);
      }
    } else if (cfg.mode == Config::FloatMode::MPFR) {
      using FT = FP_NR<mpfr_t>;
      for (unsigned t = 0; t < n_workers; t++) {
        threads.emplace_back(worker_loop<FT>,
                             std::cref(cfg), std::cref(q),
                             n, m,
                             std::ref(next_seed),
                             seed_end_exclusive);
      }
    } else { // DD
#if defined(FPLLL_WITH_QD) || defined(FPLLL_WITH_DD) || defined(FPLLL_WITH_DD_REAL)
      using FT = FP_NR<dd_real>;
      for (unsigned t = 0; t < n_workers; t++) {
        threads.emplace_back(worker_loop<FT>,
                             std::cref(cfg), std::cref(q),
                             n, m,
                             std::ref(next_seed),
                             seed_end_exclusive);
      }
#else
      throw std::runtime_error("DD requested, but this fplll build doesn't expose dd_real/QD support.");
#endif
    }

    for (auto &th : threads) th.join();
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    // Join any started threads before exit
    for (auto &th : threads) if (th.joinable()) th.join();
    return 100;
  }

  return 0;
}