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

  namespace progress {

  static std::atomic<uint64_t> completed{0};
  static std::mutex cout_mutex;

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

static std::string float_mode_to_string(const Config &cfg_clll) {
  switch (cfg_clll.mode) {
    case Config::FloatMode::D:    return "d";
    case Config::FloatMode::DD:   return "dd";
    case Config::FloatMode::MPFR: return "mpfr";
    default: return "unknown";
  }
}

static int precision_bits(const Config &cfg_clll) {
  // For "d" and many "dd" implementations, define something sensible.
  // You can adjust these if you want the log to reflect mantissa bits vs MPFR precision.
  switch (cfg_clll.mode) {
    case Config::FloatMode::D:    return 53;
    case Config::FloatMode::DD:   return 106;   // typical double-double mantissa-ish
    case Config::FloatMode::MPFR: return cfg_clll.mpfr_prec_bits;
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
    const Config &cfg_clll,
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
  line << "'n': '" << n << "', ";
  line << "'m': '" << n << "', ";
  line << "'q': '" << mpz_to_dec_string(q_mpz) << "', ";
  line << "'algorithm': " << "'" << algorithm_name << "', ";
  line << "'float': " << "'" << float_mode_to_string(cfg_clll) << "', ";
  line << "'prec_bits': " << precision_bits(cfg_clll) << ", ";
  line << "'delta': " << std::setprecision(17) << cfg_clll.delta << ", ";
  line << "'eta': " << std::setprecision(17) << cfg_clll.eta << ", ";
  line << "'lll_flags': " << cfg_clll.lll_flags << ", ";
  line << "'hlll_flags': " << cfg_clll.hlll_flags << ", ";
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
struct LLLResult {
  bool ok = false;
  double t_gso = 0.0;
  double t_alg = 0.0;
  std::vector<double> profile; // always size n (dummy if failed)
};

template <class FT>
struct HLLLResult {
  bool ok = false;
  double t_hh_build = 0.0;
  double t_alg = 0.0;
  std::vector<double> profile; // always size n (dummy if failed)
};

// ---------- Standard LLL (Cholesky / MatGSO) ----------
template <class FT>
LLLResult<FT> run_clll(const Config &cfg_clll,
                       fplll::ZZ_mat<mpz_t> &B0) {
  using namespace fplll;
  using ZT = Z_NR<mpz_t>;
  const int n = B0.get_rows();

  LLLResult<FT> res;
  res.profile = dummy_profile(n);

  Timer t;

  try {
    ZZ_mat<mpz_t> U0, U0invT; // empty => no tracking
    MatGSO<ZT, FT> gso0(B0, U0, U0invT, GSO_DEFAULT);

    t.start();
    gso0.update_gso();
    res.t_gso = t.stop_s();

    t.start();
    LLLReduction<ZT, FT> lll(gso0, cfg_clll.delta, cfg_clll.eta, cfg_clll.lll_flags);
    res.ok = lll.lll();
    res.t_alg = t.stop_s();

    if (res.ok) {
      res.profile = profile_from_gso<ZT, FT>(gso0, n);
    } else {
      std::cerr << "Standard LLL returned failure status.\n";
    }
  } catch (const std::exception &e) {
    res.ok = false;
    std::cerr << "Standard LLL threw exception: " << e.what() << "\n";
  }

  return res;
}

// ---------- HLLL (Householder) ----------
template <class FT>
HLLLResult<FT> run_hlll(const Config &cfg_hlll,
                        fplll::ZZ_mat<mpz_t> &B1) {
  using namespace fplll;
  using ZT = Z_NR<mpz_t>;
  const int n = B1.get_rows();

  HLLLResult<FT> res;
  res.profile = dummy_profile(n);

  Timer th;

  try {
    ZZ_mat<mpz_t> Uh, UTh; // empty => no tracking

    th.start();
    MatHouseholder<ZT, FT> hh(B1, Uh, UTh, /*flags=*/0);
    hh.update_R();                // (you can skip this later if desired)
    res.t_hh_build = th.stop_s();

    th.start();
    HLLLReduction<ZT, FT> hlll(hh,
                              cfg_hlll.delta, cfg_hlll.eta,
                              cfg_hlll.theta, cfg_hlll.c,
                              cfg_hlll.hlll_flags);
    res.ok = hlll.hlll();
    res.t_alg = th.stop_s();

    if (res.ok) {
      res.profile = profile_from_householder<ZT, FT>(hh, n);
    } else {
      std::cerr << "HLLL returned failure, status=" << hlll.get_status() << "\n";
    }
  } catch (const std::exception &e) {
    res.ok = false;
    std::cerr << "HLLL threw exception: " << e.what() << "\n";
  }

  return res;
}

// ---------- One lattice instance: run both algorithms with possibly different float types ----------
template <class FTc, class FTh>
int run_one_lattice(const Config &cfg_clll,
                    const Config &cfg_hlll,
                    const fplll::Z_NR<mpz_t> &q,
                    uint64_t seed,
                    fplll::ZZ_mat<mpz_t> &B0,
                    fplll::ZZ_mat<mpz_t> &B1) {
  const int n = B0.get_rows();
  const int m = n / 2;

  // Run CLLL
  auto r_clll = run_clll<FTc>(cfg_clll, B0);
  logging::append_experiment_result(
      q.get_data(), n, m, seed,
      "L2-Cholesky",
      cfg_clll,
      r_clll.ok,
      /*t_gso=*/ r_clll.t_gso,
      /*t_alg=*/ r_clll.t_alg,
      /*t_extra_stage=*/ -1.0,
      /*extra_stage_name=*/ "",
      /*profile=*/ r_clll.profile);

  // Run HLLL
  auto r_hlll = run_hlll<FTh>(cfg_hlll, B1);
  logging::append_experiment_result(
      q.get_data(), n, m, seed,
      "HLLL",
      cfg_hlll,
      r_hlll.ok,
      /*t_gso=*/ 0.0,
      /*t_alg=*/ r_hlll.t_alg,
      /*t_extra_stage=*/ r_hlll.t_hh_build,
      /*extra_stage_name=*/ "householder_R",
      /*profile=*/ r_hlll.profile);

  if (!r_clll.ok && !r_hlll.ok) return 2;
  if (!r_clll.ok || !r_hlll.ok) return 1;

  // -------------------- Report --------------------
  std::cout << "Timings (seconds)\n";
  std::cout << "  GSO.update_gso():            " << r_clll.t_gso << "\n";
  std::cout << "  Standard LLL (MatGSO):       " << r_clll.t_alg << "\n";
  std::cout << "  Householder R computation:   " << r_hlll.t_hh_build << "\n";
  std::cout << "  HLLL (Householder inside):   " << r_hlll.t_alg << "\n";

  print_profile("profile_standard (0.5*log(r_ii))", r_clll.profile);
  print_profile("profile_hlll      (log(|R_ii|))", r_hlll.profile);

  if (!r_clll.ok && !r_hlll.ok) return 2;
  if (!r_clll.ok || !r_hlll.ok) return 1;
  return 0;
}

// loop for parallelism
template <class FTc, class FTh>
void worker_loop(const Config &cfg_clll,
                 const Config &cfg_hlll,
                 const fplll::Z_NR<mpz_t> &q,
                 int n, int m,
                 std::atomic<uint64_t> &next_seed,
                 uint64_t end_seed_exclusive,
                 uint64_t total_trials,
                 uint64_t report_interval){
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

    (void)run_one_lattice<FTc, FTh>(cfg_clll, cfg_hlll, q, seed, B0, B1);
    uint64_t done = progress::completed.fetch_add(1) + 1;

    if (report_interval > 0 && done % report_interval == 0) {
      std::lock_guard<std::mutex> lock(progress::cout_mutex);
      std::cout << done << " experiments out of "
                << total_trials << " done.\n";
    }
  }
}

int main() {
  using namespace fplll;

  Config cfg_clll;
  cfg_clll.mode = Config::FloatMode::DD;   // example: DD for Cholesky
  cfg_clll.mpfr_prec_bits = 106;
  cfg_clll.delta = 0.99;
  cfg_clll.eta   = 0.51;
  cfg_clll.lll_flags  = LLL_DEFAULT;
  cfg_clll.hlll_flags = LLL_DEFAULT; // irrelevant for CLLL logs, but keep consistent

  Config cfg_hlll;
  cfg_hlll.mode = Config::FloatMode::DD;    // example: D for HLLL
  cfg_hlll.mpfr_prec_bits = 106;           // must match cfg_clll if either is MPFR
  cfg_hlll.delta = 0.99;
  cfg_hlll.eta   = 0.51;
  cfg_hlll.lll_flags  = LLL_DEFAULT;       // not used in HLLL
  cfg_hlll.hlll_flags = LLL_DEFAULT;

  // Enforce MPFR precision consistency (global)
  if (cfg_clll.mode == Config::FloatMode::MPFR || cfg_hlll.mode == Config::FloatMode::MPFR) {
    if (cfg_clll.mpfr_prec_bits != cfg_hlll.mpfr_prec_bits) {
      throw std::runtime_error("MPFR precision must be identical for cfg_clll and cfg_hlll (global setting).");
    }
    set_mpfr_prec_bits(cfg_clll.mpfr_prec_bits);
  }

  // Modulus
  Z_NR<mpz_t> q;
  mpz_set_str(q.get_data(), "8380417", 10);

  // Dimensions
  const int n  = 384;
  const int m  = n / 2;

  // Seeds
  const uint64_t seed_begin = 0;
  const uint64_t n_trials   = 20;
  const uint64_t seed_end_exclusive = seed_begin + n_trials;

  // Workers
  unsigned n_workers = std::thread::hardware_concurrency();
  if (n_workers == 0) n_workers = 4;
  n_workers = std::min<unsigned>(n_workers, 10);
  std::cout << "Using " << n_workers << " workers.\n";

  // how often to report
  const uint64_t report_interval = 2;

  std::atomic<uint64_t> next_seed(seed_begin);
  std::vector<std::thread> threads;
  threads.reserve(n_workers);

  auto spawn_threads = [&](auto tagFTc, auto tagFTh) {
    using FTc = decltype(tagFTc);
    using FTh = decltype(tagFTh);
    for (unsigned t = 0; t < n_workers; t++) {
      threads.emplace_back(worker_loop<FTc, FTh>,
                     std::cref(cfg_clll), std::cref(cfg_hlll), std::cref(q),
                     n, m,
                     std::ref(next_seed),
                     seed_end_exclusive,
                     n_trials,
                     report_interval);
    }
  };

  try {
    // Dispatch on (cfg_clll.mode, cfg_hlll.mode)
    // CLLL float:
    if (cfg_clll.mode == Config::FloatMode::D) {
      // HLLL float:
      if (cfg_hlll.mode == Config::FloatMode::D) {
        spawn_threads(FP_NR<double>{}, FP_NR<double>{});
      } else if (cfg_hlll.mode == Config::FloatMode::MPFR) {
        spawn_threads(FP_NR<double>{}, FP_NR<mpfr_t>{});
      } else {
#if defined(FPLLL_WITH_QD) || defined(FPLLL_WITH_DD) || defined(FPLLL_WITH_DD_REAL)
        spawn_threads(FP_NR<double>{}, FP_NR<dd_real>{});
#else
        throw std::runtime_error("HLLL DD requested, but dd_real/QD not available.");
#endif
      }
    } else if (cfg_clll.mode == Config::FloatMode::MPFR) {
      if (cfg_hlll.mode == Config::FloatMode::D) {
        spawn_threads(FP_NR<mpfr_t>{}, FP_NR<double>{});
      } else if (cfg_hlll.mode == Config::FloatMode::MPFR) {
        spawn_threads(FP_NR<mpfr_t>{}, FP_NR<mpfr_t>{});
      } else {
#if defined(FPLLL_WITH_QD) || defined(FPLLL_WITH_DD) || defined(FPLLL_WITH_DD_REAL)
        spawn_threads(FP_NR<mpfr_t>{}, FP_NR<dd_real>{});
#else
        throw std::runtime_error("HLLL DD requested, but dd_real/QD not available.");
#endif
      }
    } else { // cfg_clll DD
#if defined(FPLLL_WITH_QD) || defined(FPLLL_WITH_DD) || defined(FPLLL_WITH_DD_REAL)
      if (cfg_hlll.mode == Config::FloatMode::D) {
        spawn_threads(FP_NR<dd_real>{}, FP_NR<double>{});
      } else if (cfg_hlll.mode == Config::FloatMode::MPFR) {
        spawn_threads(FP_NR<dd_real>{}, FP_NR<mpfr_t>{});
      } else {
        spawn_threads(FP_NR<dd_real>{}, FP_NR<dd_real>{});
      }
#else
      throw std::runtime_error("CLLL DD requested, but dd_real/QD not available.");
#endif
    }

    for (auto &th : threads) th.join();
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    for (auto &th : threads) if (th.joinable()) th.join();
    return 100;
  }

  return 0;
}