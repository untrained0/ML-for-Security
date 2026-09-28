/**
 * mlsv_cuda — FP64 dense linear algebra on an NVIDIA GPU for the attack server.
 *
 * A Node-API addon (plain C API, no node-addon-api), built by scripts/build_cuda.mjs with nvcc and
 * loaded only by src/server/compute/cuda.ts. The engine never sees it directly: it calls the
 * ComputeBackend interface (src/engine/compute/backend.ts), whose CPU implementation is the
 * reference these routines are checked against.
 *
 * Every matrix crosses the boundary as a row-major Float64Array. cuBLAS/cuSOLVER are
 * column-major, so a row-major M is seen as Mᵀ; each routine below is written so that this needs
 * no explicit transpose:
 *   gram      AᵀA  — row-major A (m×n) is column-major B = Aᵀ (n×m), and AᵀA = B·Bᵀ (syrk)
 *   inverse   M⁻¹  — solving Mᵀ X = I gives X = (M⁻¹)ᵀ, which read back row-major is M⁻¹
 *   RankOneInverse  keeps a symmetric A⁻¹ resident on the device and applies Sherman–Morrison
 *                   updates there, so a row swap of the regression design matrix costs two
 *                   gemv + two ger launches and no host round trip.
 *
 *   DenseMatrix     keeps a matrix resident for repeated A·v / Aᵀ·v (a regression objective's
 *                   residuals and gradient, evaluated at every line-search step).
 *
 * Exports: info(), gram(A, m, n), inverse(M, n), RankOneInverse(Ainv, n)
 *          { update(u, s), apply(v), read(), dispose() }, DenseMatrix(A, m, n)
 *          { mul(v), mulT(v), dispose() }.
 */

#define NAPI_VERSION 8
#define NODE_GYP_MODULE_NAME mlsv_cuda
#include <node_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Error handling ─────────────────────────────────────────────────────

struct GpuError {
  std::string msg;
};

#define CUDA_OK(call)                                                                  \
  do {                                                                                 \
    cudaError_t e_ = (call);                                                           \
    if (e_ != cudaSuccess) throw GpuError{std::string(#call) + ": " + cudaGetErrorString(e_)}; \
  } while (0)
#define CUBLAS_OK(call)                                                                \
  do {                                                                                 \
    cublasStatus_t s_ = (call);                                                        \
    if (s_ != CUBLAS_STATUS_SUCCESS) throw GpuError{std::string(#call) + ": cuBLAS status " + std::to_string((int)s_)}; \
  } while (0)
#define CUSOLVER_OK(call)                                                              \
  do {                                                                                 \
    cusolverStatus_t s_ = (call);                                                      \
    if (s_ != CUSOLVER_STATUS_SUCCESS) throw GpuError{std::string(#call) + ": cuSOLVER status " + std::to_string((int)s_)}; \
  } while (0)

#define NAPI_OK(call)                                                                  \
  do {                                                                                 \
    if ((call) != napi_ok) throw GpuError{std::string("N-API call failed: ") + #call}; \
  } while (0)

// ── Device state (one context per process; Node calls us from one thread) ──

static cublasHandle_t g_blas = nullptr;
static cusolverDnHandle_t g_solver = nullptr;
static cudaStream_t g_stream = nullptr;

static void ensureContext() {
  if (g_blas) return;
  int count = 0;
  CUDA_OK(cudaGetDeviceCount(&count));
  if (count == 0) throw GpuError{"no CUDA device"};
  CUDA_OK(cudaSetDevice(0));
  CUDA_OK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  CUBLAS_OK(cublasCreate(&g_blas));
  CUBLAS_OK(cublasSetStream(g_blas, g_stream));
  CUSOLVER_OK(cusolverDnCreate(&g_solver));
  CUSOLVER_OK(cusolverDnSetStream(g_solver, g_stream));
}

/** Device buffer freed on scope exit (exceptions included). */
struct DevBuf {
  void* p = nullptr;
  explicit DevBuf(size_t bytes) { if (bytes) CUDA_OK(cudaMalloc(&p, bytes)); }
  ~DevBuf() { if (p) cudaFree(p); }
  double* d() const { return static_cast<double*>(p); }
  int* i() const { return static_cast<int*>(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
};

// ── N-API helpers ──────────────────────────────────────────────────────

static void throwJs(napi_env env, const std::string& msg) {
  napi_throw_error(env, "ERR_MLSV_CUDA", msg.c_str());
}

static double* float64Arg(napi_env env, napi_value v, size_t* length) {
  bool isTyped = false;
  NAPI_OK(napi_is_typedarray(env, v, &isTyped));
  if (!isTyped) throw GpuError{"expected a Float64Array"};
  napi_typedarray_type type;
  void* data;
  napi_value buffer;
  size_t offset;
  NAPI_OK(napi_get_typedarray_info(env, v, &type, length, &data, &buffer, &offset));
  if (type != napi_float64_array) throw GpuError{"expected a Float64Array"};
  return static_cast<double*>(data);
}

static int64_t intArg(napi_env env, napi_value v) {
  int64_t out;
  NAPI_OK(napi_get_value_int64(env, v, &out));
  return out;
}

static napi_value newFloat64Array(napi_env env, size_t length, double** data) {
  napi_value buffer, array;
  void* raw;
  NAPI_OK(napi_create_arraybuffer(env, length * sizeof(double), &raw, &buffer));
  NAPI_OK(napi_create_typedarray(env, napi_float64_array, length, buffer, 0, &array));
  *data = static_cast<double*>(raw);
  return array;
}

static std::vector<napi_value> args(napi_env env, napi_callback_info info, size_t expected, napi_value* self = nullptr) {
  std::vector<napi_value> argv(expected);
  size_t argc = expected;
  NAPI_OK(napi_get_cb_info(env, info, &argc, argv.data(), self, nullptr));
  if (argc < expected) throw GpuError{"expected " + std::to_string(expected) + " arguments"};
  return argv;
}

// Variadic: bodies contain commas (kernel launches, argument lists)
#define NAPI_ENTRY(...)                                   \
  try { __VA_ARGS__ }                                     \
  catch (const GpuError& e) { throwJs(env, e.msg); }      \
  return nullptr;

// ── info() ─────────────────────────────────────────────────────────────

static napi_value Info(napi_env env, napi_callback_info) {
  NAPI_ENTRY({
    ensureContext();
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    size_t freeMem = 0, totalMem = 0;
    CUDA_OK(cudaMemGetInfo(&freeMem, &totalMem));

    napi_value out, v;
    NAPI_OK(napi_create_object(env, &out));
    NAPI_OK(napi_create_string_utf8(env, prop.name, NAPI_AUTO_LENGTH, &v));
    NAPI_OK(napi_set_named_property(env, out, "device", v));
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    NAPI_OK(napi_create_string_utf8(env, cc.c_str(), NAPI_AUTO_LENGTH, &v));
    NAPI_OK(napi_set_named_property(env, out, "computeCapability", v));
    NAPI_OK(napi_create_double(env, (double)totalMem, &v));
    NAPI_OK(napi_set_named_property(env, out, "memoryBytes", v));
    NAPI_OK(napi_create_int32(env, runtime, &v));
    NAPI_OK(napi_set_named_property(env, out, "cudaRuntime", v));
    NAPI_OK(napi_create_int32(env, driver, &v));
    NAPI_OK(napi_set_named_property(env, out, "cudaDriver", v));
    return out;
  })
}

// ── gram(A, m, n) → AᵀA ────────────────────────────────────────────────

static napi_value Gram(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    auto a = args(env, info, 3);
    size_t len;
    double* A = float64Arg(env, a[0], &len);
    int64_t m = intArg(env, a[1]), n = intArg(env, a[2]);
    if (m <= 0 || n <= 0 || (size_t)(m * n) != len) throw GpuError{"gram: A must hold m·n values"};
    ensureContext();

    DevBuf dA(sizeof(double) * m * n), dC(sizeof(double) * n * n);
    CUDA_OK(cudaMemcpyAsync(dA.p, A, sizeof(double) * m * n, cudaMemcpyHostToDevice, g_stream));
    const double one = 1.0, zero = 0.0;
    // Column-major B = Aᵀ is n×m with leading dimension n; C = B·Bᵀ (lower triangle)
    CUBLAS_OK(cublasDsyrk(g_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, (int)n, (int)m,
                          &one, dA.d(), (int)n, &zero, dC.d(), (int)n));
    double* C;
    napi_value out = newFloat64Array(env, n * n, &C);
    CUDA_OK(cudaMemcpyAsync(C, dC.p, sizeof(double) * n * n, cudaMemcpyDeviceToHost, g_stream));
    CUDA_OK(cudaStreamSynchronize(g_stream));
    // syrk fills one triangle; mirror it so the result is exactly symmetric
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = j + 1; i < n; i++) C[i * n + j] = C[j * n + i];
    return out;
  })
}

// ── inverse(M, n) → M⁻¹, or null when LU finds an exactly zero pivot ────

static napi_value Inverse(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    auto a = args(env, info, 2);
    size_t len;
    double* M = float64Arg(env, a[0], &len);
    int64_t n = intArg(env, a[1]);
    if (n <= 0 || (size_t)(n * n) != len) throw GpuError{"inverse: M must hold n·n values"};
    ensureContext();

    std::vector<double> I(n * n, 0.0);
    for (int64_t i = 0; i < n; i++) I[i * n + i] = 1.0;

    DevBuf dM(sizeof(double) * n * n), dB(sizeof(double) * n * n), dPiv(sizeof(int) * n), dInfo(sizeof(int));
    CUDA_OK(cudaMemcpyAsync(dM.p, M, sizeof(double) * n * n, cudaMemcpyHostToDevice, g_stream));
    CUDA_OK(cudaMemcpyAsync(dB.p, I.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice, g_stream));
    int lwork = 0;
    CUSOLVER_OK(cusolverDnDgetrf_bufferSize(g_solver, (int)n, (int)n, dM.d(), (int)n, &lwork));
    DevBuf dWork(sizeof(double) * (lwork > 0 ? lwork : 1));
    CUSOLVER_OK(cusolverDnDgetrf(g_solver, (int)n, (int)n, dM.d(), (int)n, dWork.d(), dPiv.i(), dInfo.i()));
    int hInfo = 0;
    CUDA_OK(cudaMemcpyAsync(&hInfo, dInfo.p, sizeof(int), cudaMemcpyDeviceToHost, g_stream));
    CUDA_OK(cudaStreamSynchronize(g_stream));
    if (hInfo != 0) {
      napi_value nul;
      NAPI_OK(napi_get_null(env, &nul));
      return nul;
    }
    CUSOLVER_OK(cusolverDnDgetrs(g_solver, CUBLAS_OP_N, (int)n, (int)n, dM.d(), (int)n, dPiv.i(), dB.d(), (int)n, dInfo.i()));
    double* out;
    napi_value result = newFloat64Array(env, n * n, &out);
    CUDA_OK(cudaMemcpyAsync(out, dB.p, sizeof(double) * n * n, cudaMemcpyDeviceToHost, g_stream));
    CUDA_OK(cudaStreamSynchronize(g_stream));
    return result;
  })
}

// ── RankOneInverse ─────────────────────────────────────────────────────

/** alpha ← −s / (1 + s·uᵀA⁻¹u), or 0 when the update would be singular (CPU: skip it). */
__global__ void shermanMorrisonScale(const double* uAu, double s, double* alpha) {
  const double denom = 1.0 + s * (*uAu);
  *alpha = fabs(denom) < 1e-12 ? 0.0 : -s / denom;
}

struct RankOne {
  int n = 0;
  double* inv = nullptr;   // n×n, symmetric
  double* u = nullptr;     // input vector
  double* au = nullptr;    // A⁻¹u
  double* scalars = nullptr;  // [uᵀA⁻¹u, alpha]
  std::vector<double> host;   // staging for apply()

  void release() {
    if (inv) cudaFree(inv);
    if (u) cudaFree(u);
    if (au) cudaFree(au);
    if (scalars) cudaFree(scalars);
    inv = u = au = scalars = nullptr;
  }
};

static void finalizeRankOne(napi_env, void* data, void*) {
  auto* r = static_cast<RankOne*>(data);
  r->release();
  delete r;
}

static RankOne* unwrap(napi_env env, napi_value self) {
  void* data = nullptr;
  NAPI_OK(napi_unwrap(env, self, &data));
  auto* r = static_cast<RankOne*>(data);
  if (!r->inv) throw GpuError{"RankOneInverse used after dispose()"};
  return r;
}

static napi_value RankOneCtor(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    auto a = args(env, info, 2, &self);
    size_t len;
    double* Ainv = float64Arg(env, a[0], &len);
    int64_t n = intArg(env, a[1]);
    if (n <= 0 || (size_t)(n * n) != len) throw GpuError{"RankOneInverse: Ainv must hold n·n values"};
    ensureContext();

    auto* r = new RankOne();
    r->n = (int)n;
    r->host.resize(n);
    try {
      CUDA_OK(cudaMalloc(&r->inv, sizeof(double) * n * n));
      CUDA_OK(cudaMalloc(&r->u, sizeof(double) * n));
      CUDA_OK(cudaMalloc(&r->au, sizeof(double) * n));
      CUDA_OK(cudaMalloc(&r->scalars, sizeof(double) * 2));
      CUDA_OK(cudaMemcpyAsync(r->inv, Ainv, sizeof(double) * n * n, cudaMemcpyHostToDevice, g_stream));
      CUDA_OK(cudaStreamSynchronize(g_stream));
      NAPI_OK(napi_wrap(env, self, r, finalizeRankOne, nullptr, nullptr));
    } catch (...) {
      r->release();
      delete r;
      throw;
    }
    return self;
  })
}

/** update(u, s): A⁻¹ ← (A + s·uuᵀ)⁻¹, entirely on the device. */
static napi_value RankOneUpdate(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    auto a = args(env, info, 2, &self);
    RankOne* r = unwrap(env, self);
    size_t len;
    double* u = float64Arg(env, a[0], &len);
    double s;
    NAPI_OK(napi_get_value_double(env, a[1], &s));
    if ((int)len != r->n) throw GpuError{"update: u has the wrong length"};

    const int n = r->n;
    const double one = 1.0, zero = 0.0;
    // Pageable source: the copy is staged before this call returns, so `u` may be reused
    CUDA_OK(cudaMemcpyAsync(r->u, u, sizeof(double) * n, cudaMemcpyHostToDevice, g_stream));
    CUBLAS_OK(cublasDgemv(g_blas, CUBLAS_OP_T, n, n, &one, r->inv, n, r->u, 1, &zero, r->au, 1));
    CUBLAS_OK(cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_DEVICE));
    try {
      CUBLAS_OK(cublasDdot(g_blas, n, r->u, 1, r->au, 1, r->scalars));
      shermanMorrisonScale<<<1, 1, 0, g_stream>>>(r->scalars, s, r->scalars + 1);
      CUDA_OK(cudaGetLastError());
      CUBLAS_OK(cublasDger(g_blas, n, n, r->scalars + 1, r->au, 1, r->au, 1, r->inv, n));
    } catch (...) {
      cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST);
      throw;
    }
    CUBLAS_OK(cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST));
    return nullptr;
  })
}

/** apply(v) → A⁻¹v (synchronises). */
static napi_value RankOneApply(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    auto a = args(env, info, 1, &self);
    RankOne* r = unwrap(env, self);
    size_t len;
    double* v = float64Arg(env, a[0], &len);
    if ((int)len != r->n) throw GpuError{"apply: v has the wrong length"};
    const int n = r->n;
    const double one = 1.0, zero = 0.0;
    CUDA_OK(cudaMemcpyAsync(r->u, v, sizeof(double) * n, cudaMemcpyHostToDevice, g_stream));
    CUBLAS_OK(cublasDgemv(g_blas, CUBLAS_OP_T, n, n, &one, r->inv, n, r->u, 1, &zero, r->au, 1));
    double* out;
    napi_value result = newFloat64Array(env, n, &out);
    CUDA_OK(cudaMemcpyAsync(out, r->au, sizeof(double) * n, cudaMemcpyDeviceToHost, g_stream));
    CUDA_OK(cudaStreamSynchronize(g_stream));
    return result;
  })
}

/** read() → the current A⁻¹ (row-major; it is symmetric up to rounding). */
static napi_value RankOneRead(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    args(env, info, 0, &self);
    RankOne* r = unwrap(env, self);
    const int n = r->n;
    double* out;
    napi_value result = newFloat64Array(env, (size_t)n * n, &out);
    // The device holds the row-major bytes, i.e. (A⁻¹)ᵀ column-major; every update adds the
    // symmetric α·(A⁻¹u)(A⁻¹u)ᵀ, so the bytes still read back as the row-major A⁻¹.
    CUDA_OK(cudaMemcpyAsync(out, r->inv, sizeof(double) * n * n, cudaMemcpyDeviceToHost, g_stream));
    CUDA_OK(cudaStreamSynchronize(g_stream));
    return result;
  })
}

/** dispose(): free device memory now rather than at garbage collection. */
static napi_value RankOneDispose(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    args(env, info, 0, &self);
    void* data = nullptr;
    NAPI_OK(napi_unwrap(env, self, &data));
    static_cast<RankOne*>(data)->release();
    return nullptr;
  })
}

// ── DenseMatrix ────────────────────────────────────────────────────────

/**
 * A row-major m × n matrix kept on the device for repeated products. Its bytes are Aᵀ in
 * column-major order (n × m, leading dimension n), so A·v is gemv(OP_T) and Aᵀ·v is gemv(OP_N).
 */
struct Dense {
  int m = 0, n = 0;
  double* a = nullptr;
  double* in = nullptr;    // max(m, n)
  double* out = nullptr;   // max(m, n)

  void release() {
    if (a) cudaFree(a);
    if (in) cudaFree(in);
    if (out) cudaFree(out);
    a = in = out = nullptr;
  }
};

static void finalizeDense(napi_env, void* data, void*) {
  auto* d = static_cast<Dense*>(data);
  d->release();
  delete d;
}

static Dense* unwrapDense(napi_env env, napi_value self) {
  void* data = nullptr;
  NAPI_OK(napi_unwrap(env, self, &data));
  auto* d = static_cast<Dense*>(data);
  if (!d->a) throw GpuError{"DenseMatrix used after dispose()"};
  return d;
}

static napi_value DenseCtor(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    auto a = args(env, info, 3, &self);
    size_t len;
    double* A = float64Arg(env, a[0], &len);
    int64_t m = intArg(env, a[1]), n = intArg(env, a[2]);
    if (m <= 0 || n <= 0 || (size_t)(m * n) != len) throw GpuError{"DenseMatrix: A must hold m·n values"};
    ensureContext();
    auto* d = new Dense();
    d->m = (int)m;
    d->n = (int)n;
    const size_t k = (size_t)(m > n ? m : n);
    try {
      CUDA_OK(cudaMalloc(&d->a, sizeof(double) * m * n));
      CUDA_OK(cudaMalloc(&d->in, sizeof(double) * k));
      CUDA_OK(cudaMalloc(&d->out, sizeof(double) * k));
      CUDA_OK(cudaMemcpyAsync(d->a, A, sizeof(double) * m * n, cudaMemcpyHostToDevice, g_stream));
      CUDA_OK(cudaStreamSynchronize(g_stream));
      NAPI_OK(napi_wrap(env, self, d, finalizeDense, nullptr, nullptr));
    } catch (...) {
      d->release();
      delete d;
      throw;
    }
    return self;
  })
}

/** y = op(A)·v; `transpose` selects Aᵀ·v. Synchronises. */
static napi_value denseProduct(napi_env env, napi_callback_info info, bool transpose) {
  napi_value self;
  auto a = args(env, info, 1, &self);
  Dense* d = unwrapDense(env, self);
  size_t len;
  double* v = float64Arg(env, a[0], &len);
  const int inLen = transpose ? d->m : d->n, outLen = transpose ? d->n : d->m;
  if ((int)len != inLen) throw GpuError{"DenseMatrix: vector has the wrong length"};
  const double one = 1.0, zero = 0.0;
  CUDA_OK(cudaMemcpyAsync(d->in, v, sizeof(double) * inLen, cudaMemcpyHostToDevice, g_stream));
  CUBLAS_OK(cublasDgemv(g_blas, transpose ? CUBLAS_OP_N : CUBLAS_OP_T, d->n, d->m,
                        &one, d->a, d->n, d->in, 1, &zero, d->out, 1));
  double* out;
  napi_value result = newFloat64Array(env, outLen, &out);
  CUDA_OK(cudaMemcpyAsync(out, d->out, sizeof(double) * outLen, cudaMemcpyDeviceToHost, g_stream));
  CUDA_OK(cudaStreamSynchronize(g_stream));
  return result;
}

static napi_value DenseMul(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({ return denseProduct(env, info, false); })
}

static napi_value DenseMulT(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({ return denseProduct(env, info, true); })
}

static napi_value DenseDispose(napi_env env, napi_callback_info info) {
  NAPI_ENTRY({
    napi_value self;
    args(env, info, 0, &self);
    void* data = nullptr;
    NAPI_OK(napi_unwrap(env, self, &data));
    static_cast<Dense*>(data)->release();
    return nullptr;
  })
}

// ── Module ─────────────────────────────────────────────────────────────

static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor fns[] = {
    {"info", nullptr, Info, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"gram", nullptr, Gram, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"inverse", nullptr, Inverse, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(fns) / sizeof(fns[0]), fns);

  napi_property_descriptor methods[] = {
    {"update", nullptr, RankOneUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"apply", nullptr, RankOneApply, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"read", nullptr, RankOneRead, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dispose", nullptr, RankOneDispose, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_value cls;
  napi_define_class(env, "RankOneInverse", NAPI_AUTO_LENGTH, RankOneCtor, nullptr,
                    sizeof(methods) / sizeof(methods[0]), methods, &cls);
  napi_set_named_property(env, exports, "RankOneInverse", cls);

  napi_property_descriptor denseMethods[] = {
    {"mul", nullptr, DenseMul, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"mulT", nullptr, DenseMulT, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"dispose", nullptr, DenseDispose, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_value denseCls;
  napi_define_class(env, "DenseMatrix", NAPI_AUTO_LENGTH, DenseCtor, nullptr,
                    sizeof(denseMethods) / sizeof(denseMethods[0]), denseMethods, &denseCls);
  napi_set_named_property(env, exports, "DenseMatrix", denseCls);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
