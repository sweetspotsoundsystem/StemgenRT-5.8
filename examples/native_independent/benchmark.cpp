// Native worker probe for independent DBV and physical residual Other. Not a DAW test.
#include <onnxruntime_cxx_api.h>
#include "scheduling.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {
using Clock = scheduling::Clock;
#if defined(__APPLE__)
constexpr const char* kPlatform = "macos";
#else
constexpr const char* kPlatform = "portable";
#endif
constexpr size_t kMaxCount = 6;
constexpr std::array<const char*, kMaxCount> inputs{
  "audio_chunk", "audio_history", "local_hidden", "global_hidden", "spectral_numerator_tail", "waveform_numerator_tail"};
constexpr std::array<const char*, kMaxCount> outputs{
  "separated_chunk", "next_audio_history", "next_local_hidden", "next_global_hidden", "next_spectral_numerator_tail", "next_waveform_numerator_tail"};
void require(bool condition, const char* why) { if (!condition) throw std::runtime_error(why); }
size_t elements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}
double ms(Clock::duration value) { return std::chrono::duration<double, std::milli>(value).count(); }
double threadSeconds() {
  timespec ts{};
  require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0, "Thread clock failed");
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}
void disableDenormals() {
#if defined(__aarch64__)
  uint64_t fpcr = 0;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= uint64_t{1} << 24;
  asm volatile("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__SSE2__)
  _mm_setcsr(_mm_getcsr() | 0x8040U);
#else
  throw std::runtime_error("Unsupported denormal-control architecture");
#endif
}
void stats(std::vector<double> values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.) / double(values.size());
  std::sort(values.begin(), values.end());
  auto percentile = [&](double fraction) {
    const double index = fraction * double(values.size() - 1);
    const auto a = static_cast<size_t>(index), b = std::min(a + 1, values.size() - 1);
    return values[a] + (index - double(a)) * (values[b] - values[a]);
  };
  std::cout << "{\"mean_ms\":" << mean << ",\"p50_ms\":" << percentile(.5)
    << ",\"p95_ms\":" << percentile(.95) << ",\"p99_ms\":" << percentile(.99)
    << ",\"p999_ms\":" << percentile(.999) << ",\"maximum_ms\":" << values.back()
    << ",\"mean_fraction_of_hop\":" << mean / (128000. / 44100.) << '}';
}
void writeFloats(const std::string& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  require(bool(stream), "Cannot create output trace");
  stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * 4));
  stream.close();
  require(bool(stream), "Trace write failed");
}

struct Graph {
  Ort::Session session;
  size_t count = 0;
  std::string kind;
  std::vector<std::vector<float>> in, out;
  std::vector<Ort::Value> iv, ov;
  std::vector<float> delivered;
  explicit Graph(Ort::Env& env, const char* path, const Ort::SessionOptions& options) : session(env, path, options) {
    count = session.GetInputCount();
    require((count == 5 || count == 6) && session.GetOutputCount() == count, "Expected four or five recurrent states");
    Ort::AllocatorWithDefaultOptions allocator;
    const auto type = session.GetOutputTypeInfo(0);
    const auto audio = type.GetTensorTypeAndShapeInfo().GetShape();
    require(audio.size() == 4 && audio[0] == 1 && (audio[1] == 1 || audio[1] == 4)
            && audio[2] == 2 && audio[3] == 128, "Wrong source/audio geometry");
    const auto historyType = session.GetInputTypeInfo(1);
    const auto history = historyType.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> expectedHistory, expectedLocal, expectedGlobal;
    if (count == 6) {
      require(audio[1] == 1, "The drums waveform specialist has one output");
      kind = "drums";
      expectedHistory = {1, 2, 896}; expectedLocal = {2, 20, 96}; expectedGlobal = {2, 1, 192};
    } else if (history == std::vector<int64_t>{1, 2, 3968}) {
      require(audio[1] == 1, "The bass specialist has one output");
      kind = "bass";
      expectedHistory = {1, 2, 3968}; expectedLocal = {2, 17, 80}; expectedGlobal = {2, 1, 160};
    } else {
      kind = audio[1] == 1 ? "vocals" : "joint";
      expectedHistory = {1, 2, 896}; expectedLocal = {2, 20, 96}; expectedGlobal = {2, 1, 192};
    }
    std::vector<std::vector<int64_t>> inputShapes{
      {1, 2, 128}, expectedHistory, expectedLocal, expectedGlobal, {1, audio[1], 2, 128}};
    if (count == 6) inputShapes.push_back({1, 1, 2, 128});
    auto outputShapes = inputShapes;
    outputShapes[0] = audio;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    in.resize(count); out.resize(count); iv.reserve(count); ov.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto inputName = session.GetInputNameAllocated(i, allocator);
      auto outputName = session.GetOutputNameAllocated(i, allocator);
      auto it = session.GetInputTypeInfo(i), ot = session.GetOutputTypeInfo(i);
      auto ii = it.GetTensorTypeAndShapeInfo(), oi = ot.GetTensorTypeAndShapeInfo();
      require(std::string(inputName.get()) == inputs[i] && std::string(outputName.get()) == outputs[i]
          && ii.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
          && oi.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
          && ii.GetShape() == inputShapes[i] && oi.GetShape() == outputShapes[i], "Selected architecture ABI differs");
      in[i].resize(elements(inputShapes[i])); out[i].resize(elements(outputShapes[i]));
      iv.push_back(Ort::Value::CreateTensor<float>(memory, in[i].data(), in[i].size(), inputShapes[i].data(), inputShapes[i].size()));
      ov.push_back(Ort::Value::CreateTensor<float>(memory, out[i].data(), out[i].size(), outputShapes[i].data(), outputShapes[i].size()));
    }
    delivered.resize(out[0].size());
  }
  double run(const float* audio, const Ort::RunOptions& options) {
    std::copy_n(audio, 256, in[0].data());
    const auto began = Clock::now();
    session.Run(options, inputs.data(), iv.data(), count, outputs.data(), ov.data(), count);
    const auto ended = Clock::now();
    for (size_t i = 1; i < count; ++i) std::copy(out[i].begin(), out[i].end(), in[i].begin());
    std::copy(out[0].begin(), out[0].end(), delivered.begin());
    return ms(ended - began);
  }
  void check() {
    for (size_t i = 0; i < count; ++i) {
      require(ov[i].GetTensorMutableData<float>() == out[i].data(), "ORT replaced an output buffer");
      for (float v : out[i]) require(std::isfinite(v), "Nonfinite output/state");
    }
  }
  size_t snapshotSize() const {
    size_t n = 0; for (const auto& tensor : out) n += tensor.size(); return n;
  }
  float* snapshot(float* destination) const {
    for (const auto& tensor : out) destination = std::copy(tensor.begin(), tensor.end(), destination);
    return destination;
  }
};

} // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 8 && argc <= 10, "Usage: benchmark INPUT.f32 WARMUP MEASURED check|measure|check_combined|measure_combined samples.csv final.bin MODEL [MODEL MODEL]");
    require(std::endian::native == std::endian::little && sizeof(float) == 4, "Require little-endian FP32");
    const int warmup = std::stoi(argv[2]), measured = std::stoi(argv[3]);
    const std::string action = argv[4];
    const bool combined = action == "check_combined" || action == "measure_combined";
    const bool check = action == "check" || action == "check_combined";
    require(check || action == "measure" || action == "measure_combined", "Unknown action");
    require(warmup >= 16 && warmup <= 4096 && measured >= 64 && measured <= 100000, "Invalid bounded hop counts");
    require(!check || warmup + measured <= 4096, "Parity trace too large");
    require(std::string(OrtGetApiBase()->GetVersionString()) == "1.26.0", "Require pinned ORT 1.26.0");
    disableDenormals(); Clock::initialize(); scheduling::Scheduler::configureQos();
    Dl_info runtime{};
    require(dladdr(dlsym(RTLD_DEFAULT, "OrtGetApiBase"), &runtime) != 0 && runtime.dli_fname, "Cannot identify loaded runtime");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "independent-stem-architecture-probe");
    Ort::SessionOptions options;
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    options.AddConfigEntry("mlas.disable_kleidiai", "1");
    std::vector<std::unique_ptr<Graph>> graphs;
    for (int i = 7; i < argc; ++i) graphs.push_back(std::make_unique<Graph>(env, argv[i], options));
    require(combined ? graphs.size() == 3 : graphs.size() == 1, "Declare one graph or the DBV assembly explicitly");
    if (combined) require(graphs[0]->kind == "drums" && graphs[1]->kind == "bass" && graphs[2]->kind == "vocals",
                          "Combined order must be the selected drums, bass, vocal architectures");
    std::array<float, 256> physicalHistory{}, other{};
    const size_t hops = static_cast<size_t>(warmup + measured);
    std::vector<float> audio(hops * 256);
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "Cannot open synthetic input");
    input.read(reinterpret_cast<char*>(audio.data()), static_cast<std::streamsize>(audio.size() * 4));
    require(bool(input), "Synthetic input is too short");
    size_t snapshotSize = combined ? 512 : 0;
    for (const auto& graph : graphs) snapshotSize += graph->snapshotSize();
    std::vector<float> snapshot(snapshotSize), trace(check ? hops * snapshotSize : 0);
    std::vector<double> runTimes(measured), loopTimes(measured), cpuTimes(measured), lateness(measured), completion(measured), budgets(measured);
    uint64_t misses = 0, overBudget = 0;
    Ort::RunOptions runOptions;
    scheduling::Scheduler scheduler("qos");
    scheduler.start();
    const auto epoch = Clock::now() + std::chrono::milliseconds(10);
    for (size_t hop = 0; hop < hops; ++hop) {
      const auto due = scheduling::arrival(epoch, hop), deadline = scheduling::arrival(epoch, hop + 1);
      scheduling::waitUntil(due);
      const double cpuBegan = threadSeconds();
      const auto began = Clock::now();
      double runMs = 0;
      for (auto& graph : graphs) runMs += graph->run(audio.data() + hop * 256, runOptions);
      if (combined) {
        for (size_t i = 0; i < 256; ++i)
          other[i] = physicalHistory[i] - ((graphs[0]->delivered[i] + graphs[1]->delivered[i]) + graphs[2]->delivered[i]);
        std::copy_n(audio.data() + hop * 256, 256, physicalHistory.data());
      }
      const auto ended = Clock::now();
      const double cpuEnded = threadSeconds();
      if (check || hop < static_cast<size_t>(warmup) || hop == hops - 1)
        for (auto& graph : graphs) graph->check();
      if (check) {
        float* next = trace.data() + hop * snapshotSize;
        for (const auto& graph : graphs) next = graph->snapshot(next);
        if (combined) {
          next = std::copy(other.begin(), other.end(), next);
          std::copy(physicalHistory.begin(), physicalHistory.end(), next);
        }
      }
      if (hop >= static_cast<size_t>(warmup)) {
        const auto i = hop - static_cast<size_t>(warmup);
        runTimes[i] = runMs; loopTimes[i] = ms(ended - began); cpuTimes[i] = 1000 * (cpuEnded - cpuBegan);
        lateness[i] = ms(began - due); completion[i] = ms(ended - due); budgets[i] = ms(deadline - due);
        misses += ended > deadline; overBudget += ended - began > deadline - due;
      }
    }
    scheduler.stop();
    float* next = snapshot.data();
    for (const auto& graph : graphs) next = graph->snapshot(next);
    if (combined) {
      for (float value : other) require(std::isfinite(value), "Nonfinite residual Other");
      next = std::copy(other.begin(), other.end(), next);
      std::copy(physicalHistory.begin(), physicalHistory.end(), next);
    }
    writeFloats(argv[6], snapshot);
    if (check) writeFloats(std::string(argv[6]) + ".trace", trace);
    std::ofstream samples(argv[5]);
    samples.precision(12);
    samples << "hop,run_wall_ms,loop_wall_ms,thread_cpu_ms,start_lateness_ms,completion_from_arrival_ms,hop_budget_ms\n";
    for (int i = 0; i < measured; ++i)
      samples << i << ',' << runTimes[i] << ',' << loopTimes[i] << ',' << cpuTimes[i] << ',' << lateness[i] << ',' << completion[i] << ',' << budgets[i] << '\n';
    samples.close(); require(bool(samples), "Cannot write samples");
    std::cout.precision(12);
    std::cout << "{\"status\":\"completed\",\"paced\":true,\"condition\":\"qos\",\"graphs\":" << graphs.size()
      << ",\"warmup_hops\":" << warmup << ",\"measured_hops\":" << measured << ",\"parity_trace\":" << (check ? "true" : "false")
      << ",\"physical_residual_other\":" << (combined ? "true" : "false")
      << ",\"graph_alignment_samples\":128,\"host_queue_measured\":false"
      << ",\"runtime_platform\":" << std::quoted(kPlatform)
      << ",\"onnxruntime_version\":\"1.26.0\",\"loaded_runtime\":" << std::quoted(runtime.dli_fname)
      << ",\"intra_op_threads\":1,\"inter_op_threads\":1,\"spinning\":false,\"kleidiai\":false"
      << ",\"execution\":\"sequential\",\"provider\":\"CPU\",\"preallocated_tensors\":true"
      << ",\"initial_default_policy\":" << (scheduler.initial.isDefault ? "true" : "false")
      << ",\"final_default_policy\":" << (scheduler.final.isDefault ? "true" : "false")
      << ",\"mach_timebase_numer\":" << Clock::numer << ",\"mach_timebase_denom\":" << Clock::denom
      << ",\"deadline_misses\":" << misses << ",\"compute_over_budget\":" << overBudget << ",\"run\":";
    stats(runTimes); std::cout << ",\"loop\":"; stats(loopTimes); std::cout << ",\"thread_cpu\":"; stats(cpuTimes);
    std::cout << ",\"start_lateness\":"; stats(lateness); std::cout << ",\"completion\":"; stats(completion);
    std::cout << ",\"quality_measured\":false,\"plugin_deadlines_qualified\":false}\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
