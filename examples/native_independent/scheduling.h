#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <os/workgroup.h>
#include <os/object.h>
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace scheduling {
inline void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Use one clock for pacing, measurements and workgroup timestamps. A Mach tick
// is not necessarily a nanosecond. Conversion rounds down by less than one tick.
struct Clock {
  using rep = int64_t;
  using period = std::nano;
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<Clock>;
  static constexpr bool is_steady = true;
  static inline uint32_t numer = 1, denom = 1;

  static void initialize() {
#if defined(__APPLE__)
    mach_timebase_info_data_t info{};
    check(mach_timebase_info(&info) == KERN_SUCCESS && info.numer && info.denom,
          "mach_timebase_info failed");
    numer = info.numer;
    denom = info.denom;
#endif
  }

  static uint64_t ticks(uint64_t ns) {
    return static_cast<uint64_t>(static_cast<__uint128_t>(ns) * denom / numer);
  }

  static time_point now() noexcept {
#if defined(__APPLE__)
    const auto ns = static_cast<uint64_t>(
        static_cast<__uint128_t>(mach_absolute_time()) * numer / denom);
    return time_point(duration(static_cast<int64_t>(ns)));
#else
    return time_point(std::chrono::duration_cast<duration>(
        std::chrono::steady_clock::now().time_since_epoch()));
#endif
  }
};

inline Clock::time_point arrival(Clock::time_point epoch, uint64_t hop) {
  // Calculate from the origin, avoiding accumulated rounding drift.
  const auto ns = static_cast<uint64_t>(
      static_cast<__uint128_t>(hop) * 128 * 1000000000 / 44100);
  return epoch + Clock::duration(ns);
}

inline void waitUntil(Clock::time_point due) {
  // Identical timer mechanism for all three conditions. No spin or timer tuning.
  std::this_thread::sleep_until(due);
}

struct Policy {
  uint32_t period = 0, computation = 0, constraint = 0;
  bool preemptible = false, isDefault = true;
};

class Scheduler {
 public:
  explicit Scheduler(std::string mode) : mode_(std::move(mode)) {
    check(mode_ == "qos" || mode_ == "realtime" || mode_ == "workgroup",
          "Unknown scheduling mode");
  }
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  ~Scheduler() { cleanup(); }

  static void configureQos() {
#if defined(__APPLE__)
    check(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0,
          "Cannot apply user-interactive QoS");
    qos_class_t observed{};
    int relative = 0;
    check(pthread_get_qos_class_np(pthread_self(), &observed, &relative) == 0 &&
              observed == QOS_CLASS_USER_INTERACTIVE,
          "User-interactive QoS readback failed");
#endif
  }

  void start() {
#if defined(__APPLE__)
    if (mode_ == "workgroup") {
      group_ = AudioWorkIntervalCreate("StemgenRT scheduling experiment",
                                       OS_CLOCK_MACH_ABSOLUTE_TIME, nullptr);
      check(group_ != nullptr, "AudioWorkIntervalCreate failed");
    }
    if (mode_ != "qos") {
      const auto period = Clock::ticks(128ULL * 1000000000 / 44100);
      const auto computation = Clock::ticks(2000000);  // Frozen 2 ms CPU estimate.
      check(period <= std::numeric_limits<uint32_t>::max(), "Policy tick overflow");
      thread_time_constraint_policy_data_t policy{};
      policy.period = static_cast<uint32_t>(period);
      policy.computation = static_cast<uint32_t>(computation);
      policy.constraint = policy.period;
      policy.preemptible = TRUE;
      check(thread_policy_set(pthread_mach_thread_np(pthread_self()),
                              THREAD_TIME_CONSTRAINT_POLICY,
                              reinterpret_cast<thread_policy_t>(&policy),
                              THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS,
            "Cannot apply Mach real-time policy");
      realtime_ = true;
    }
    if (group_) {
      check(os_workgroup_join(group_, &token_) == 0, "Audio workgroup join failed");
      joined_ = true;
    }
#else
    check(mode_ == "qos", "Real-time/workgroup modes require macOS");
#endif
    initial = readPolicy();
    verifyPolicy(initial);
  }

  void begin(Clock::time_point due, Clock::time_point deadline) {
#if defined(__APPLE__)
    if (group_) {
      // Preserve the original due/deadline even if both are already in the past.
      // Replacing deadline with now + period would hide the misses under test.
      check(os_workgroup_interval_start(
                group_, Clock::ticks(due.time_since_epoch().count()),
                Clock::ticks(deadline.time_since_epoch().count()), nullptr) == 0,
            "Audio workgroup interval start failed");
      active_ = true;
      ++intervalStarts;
    }
#else
    (void)due;
    (void)deadline;
#endif
  }

  void finish() {
#if defined(__APPLE__)
    if (active_) {
      check(os_workgroup_interval_finish(group_, nullptr) == 0,
            "Audio workgroup interval finish failed");
      active_ = false;
      ++intervalFinishes;
    }
#endif
  }

  void stop() {
    final = readPolicy();
    verifyPolicy(final);
    cleanup();
    check(restoreSucceeded_, "Could not restore standard thread scheduling");
    configureQos();
  }

  Policy initial{}, final{};
  uint64_t intervalStarts = 0, intervalFinishes = 0;

 private:
  Policy readPolicy() const {
    Policy result{};
#if defined(__APPLE__)
    thread_time_constraint_policy_data_t policy{};
    mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
    boolean_t getDefault = FALSE;
    check(thread_policy_get(pthread_mach_thread_np(pthread_self()),
                            THREAD_TIME_CONSTRAINT_POLICY,
                            reinterpret_cast<thread_policy_t>(&policy), &count,
                            &getDefault) == KERN_SUCCESS &&
              count == THREAD_TIME_CONSTRAINT_POLICY_COUNT,
          "Mach scheduling policy readback failed");
    result = {policy.period, policy.computation, policy.constraint,
              policy.preemptible != FALSE, getDefault != FALSE};
#endif
    return result;
  }

  void verifyPolicy(const Policy& policy) const {
#if defined(__APPLE__)
    if (mode_ != "qos") {
      check(!policy.isDefault && policy.preemptible &&
                policy.period == Clock::ticks(128ULL * 1000000000 / 44100) &&
                policy.constraint == policy.period &&
                policy.computation == Clock::ticks(2000000),
            "Real-time scheduling policy readback mismatch");
    } else {
      check(policy.isDefault, "QoS control unexpectedly has real-time policy");
    }
#else
    (void)policy;
#endif
  }

  void cleanup() noexcept {
#if defined(__APPLE__)
    if (active_) {
      const int result = os_workgroup_interval_finish(group_, nullptr);
      (void)result;
      active_ = false;
    }
    if (joined_) {
      os_workgroup_leave(group_, &token_);
      joined_ = false;
    }
    if (group_) {
      os_release(group_);
      group_ = nullptr;
    }
    if (realtime_) {
      restoreSucceeded_ = thread_policy_set(
          pthread_mach_thread_np(pthread_self()), THREAD_STANDARD_POLICY,
          nullptr, THREAD_STANDARD_POLICY_COUNT) == KERN_SUCCESS;
      realtime_ = false;
    }
#endif
  }

  std::string mode_;
  bool restoreSucceeded_ = true;
#if defined(__APPLE__)
  os_workgroup_interval_t group_ = nullptr;
  os_workgroup_join_token_s token_{};
  bool joined_ = false, active_ = false, realtime_ = false;
#endif
};
}  // namespace scheduling
