// Actual macOS API regression for cleanup after a Mach real-time policy.
// This checks lifecycle correctness, not model speed or target-Mac deadlines.
#include "scheduling.h"

#include <exception>
#include <iostream>

int main() {
  try {
    scheduling::Clock::initialize();
#if defined(__APPLE__)
    const char* modes[] = {"qos", "realtime", "workgroup", "qos"};
#else
    const char* modes[] = {"qos"};
#endif
    for (const char* mode : modes) {
      std::exception_ptr failure;
      std::thread worker([&] {
        try {
          scheduling::Scheduler::configureQos();
          scheduling::Scheduler scheduler(mode, 2700000);
          scheduler.start();
          const auto due = scheduling::Clock::now();
          scheduler.begin(due, scheduling::arrival(due, 1));
          scheduler.finish();
          scheduler.stop();
          const bool control = std::string(mode) == "qos";
          scheduling::check(scheduler.initial.isDefault == control &&
                                scheduler.final.isDefault == control,
                            "Measured policy readback differs");
          const uint64_t intervals = std::string(mode) == "workgroup" ? 1 : 0;
          scheduling::check(scheduler.intervalStarts == intervals &&
                                scheduler.intervalFinishes == intervals,
                            "Workgroup interval was not closed");
        } catch (...) {
          failure = std::current_exception();
        }
      });
      worker.join();
      if (failure) std::rethrow_exception(failure);
      std::cout << mode << ": initial/final policy and standard-policy cleanup passed\n";
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
