#ifndef __KIT_TIMER_H__
#define __KIT_TIMER_H__

#include <chrono>

namespace kitsune
{
  class timer {
    public:
      inline timer() { reset(); }
      inline ~timer() = default;
      
      inline void reset() {
	using namespace std::chrono;
	stamp = high_resolution_clock::now();
      };

      inline double seconds() const {
	using namespace std::chrono;
	high_resolution_clock::time_point now;
	now = high_resolution_clock::now();
	return duration_cast<duration<double>>(now - stamp).count();
      }

    private:
      timer(const timer&);
      timer& operator=(const timer&);
      std::chrono::high_resolution_clock::time_point stamp;
  };
}

#endif

