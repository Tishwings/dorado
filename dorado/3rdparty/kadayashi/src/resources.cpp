#include "resources.h"

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__) || defined(_POSIX_VERSION)
#include <stddef.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace kadayashi {

bool KDY_VERBOSE = false;

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__) || defined(_POSIX_VERSION)
double Get_T(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

double Get_U(void) {
    struct rusage s;
    getrusage(RUSAGE_SELF, &s);
    return (double)s.ru_maxrss / 1048576.0;  // GB
}

#else
double Get_T(void) { return 0.0f; }
double Get_U(void) { return 0.0f; }
#endif

}  // namespace kadayashi
