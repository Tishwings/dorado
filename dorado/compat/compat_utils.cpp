#include "compat/compat_utils.h"

// For some reason AppleClang 2100.0.123.102 confuses ninja when this file is compiled, leading
// to it being rebuilt and hence relinked into everything. Presumably this is because it's
// completely empty after preprocessing. Adding this comment "fixes" the issue.

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <ctime>
#include <iomanip>
#include <sstream>

namespace dorado::utils {
char* strptime(const char* s, const char* f, tm* tm) {
    std::istringstream input(s);
    input.imbue(std::locale(setlocale(LC_ALL, nullptr)));
    input >> std::get_time(tm, f);
    if (input.fail()) {
        return nullptr;
    }
    return (char*)(s + input.tellg());
}

int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite) {
        size_t envsize = 0;
        int errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) {
            return errcode;
        }
    }
    return _putenv_s(name, value);
}

}  // namespace dorado::utils
#endif  // _WIN32
