//
// Created by hutao on 19-4-2.
//
#pragma once

#include "real.h"
#include <algorithm>
#include <fstream>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define FASTTEXTMINE_DEPRECATED(msg) __attribute_((__deprecated__(msg)))
#elif defined(_MSC_VER)
#define FASTTEXTMINE_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define FASTTEXTMINE_DEPRECATED(msg)
#endif

namespace fasttext{

    using Predictions = std::vector<std::pair<real, int32_t >>;

    namespace utils{
        int64_t size(std::ifstream&);
        void seek(std::ifstream&, int64_t);

        template <typename T>
        bool contains(const std::vector<T>& container, const T& value){
            return std::find(container.begin(), container.end(), value) != container.end();
        }
    }
}
