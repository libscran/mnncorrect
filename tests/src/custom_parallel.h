#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H
#ifdef TEST_CUSTOM_PARALLEL

#include <cmath>
#include <vector>
#include <thread>

template<class Function>
void parallelize(size_t n, Function f, size_t nthreads) {
    size_t jobs_per_worker = std::ceil(static_cast<double>(n) / nthreads);
    size_t start = 0;
    std::vector<std::thread> jobs;
    jobs.reserve(nthreads);

    for (size_t w = 0; w < nthreads; ++w) {
        if (start == n) {
            break;
        }
        size_t length = std::min(n - start, jobs_per_worker);
        jobs.emplace_back(f, start, length);
        start += length;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define MNNCORRECT_CUSTOM_PARALLEL parallelize
#endif
#endif
