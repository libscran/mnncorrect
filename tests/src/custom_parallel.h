#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H
#ifdef TEST_CUSTOM_PARALLEL

#include <cmath>
#include <vector>
#include <thread>

template<class Function>
void default_parallelize(int nthreads, size_t n, Function f) {
    size_t jobs_per_worker = n / nthreads + (n % nthreads > 0);
    size_t start = 0;
    std::vector<std::thread> jobs;
    jobs.reserve(nthreads);

    for (int w = 0; w < nthreads && start < n; ++w) {
        size_t length = std::min(n - start, jobs_per_worker);
        jobs.emplace_back(f, w, start, length);
        start += length;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define MNNCORRECT_CUSTOM_PARALLEL default_parallelize
#endif
#endif
