#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
    // num_threads == 0 -> auto-detect, capped at 8.
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return workers_.size(); }

    // Wake worker threads once for an active team region.
    void enter_team();

    // Put worker threads back to sleep after the team region.
    void leave_team();

    // All team threads (workers + leader) meet on a two-phase barrier.
    template <typename F>
    void parallel_for(int begin, int end, F&& f);

private:
    void worker_loop(size_t worker_id);

    struct alignas(64) Barrier {
        std::mutex mtx;
        std::condition_variable cv;
        size_t count = 0;        // arrivals in the current phase
        size_t generation = 0;   // completed barrier phase
        size_t expected = 0;

        Barrier() = default;
        explicit Barrier(size_t n) : expected(n) {}

        void set_expected(size_t n) { expected = n; }

        void arrive_and_wait() {
            std::unique_lock<std::mutex> lock(mtx);
            size_t gen = generation;
            if (++count == expected) {
                count = 0;
                ++generation;
                cv.notify_all();
            } else {
                cv.wait(lock, [this, gen] { return generation != gen; });
            }
        }
    };

    struct WorkItem {
        int begin = 0;
        int end = 0;
        std::function<void(int, int)> fn;
    };

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_team_;
    std::condition_variable cv_work_;
    bool stop_ = false;
    std::atomic<bool> team_active_{false};

    std::atomic<size_t> work_gen_{0};

    Barrier barrier_;
    WorkItem work_;
    inline static thread_local int tls_team_id_ = -1;
    inline static thread_local size_t tls_work_gen_ = 0;
};

inline ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        unsigned hc = std::thread::hardware_concurrency();
        num_threads = std::max(1u, std::min(hc, 8u));
    }
    barrier_.set_expected(num_threads + 1);
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        team_active_ = false;
    }
    cv_team_.notify_all();
    cv_work_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

inline void ThreadPool::worker_loop(size_t worker_id) {
    tls_team_id_ = static_cast<int>(worker_id);
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_team_.wait(lock, [this] { return stop_ || team_active_.load(std::memory_order_acquire); });
        if (stop_) return;
        lock.unlock();

        while (true) {
            std::unique_lock<std::mutex> wlock(mutex_);
            cv_work_.wait(wlock, [this] { return stop_ || !team_active_.load(std::memory_order_acquire) || work_gen_.load(std::memory_order_acquire) != tls_work_gen_; });
            if (stop_) return;
            if (!team_active_.load(std::memory_order_acquire)) break;
            tls_work_gen_ = work_gen_.load(std::memory_order_acquire);
            wlock.unlock();

            barrier_.arrive_and_wait(); // start of a parallel_for
            if (!team_active_.load(std::memory_order_acquire)) break;

            WorkItem& w = work_;
            int total = w.end - w.begin;
            if (total > 0) {
                size_t team_size = barrier_.expected;
                int chunk = (total + (int)team_size - 1) / (int)team_size;
                if (chunk <= 0) chunk = 1;
                int s = w.begin + tls_team_id_ * chunk;
                int e = std::min(s + chunk, w.end);
                if (s < e && w.fn) w.fn(s, e);
            }

            barrier_.arrive_and_wait(); // end of a parallel_for
        }
    }
}

inline void ThreadPool::enter_team() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        team_active_.store(true, std::memory_order_release);
    }
    cv_team_.notify_all();
}

inline void ThreadPool::leave_team() {
    std::unique_lock<std::mutex> lock(mutex_);
    team_active_.store(false, std::memory_order_release);
    lock.unlock();
    cv_team_.notify_all();
    cv_work_.notify_all();
}

template <typename F>
inline void ThreadPool::parallel_for(int begin, int end, F&& f) {
    if (!team_active_.load(std::memory_order_acquire)) {
        enter_team();
        parallel_for(begin, end, std::forward<F>(f));
        leave_team();
        return;
    }

    tls_team_id_ = static_cast<int>(workers_.size());

    {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t next_gen = work_gen_.fetch_add(1, std::memory_order_acq_rel) + 1;
        work_.begin = begin;
        work_.end = end;
        work_.fn = std::forward<F>(f);
        tls_work_gen_ = next_gen;
    }
    cv_work_.notify_all();

    barrier_.arrive_and_wait(); // start

    WorkItem& w = work_;
    int total = w.end - w.begin;
    if (total > 0) {
        size_t team_size = barrier_.expected;
        int chunk = (total + (int)team_size - 1) / (int)team_size;
        if (chunk <= 0) chunk = 1;
        int s = w.begin + tls_team_id_ * chunk;
        int e = std::min(s + chunk, w.end);
        if (s < e && w.fn) w.fn(s, e);
    }

    barrier_.arrive_and_wait(); // end
}
