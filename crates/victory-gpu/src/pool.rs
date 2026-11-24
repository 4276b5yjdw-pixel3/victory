//! A few worker threads made once, for big copies and compares of a
//! submission. Threads made and ended per call would each run the title
//! image's thread-attach and detach callbacks (the title is a DLL of this
//! process), which crashed the title; these never end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    jobs: Mutex<Sender<Job>>,
}

/// Threads in the pool: a third of the machine's, at most 8, at least 1.
pub(crate) fn workers() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get() / 3)
            .clamp(1, 8)
    })
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        let rx: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(rx));
        for i in 0..workers() {
            let rx = rx.clone();
            let _ = std::thread::Builder::new()
                .name(format!("victory-copy-{i}"))
                .spawn(move || {
                    loop {
                        let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        match job {
                            Ok(job) => job(),
                            Err(_) => return,
                        }
                    }
                });
        }
        Pool {
            jobs: Mutex::new(tx),
        }
    })
}

/// Makes the threads now (at startup, not in the middle of a frame).
pub(crate) fn start() {
    pool();
}

/// Runs `tasks` on the pool (the first on the calling thread) and returns
/// when all are done. A task that panics makes this panic after the others
/// finish.
pub(crate) fn run_all<'a>(mut tasks: Vec<Box<dyn FnOnce() + Send + 'a>>) {
    if tasks.is_empty() {
        return;
    }
    let first = tasks.remove(0);
    if tasks.is_empty() {
        return first();
    }
    let pending = Arc::new((Mutex::new(tasks.len()), Condvar::new()));
    let failed = Arc::new(AtomicBool::new(false));
    {
        let sender = pool().jobs.lock().unwrap_or_else(|e| e.into_inner());
        for task in tasks {
            // SAFETY: this function waits below for every task it sent, so
            // what the task borrows outlives its run.
            let task: Job = unsafe {
                std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Job>(task)
            };
            let (pending, failed) = (pending.clone(), failed.clone());
            let job: Job = Box::new(move || {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
                let (count, done) = &*pending;
                *count.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
                done.notify_all();
            });
            // The pool keeps its receiver for the life of the process.
            let _ = sender.send(job);
        }
    }
    let own = std::panic::catch_unwind(std::panic::AssertUnwindSafe(first));
    let (count, done) = &*pending;
    let mut left = count.lock().unwrap_or_else(|e| e.into_inner());
    while *left > 0 {
        left = done.wait(left).unwrap_or_else(|e| e.into_inner());
    }
    drop(left);
    assert!(own.is_ok() && !failed.load(Ordering::Relaxed), "a copy task panicked");
}
