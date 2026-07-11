//! Profiling shim: Tracy zones behind the `tracy` cargo feature. Call sites
//! never see cfg — `crate::zone!` / `crate::plot!` and the functions here
//! compile to nothing when the feature is off (the tracy-client dependency
//! is not even built then).
//!
//! Placement rule (see CLAUDE.md, Profiling): frame-phase granularity only —
//! never per-tile or per-pixel (tens of thousands of zones per frame would
//! themselves be ms-class). The breakdown INSIDE a trace zone comes from
//! Tracy's statistical sampling, which is why release builds carry line
//! tables.

/// Start the Tracy client and name the threads. Call once, first thing in
/// main — `span!` requires a running client. No-op without the feature.
pub fn init() {
    #[cfg(feature = "tracy")]
    {
        // Leak one client handle for the process lifetime (refcounted; the
        // profiler must outlive every zone on every thread).
        std::mem::forget(tracy_client::Client::start());
        tracy_client::set_thread_name!("main");
        // Build the global rayon pool ourselves so workers get names in the
        // GUI. Errors only if something already built it — then workers just
        // show unnamed, which is fine.
        let _ = rayon::ThreadPoolBuilder::new()
            .start_handler(|_i| tracy_client::set_thread_name!("rayon"))
            .build_global();
    }
}

/// End-of-frame marker (once per presented frame).
pub fn frame_mark() {
    #[cfg(feature = "tracy")]
    if let Some(c) = tracy_client::Client::running() {
        c.frame_mark();
    }
}

/// Scoped zone: `crate::zone!("name");` opens a span that closes at the end
/// of the enclosing block.
#[macro_export]
macro_rules! zone {
    ($name:literal) => {
        #[cfg(feature = "tracy")]
        let _prof_zone = ::tracy_client::span!($name);
    };
}

/// Plot a per-frame value: `crate::plot!("frame ms", ms);`
#[macro_export]
macro_rules! plot {
    ($name:literal, $v:expr) => {{
        #[cfg(feature = "tracy")]
        if let Some(c) = ::tracy_client::Client::running() {
            c.plot(::tracy_client::plot_name!($name), $v as f64);
        }
        #[cfg(not(feature = "tracy"))]
        let _ = &$v;
    }};
}
