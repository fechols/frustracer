//! Audio ambience — per-island biome loops + a speed-scaled procedural wind.
//!
//! Three sound sources, one SDL3 callback stream mixing them all:
//! - WORLD mode: one CC0 ambience loop per curated island (`AMBIENCE`, keyed
//!   by the `world::CURATED` island name), crossfaded by camera proximity
//!   with the inverse-square weight the TOD attractors use — but over FULL 3D
//!   distance, deliberately unlike `flycam::attractor_hour`'s x/z-only weight
//!   (climbing away from an island should fade its sounds; the clock has no
//!   business caring about altitude, the ambience does).
//! - A directly-loaded curated scene (`san-miguel.obj` &c) plays its own
//!   loop at a steady gain (`match_scene_path` maps the resolved positional
//!   arg back to a curated name; anything unmatched is wind-only).
//! - WIND: synthesized (hash-noise through a one-pole low-pass), no asset —
//!   gain AND filter cutoff track the real camera speed, published by the
//!   500 Hz flycam thread as an atomic so the swish stays responsive while
//!   the main thread is blocked in a 100+ ms trace.
//!
//! DISPLAY-ONLY in the strongest sense: nothing here touches the render
//! path, any buffer a gate reads, or any rng stream — the same-seed/replay/
//! VisCtl contracts are structurally untouched. The device half lives only
//! in `run_window` sessions, so every `--check*`/`--spin` is silent, DLL-free
//! (lewton and sdl3 are static crates), and byte-identical by construction.
//! `--no-audio` is the kill lever (the subsystem is never initialized).
//!
//! Failure modes are all one loud line + silence, never an error: no audio
//! device -> `AudioSys::new` returns None; a missing/corrupt OGG -> that
//! island silent (the world-mode missing-island degrade).
//!
//! Follow-on (documented, not v1): TOD modulation of the loops — rungholt's
//! birds should hand over to crickets after dusk the way the fireflies fade
//! in; today a loop plays the same at every hour.

use glam::Vec3A;

/// Mixer output rate. SDL converts to whatever the device wants; fixing ours
/// keeps every constant below (smoothing coefficients, seam length, cutoff
/// coefficients) a function of one number.
pub const MIX_RATE: u32 = 48_000;
/// Loop-seam crossfade length, milliseconds (equal-power, baked at load).
const SEAM_MS: u32 = 50;
/// Master gains. Ambience + wind at full tilt sum comfortably under the
/// soft-clip knee, so the limiter is a guard, not a tone control.
const AMB_MASTER: f32 = 0.6;
/// Retuned twice on user feedback: 0.35 -> 0.22 ("too loud", together with
/// the 2-pole/lower-cutoff retune below), then 0.22 -> 0.12 ("even more
/// quiet") — the wind is a presence cue under the ambience, not a voice.
const WIND_MASTER: f32 = 0.12;
/// Gain-smoothing time constant, seconds (~150 ms: fast enough to track a
/// flight, slow enough that the per-frame gain updates never zipper).
const SMOOTH_TC_S: f32 = 0.15;
/// Wind low-pass cutoff range: idle-adjacent rumble to full-speed whoosh.
/// Retuned 300->4000 down to 150->1200 ("too harsh"): wind heard from a
/// moving viewpoint is low-mid energy — 4 kHz noise reads as hiss/static,
/// not airflow. The steepness fix rides WIND_LP_STAGES. Then 150->1200
/// down to 100->600 ("a bit lower frequency"): the top end is what full
/// flight speed sounds like, so the max is the knob — an octave down
/// turns swish into swoosh.
const WIND_FC_MIN: f32 = 100.0;
const WIND_FC_MAX: f32 = 600.0;
/// Cascaded one-pole stages per channel. TWO, not one: a single pole falls
/// only −6 dB/octave, which leaves enough top-octave noise past any cutoff
/// to read as harsh static; the cascade's −12 dB/octave is where the sound
/// turns from "detuned radio" into "air". (The other half of the same
/// complaint is the cutoff range above — tune them together.)
const WIND_LP_STAGES: usize = 2;
/// Loudness-normalization target for the loops (linear RMS, ~-22 dBFS) —
/// downloaded assets arrive at wildly different levels; normalizing at load
/// beats hand-tuning a per-file gain table.
const TARGET_RMS: f32 = 0.08;
/// Soft-clip knee: identity below, bounded tanh tail above (the tone.rs
/// curve philosophy — C1 at the knee, asymptotic to 1.0).
const CLIP_KNEE: f32 = 0.75;
/// Gain flush-to-zero threshold (−120 dB). An exponential smoother never
/// reaches 0.0 in f32 — the decay stalls at a denormal (~5e-42, where the
/// per-sample decrement drops below half the denormal spacing) — so without
/// a flush the exact-silence fast paths latch only in a session that never
/// made a sound. A gain this far below audibility whose TARGET is exactly
/// 0.0 snaps to exact 0.0 instead: the fast paths re-latch and denormal
/// multiplies stay off the audio thread (the classic filter-tail issue).
const GAIN_FLUSH: f32 = 1e-6;

/// name -> committed loop path + loudness gain, keyed by `world::CURATED`
/// island name. A fresh checkout without `git lfs pull` (or before the
/// assets land) misses per file with one loud line — the world-mode
/// missing-island pattern.
///
/// The gain multiplies TARGET_RMS at load, AFTER the per-file RMS
/// equalization — file loudness is deliberately not a knob (normalize_rms
/// would undo it), so this column is the one place a loop can be balanced
/// against the others. Equal RMS is not equal loudness: helmet's bed is
/// drone/sub-heavy (most of its energy below ~110 Hz, where the ear is
/// least sensitive), so it reads well quieter than the bird/crowd loops at
/// the same RMS and gets a boost. normalize_rms's 0.95 peak guard bounds
/// any gain here — a too-hot entry lands loud, never clipped.
pub const AMBIENCE: &[(&str, &str, f32)] = &[
    ("powerplant", "scenes/audio/powerplant.ogg", 1.0), // factory/machinery hum
    ("sponza", "scenes/audio/sponza.ogg", 1.0),         // morning courtyard birds
    ("rungholt", "scenes/audio/rungholt.ogg", 1.0),     // late-morning bird chorus
    ("helmet", "scenes/audio/helmet.ogg", 1.8),         // synthesized sci-fi bed (tools/helmet-ambience)
    ("san-miguel", "scenes/audio/san-miguel.ogg", 1.0), // patio fountain + birds
    ("bistro", "scenes/audio/bistro.ogg", 1.0),         // cafe crowd chatter
    ("vokselia", "scenes/audio/vokselia.ogg", 1.0),     // night crickets
];

/// One spatial ambience source: an island's loop, positioned at its ring
/// center with its content half-footprint as the falloff scale (the same
/// radius the TOD attractor uses).
pub struct Cue {
    pub name: String,
    pub pos: Vec3A,
    pub radius: f32,
}

/// What the session should play. Built once in `main()` beside the TOD
/// attractors; `None` still plays wind (the kill lever is `--no-audio`,
/// which never constructs the device at all).
pub enum Cues {
    None,
    /// A directly-loaded curated scene: its loop at steady full gain.
    Steady(&'static str),
    /// World mode: one positioned cue per island present on disk.
    World(Vec<Cue>),
}

/// Map a resolved scene path back to a curated ambience name. Substring
/// match over a normalized (lowercase, forward-slash) path, so the
/// `.obj.zst` sibling resolution, `-low-poly` variants, bistro Interior,
/// and intel-sponza (the same atrium — same courtyard air) all land on the
/// right loop. Unmatched (procedural, stress, arbitrary OBJ) = wind only.
pub fn match_scene_path(path: &str) -> Option<&'static str> {
    let p = path.to_ascii_lowercase().replace('\\', "/");
    const KEYS: &[(&str, &str)] = &[
        ("powerplant", "powerplant"),
        ("sponza", "sponza"),
        ("rungholt", "rungholt"),
        ("damaged-helmet", "helmet"),
        ("damagedhelmet", "helmet"),
        ("san-miguel", "san-miguel"),
        ("bistro", "bistro"),
        ("vokselia", "vokselia"),
    ];
    KEYS.iter().find(|(k, _)| p.contains(k)).map(|&(_, n)| n)
}

/// Proximity gain: `r² / (d² + r²)` — exactly 1.0 at the island center,
/// exactly 0.5 at d = r, inverse-square beyond. The attractor weight's
/// shape, normalized to a gain. No sum-normalization across islands on
/// purpose: between two islands you hear both quietly, far outside the ring
/// everything fades and only wind remains.
pub fn proximity_gain(d2: f32, radius: f32) -> f32 {
    let r2 = (radius * radius).max(1e-6);
    r2 / (d2 + r2)
}

/// Wind gain from normalized speed (norm ∈ [0,1]; norm^1.5 — quiet creep
/// stays near-silent, the top end fills in fast). Exactly 0.0 when still:
/// a parked camera contributes digital silence, not a noise floor.
pub fn wind_gain(norm: f32) -> f32 {
    let n = norm.clamp(0.0, 1.0);
    n * n.sqrt()
}

/// Wind low-pass cutoff from normalized speed: log-lerp WIND_FC_MIN ->
/// WIND_FC_MAX, so speed brightens the swish as well as louder-ing it —
/// the audible cue that this is airflow, not a volume knob.
pub fn wind_cutoff(norm: f32) -> f32 {
    let n = norm.clamp(0.0, 1.0);
    WIND_FC_MIN * (WIND_FC_MAX / WIND_FC_MIN).powf(n)
}

/// One-pole low-pass coefficient for cutoff `fc` at sample rate `fs`:
/// `y += a·(x − y)`, `a = 1 − exp(−2π·fc/fs)` — a ∈ (0, 1) for any fc > 0.
pub fn onepole_coeff(fc: f32, fs: f32) -> f32 {
    1.0 - (-std::f32::consts::TAU * fc / fs).exp()
}

/// Bounded soft clip: identity through CLIP_KNEE, tanh tail asymptotic to
/// 1.0, slope exactly 1 at the knee (the tone.rs rolloff philosophy — a
/// break here rings on every transient). The normal path is bit-transparent:
/// |x| ≤ knee returns x verbatim.
pub fn soft_clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= CLIP_KNEE {
        x
    } else {
        let t = (a - CLIP_KNEE) / (1.0 - CLIP_KNEE);
        (CLIP_KNEE + (1.0 - CLIP_KNEE) * t.tanh()).copysign(x)
    }
}

/// Deterministic noise in [-1, 1): splitmix64 finalizer of the sample index
/// (the sky::pcg_mix / twinkle-is-f(frame-index) idiom — stateless, zero rng
/// draws, byte-reproducible).
pub fn hash_noise(i: u64) -> f32 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) * (2.0 / 16_777_216.0) - 1.0
}

/// Linear resample `src` (mono i16) from `src_rate` to `dst_rate`. Equal
/// rates return the input verbatim (bit-identical — the compatibility
/// contract, like every other off-state here). WRAP-aware: the final
/// stretch interpolates toward src[0], because every consumer is a LOOP —
/// an end-clamp would hold the last sample flat for up to one source step
/// (measured as an 81/16000 outlier on the self-test sine, 10× the
/// interpolation error everywhere else). Linear is plenty for ambience
/// beds; it is pure math the self-test can pin.
pub fn resample_linear(src: &[i16], src_rate: u32, dst_rate: u32) -> Vec<i16> {
    if src_rate == dst_rate || src.is_empty() {
        return src.to_vec();
    }
    let n_dst = ((src.len() as u64 * dst_rate as u64) / src_rate as u64) as usize;
    let step = src_rate as f64 / dst_rate as f64;
    (0..n_dst)
        .map(|j| {
            let x = j as f64 * step;
            let i = (x as usize).min(src.len() - 1);
            let frac = (x - i as f64) as f32;
            let a = src[i] as f32;
            let b = src[(i + 1) % src.len()] as f32;
            (a + (b - a) * frac).round() as i16
        })
        .collect()
}

/// Scale to TARGET_RMS with a peak guard (never past 0.95 of full scale) —
/// the per-file loudness equalizer.
pub fn normalize_rms(buf: &mut [i16], target_rms: f32) {
    if buf.is_empty() {
        return;
    }
    let mut sum = 0.0f64;
    let mut peak = 0i32;
    for &s in buf.iter() {
        let v = s as f64 / 32768.0;
        sum += v * v;
        peak = peak.max((s as i32).abs());
    }
    let rms = (sum / buf.len() as f64).sqrt() as f32;
    if rms <= 1e-6 {
        return; // silence stays silence
    }
    let mut k = target_rms / rms;
    let peak_f = peak as f32 / 32768.0;
    if peak_f * k > 0.95 {
        k = 0.95 / peak_f;
    }
    for s in buf.iter_mut() {
        *s = ((*s as f32) * k).round().clamp(-32768.0, 32767.0) as i16;
    }
}

/// Bake an equal-power crossfade of the loop's tail into its head, then
/// truncate the tail — the wrap becomes continuous by construction, so
/// playback needs no runtime fade logic. Weights: head' [i] =
/// sin(t·π/2)·head[i] + cos(t·π/2)·tail[i], t = i/(K−1) — at i = 0 the
/// sample IS the (old) first tail sample, continuous with the new last
/// sample; at i = K−1 the head is fully restored, continuous with buf[K].
pub fn bake_loop_seam(buf: &mut Vec<i16>, rate: u32) {
    let k = (rate as usize * SEAM_MS as usize) / 1000;
    if k < 2 || buf.len() < 3 * k {
        return; // too short to seam — play as-is
    }
    let n = buf.len();
    let tail: Vec<i16> = buf[n - k..].to_vec();
    for i in 0..k {
        let t = i as f32 / (k - 1) as f32;
        let (s, c) = (t * std::f32::consts::FRAC_PI_2).sin_cos();
        let v = s * buf[i] as f32 + c * tail[i] as f32;
        buf[i] = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    buf.truncate(n - k);
}

/// One in-memory loop: mono i16 at MIX_RATE, seam-baked. `pos` freezes while
/// the loop is inaudible (both gain and target exactly 0), so re-approaching
/// an island resumes where it left off instead of skipping ahead.
pub struct LoopBuf {
    pub samples: Vec<i16>,
    pub pos: usize,
}

/// The mixer core — pure math, no SDL types, so `self_test` can drive it
/// sample-exactly without a device. The `#[cfg(windows)]` SDL callback below
/// is a thin shell: load the atomics, call `render`, push the block.
pub struct MixCore {
    pub loops: Vec<LoopBuf>,
    /// Smoothed per-loop gains (targets come in per render call).
    g: Vec<f32>,
    /// Wind state: smoothed gain, smoothed cutoff, L/R cascaded one-pole
    /// filter stages (WIND_LP_STAGES per channel — see the const).
    wind_g: f32,
    wind_fc: f32,
    wind_lp: [[f32; WIND_LP_STAGES]; 2],
    sample_idx: u64,
    /// Scene diagonal — normalizes the flycam's world-units/s speed against
    /// the full-tilt flight speed (diag·0.25/s, flycam.rs).
    diag: f32,
}

impl MixCore {
    pub fn new(loops: Vec<LoopBuf>, diag: f32) -> Self {
        let n = loops.len();
        Self {
            loops,
            g: vec![0.0; n],
            wind_g: 0.0,
            wind_fc: WIND_FC_MIN,
            wind_lp: [[0.0; WIND_LP_STAGES]; 2],
            sample_idx: 0,
            diag: diag.max(1e-3),
        }
    }

    /// Render `out.len()/2` stereo frames (interleaved L R). `targets` are
    /// the per-loop gain targets in [0, 1] (proximity gains — AMB_MASTER is
    /// applied here); `wind_speed` is world units/s. Deterministic: a pure
    /// function of (state, targets, wind_speed).
    pub fn render(&mut self, targets: &[f32], wind_speed: f32, out: &mut [f32]) {
        let fs = MIX_RATE as f32;
        // Per-sample gain smoother (~SMOOTH_TC_S). Per-call is fine for the
        // exp: block sizes vary but the per-sample coefficient does not.
        let a_g = 1.0 - (-1.0 / (SMOOTH_TC_S * fs)).exp();
        let norm = (wind_speed / (0.25 * self.diag)).clamp(0.0, 1.0);
        // Gust: two incommensurate slow sines ride the TARGET, so the
        // per-sample gain smoother absorbs the block-rate steps. Zero rng.
        let t_now = self.sample_idx as f64 / fs as f64;
        let gust = 1.0
            + 0.15 * (std::f64::consts::TAU * 0.23 * t_now).sin() as f32
            + 0.10 * (std::f64::consts::TAU * 0.61 * t_now).sin() as f32;
        let wind_target = wind_gain(norm) * WIND_MASTER * gust;
        // Cutoff smoothed per block (it only feeds a coefficient); the
        // filter coefficient derives from the smoothed value.
        let frames = out.len() / 2;
        let a_blk = 1.0 - (-(frames as f32) / (SMOOTH_TC_S * fs)).exp();
        self.wind_fc += a_blk * (wind_cutoff(norm) - self.wind_fc);
        let a_lp = onepole_coeff(self.wind_fc, fs);
        // Exact-silence fast path state: wind fully off contributes literal
        // 0.0 (and freezes its phase clock), parked loops contribute nothing.
        let wind_silent = self.wind_g == 0.0 && wind_target == 0.0;
        let nl = self.loops.len();
        for frame in out.chunks_exact_mut(2) {
            let mut amb = 0.0f32;
            for li in 0..nl {
                let tg = targets.get(li).copied().unwrap_or(0.0);
                let g0 = self.g[li];
                if g0 == 0.0 && tg == 0.0 {
                    continue; // inaudible: skip AND freeze pos (resume-in-place)
                }
                let mut g = g0 + a_g * (tg - g0);
                if tg == 0.0 && g < GAIN_FLUSH {
                    g = 0.0; // flush: re-latch the skip-and-freeze fast path
                }
                self.g[li] = g;
                let lb = &mut self.loops[li];
                let s = lb.samples[lb.pos] as f32 * (1.0 / 32768.0);
                lb.pos += 1;
                if lb.pos == lb.samples.len() {
                    lb.pos = 0;
                }
                amb += s * g;
            }
            amb *= AMB_MASTER;
            let (mut l, mut r) = (amb, amb);
            if !wind_silent {
                let mut wg = self.wind_g + a_g * (wind_target - self.wind_g);
                if wind_target == 0.0 && wg < GAIN_FLUSH {
                    wg = 0.0; // flush: wind_silent re-latches on the next call
                }
                self.wind_g = wg;
                let i = self.sample_idx;
                self.sample_idx = i + 1;
                for (ch, st) in self.wind_lp.iter_mut().enumerate() {
                    // Cascade: noise -> stage 0 -> stage 1 -> ... (−6 dB/oct
                    // per stage; the last stage is the output).
                    let mut x = hash_noise(i * 2 + ch as u64);
                    for s in st.iter_mut() {
                        *s += a_lp * (x - *s);
                        x = *s;
                    }
                }
                l += self.wind_lp[0][WIND_LP_STAGES - 1] * wg;
                r += self.wind_lp[1][WIND_LP_STAGES - 1] * wg;
            }
            frame[0] = soft_clip(l);
            frame[1] = soft_clip(r);
        }
    }
}

/// Pure-math gate suite, run by --check (device-free by construction).
pub fn self_test() -> Result<(), String> {
    use std::f32::consts::TAU;
    // --- resampler: equal rates are bit-identical (the off-state contract).
    let src: Vec<i16> = (0..4801).map(|i| ((i * 37) % 20011) as i16 - 10000).collect();
    if resample_linear(&src, 48_000, 48_000) != src {
        return Err("resample: equal-rate passthrough not bit-identical".into());
    }
    // --- resampler: a 440 Hz sine at 44.1k resampled to 48k matches the
    // directly synthesized 48k sine (linear-interp error at this f/fs is
    // tiny; the bounds are loose enough to be platform-safe).
    // f64 phase: TAU·440·i reaches ~1.2e8 where an f32 ulp is 8 RADIANS —
    // an f32 probe would gate the probe's own rounding, not the resampler.
    let sine = |rate: f64, n: usize| -> Vec<i16> {
        (0..n)
            .map(|i| {
                (16000.0 * (std::f64::consts::TAU * 440.0 * i as f64 / rate).sin()).round() as i16
            })
            .collect()
    };
    let rs = resample_linear(&sine(44_100.0, 44_100), 44_100, 48_000);
    let refr = sine(48_000.0, rs.len());
    let (mut worst, mut mean) = (0.0f64, 0.0f64);
    for (a, b) in rs.iter().zip(refr.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        worst = worst.max(d);
        mean += d;
    }
    mean /= rs.len() as f64;
    if worst > 64.0 || mean > 16.0 {
        return Err(format!("resample: sine error worst {worst:.1} mean {mean:.2}"));
    }
    // --- seam bake: a sine holding a non-integer period count has a hard
    // wrap discontinuity raw (anti-vacuity) and a smooth one baked.
    let n = 48_000;
    let raw: Vec<i16> =
        (0..n).map(|i| (12000.0 * (TAU * 3.37 * i as f32 / n as f32).sin()).round() as i16).collect();
    let step_max = raw.windows(2).map(|w| (w[1] as i32 - w[0] as i32).abs()).max().unwrap();
    let raw_wrap = (raw[0] as i32 - raw[n - 1] as i32).abs();
    if raw_wrap <= 4 * step_max {
        return Err("seam: probe signal has no raw discontinuity (vacuous)".into());
    }
    let mut baked = raw.clone();
    bake_loop_seam(&mut baked, 48_000);
    let m = baked.len();
    let baked_wrap = (baked[0] as i32 - baked[m - 1] as i32).abs();
    if baked_wrap > 4 * step_max {
        return Err(format!("seam: wrap step {baked_wrap} vs in-run max {step_max}"));
    }
    // --- loudness normalize: a quiet sine lands on TARGET_RMS; silence is
    // untouched (no divide-by-noise-floor blowup).
    {
        let mut quiet: Vec<i16> =
            (0..48_000).map(|i| (900.0 * (TAU * 220.0 * i as f32 / 48_000.0).sin()) as i16).collect();
        normalize_rms(&mut quiet, TARGET_RMS);
        let rms = (quiet.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum::<f64>()
            / quiet.len() as f64)
            .sqrt();
        if (rms - TARGET_RMS as f64).abs() / TARGET_RMS as f64 > 0.02 {
            return Err(format!("normalize: rms {rms:.4} vs target {TARGET_RMS}"));
        }
        let mut silent = vec![0i16; 1000];
        normalize_rms(&mut silent, TARGET_RMS);
        if silent.iter().any(|&s| s != 0) {
            return Err("normalize: silence not preserved".into());
        }
    }
    // --- proximity gain anchors: exact at center and at d = r; monotone.
    for r in [0.5f32, 3.0, 40.0] {
        if proximity_gain(0.0, r) != 1.0 {
            return Err("proximity: center gain != exactly 1.0".into());
        }
        if proximity_gain(r * r, r) != 0.5 {
            return Err("proximity: d = r gain != exactly 0.5".into());
        }
        let mut prev = f32::INFINITY;
        for k in 0..100 {
            let g = proximity_gain(k as f32 * r * r * 0.25, r);
            if g > prev {
                return Err("proximity: not monotone decreasing".into());
            }
            prev = g;
        }
    }
    // --- wind anchors: parked camera is digital silence; gain monotone;
    // cutoff endpoints; filter coefficient in (0, 1) over the sweep.
    if wind_gain(0.0) != 0.0 {
        return Err("wind: gain at norm 0 != exactly 0.0".into());
    }
    let mut prev = -1.0f32;
    for k in 0..=100 {
        let g = wind_gain(k as f32 / 100.0);
        if g < prev {
            return Err("wind: gain not monotone".into());
        }
        prev = g;
    }
    if (wind_cutoff(0.0) - WIND_FC_MIN).abs() > 1e-2
        || (wind_cutoff(1.0) - WIND_FC_MAX).abs() / WIND_FC_MAX > 1e-3
    {
        return Err("wind: cutoff endpoints off".into());
    }
    for k in 0..=100 {
        let a = onepole_coeff(wind_cutoff(k as f32 / 100.0), MIX_RATE as f32);
        if !(a > 0.0 && a < 1.0) {
            return Err("wind: one-pole coefficient out of (0,1)".into());
        }
    }
    // --- soft clip: identity below the knee, bounded and monotone above.
    if soft_clip(0.5) != 0.5 || soft_clip(-0.5) != -0.5 {
        return Err("clip: below-knee not identity".into());
    }
    let mut prev = 0.0f32;
    for k in 0..200 {
        let y = soft_clip(k as f32 * 0.1);
        // <= 1.0: f32 tanh legitimately saturates to exactly 1.0 far out on
        // the asymptote — the bound is "never past full scale", not "never
        // reaches it".
        if y < prev || y > 1.0 {
            return Err("clip: tail not monotone-bounded".into());
        }
        prev = y;
    }
    // --- mixer must-fires (no device): a live loop produces output, all-off
    // produces EXACT 0.0 (the off-bit-identity flavor), everything bounded,
    // and two identical runs are byte-equal (determinism).
    let mklp = || LoopBuf { samples: vec![8192i16; 4800], pos: 0 };
    let run = |targets: &[f32], speed: f32| -> Vec<f32> {
        let mut core = MixCore::new(vec![mklp()], 10.0);
        let mut out = vec![0.0f32; 2 * 48_000];
        core.render(targets, speed, &mut out);
        out
    };
    let live = run(&[1.0], 1.25); // half of full-tilt speed at diag 10
    if !live.iter().all(|x| x.is_finite() && x.abs() <= 1.0) {
        return Err("mix: unbounded or non-finite output".into());
    }
    if live[2 * 47_000] == 0.0 {
        return Err("mix: live loop + wind produced silence (must-fire)".into());
    }
    let off = run(&[0.0], 0.0);
    if off.iter().any(|&x| x != 0.0) {
        return Err("mix: all-off output not exactly 0.0".into());
    }
    let a = run(&[0.7], 0.6);
    let b = run(&[0.7], 0.6);
    if a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits()) {
        return Err("mix: two identical runs not byte-equal".into());
    }
    // --- post-activity re-latch: after a second of live output, all-off
    // targets must decay to EXACT 0.0 (the GAIN_FLUSH snap). Without the
    // flush the smoothers stall at a denormal and the silence fast paths
    // never re-latch — the all-off gate above only covers a NEVER-live core.
    // Timing: decay from gain ~1.0 to 1e-6 is ln(1e6)·SMOOTH_TC_S ≈ 2.1 s,
    // so second 1 of the tail is still live (anti-vacuity) and second 4 is
    // past the flush.
    {
        let mut core = MixCore::new(vec![mklp()], 10.0);
        let mut live = vec![0.0f32; 2 * 48_000];
        core.render(&[1.0], 2.5, &mut live);
        let mut tail = vec![0.0f32; 2 * 4 * 48_000];
        core.render(&[0.0], 0.0, &mut tail);
        if tail[..2 * 48_000].iter().all(|&x| x == 0.0) {
            return Err("mix: decay tail silent from sample 0 (re-latch gate vacuous)".into());
        }
        if tail[2 * 3 * 48_000..].iter().any(|&x| x != 0.0) {
            return Err("mix: post-activity all-off not exact 0.0 (gain flush failed)".into());
        }
    }
    // --- loop wrap accounting: pos advances modulo the buffer exactly.
    {
        let mut core = MixCore::new(vec![LoopBuf { samples: vec![100i16; 1000], pos: 0 }], 10.0);
        let mut out = vec![0.0f32; 2 * 2500];
        core.render(&[1.0], 0.0, &mut out);
        if core.loops[0].pos != 2500 % 1000 {
            return Err(format!("mix: loop pos {} != 2500 mod 1000", core.loops[0].pos));
        }
    }
    // --- scene-path mapping: the aliases that must hold.
    let cases = [
        (r"scenes\san-miguel\san-miguel-low-poly.obj", Some("san-miguel")),
        ("scenes/sponza-khronos/Sponza.gltf", Some("sponza")),
        ("scenes/intel-sponza/main_sponza/NewSponza_Main_glTF_003.gltf", Some("sponza")),
        ("scenes/damaged-helmet/DamagedHelmet.glb", Some("helmet")),
        ("scenes/bistro/Interior/interior.obj", Some("bistro")),
        ("scenes/hairball/hairball.obj", None),
        ("model.obj", None),
    ];
    for (p, want) in cases {
        if match_scene_path(p) != want {
            return Err(format!("path map: {p} -> {:?}, wanted {:?}", match_scene_path(p), want));
        }
    }
    // Every curated island must have an ambience mapping (a new island
    // without a loop should fail here, not silently play nothing).
    for spec in crate::world::CURATED {
        if !AMBIENCE.iter().any(|(n, _, _)| *n == spec.name) {
            return Err(format!("ambience: no loop mapped for curated island {}", spec.name));
        }
    }
    // Gain sanity: finite, positive, and bounded — a 4x gain on TARGET_RMS
    // is already inside peak-guard territory for any real loop; past that
    // the entry is a typo, not a balance choice.
    for &(name, _, gain) in AMBIENCE {
        if !(gain.is_finite() && gain > 0.0 && gain <= 4.0) {
            return Err(format!("ambience: {name} gain {gain} out of (0, 4]"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Device half — SDL3 callback stream + lewton decode. Windows-only, and only
// `run_window` constructs it: every headless path is structurally silent.
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod device {
    use super::*;
    use sdl3::audio::{AudioCallback, AudioFormat, AudioSpec, AudioStream, AudioStreamWithCallback};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

    /// The SDL callback shell around MixCore: load the per-cue gain targets
    /// and the flycam's speed atomic, render, push. The scratch buffer grows
    /// to the device's block size once and is then steady-state — no
    /// allocation on the audio thread after warmup.
    pub struct SdlMixer {
        core: MixCore,
        gains: Arc<Vec<AtomicU32>>,
        speed: Arc<AtomicU32>,
        targets: Vec<f32>,
        scratch: Vec<f32>,
    }

    impl AudioCallback<f32> for SdlMixer {
        fn callback(&mut self, stream: &mut AudioStream, requested: i32) {
            // `requested` counts f32 VALUES (the crate divides bytes by the
            // sample size), not frames — round up to whole stereo frames.
            let n = ((requested.max(0) as usize) + 1) & !1;
            if self.scratch.len() < n {
                self.scratch.resize(n, 0.0);
            }
            for i in 0..self.targets.len() {
                self.targets[i] = f32::from_bits(self.gains[i].load(Relaxed));
            }
            let speed = f32::from_bits(self.speed.load(Relaxed));
            let (targets, out) = (&self.targets, &mut self.scratch[..n]);
            self.core.render(targets, speed, out);
            let _ = stream.put_data_f32(out);
        }
    }

    /// The session's audio system: owns the device stream (drop = silence).
    /// Lives in `run_window` beside the FlyCam, so it survives resize
    /// re-entries exactly like the pose does.
    pub struct AudioSys {
        _stream: AudioStreamWithCallback<SdlMixer>,
        gains: Arc<Vec<AtomicU32>>,
        cues: Vec<Cue>,
        steady: bool,
    }

    /// Detach the C callback BEFORE the stream field drops. The sdl3 crate's
    /// own `Drop for AudioStreamWithCallback` frees the boxed `SdlMixer` in its
    /// body and only THEN drops `base_stream` — i.e. it calls
    /// `SDL_DestroyAudioStream`, the call that actually stops the device thread
    /// and joins any in-flight callback, one statement too late (Rust runs a
    /// Drop body before the struct's fields). That window is exactly one
    /// callback wide, and a device thread that walks into it reads a freed
    /// `LoopBuf::samples` — multi-MB, so the allocator returns it to the OS and
    /// the read FAULTS rather than glitching once (observed at shutdown as an
    /// access violation inside `MixCore::render`, reported by src/crash.rs).
    /// `SDL_SetAudioStreamGetCallback` takes the stream's own lock, so it
    /// blocks behind an in-flight callback and, once it returns, nothing can
    /// reach the box again — after which the crate's drop order is harmless.
    impl Drop for AudioSys {
        fn drop(&mut self) {
            let s = self._stream.stream();
            if !s.is_null() {
                // SAFETY: `s` is this stream, still alive (we are ahead of its
                // own drop); a None callback with a null userdata is the
                // documented detach, and SDL states the call is thread-safe.
                unsafe {
                    sdl3::sys::audio::SDL_SetAudioStreamGetCallback(
                        s,
                        None,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
    }

    impl AudioSys {
        /// Build the whole pipeline. Every failure is one loud line + None —
        /// a session without audio is a session, never an error.
        pub fn new(
            sdl: &sdl3::Sdl,
            cues: Cues,
            speed: Arc<AtomicU32>,
            diag: f32,
        ) -> Option<AudioSys> {
            let audio = match sdl.audio() {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("audio: SDL audio subsystem unavailable ({e}) — session is silent");
                    return None;
                }
            };
            let (kept, loops, steady) = load_cues(cues);
            let gains: Arc<Vec<AtomicU32>> =
                Arc::new((0..loops.len()).map(|_| AtomicU32::new(0)).collect());
            let mixer = SdlMixer {
                core: MixCore::new(loops, diag),
                gains: gains.clone(),
                speed,
                targets: vec![0.0; kept.len()],
                scratch: Vec::new(),
            };
            let spec = AudioSpec {
                freq: Some(MIX_RATE as i32),
                channels: Some(2),
                format: Some(AudioFormat::f32_sys()),
            };
            let stream = match audio.open_playback_stream(&spec, mixer) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("audio: playback stream failed ({e}) — session is silent");
                    return None;
                }
            };
            if let Err(e) = stream.resume() {
                eprintln!("audio: stream resume failed ({e}) — session is silent");
                return None;
            }
            eprintln!(
                "audio: {} ambience loop(s) + speed-scaled wind (--no-audio disables)",
                kept.len()
            );
            Some(AudioSys { _stream: stream, gains, cues: kept, steady })
        }

        /// Per-frame gain update from the session's ONE camera snapshot.
        /// Full-3D distance (see the module header). Cheap: N atomic stores.
        pub fn update(&self, cam_pos: Vec3A) {
            if self.steady {
                if let Some(g) = self.gains.first() {
                    g.store(1.0f32.to_bits(), Relaxed);
                }
                return;
            }
            for (i, cue) in self.cues.iter().enumerate() {
                let d2 = (cam_pos - cue.pos).length_squared();
                let g = proximity_gain(d2, cue.radius);
                self.gains[i].store(g.to_bits(), Relaxed);
            }
        }
    }

    /// Resolve Cues into (kept cues, decoded loops, steady?) — cue i and
    /// loop i stay index-aligned by construction (a failed decode drops
    /// both). Missing files are the expected degrade, so the line says what
    /// to do about it.
    fn load_cues(cues: Cues) -> (Vec<Cue>, Vec<LoopBuf>, bool) {
        let mut kept = Vec::new();
        let mut loops = Vec::new();
        let mut steady = false;
        let mut load = |name: &str, cue: Cue| {
            let Some(&(_, path, gain)) = AMBIENCE.iter().find(|(n, _, _)| *n == name) else {
                eprintln!("audio: no ambience mapped for {name} — silent");
                return;
            };
            match load_loop(path, gain) {
                Ok(samples) => {
                    kept.push(cue);
                    loops.push(LoopBuf { samples, pos: 0 });
                }
                Err(e) => eprintln!("audio: {path}: {e} — {name} silent (git lfs pull?)"),
            }
        };
        match cues {
            Cues::None => {}
            Cues::Steady(name) => {
                steady = true;
                load(name, Cue { name: name.to_string(), pos: Vec3A::ZERO, radius: 1.0 });
            }
            Cues::World(v) => {
                for cue in v {
                    let name = cue.name.clone();
                    load(&name, cue);
                }
            }
        }
        (kept, loops, steady)
    }

    /// Decode one OGG into the mixer's format: lewton (pure Rust) -> mono
    /// downmix -> resample to MIX_RATE -> RMS normalize (x the island's
    /// balance gain — see AMBIENCE) -> seam bake.
    fn load_loop(path: &str, gain: f32) -> Result<Vec<i16>, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("open failed ({e})"))?;
        let mut r = lewton::inside_ogg::OggStreamReader::new(f)
            .map_err(|e| format!("vorbis header ({e:?})"))?;
        let ch = r.ident_hdr.audio_channels as usize;
        let rate = r.ident_hdr.audio_sample_rate;
        if ch == 0 || rate == 0 {
            return Err("degenerate vorbis header".into());
        }
        let mut mono: Vec<i16> = Vec::new();
        loop {
            match r.read_dec_packet_itl() {
                Ok(Some(pkt)) => {
                    for frame in pkt.chunks_exact(ch) {
                        let s: i32 = frame.iter().map(|&x| x as i32).sum();
                        mono.push((s / ch as i32) as i16);
                    }
                }
                Ok(None) => break,
                Err(e) => return Err(format!("vorbis decode ({e:?})")),
            }
        }
        if mono.len() < MIX_RATE as usize {
            return Err(format!("too short ({} samples < 1 s)", mono.len()));
        }
        let mut m = resample_linear(&mono, rate, MIX_RATE);
        normalize_rms(&mut m, TARGET_RMS * gain);
        bake_loop_seam(&mut m, MIX_RATE);
        Ok(m)
    }
}

#[cfg(windows)]
pub use device::AudioSys;
