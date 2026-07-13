//! The temporal-upscaler fallback chain.
//!
//! Upscaling is always on: every session probes the chain in the fixed order
//! DLSS-RR → FSR4-Ray-Regen → XeSS → FSR3 (upscale-only) and wires the first
//! level whose availability probe passes. `UpChain` is the *intent* half —
//! which levels the CLI allows — kept as plain data so `--check` can gate the
//! resolution logic DLL-free (availability is injected, never probed here).
//! The probe itself lives in `GpuContext::new`; exactly one upscaler is wired
//! per session (DLSS decides the SL-proxy-vs-native device split up front, and
//! the native levels are mutually exclusive by first-hit-wins).
//!
//! Flag algebra: `--<x>` *forces* the chain to start at level x (everything
//! above x is disabled; everything below stays as fall-through), `--no-<x>`
//! *skips* that one level (`--no-fsr` skips both FSR levels at the parse
//! site), `--no-upscale` is `NONE` — the only plain-path spelling. Later flags
//! win, matching left-to-right parse order.

/// One level of the chain, in probe order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UpLevel {
    Dlss,
    Fsr4,
    Xess,
    Fsr3,
}

impl UpLevel {
    /// Probe order — the chain IS this array.
    pub const ORDER: [UpLevel; 4] = [UpLevel::Dlss, UpLevel::Fsr4, UpLevel::Xess, UpLevel::Fsr3];

    /// Short name for log lines.
    pub fn name(self) -> &'static str {
        match self {
            UpLevel::Dlss => "dlss",
            UpLevel::Fsr4 => "fsr4",
            UpLevel::Xess => "xess",
            UpLevel::Fsr3 => "fsr3",
        }
    }
}

/// Which levels of the chain may be probed. Pure data — no SDK, no GPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UpChain {
    pub dlss: bool,
    pub fsr4: bool,
    pub xess: bool,
    pub fsr3: bool,
}

impl UpChain {
    /// The default: every level enabled, first supported wins.
    pub const ALL: UpChain = UpChain { dlss: true, fsr4: true, xess: true, fsr3: true };
    /// `--no-upscale`: plain presentation, nothing probed.
    pub const NONE: UpChain = UpChain { dlss: false, fsr4: false, xess: false, fsr3: false };

    fn enabled(&self, level: UpLevel) -> bool {
        match level {
            UpLevel::Dlss => self.dlss,
            UpLevel::Fsr4 => self.fsr4,
            UpLevel::Xess => self.xess,
            UpLevel::Fsr3 => self.fsr3,
        }
    }

    fn set(&mut self, level: UpLevel, on: bool) {
        match level {
            UpLevel::Dlss => self.dlss = on,
            UpLevel::Fsr4 => self.fsr4 = on,
            UpLevel::Xess => self.xess = on,
            UpLevel::Fsr3 => self.fsr3 = on,
        }
    }

    /// Resolve the chain: the first enabled level whose availability probe
    /// passes. `avail` is injected so the ordering logic is gateable without
    /// any SDK on disk; the real probes run in `GpuContext::new` in this same
    /// order, short-circuiting on the first success.
    pub fn resolve(&self, avail: impl Fn(UpLevel) -> bool) -> Option<UpLevel> {
        UpLevel::ORDER.iter().copied().find(|&l| self.enabled(l) && avail(l))
    }

    /// `--<x>`: force the chain to start at `level` — every level ABOVE it is
    /// disabled, `level` itself re-enabled. Levels below are untouched (they
    /// remain the fall-through).
    pub fn force(&mut self, level: UpLevel) {
        for l in UpLevel::ORDER {
            if l == level {
                self.set(l, true);
                break;
            }
            self.set(l, false);
        }
    }

    /// `--no-<x>`: skip one level of the chain (`--no-fsr` calls this for
    /// both FSR levels at the parse site).
    pub fn skip(&mut self, level: UpLevel) {
        self.set(level, false);
    }
}

/// Chain-resolution gates: probe order under every availability mask, the
/// force/skip flag algebra (including flag-order compositions the parser
/// produces), and the NONE escape. Pure and DLL-free — run by `--check`.
pub fn self_test() -> Result<(), String> {
    use UpLevel::*;
    let all = |_: UpLevel| true;
    let none = |_: UpLevel| false;
    let only = |lv: UpLevel| move |l: UpLevel| l == lv;
    let not = |lv: UpLevel| move |l: UpLevel| l != lv;

    // Probe order: first enabled+available wins, straight down the chain.
    if UpChain::ALL.resolve(all) != Some(Dlss) {
        return Err("ALL/all-avail must resolve Dlss".into());
    }
    if UpChain::ALL.resolve(not(Dlss)) != Some(Fsr4) {
        return Err("dlss unavailable must fall to Fsr4".into());
    }
    if UpChain::ALL.resolve(|l| l != Dlss && l != Fsr4) != Some(Xess) {
        return Err("dlss+fsr4 unavailable must fall to Xess".into());
    }
    if UpChain::ALL.resolve(only(Fsr3)) != Some(Fsr3) {
        return Err("only fsr3 available must resolve Fsr3".into());
    }
    if UpChain::ALL.resolve(none) != None {
        return Err("nothing available must resolve None".into());
    }
    if UpChain::NONE.resolve(all) != None {
        return Err("NONE (--no-upscale) must resolve None even with all available".into());
    }

    // force(x): disables everything above, keeps the fall-through below.
    let mut c = UpChain::ALL;
    c.force(Xess);
    if (c.dlss, c.fsr4, c.xess, c.fsr3) != (false, false, true, true) {
        return Err(format!("force(Xess) wrong mask: {c:?}"));
    }
    if c.resolve(all) != Some(Xess) {
        return Err("force(Xess)/all-avail must resolve Xess".into());
    }
    if c.resolve(not(Xess)) != Some(Fsr3) {
        return Err("force(Xess) with xess unavailable must fall to Fsr3".into());
    }

    // skip(x): one level out, the rest of the chain intact.
    let mut c = UpChain::ALL;
    c.skip(Dlss);
    if c.resolve(all) != Some(Fsr4) {
        return Err("skip(Dlss) must resolve Fsr4".into());
    }
    c.force(Fsr4); // `--no-dlss --fsr`
    if (c.dlss, c.fsr4, c.xess, c.fsr3) != (false, true, true, true) {
        return Err(format!("skip(Dlss)+force(Fsr4) wrong mask: {c:?}"));
    }

    // Parser flag-order compositions (later flag wins).
    // `--fsr --no-fsr`: force(Fsr4), then skip both FSR levels → Xess.
    let mut c = UpChain::ALL;
    c.force(Fsr4);
    c.skip(Fsr4);
    c.skip(Fsr3);
    if c.resolve(all) != Some(Xess) {
        return Err("--fsr --no-fsr must resolve Xess".into());
    }
    // `--no-fsr --fsr`: skip both, then force re-enables Fsr4 only.
    let mut c = UpChain::ALL;
    c.skip(Fsr4);
    c.skip(Fsr3);
    c.force(Fsr4);
    if (c.dlss, c.fsr4, c.xess, c.fsr3) != (false, true, true, false) {
        return Err(format!("--no-fsr --fsr wrong mask: {c:?}"));
    }
    if c.resolve(not(Fsr4)) != Some(Xess) {
        return Err("--no-fsr --fsr off-RDNA4 must fall to Xess (fsr3 stays skipped)".into());
    }
    // `--xess --fsr`: force(Xess) then force(Fsr4) re-enables Fsr4 above Xess.
    let mut c = UpChain::ALL;
    c.force(Xess);
    c.force(Fsr4);
    if c.resolve(all) != Some(Fsr4) {
        return Err("--xess --fsr must resolve Fsr4".into());
    }
    if c.dlss {
        return Err("--xess --fsr must keep dlss disabled".into());
    }

    Ok(())
}
