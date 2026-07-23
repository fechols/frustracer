#!/usr/bin/env python3
"""Synthesize scenes/audio/helmet.ogg — the DamagedHelmet island's sci-fi
ambience bed (src/audio.rs::AMBIENCE).

Fully synthesized, so the asset is self-authored (CC0 by dedication, see
scenes/audio/credits_license.txt) and deterministic — the DSP and the
intermediate helmet.wav are byte-reproducible; the .ogg additionally depends
on the local ffmpeg/libvorbis build, so the COMMITTED file is the artifact of
record. Every oscillator and LFO frequency sits on the 1/T grid (integer
cycles per loop), every echo is a circular np.roll, and all noise comes from
one seeded generator — the loop is seamless BY CONSTRUCTION, before audio.rs
even bakes its 50 ms seam.

Layers (mixed, then peak-normalized; audio.rs RMS-normalizes at load, so only
the relative balance here matters):
  - reactor drone: detuned sine stack at 55 Hz + fifth (slow beating)
  - sub thrum: a 42->36 Hz decaying pulse every 4 s — the ship's heartbeat
  - life-support air: low-passed noise with an 8-cycles/loop breathing LFO
  - vent hiss: a narrow ~3 kHz noise band, slowly waxing and waning
  - shimmer pad: quiet detuned high pairs with slow independent tremolos
  - telemetry bleeps: sparse 3-note sine sequences, hash-picked pitches, echo
  - sonar ping: a descending 1040->660 Hz glide with echo, twice per loop

Usage:  python gen.py            (writes helmet.wav next to this script, then
                                  ffmpeg-encodes scenes/audio/helmet.ogg —
                                  mono 48 kHz Ogg Vorbis q3, the credits-file
                                  convention)
Needs:  numpy, ffmpeg on PATH.
"""

import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 48_000
T = 48.0  # seconds — the ~48 s bed convention from credits_license.txt
N = int(SR * T)
t = np.arange(N) / SR


def cyc(k, phase=0.0):
    """Sine LFO with exactly k cycles over the loop (seamless)."""
    return np.sin(2.0 * np.pi * k * t / T + phase)


def osc(freq, phase=0.0):
    """Sine oscillator; freq must be a multiple of 1/T to loop cleanly."""
    assert abs(freq * T - round(freq * T)) < 1e-6, f"{freq} Hz off the 1/T grid"
    return np.sin(2.0 * np.pi * freq * t + phase)


def hash32(i):
    """Deterministic integer hash (splitmix-style, the audio.rs idiom)."""
    z = (i * 0x9E3779B9 + 0x85EBCA6B) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x21F0AAAD) & 0xFFFFFFFF
    z = ((z ^ (z >> 15)) * 0x735A2D97) & 0xFFFFFFFF
    return z ^ (z >> 15)


def add_wrapped(buf, start_s, sig):
    """Add an event at start_s with modulo-N indexing (loop-safe tails)."""
    i0 = int(round(start_s * SR))
    idx = (i0 + np.arange(len(sig))) % N
    buf[idx] += sig


def echo(x, delay_s, gains=(0.38, 0.16)):
    """Circular feedback-style echo — np.roll keeps the loop seamless."""
    d = int(delay_s * SR)
    out = x.copy()
    for i, g in enumerate(gains, 1):
        out += g * np.roll(x, i * d)
    return out


# --- reactor drone: detuned pairs beat at ~0.2-0.3 Hz; all freqs = k/48.
drone = (
    0.50 * osc(55.0)
    + 0.42 * osc(55.3125, 1.7)
    + 0.30 * osc(82.5, 0.6)
    + 0.24 * osc(82.7083333333, 2.9)  # 3970/48
    + 0.18 * osc(110.0, 4.1)
    + 0.08 * osc(165.0, 0.9)
)
drone *= 0.85 + 0.15 * cyc(5, 0.4)  # slow power swell

# --- sub thrum: 12 pulses, one per 4 s, slight downward chirp, long decay.
sub = np.zeros(N)
dur = 2.2
tt = np.arange(int(dur * SR)) / SR
env = np.minimum(tt / 0.05, 1.0) * np.exp(-tt * 3.2)
freq = 42.0 - 6.0 * np.minimum(tt / 0.8, 1.0)  # 42 -> 36 Hz glide
phase = 2.0 * np.pi * np.cumsum(freq) / SR
pulse = env * np.sin(phase)
for i in range(12):
    add_wrapped(sub, i * 4.0 + 0.001 * (hash32(i) % 200), pulse)

# --- noise beds, filtered in the frequency domain (circular == seamless).
rng = np.random.default_rng(0xF16_5C1F1)
f = np.fft.rfftfreq(N, 1.0 / SR)

W_air = np.fft.rfft(rng.standard_normal(N))
air = np.fft.irfft(W_air / (1.0 + (f / 450.0) ** 4), N)
air /= np.sqrt(np.mean(air * air))
air *= 0.72 + 0.28 * cyc(8)  # life-support breathing, 6 s period

W_hiss = np.fft.rfft(rng.standard_normal(N))
H_hiss = np.exp(-0.5 * ((np.log(f + 1e-9) - np.log(3000.0)) / 0.35) ** 2)
hiss = np.fft.irfft(W_hiss * H_hiss, N)
hiss /= np.sqrt(np.mean(hiss * hiss))
hiss *= 0.5 + 0.5 * cyc(3, 2.0)

# --- shimmer pad: airy detuned pairs, independent slow tremolos.
pad = (
    (0.9 + 0.1 * cyc(7)) * (osc(660.0) + osc(660.25, 0.8))
    + (0.9 + 0.1 * cyc(11, 1.3)) * (osc(990.0, 2.1) + osc(990.3125, 4.0))
    + (0.9 + 0.1 * cyc(13, 2.6)) * 0.6 * (osc(1320.0, 1.1) + osc(1320.1875, 3.3))
)

# --- telemetry bleeps: 8 sparse 3-note sequences, pitches hash-picked from a
# high pentatonic-ish set; short sines with echo.
NOTES = [1568.0, 1760.0, 2093.0, 2349.0, 2637.0]
bleeps = np.zeros(N)
nb = int(0.06 * SR)
nt = np.arange(nb) / SR
nenv = np.minimum(nt / 0.005, 1.0) * np.exp(-nt * 55.0)
for e in range(8):
    t0 = e * 6.0 + 1.0 + 0.001 * (hash32(100 + e) % 2500)
    for k in range(3):
        fr = NOTES[hash32(e * 7 + k) % len(NOTES)]
        add_wrapped(bleeps, t0 + 0.09 * k, nenv * np.sin(2.0 * np.pi * fr * nt))
bleeps = echo(bleeps, 0.31)

# --- sonar ping: descending glide, twice per loop, with echo.
ping = np.zeros(N)
pd = int(1.2 * SR)
pt = np.arange(pd) / SR
penv = np.minimum(pt / 0.01, 1.0) * np.exp(-pt * 3.8)
pfreq = 1040.0 - 380.0 * np.minimum(pt / 0.9, 1.0)
pph = 2.0 * np.pi * np.cumsum(pfreq) / SR
psig = penv * np.sin(pph)
add_wrapped(ping, 11.3, psig)
add_wrapped(ping, 35.7, psig)
ping = echo(ping, 0.45, gains=(0.5, 0.25, 0.12))

# --- mix + peak normalize (loudness itself is audio.rs's job at load).
mix = (
    0.32 * drone / np.abs(drone).max()
    + 0.30 * sub / max(np.abs(sub).max(), 1e-9)
    + 0.16 * air / np.abs(air).max()
    + 0.045 * hiss / np.abs(hiss).max()
    + 0.055 * pad / np.abs(pad).max()
    + 0.06 * bleeps / max(np.abs(bleeps).max(), 1e-9)
    + 0.07 * ping / max(np.abs(ping).max(), 1e-9)
)
mix *= 0.85 / np.abs(mix).max()

here = Path(__file__).parent
wav_path = here / "helmet.wav"
with wave.open(str(wav_path), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767.0).round().astype(np.int16).tobytes())

ogg_path = here / ".." / ".." / "scenes" / "audio" / "helmet.ogg"
subprocess.run(
    [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-ac", "1", "-ar", str(SR), "-c:a", "libvorbis", "-q:a", "3",
        str(ogg_path.resolve()),
    ],
    check=True,
)
print(f"wrote {ogg_path.resolve()} ({T:.0f} s, mono {SR} Hz, vorbis q3)")
