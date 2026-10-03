"""Every sound in the game, synthesized from numpy waveforms at startup. No audio files.

The toolkit is small:
- sine waves with falling pitch for thuds and booms,
- band-filtered noise for scrapes, rain, waves and thunder,
- impulse trains through resonant filters for wood creaks,
- stacks of inharmonic partials for the ship's bell.

Filtering is done in the frequency domain (FFT -> multiply -> inverse FFT). A side effect
is that the filtered noise wraps around seamlessly, which makes the rain loop click-free.
"""

from __future__ import annotations

import numpy as np

SR = 44_100


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _band(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    """Soft band-pass (Butterworth-shaped magnitude) applied via FFT."""
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    f[0] = 1e-6
    mask = 1 / (1 + (lo / f) ** (2 * order)) / (1 + (f / hi) ** (2 * order))
    return np.fft.irfft(spectrum * mask, len(x))


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def _delay(x: np.ndarray, seconds: float, total: int | None = None) -> np.ndarray:
    pad = int(seconds * SR)
    out = np.zeros(max(total or 0, pad + len(x)))
    out[pad : pad + len(x)] = x
    return out


def _mix(*parts: np.ndarray) -> np.ndarray:
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[: len(p)] += p
    return out


def _fade(x: np.ndarray, attack: float = 0.005, release: float = 0.02) -> np.ndarray:
    a, r = int(attack * SR), int(release * SR)
    env = np.ones(len(x))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return x * env


# --- individual sounds ---------------------------------------------------------------


def thud(rng, weight: float = 1.0) -> np.ndarray:
    """A crate hitting timber: a falling-pitch body plus a short knock."""
    t = _t(0.45)
    pitch = 48 + 90 * np.exp(-t * 20)
    body = np.sin(2 * np.pi * np.cumsum(pitch) / SR) * np.exp(-t * 10)
    knock = _norm(_band(rng.standard_normal(len(t)), 180, 1600)) * np.exp(-t * 70)
    return _fade(_norm(body + 0.45 * weight * knock))


def latch(rng) -> np.ndarray:
    """The metallic double-click of a crate locking into its slot."""

    def ping(freqs, decay):
        t = _t(0.16)
        tone = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * decay * (1 + i * 0.4)) for i, f in enumerate(freqs))
        click = _band(rng.standard_normal(len(t)), 3000, 9000) * np.exp(-t * 500)
        return _norm(tone) * 0.7 + _norm(click) * 0.5

    return _fade(_norm(_mix(ping([2350, 3580, 5150], 55), _delay(ping([2600, 3900], 70) * 0.55, 0.045))))


def lock(rng, distance: float) -> np.ndarray:
    """Landing = thud, then the latch catches."""
    weight = min(1.0, 0.4 + distance / 8)
    return _norm(_mix(thud(rng, weight) * weight, _delay(latch(rng) * 0.32, 0.035)))


def scrape(rng, seconds: float, distance: float, falling: bool) -> np.ndarray:
    """Wood sliding on wood (or the whoosh of a falling crate). Pitch rises with distance."""
    t = _t(seconds + 0.06)
    noise = rng.standard_normal(len(t))
    if falling:
        lo, hi = 300 + 50 * distance, 1500 + 260 * distance
    else:
        lo, hi = 160 + 35 * distance, 900 + 150 * distance
    x = _band(noise, lo, hi)
    grain = 1 + 0.7 * _norm(_band(rng.standard_normal(len(t)), 6, 35))  # stick-slip roughness
    swell = np.minimum(1, t / max(seconds, 0.05)) ** (2 if falling else 0.6)  # louder as it speeds up
    return _fade(_norm(x * (grain if not falling else 1) * swell), 0.01, 0.05)


def thunder(rng) -> np.ndarray:
    t = _t(3.2)
    crack = _norm(_band(rng.standard_normal(len(t)), 500, 5000)) * np.exp(-t * 18)
    brown = np.cumsum(rng.standard_normal(len(t)))  # drift is removed by the band-pass below
    rumble = _norm(_band(brown, 25, 220))
    swells = 1 + 0.6 * _norm(_band(rng.standard_normal(len(t)), 0.5, 4))
    env = np.minimum(1, t / 0.12) * np.exp(-t * 1.1)
    return _fade(_norm(0.6 * crack + rumble * env * swells), 0.002, 0.3)


def wave(rng) -> np.ndarray:
    """A wave slamming the hull: a swelling wash of noise over a low boom."""
    t = _t(2.2)
    wash = _norm(_band(rng.standard_normal(len(t)), 140, 3200))
    env = np.where(t < 0.35, (t / 0.35) ** 2, np.exp(-(t - 0.35) * 2.2))
    boom = np.sin(2 * np.pi * np.cumsum(38 + 30 * np.exp(-t * 6)) / SR) * np.exp(-t * 3) * (t > 0.3)
    return _fade(_norm(wash * env + 0.8 * boom), 0.01, 0.3)


def creak(rng, seconds: float = 1.0) -> np.ndarray:
    """Timber groaning: a wobbling-rate impulse train through wood-like resonances."""
    t = _t(seconds)
    rate = 120 + 70 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 6)) + 25 * _norm(_band(rng.standard_normal(len(t)), 1, 8))
    pulses = np.diff(np.floor(np.cumsum(rate) / SR), prepend=0.0)
    body = _band(pulses, 450, 1300, order=3) + 0.5 * _band(pulses, 1700, 2500, order=3)
    env = np.sin(np.pi * t / seconds) ** 0.7
    return _fade(_norm(body * env), 0.02, 0.05)


def bell(f0: float, seconds: float = 2.6) -> np.ndarray:
    """A ship's bell: inharmonic partials (hum, prime, tierce, quint, nominal...)."""
    t = _t(seconds)
    partials = [(0.5, 0.35, 1.2), (1.0, 1.0, 1.6), (1.19, 0.45, 2.4), (1.5, 0.3, 2.8), (2.0, 0.5, 3.0), (2.74, 0.25, 4.5), (3.0, 0.2, 5.0)]
    tone = sum(a * np.sin(2 * np.pi * f0 * m * t) * np.exp(-t * d) for m, a, d in partials)
    strike = np.exp(-t * 300) * np.sin(2 * np.pi * f0 * 4.2 * t) * 0.3
    return _fade(_norm(tone + strike), 0.001, 0.2)


def victory() -> np.ndarray:
    notes = [523.25, 659.25, 783.99, 1046.5]
    return _norm(_mix(*[_delay(bell(f) * (0.9 - 0.1 * i), 0.14 * i) for i, f in enumerate(notes)]))


def stalemate() -> np.ndarray:
    return _norm(_mix(bell(392.0), _delay(bell(349.23) * 0.8, 0.32)))


def tick(rng) -> np.ndarray:
    t = _t(0.05)
    x = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 120) + 0.4 * _band(rng.standard_normal(len(t)), 2000, 7000) * np.exp(-t * 200)
    return _fade(_norm(x))


def rain_loop(rng, seconds: float = 6.0) -> np.ndarray:
    """Rain on deck plus a low sea wash. FFT filtering makes it loop seamlessly."""
    n = int(seconds * SR)
    hiss = _norm(_band(rng.standard_normal(n), 1500, 9000))
    drops = np.zeros(n)
    idx = rng.integers(0, n, size=int(seconds * 90))
    drops[idx] = rng.uniform(0.3, 1.0, size=len(idx))
    drops = _norm(_band(drops, 1800, 6000))
    sea = _norm(_band(rng.standard_normal(n), 60, 400)) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / n * 2))
    return _norm(0.35 * hiss + 0.35 * drops + 0.5 * sea)


# --- playback ------------------------------------------------------------------------


class SoundBank:
    """Builds every sound once at startup and plays them. Becomes silent (not broken)
    when no audio device is available, e.g. in headless tests."""

    DISTANCES = range(1, 9)

    def __init__(self, enabled: bool = True, volume: float = 0.8):
        import pygame

        self.pygame = pygame
        self.muted = False
        self.volume = volume
        init = pygame.mixer.get_init() if enabled else None
        self.enabled = init is not None
        self.sounds: dict[str, object] = {}
        self._rain_channel = None
        self._rain_level = 0.0
        if not self.enabled:
            return
        self._channels = init[2]
        pygame.mixer.set_num_channels(32)
        rng = np.random.default_rng(2026)
        from cargostorm.ui.tween import fall_duration

        raw = {
            "thunder": thunder(rng),
            "wave": wave(rng),
            "creak": creak(rng),
            "creak_short": creak(rng, 0.5),
            "victory": victory(),
            "draw": stalemate(),
            "tick": tick(rng),
            "rain": rain_loop(rng),
        }
        for d in self.DISTANCES:
            raw[f"lock{d}"] = lock(rng, d)
            raw[f"thud{d}"] = thud(rng, min(1.0, 0.3 + d / 8))
            raw[f"fall{d}"] = scrape(rng, fall_duration(d), d, falling=True)
            raw[f"slide{d}"] = scrape(rng, fall_duration(d), d, falling=False)
        self.sounds = {name: self._make(x) for name, x in raw.items()}
        self._rain_channel = self.sounds["rain"].play(loops=-1)
        self.set_rain(0.25)

    def _make(self, x: np.ndarray):
        pcm = (np.clip(x, -1, 1) * 0.9 * 32767).astype(np.int16)
        pcm = np.ascontiguousarray(np.repeat(pcm[:, None], self._channels, axis=1))
        return self.pygame.sndarray.make_sound(pcm)

    def play(self, name: str, volume: float = 1.0, pan: float = 0.0) -> None:
        """`pan` runs from -1 (left) to 1 (right)."""
        if not self.enabled or self.muted or name not in self.sounds:
            return
        channel = self.sounds[name].play()
        if channel is not None:
            v = max(0.0, min(1.0, volume * self.volume))
            left = v * min(1.0, 1 - pan)
            right = v * min(1.0, 1 + pan)
            channel.set_volume(left, right)

    def play_distance(self, kind: str, distance: float, volume: float = 1.0, pan: float = 0.0) -> None:
        d = int(min(max(round(distance), self.DISTANCES.start), self.DISTANCES.stop - 1))
        self.play(f"{kind}{d}", volume, pan)

    def set_rain(self, level: float) -> None:
        self._rain_level = level
        if self._rain_channel is not None:
            self._rain_channel.set_volume(0.0 if self.muted else level * self.volume)

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self.set_rain(self._rain_level)
