/// std/dsp/utils: small audio unit conversions.
///
/// Helpers for moving between the units DSP code mixes constantly: linear
/// amplitude vs decibels, and MIDI note numbers vs frequency in hertz. All
/// single precision.

import std/math { powf, log10f, log2f }

// TODO: the compiler should be able to infer these are `@alloc(false)`.

/// Convert a gain in decibels to a linear amplitude multiplier (0 dB -> 1.0).
pub proc db_to_lin(db: f32) f32 { return powf(10.0, db / 20.0); }
/// Convert a linear amplitude multiplier to decibels (1.0 -> 0 dB).
pub proc lin_to_db(lin: f32) f32 { return 20.0 * log10f(lin); }
/// Convert a MIDI note number to frequency in hertz (A4 = note 69 = 440 Hz).
pub proc midi_to_hz(note: f32) f32 { return 440.0 * powf(2.0, (note - 69.0) / 12.0); }
/// Convert a frequency in hertz to a (fractional) MIDI note number.
pub proc hz_to_midi(hz: f32) f32 { return 69.0 + 12.0 * log2f(hz / 440.0); }
