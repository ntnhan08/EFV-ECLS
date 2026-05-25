"""
EFV – Usage Examples
====================
Demonstrates every major feature of the EFV library.
"""
import efv

print(f"EFV v{efv.__version__}")
print(f"Backend : {efv.backend_name()}")
print(f"GPU     : {efv.gpu_available()}")
print()

# ══════════════════════════════════════════════════════════════════════════
#  1.  Fluent / chained API
# ══════════════════════════════════════════════════════════════════════════

def example_chain():
    """Full pipeline using the fluent Video class."""
    (efv.Video("input.avi", quality=90)

        # ── Timeline ──────────────────────────────────────────────────
        .trim(start_sec=2.0, end_sec=30.0)      # Keep seconds 2-30
        .cut(start_sec=10.0, end_sec=12.0)      # Remove 10-12 s glitch

        # ── Geometry ──────────────────────────────────────────────────
        .resize(1280, 720)                       # 720p
        .crop(0, 0, 1280, 720)                  # Crop (already 720p here)
        .rotate(0)                               # No rotation
        .flip(horizontal=False, vertical=False)  # No flip

        # ── Audio ─────────────────────────────────────────────────────
        .volume(1.2)                             # Boost audio 20%
        .fade_audio(in_sec=0.5, out_sec=1.0)    # Audio fades

        # ── Color ─────────────────────────────────────────────────────
        .brightness(1.05)                        # Slightly brighter
        .contrast(1.1)                           # More punch
        .saturation(1.15)                        # More vivid colours

        # ── Filters ───────────────────────────────────────────────────
        .sharpen(0.8)                            # Subtle sharpening
        .vignette(strength=0.3, radius=0.75)    # Mild vignette

        # ── Overlays ──────────────────────────────────────────────────
        .add_logo("logo.png",
                  position="topright",
                  scale=0.15,
                  opacity=0.85,
                  margin=12)
        .add_watermark("© My Channel 2025",
                       opacity=0.25,
                       font_size=28)
        .add_text("EFV Demo",
                  x=20, y=20,
                  font_size=32,
                  color=(255, 220, 0),
                  duration_sec=3.0)
        .add_subtitle("Welcome to EFV!",
                      start_sec=1.0, end_sec=4.0,
                      font_size=30)

        # ── Transitions ───────────────────────────────────────────────
        .fade_in(0.5)
        .fade_out(1.0)

        # ── Save ──────────────────────────────────────────────────────
        .save("output_chain.avi")
    )


# ══════════════════════════════════════════════════════════════════════════
#  2.  Standalone one-liner functions
# ══════════════════════════════════════════════════════════════════════════

def example_standalone():
    """Every standalone function demonstrated."""

    # Audio
    efv.mute("input.avi",        "muted.avi")
    efv.add_audio("input.avi",   "with_music.avi",  audio_path="music.wav", volume=0.8)
    efv.volume("input.avi",      "louder.avi",       factor=1.5)
    efv.fade_audio("input.avi",  "fade.avi",         in_sec=1.0, out_sec=2.0)

    # Overlays
    efv.add_logo("input.avi",      "logo.avi",      logo_path="logo.png", position="bottomright")
    efv.add_watermark("input.avi", "wm.avi",        text="© 2025")
    efv.add_text("input.avi",      "text.avi",      text="Hello!", x=50, y=50, font_size=40)
    efv.add_subtitle("input.avi",  "sub.avi",       text="Subtitle here", start_sec=5.0, end_sec=10.0)

    # Geometry
    efv.resize("input.avi",   "1080p.avi",     1920, 1080)
    efv.crop("input.avi",     "cropped.avi",   100, 50, 1280, 720)
    efv.zoom("input.avi",     "zoomed.avi",    factor=1.5)
    efv.rotate("input.avi",   "rotated.avi",   degrees=90)
    efv.flip("input.avi",     "flipped.avi",   horizontal=True)

    # Color
    efv.brightness("input.avi",  "bright.avi",  factor=1.2)
    efv.contrast("input.avi",    "contrast.avi", factor=1.3)
    efv.saturation("input.avi",  "vivid.avi",   factor=1.4)
    efv.greyscale("input.avi",   "grey.avi")

    # LUT (colour grade)
    lut = efv.build_gamma_lut(gamma_r=1.1, gamma_g=1.0, gamma_b=0.9)  # warm tone
    efv.apply_lut("input.avi",   "graded.avi",  lut=lut)

    # Filters
    efv.blur("input.avi",        "blurred.avi", radius=3)
    efv.sharpen("input.avi",     "sharp.avi",   strength=1.2)
    efv.denoise("input.avi",     "clean.avi",   strength=1.5)
    efv.vignette("input.avi",    "vignette.avi", strength=0.4)

    # Timeline
    efv.trim("input.avi",    "trimmed.avi",  start_sec=5.0, end_sec=20.0)
    efv.cut("input.avi",     "cut.avi",      start_sec=8.0, end_sec=10.0)
    efv.speed("input.avi",   "fast.avi",     factor=2.0)
    efv.reverse("input.avi", "reversed.avi")
    efv.loop("input.avi",    "loop3x.avi",   times=3)

    # Transitions
    efv.fade_in("input.avi",    "fadein.avi",  duration_sec=1.0)
    efv.fade_out("input.avi",   "fadeout.avi", duration_sec=1.0)
    efv.crossfade("input.avi",  "xfade.avi",   other_path="input2.avi", duration_sec=1.5)


# ══════════════════════════════════════════════════════════════════════════
#  3.  Get video info
# ══════════════════════════════════════════════════════════════════════════

def example_info():
    meta = efv.info("input.avi")
    print(f"  Size    : {meta['width']}×{meta['height']}")
    print(f"  FPS     : {meta['fps']:.2f}")
    print(f"  Frames  : {meta['frame_count']}")
    print(f"  Duration: {meta['duration_sec']:.2f} s")
    print(f"  Codec   : {meta['codec']}")
    print(f"  Audio   : {meta['has_audio']}  ({meta['sample_rate']} Hz, {meta['channels']}ch)")


# ══════════════════════════════════════════════════════════════════════════
#  4.  Pending ops introspection
# ══════════════════════════════════════════════════════════════════════════

def example_inspect():
    v = (efv.Video("input.avi")
             .mute()
             .resize(1280, 720)
             .brightness(1.1))
    print(f"Pending ops: {v.pending_ops()}")


if __name__ == "__main__":
    import os
    if os.path.exists("input.avi"):
        example_chain()
        example_standalone()
        example_info()
    else:
        print("Place an 'input.avi' in the current directory to run these examples.")
    example_inspect()
