# Capture Guide

Original VHS capture hardware/software reference, preserved from legacy step_1 documentation.

**Panasonic AG-1970P VCR settings:**

| Setting | Value |
|---|---|
| Phone Level | Neutral |
| Picture | Neutral |
| Hi-fi Rec Level | Neutral |
| Noise Filter | Off |
| TBC | On |
| Search Sound | Off |
| HiFi/NormalMix | Off |
| Tape Select | T120 |
| Mono | On |
| MTS | MTS |

A1 connectors (back of VCR): S-Video out to Osprey 260e S-Video In; Right and Left Audio to Osprey 260e Unbalanced Audio In.

**Osprey 260e settings (all defaults except):**

- RefSize → Horizontal Format: CCIR-601, Source Width: 720
- Input → Video Input: S-Video
- Video Decoder → Video Standard: NTSC_M
- Filters → SimulStream: Unchecked

**Software:** Install UT Video driver, Osprey driver, and unzip VirtualDub from the `software/` directory.

**VirtualDub Capture Settings:**

1. Capture framerate: 29.97 FPS
2. Audio → Compression: PCM 48.000 kHz 16-bit mono (avoids blank-track noise from stereo mix)
3. Video → Compression: `UtVideo YUV422 BT.601.VCM`
4. Capture → Settings → Abort options → uncheck "Abort on left mouse button" (use ESC instead)
5. Capture → Timing → Internal capture mode synchronization: no correction
6. Capture → Stop conditions → Capture time exceeds: 5400 seconds (max for VHS-C EP mode)
7. File → Set Capture File every time (it overwrites the last file by default)

**Notes:**

- UT Video is the only encoder that reliably avoided dropped frames and kept color in VHS ranges.
- T2 UtVideo YUV422 BT.601.VCM is faster but VLC doesn't support it for validation.
- If the VCR stops playing a VHS-C tape in its cassette adapter, the battery may be low.
