# AI Target Speaker Isolation Microphone

## Phase 1: Project Setup (start here)

Run these in **PowerShell**, on your Windows machine, from wherever you want the project folder to live (e.g. `D:\Projects\`).

### 1. Create the venv

```powershell
git clone <your-empty-repo-url> voice-isolation
cd voice-isolation
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Use Python 3.11 (3.12 also fine) -- avoid 3.13 for now, some audio/ML wheels lag behind on Windows.

### 2. Install PyTorch with CUDA first

Do this **before** `requirements.txt`, otherwise pip may grab the CPU-only build:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This matches your RTX 4070 Super (Ada Lovelace, CUDA 12.x compatible). If you already have a newer NVIDIA driver and want CUDA 12.4 wheels instead, swap `cu121` for `cu124` in the URL -- either works fine on a 4070 Super.

### 3. Install the rest

```powershell
pip install -r requirements.txt
```

### 4. Verify

```powershell
python verify_setup.py
```

You're looking for:
- `CUDA available: True`
- Your RTX 4070 Super listed under CUDA devices
- A GPU matmul time under ~15ms
- Your PD100X (and other input devices) listed under `sounddevice`

If `CUDA available` is `False`, it's almost always because step 2 was skipped and pip pulled the CPU-only torch wheel -- run `pip uninstall torch torchaudio` and redo step 2.

Once `verify_setup.py` is clean, Phase 1's goal ("load AI model successfully" -- in our case, "environment can run GPU inference at all") is done, and we move to Phase 2 (audio device manager).

---

## Model decision (read before Phase 4)

Your original plan listed **Waveformer** and **ClearerVoice-Studio** as candidates. Neither actually does what "Voice Enrollment" (Phase 5) describes:

- **Waveformer** extracts fixed *sound event classes* (footsteps, barking, typing...), not a specific person's voice. There's no speaker embedding in it at all.
- **ClearerVoice-Studio**'s target speaker extraction model (`AV_MossFormer2_TSE_16K`) is **audio-visual** -- it needs a synced video of your face to know who to extract, not an audio-only voice enrollment.

The models that actually do "enroll 10-20s of my voice, then extract just me from a mix" are research-grade, not one-line-installs:

- **SpEx+** (`gemengtju/SpEx_Plus`) -- time-domain, causal-friendly, pretrained checkpoints exist. This is what we're building on.
- **WeSep** (`wenet-e2e/wesep`) -- newer, actively maintained toolkit built specifically for this task, integrates WeSpeaker embeddings. Fallback if SpEx+ checkpoint quality disappoints once we test it.

Neither ships a ready streaming/real-time wrapper -- that's genuinely on us to build in Phase 6. Realistic first-pass latency target is **80-150ms**, with 25-40ms as a later optimization goal once the pipeline is proven, not a week-one expectation.

## VB-CABLE quick note (since it's just sitting there unused)

VB-CABLE creates two virtual audio devices: **"CABLE Input"** (a virtual playback device) and **"CABLE Output"** (a virtual recording device). Anything played to CABLE Input instantly appears as a recording source on CABLE Output. So the trick is:

1. Our Python app's audio *output* goes to **CABLE Input**.
2. In Discord/Zoom/OBS/games, you select **CABLE Output** as your microphone.

We'll wire this up concretely in Phase 7, but it's worth setting your Windows Sound settings so you understand the routing now: Settings > Sound > check that "CABLE Input" and "CABLE Output" both show up as devices. Don't set either as your default -- our app will target them explicitly.
