"""
Phase 2: Audio Device Manager.

Handles:
  - Enumerating Windows audio devices, filtered to WASAPI (most reliable
    host API for this use case -- lower latency than MME/DirectSound,
    and unlike WDM-KS it doesn't require exclusive device access, so it
    won't fight with other apps that also want the mic).
  - An interactive CLI picker for input/output devices.
  - Saving the picked devices (by name, not index -- Windows device
    indices shift between reboots/reconnects) into config/settings.yaml.
  - On subsequent runs, offering "press 1 to reuse last picks" instead
    of forcing a re-pick every time.

Run directly to exercise the Phase 2 goal ("reliable audio routing"):
    python -m audio.devices
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import sounddevice as sd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
PREFERRED_HOST_API_SUBSTRING = "WASAPI"


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


def _get_preferred_hostapi_index() -> int:
    """Find the host API index whose name contains our preferred substring.
    Falls back to the system default host API if WASAPI isn't found for
    some reason (shouldn't happen on a normal Windows install)."""
    for i, api in enumerate(sd.query_hostapis()):
        if PREFERRED_HOST_API_SUBSTRING.lower() in api["name"].lower():
            return i
    print(f"WARNING: no host API matched '{PREFERRED_HOST_API_SUBSTRING}', "
          f"falling back to default host API.")
    return sd.default.hostapi


def list_devices(kind: str | None = None) -> list[DeviceInfo]:
    """List devices on the preferred host API.

    kind: None for all, "input" for input-capable, "output" for output-capable.
    """
    hostapi_idx = _get_preferred_hostapi_index()
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] != hostapi_idx:
            continue
        info = DeviceInfo(
            index=i,
            name=dev["name"],
            max_input_channels=dev["max_input_channels"],
            max_output_channels=dev["max_output_channels"],
            default_samplerate=dev["default_samplerate"],
        )
        if kind == "input" and not info.is_input:
            continue
        if kind == "output" and not info.is_output:
            continue
        devices.append(info)
    return devices


def _print_device_list(devices: list[DeviceInfo]) -> None:
    for n, dev in enumerate(devices, start=1):
        print(f"  {n}. {dev.name}  "
              f"(default rate: {int(dev.default_samplerate)} Hz)")


def _prompt_pick(devices: list[DeviceInfo], label: str) -> DeviceInfo:
    if not devices:
        print(f"No {label} devices found on {PREFERRED_HOST_API_SUBSTRING}. "
              f"Is the device connected / enabled in Windows Sound settings?")
        sys.exit(1)

    print(f"\nAvailable {label} devices (WASAPI):")
    _print_device_list(devices)
    while True:
        choice = input(f"Pick {label} device [1-{len(devices)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(devices):
            return devices[int(choice) - 1]
        print("Invalid choice, try again.")


def load_settings() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_settings(settings: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)


def _find_saved_device(saved_name: str, devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Resolve a saved device name against the current device list.
    Returns None if the device is no longer present (unplugged, renamed,
    driver reinstalled, etc.) so the caller can fall back to a manual pick."""
    for dev in devices:
        if dev.name == saved_name:
            return dev
    return None


def select_devices(force_repick: bool = False) -> tuple[DeviceInfo, DeviceInfo]:
    """Returns (input_device, output_device).

    If a previous selection is saved and still valid, offers a quick
    "press 1 to reuse" shortcut. Otherwise runs the full interactive picker.
    """
    settings = load_settings()
    audio_cfg = settings.get("audio", {}) or {}
    saved_input_name = audio_cfg.get("input_device")
    saved_output_name = audio_cfg.get("output_device")

    input_devices = list_devices("input")
    output_devices = list_devices("output")

    saved_input = _find_saved_device(saved_input_name, input_devices) if saved_input_name else None
    saved_output = _find_saved_device(saved_output_name, output_devices) if saved_output_name else None

    if not force_repick and saved_input and saved_output:
        print("Last used devices:")
        print(f"  Input:  {saved_input.name}")
        print(f"  Output: {saved_output.name}")
        choice = input("Press 1 to reuse these, or 2 to pick new devices: ").strip()
        if choice != "2":
            return saved_input, saved_output

    input_dev = _prompt_pick(input_devices, "input")
    output_dev = _prompt_pick(output_devices, "output")

    audio_cfg["input_device"] = input_dev.name
    audio_cfg["output_device"] = output_dev.name
    settings["audio"] = audio_cfg
    save_settings(settings)

    print(f"\nSaved selection to {CONFIG_PATH}")
    return input_dev, output_dev


def pick_device_now(kind: str, label: str) -> DeviceInfo:
    """One-off interactive pick, bypassing the saved-selection shortcut.
    Useful for scripts that need a specific device for a single run
    (e.g. picking speakers/headphones for a test, separate from the
    CABLE Input saved as the app's normal output_device)."""
    devices = list_devices(kind)
    return _prompt_pick(devices, label)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Audio Device Manager")
    print("=" * 60)
    in_dev, out_dev = select_devices()
    print("\nFinal selection:")
    print(f"  Input:  [{in_dev.index}] {in_dev.name} @ {int(in_dev.default_samplerate)} Hz")
    print(f"  Output: [{out_dev.index}] {out_dev.name} @ {int(out_dev.default_samplerate)} Hz")
    print("\nPhase 2 goal met if both devices resolved correctly above.")
    print("Next: Phase 3, live audio capture (ring buffer, 20ms chunks).")
