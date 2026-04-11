from __future__ import annotations

import sounddevice as sd


def print_devices() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, dev in enumerate(devices):
        ha = hostapis[dev["hostapi"]]["name"]
        in_ch = dev["max_input_channels"]
        out_ch = dev["max_output_channels"]
        default_in = i == sd.default.device[0]
        default_out = i == sd.default.device[1]
        flags = []
        if default_in:
            flags.append("default in")
        if default_out:
            flags.append("default out")
        suffix = f" ({', '.join(flags)})" if flags else ""
        print(f"{i}: {dev['name']!r} — hostapi={ha!r} in={in_ch} out={out_ch}{suffix}")
