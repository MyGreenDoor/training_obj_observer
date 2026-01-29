#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point wrapper for panoptic stereo training (non-SDF).
"""
import train_panoptic_utils_rot as _panoptic


def __getattr__(name: str):
    return getattr(_panoptic, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(_panoptic)))


def main() -> None:
    _panoptic.main()


if __name__ == "__main__":
    main()
