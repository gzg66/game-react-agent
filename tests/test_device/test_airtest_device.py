"""Tests for AirtestDevice screenshot normalization and fallback."""

from __future__ import annotations

from game_agent.device.airtest_device import AirtestDevice
from game_agent.exceptions import PerceptionError


PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


class _FakeAirtestDevice:
    def __init__(self, result):
        self._result = result

    def snapshot(self):
        return self._result


def _make_device(raw_device=None) -> AirtestDevice:
    device = object.__new__(AirtestDevice)
    device._device = raw_device
    device._adb_screenshot = lambda: None
    return device


def test_screenshot_prefers_raw_device_snapshot(monkeypatch):
    device = _make_device(_FakeAirtestDevice(PNG_BYTES))
    monkeypatch.setattr("game_agent.device.airtest_device.snapshot", lambda filename=None: None)

    data = device.screenshot()

    assert data == PNG_BYTES


def test_screenshot_supports_airtest_snapshot_file_dict(monkeypatch, tmp_path):
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(PNG_BYTES)
    device = _make_device()
    monkeypatch.setattr(
        "game_agent.device.airtest_device.snapshot",
        lambda filename=None: {"screen": str(screenshot_path)},
    )

    data = device.screenshot()

    assert data == PNG_BYTES


def test_screenshot_falls_back_to_adb(monkeypatch):
    device = _make_device(_FakeAirtestDevice(None))
    monkeypatch.setattr("game_agent.device.airtest_device.snapshot", lambda filename=None: None)
    device._adb_screenshot = lambda: PNG_BYTES

    data = device.screenshot()

    assert data == PNG_BYTES


def test_screenshot_raises_when_all_providers_fail(monkeypatch):
    device = _make_device(_FakeAirtestDevice(None))
    monkeypatch.setattr("game_agent.device.airtest_device.snapshot", lambda filename=None: None)

    try:
        device.screenshot()
    except PerceptionError as exc:
        assert "ADB" in str(exc)
    else:
        raise AssertionError("expected PerceptionError when all screenshot providers fail")


def test_snapshot_to_file_reuses_screenshot_chain(monkeypatch, tmp_path):
    output_path = tmp_path / "nested" / "screen.png"
    device = _make_device(_FakeAirtestDevice(PNG_BYTES))
    monkeypatch.setattr("game_agent.device.airtest_device.snapshot", lambda filename=None: None)

    ok = device.snapshot_to_file(str(output_path))

    assert ok is True
    assert output_path.read_bytes() == PNG_BYTES


def test_snapshot_to_file_returns_false_when_screenshot_fails(monkeypatch, tmp_path):
    output_path = tmp_path / "screen.png"
    device = _make_device(_FakeAirtestDevice(None))
    monkeypatch.setattr("game_agent.device.airtest_device.snapshot", lambda filename=None: None)

    ok = device.snapshot_to_file(str(output_path))

    assert ok is False
    assert output_path.exists() is False
