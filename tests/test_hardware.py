"""Hardware detection: budget math + describe() shape (detectors mocked)."""

import hardware


def test_total_ram_positive():
    assert hardware.total_ram_gb() > 0


def test_memory_budget_adds_vram(monkeypatch):
    monkeypatch.setattr(hardware, "total_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "gpu_vram_gb", lambda: 8.0)
    assert hardware.memory_budget_gb() == 24.0


def test_memory_budget_ram_only_when_no_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "total_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "gpu_vram_gb", lambda: None)
    assert hardware.memory_budget_gb() == 16.0


def test_describe_shape(monkeypatch):
    monkeypatch.setattr(hardware, "detect_gpus", lambda: ())
    monkeypatch.setattr(hardware, "gpu_vram_gb", lambda: None)
    monkeypatch.setattr(hardware, "has_qualcomm_npu", lambda: False)
    monkeypatch.setattr(hardware, "has_intel_npu", lambda: False)
    monkeypatch.setattr(hardware, "has_amd_npu", lambda: False)
    d = hardware.describe()
    assert {"ram_gb", "gpu", "gpus", "vram_gb", "npus", "budget_gb"} <= set(d)
    assert d["gpu"] is False and d["gpus"] == []
    assert set(d["npus"]) == {"qualcomm", "intel", "amd"}


def test_describe_reports_gpu(monkeypatch):
    g = hardware.Gpu(vendor="nvidia", vram_gb=24.0, name="RTX 4090")
    monkeypatch.setattr(hardware, "detect_gpus", lambda: (g,))
    monkeypatch.setattr(hardware, "gpu_vram_gb", lambda: 24.0)
    monkeypatch.setattr(hardware, "has_qualcomm_npu", lambda: False)
    monkeypatch.setattr(hardware, "has_intel_npu", lambda: False)
    monkeypatch.setattr(hardware, "has_amd_npu", lambda: False)
    d = hardware.describe()
    assert d["gpu"] is True
    assert d["gpus"][0]["vendor"] == "nvidia" and d["gpus"][0]["vram_gb"] == 24.0
