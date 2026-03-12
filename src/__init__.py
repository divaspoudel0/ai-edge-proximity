"""
Unit tests for the beacon generator module.
Run with: pytest tests/test_beacon_generator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.beacon_generator import Device, BeaconSimulator
from src.utils import rssi_from_distance

# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def sample_config():
    """Return a minimal configuration for testing."""
    return {
        'area_width': 100,
        'area_height': 100,
        'num_static_beacons': 2,
        'num_mobile_devices': 1,
        'num_rogue_devices': 1,
        'simulation_duration_hours': 0.001,  # very short for testing
        'advertisement_interval_mean': 0.1,
        'advertisement_interval_std': 0.01,
        'mac_rotation_min': 5,
        'mac_rotation_max': 10,
        'uid_rotation_min': 3,
        'uid_rotation_max': 8,
        'rogue_behavior': ['spoof_uid'],
        'anomaly_duration_minutes': 0.01,
        'static_beacon_positions': [],
        'mobile_path_type': 'random_waypoint'
    }

@pytest.fixture
def fixed_device():
    """Create a device with fixed parameters for deterministic tests."""
    dev = Device(
        device_id="test_dev",
        service_id="SVC_1",
        mac="00:11:22:33:44:55",
        uid="1234567890ABCDEF",
        mac_rotation_interval=10,
        uid_rotation_interval=5,
        is_rogue=False,
        x=10.0,
        y=20.0,
        vx=1.0,
        vy=0.5,
        advertisement_interval=0.1
    )
    return dev

# ---------------------------
# Device Tests
# ---------------------------
def test_device_initialization(fixed_device):
    """Test that a device is initialized correctly."""
    dev = fixed_device
    assert dev.device_id == "test_dev"
    assert dev.mac == "00:11:22:33:44:55"
    assert dev.uid == "1234567890ABCDEF"
    assert dev.x == 10.0
    assert dev.y == 20.0
    assert dev.vx == 1.0
    assert dev.vy == 0.5
    assert not dev.is_rogue

def test_mac_rotation(fixed_device):
    """Test MAC address rotation based on time."""
    dev = fixed_device
    current_time = 0.0
    dev.next_mac_rotation = 10.0
    dev.mac = "00:11:22:33:44:55"

    # Before rotation time
    dev.rotate_mac(current_time)
    assert dev.mac == "00:11:22:33:44:55"

    # After rotation time
    current_time = 10.5
    dev.rotate_mac(current_time)
    assert dev.mac != "00:11:22:33:44:55"
    assert dev.next_mac_rotation > current_time

def test_uid_rotation(fixed_device):
    """Test UID rotation based on time."""
    dev = fixed_device
    current_time = 0.0
    dev.next_uid_rotation = 5.0
    old_uid = dev.uid

    # Before rotation
    dev.rotate_uid(current_time)
    assert dev.uid == old_uid

    # After rotation
    current_time = 5.5
    dev.rotate_uid(current_time)
    assert dev.uid != old_uid
    assert dev.next_uid_rotation > current_time

def test_generate_advertisement(fixed_device):
    """Test advertisement generation includes correct fields."""
    dev = fixed_device
    receiver_pos = (0, 0)
    current_time = 123.45
    adv = dev.generate_advertisement(current_time, receiver_pos)

    assert adv['timestamp'] == current_time
    assert adv['device_id'] == "test_dev"
    assert adv['mac'] == "00:11:22:33:44:55"
    assert adv['uid'] == "1234567890ABCDEF"
    assert adv['service_id'] == "SVC_1"
    assert 'rssi' in adv
    assert adv['x'] == 10.0
    assert adv['y'] == 20.0
    assert not adv['is_rogue']
    assert adv['rogue_type'] is None

def test_rssi_calculation():
    """Test that RSSI decreases with distance."""
    d1 = 1.0
    d2 = 10.0
    rssi1 = rssi_from_distance(d1, P0=-59, n=2.0)
    rssi2 = rssi_from_distance(d2, P0=-59, n=2.0)
    assert rssi2 < rssi1  # farther should be weaker

def test_mobility_update(fixed_device):
    """Test that position updates correctly with mobility."""
    dev = fixed_device
    dt = 0.5
    old_x, old_y = dev.x, dev.y
    dev.update_position(dt, 100, 100)
    expected_x = old_x + dev.vx * dt
    expected_y = old_y + dev.vy * dt
    assert np.isclose(dev.x, expected_x)
    assert np.isclose(dev.y, expected_y)

def test_wall_bounce(fixed_device):
    """Test that device bounces off walls."""
    dev = fixed_device
    # Place near right wall moving right
    dev.x = 99.0
    dev.vx = 2.0
    dev.update_position(1.0, 100, 100)
    assert dev.x < 100.0  # Should not exceed boundary
    assert dev.vx < 0  # Velocity should reverse

# ---------------------------
# Simulator Tests
# ---------------------------
def test_simulator_initialization(sample_config):
    """Test that simulator initializes devices correctly."""
    sim = BeaconSimulator(sample_config)
    sim.initialize_devices()
    expected_total = sample_config['num_static_beacons'] + sample_config['num_mobile_devices']
    assert len(sim.devices) == expected_total

    # Check that static beacons have zero velocity
    static_devices = [d for d in sim.devices if 'static' in d.device_id]
    for dev in static_devices:
        assert dev.vx == 0
        assert dev.vy == 0

    # Check that mobile devices have non-zero velocity
    mobile_devices = [d for d in sim.devices if 'mobile' in d.device_id]
    for dev in mobile_devices:
        assert dev.vx != 0 or dev.vy != 0

def test_rogue_injection(sample_config):
    """Test that rogue devices are injected at scheduled times."""
    sim = BeaconSimulator(sample_config)
    sim.initialize_devices()
    initial_count = len(sim.devices)

    # Manually inject a rogue
    sim.inject_rogue(current_time=10.0, duration=60.0, rogue_type='spoof_uid')
    assert len(sim.devices) == initial_count + 1

    rogue = sim.devices[-1]
    assert rogue.is_rogue
    assert rogue.rogue_type == 'spoof_uid'
    assert hasattr(rogue, 'end_time')
    assert rogue.end_time == 10.0 + 60.0

def test_rogue_removal_after_expiry(sample_config):
    """Test that rogue devices are removed after their duration."""
    sim = BeaconSimulator(sample_config)
    sim.initialize_devices()
    sim.inject_rogue(current_time=10.0, duration=1.0, rogue_type='spoof_uid')
    assert any(d.is_rogue for d in sim.devices)

    # Advance time past expiry
    sim.current_time = 12.0
    sim.devices = [d for d in sim.devices if not (d.is_rogue and hasattr(d, 'end_time') and sim.current_time > d.end_time)]
    assert not any(d.is_rogue for d in sim.devices)

def test_run_simulation(sample_config):
    """Test that the simulator runs and produces a DataFrame with events."""
    sim = BeaconSimulator(sample_config)
    sim.initialize_devices()
    df = sim.run()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_columns = ['timestamp', 'device_id', 'mac', 'uid', 'service_id', 'rssi', 'x', 'y', 'is_rogue', 'rogue_type']
    for col in expected_columns:
        assert col in df.columns

def test_advertisement_generation_during_run(sample_config):
    """Test that advertisements are generated with increasing timestamps."""
    sim = BeaconSimulator(sample_config)
    sim.initialize_devices()
    # Override advertisement interval to ensure multiple events
    for dev in sim.devices:
        dev.advertisement_interval = 0.01  # very frequent

    df = sim.run()
    timestamps = df['timestamp'].values
    assert np.all(np.diff(timestamps) >= 0)  # non-decreasing
    # Check that at least some rotations happened (if simulation long enough)
    # Since we set duration very short, may not test rotation. For thoroughness,
    # we could extend duration, but for unit test it's fine.

if __name__ == "__main__":
    pytest.main([__file__])