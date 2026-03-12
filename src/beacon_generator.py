import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import uuid

@dataclass
class Device:
    """Represents a physical device (beacon or mobile)."""
    device_id: str
    service_id: str
    mac: str
    uid: str
    mac_rotation_interval: float      # seconds
    uid_rotation_interval: float
    is_rogue: bool = False
    rogue_type: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    next_mac_rotation: float = 0.0
    next_uid_rotation: float = 0.0
    advertisement_interval: float = 0.5
    last_advertisement: float = 0.0

    def update_position(self, dt, width, height):
        """Simple random waypoint mobility."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Bounce off walls
        if self.x < 0 or self.x > width:
            self.vx *= -1
            self.x = np.clip(self.x, 0, width)
        if self.y < 0 or self.y > height:
            self.vy *= -1
            self.y = np.clip(self.y, 0, height)
        # Occasionally change direction
        if np.random.rand() < 0.01:
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(0.5, 2.0)
            self.vx = speed * np.cos(angle)
            self.vy = speed * np.sin(angle)

    def rotate_mac(self, current_time):
        if current_time >= self.next_mac_rotation:
            # Generate new random MAC
            self.mac = "02:%02x:%02x:%02x:%02x:%02x" % tuple(np.random.randint(0,256,5))
            self.next_mac_rotation = current_time + self.mac_rotation_interval

    def rotate_uid(self, current_time):
        if current_time >= self.next_uid_rotation:
            # Generate new random UID (16 hex chars)
            self.uid = uuid.uuid4().hex[:16].upper()
            self.next_uid_rotation = current_time + self.uid_rotation_interval

    def generate_advertisement(self, current_time, receiver_pos):
        """Return a dict with advertisement data."""
        # Compute RSSI based on distance to receiver (assume receiver at origin for simplicity)
        dist = np.sqrt((self.x)**2 + (self.y)**2)  # receiver at (0,0) for simulation
        from .utils import rssi_from_distance
        rssi = rssi_from_distance(dist)
        return {
            'timestamp': current_time,
            'device_id': self.device_id,
            'mac': self.mac,
            'uid': self.uid,
            'service_id': self.service_id,
            'rssi': rssi,
            'x': self.x,
            'y': self.y,
            'is_rogue': self.is_rogue,
            'rogue_type': self.rogue_type
        }

class BeaconSimulator:
    def __init__(self, config):
        self.config = config
        self.devices = []
        self.receiver_pos = (0, 0)  # edge device at origin
        self.current_time = 0.0
        self.events = []

    def initialize_devices(self):
        # Static beacons
        static_pos = self.config.get('static_beacon_positions', [])
        if not static_pos:
            static_pos = np.random.rand(self.config['num_static_beacons'], 2) * [self.config['area_width'], self.config['area_height']]
        for i, (x, y) in enumerate(static_pos):
            device = Device(
                device_id=f"static_{i}",
                service_id=f"SVC_{i%3+1}",
                mac=f"00:11:22:33:{i:02x}:01",
                uid=uuid.uuid4().hex[:16].upper(),
                mac_rotation_interval=float('inf'),  # never rotate for static beacons
                uid_rotation_interval=float('inf'),
                is_rogue=False,
                x=x,
                y=y,
                vx=0,
                vy=0
            )
            self.devices.append(device)

        # Mobile devices
        for i in range(self.config['num_mobile_devices']):
            device = Device(
                device_id=f"mobile_{i}",
                service_id=f"SVC_{i%3+1}",
                mac=f"00:11:22:33:{i+10:02x}:01",
                uid=uuid.uuid4().hex[:16].upper(),
                mac_rotation_interval=np.random.uniform(self.config['mac_rotation_min'], self.config['mac_rotation_max']),
                uid_rotation_interval=np.random.uniform(self.config['uid_rotation_min'], self.config['uid_rotation_max']),
                is_rogue=False,
                x=np.random.uniform(0, self.config['area_width']),
                y=np.random.uniform(0, self.config['area_height']),
                vx=np.random.uniform(-1, 1),
                vy=np.random.uniform(-1, 1),
                advertisement_interval=np.random.normal(self.config['advertisement_interval_mean'], self.config['advertisement_interval_std'])
            )
            self.devices.append(device)

        # Rogue devices (to be injected later)
        self.rogue_schedule = []  # list of (inject_time, duration, rogue_type)
        for _ in range(self.config['num_rogue_devices']):
            inject_time = np.random.uniform(0, self.config['simulation_duration_hours']*3600)
            rogue_type = np.random.choice(self.config['rogue_behavior'])
            self.rogue_schedule.append((inject_time, self.config['anomaly_duration_minutes']*60, rogue_type))

    def inject_rogue(self, current_time, duration, rogue_type):
        """Create a temporary rogue device."""
        # Simple spoof: copy a legitimate device's UID
        legitimate = np.random.choice([d for d in self.devices if not d.is_rogue])
        rogue = Device(
            device_id=f"rogue_{uuid.uuid4().hex[:4]}",
            service_id=legitimate.service_id,
            mac=f"ff:ff:ff:{np.random.randint(0,256):02x}:{np.random.randint(0,256):02x}:{np.random.randint(0,256):02x}",
            uid=legitimate.uid if rogue_type == "spoof_uid" else uuid.uuid4().hex[:16].upper(),
            mac_rotation_interval=10 if rogue_type == "erratic_timing" else 300,
            uid_rotation_interval=10 if rogue_type == "erratic_timing" else 300,
            is_rogue=True,
            rogue_type=rogue_type,
            x=np.random.uniform(0, self.config['area_width']),
            y=np.random.uniform(0, self.config['area_height']),
            vx=np.random.uniform(-2, 2),
            vy=np.random.uniform(-2, 2),
            advertisement_interval=0.1 if rogue_type == "erratic_timing" else 0.5
        )
        rogue.end_time = current_time + duration
        self.devices.append(rogue)

    def step(self, dt=0.1):
        self.current_time += dt
        # Update positions for mobile devices
        for d in self.devices:
            if d.vx != 0 or d.vy != 0:  # mobile
                d.update_position(dt, self.config['area_width'], self.config['area_height'])
            # Rotate identifiers
            d.rotate_mac(self.current_time)
            d.rotate_uid(self.current_time)

        # Check for rogue injection
        for inj in self.rogue_schedule[:]:
            if self.current_time >= inj[0]:
                self.inject_rogue(inj[0], inj[1], inj[2])
                self.rogue_schedule.remove(inj)

        # Remove expired rogue devices
        self.devices = [d for d in self.devices if not (d.is_rogue and hasattr(d, 'end_time') and self.current_time > d.end_time)]

        # Generate advertisements from each device if it's time
        for d in self.devices:
            if self.current_time - d.last_advertisement >= d.advertisement_interval:
                adv = d.generate_advertisement(self.current_time, self.receiver_pos)
                self.events.append(adv)
                d.last_advertisement = self.current_time

    def run(self):
        total_steps = int(self.config['simulation_duration_hours'] * 3600 / 0.1)  # dt=0.1s
        for _ in range(total_steps):
            self.step()
        # Convert events to DataFrame
        df = pd.DataFrame(self.events)
        return df