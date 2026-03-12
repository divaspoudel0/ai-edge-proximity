# AI-Augmented Edge Intelligence for Proximity Services

This repository contains the complete simulation code for AI-Augmented Edge Intelligence for Predictive Proximity Services Using Rotating Identifiers:

## Overview

The project simulates a network of BLE beacons and mobile devices that broadcast advertisements containing a MAC address, a rotating unique identifier (UID), and a service identifier. An edge processor listens to these advertisements, builds contextual fingerprints, detects anomalies (spoofing, erratic behavior), predicts user intent (next zone), and maintains session continuity across identifier rotations.

## Features

- Realistic beacon traffic generator with:
  - Multiple static and mobile devices
  - Time-based MAC and UID rotation
  - Mobility (random waypoint)
  - Injection of rogue devices with different attack types
- Edge AI module with:
  - Isolation Forest for anomaly detection
  - Hidden Markov Model for intent prediction
  - Session continuity manager using fingerprint similarity
- Mock cloud server for identifier validation
- Evaluation scripts for precision, recall, F1, and prediction accuracy
- Plot generation for figures used in the paper

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
