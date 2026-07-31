# Tiangong distributed recorder

This service runs on the Orin board. It records the three local ROS 2 camera
topics, receives 50 Hz robot state/action entries from the x86 teleoperation
process, aligns them by capture timestamp, keeps the completed episode in
memory, and writes the existing `List[Dict]` pickle format after `STOP_SAVE`.

## Start

```bash
./scripts/start_recorder.sh
```

The default control endpoint is `tcp://0.0.0.0:5560`. Generated datasets are
written outside this repository to `/home/nvidia/teleop_logs`.

Start this service before running
`start_tiangong_teleop_distributed_recording.sh` on the x86 board.

The service uses ROS image header timestamps and reports per-camera alignment
statistics after every saved episode. The x86 client also estimates the clock
offset before recording. PTP or another system clock synchronizer must keep
both boards in the same wall-clock domain.

## Test

```bash
python3 -m pytest -q
python3 -m compileall tiangong_recorder
```
