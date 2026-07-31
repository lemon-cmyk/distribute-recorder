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

## Test

```bash
python3 -m pytest -q
python3 -m compileall tiangong_recorder
```

