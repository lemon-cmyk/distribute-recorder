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

## Convert PKL recordings to LeRobot v2.1

Create the isolated converter environment once, then start the independent
scanner. The recorder can continue saving PKLs while this process converts
completed episodes serially.

```bash
./scripts/setup_converter_env.sh
./scripts/start_converter.sh \
  --dataset-name tiangong_pick_cube \
  --task "Pick up the cube and place it in the tray"
```

The converter reads final `/home/nvidia/teleop_logs/*.pkl` files and ignores
`.pkl.tmp` files. It writes an MP4-backed LeRobot v2.1 dataset to
`/home/nvidia/lerobot_datasets/<dataset-name>`. Images stay as NumPy arrays in
memory and are streamed directly to Jetson AGX Orin's AV1 hardware encoder; no
temporary PNG files are created. The encoder writes a short-lived IVF file with
a two-frame GOP, then FFmpeg losslessly remuxes it into a PyAV-seekable MP4 and
removes the IVF file. The output uses the target Tiangong2 layout:
`front`, `left_wrist`, and `right_wrist` videos, interleaved arm/gripper
16-dimensional vectors, `list<float>` Parquet columns, and global
`norm_stats.json`. A source PKL is deleted only after
`save_episode()` succeeds, the saved episode/frame count is verified, and
durable conversion state is committed. Failed PKLs remain in place and their
errors are recorded in the SQLite state database.

`--dataset-name` derives all dataset-specific paths together. For the example
above, the output is `tiangong_pick_cube/`, the state database is
`.tiangong_pick_cube_converter.sqlite3`, the process lock is
`.tiangong_pick_cube_converter.lock`, and the repository ID is
`local/tiangong_pick_cube`, all under `/home/nvidia/lerobot_datasets` where
applicable. `--task` is written into `meta/tasks.jsonl`. Both options are
optional; omitting them uses `config/converter.yaml`. The dataset name must be
new: if its output directory already exists, startup fails without changing or
deleting it. A new dataset/state pair treats every PKL currently in the input
directory as unconverted.

Pressing `Ctrl+C` immediately interrupts the current file and exits with code
130; it does not wait for the rest of the current directory scan or video
encoding. The source PKL is retained. An interrupted run may leave its dataset
directory behind, so that dataset name is intentionally rejected on the next
launch; use a new name or remove the incomplete directory manually. During
stage 7, each camera's MP4 encoder displays an in-place frame progress bar in
an interactive terminal; redirected logs receive one line per 10%.

Use `./scripts/start_converter.sh --retry-errors --once` after fixing a failed
file, or add `--keep-source --once` for a non-deleting validation run. Head
images are treated as RGB; wrist images are converted from BGR to RGB without
resizing. MP4 encoding is lossy, so the deleted PKL cannot serve as a lossless
image backup after a successful conversion.
