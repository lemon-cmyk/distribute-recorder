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

## G200 x86: convert PKL to LeRobot v2.1

### NumPy and JPEG image inputs

The converter automatically detects each `image[camera]["color"]` payload:
legacy HWC uint8 NumPy arrays keep the existing `camera_color_order` RGB/BGR
configuration; JPEG `bytes` (also `bytearray` and `memoryview`) are decoded to
RGB before entering the same MP4 encoder. JPEG channel order is determined by
the codec, so `camera_color_order` applies only to legacy pixel arrays.
No mode flag or format version is required, and the two representations may
coexist across cameras, frames, and source PKLs.

JPEG payloads may include `color_encoding: "jpeg"`; the marker is optional.
If supplied it must identify JPEG (`jpeg`, `jpg`, or `image/jpeg`), and the
actual payload must decode as JPEG. Corrupt images, unsupported byte formats,
and dimension mismatches fail validation before creating or appending an
episode. Image size checks, NumPy dtype checks, state/action mapping, HIL
annotations, timestamps, source verification/deletion policy, statistics, and
AV1-in-MP4 settings retain their existing logic. No temporary PNGs are written.


The converter automatically preserves available HIL annotations as per-frame
Parquet columns, using the original dotted names:
`complementary_info.policy_action`, `complementary_info.policy_action_valid`,
`complementary_info.is_intervention`, `complementary_info.state`,
`complementary_info.collector_policy_id`, and `episode_success`.
No extra flag is required. Episodes without these fields keep the original
conversion path and seven-column Parquet schema. Older HIL recordings without
`policy_action_valid` are supported; only fields actually present are added.

Values are copied in frame order, without inferring intervention from action
values, changing success labels, or filtering frames. Source scalar labels and
single-element NumPy arrays are accepted. Missing or invalid annotations within
an episode are reported before writing it, rather than inventing labels.
All episodes in one output dataset must have the same annotation fields; use a
new `--dataset-name` when converting a different annotation schema.


```bash
./scripts/setup_converter_env.sh
./scripts/start_converter.sh --dataset-name g200_test --task "Describe the task" --once
```

Defaults read `/home/grg/XRoboToolkit-Teleop-Sample-Python/logs/g200` and write
`/home/grg/lerobot_datasets/<dataset-name>`. Source PKLs are retained by default.
Use a new dataset name for each conversion run. `.pkl.tmp` files are ignored.

Use `--no-source-verification` (or YAML `verify_source: false`) to disable both
SHA256 and file-metadata comparisons, including checks before source deletion.
This overrides `compute_source_sha256`. Completed files are deduplicated by path;
changes at an already completed path are ignored, and source deletion will not
detect concurrent writes or replacement. PKL loading, frame validation, and
output episode verification still run. Source verification is enabled by default.
Use a new dataset name when switching back to verification, because unchecked
records have no fingerprint to compare.

```bash
./scripts/start_converter.sh --dataset-name g200_test_no_checks --no-source-verification
```

The converter writes LeRobot v2.1 (`lerobot==0.3.3`), robot type `g200`, at 50 Hz.
Timestamps are frame index / 50, without resampling the source timestamps.
State/action order is left arm (7), left gripper (1), right arm (7), right gripper (1).
State uses current measured positions; action uses current commanded positions.
Camera keys are front, left_wrist, right_wrist. Legacy array RGB/BGR handling
uses `camera_color_order` in the selected configuration; JPEG bytes always decode
to RGB. Width and height default to null (infer from the
first image); later images must match. No resizing or rotation is performed.

NumPy frames go directly through PyAV into MP4 using CPU `libsvtav1`, yuv420p,
GOP 2 and CRF 30, matching the official default encoder settings. There are no
PNG intermediates and no Jetson/GStreamer dependency. The PyAV build must expose
libsvtav1. Encoding speed depends on CPU; 50 Hz is the dataset rate, not a promise
of real-time conversion. Episode statistics are retained; no extra root-level
norm_stats.json is generated.

Interrupting conversion retains the source PKL, but may leave an incomplete
output directory. Use a new dataset name for the next run. Initial validation
should always retain the original PKLs.

Validated on this x86 board with a 920-frame G200 PKL: three 640x480 AV1 videos,
50 Hz, official LeRobot random-access reads using `video_backend="pyav"`, and
state/action equality checks passed. Conversion plus validation took 41.2 s,
peak process RSS was approximately 3.42 GiB. These figures describe that sample,
not a throughput guarantee. The raw PKL was retained.

To read the result explicitly with the verified decoder:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/g200_test", root="/home/grg/lerobot_datasets/g200_test",
                         video_backend="pyav")
```
