# Debug Session: serve-policy-segfault

Status: [OPEN]

## Symptom

Running `scripts/serve_policy.py` exits immediately after uv deprecation warning, with shell exit code `139`.

Command observed:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=checkpoints/data_toiletButton_0626_quaternion_10steps/10000 --port=8123
```

## Meaning of Exit Code

Exit code 139 usually means SIGSEGV / segmentation fault. This is a native crash, not a normal Python exception.

## Hypotheses

1. CUDA/JAX/XLA crashes during model initialization before Python logging reaches `Creating server`.
2. The selected GPU `CUDA_VISIBLE_DEVICES=1` has driver/runtime incompatibility, memory issue, or device-specific failure.
3. Checkpoint/config combination `pi05_piper` with the given checkpoint triggers a native shape/load path crash.
4. A native dependency installed by `uv run --python python3.11` differs from the active `(pi)` environment and crashes at import/init time.
5. The process crashes during checkpoint deserialization before Python can emit an exception.

## Evidence Needed

- Whether crash happens during import, config load, policy creation, or server creation.
- Whether CPU-only or another GPU changes the behavior.
- Whether `faulthandler` or native stack output identifies JAX/CUDA/XLA.

## Next Step

Collect runtime evidence without changing business logic first.

## Evidence Collected

### Import/init probe

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 uv run --python python3.11 python -X faulthandler -c "from openpi.training import config as c; from openpi.policies import policy_config as pc; print('before config'); cfg=c.get_config('pi05_piper'); print('before policy'); p=pc.create_trained_policy(cfg, 'checkpoints/data_toiletButton_0626_quaternion_10steps/10000'); print('policy created')"
```

Observed:

- Only uv deprecation warning printed.
- `before config` was not printed.
- Exit code: `139`.

Conclusion: crash happens before runtime reaches user probe code, most likely during import of `openpi.training.config`, `openpi.policies.policy_config`, or one of their native dependencies.

### Cross-repo comparison

User reports this command works normally in another repo `/home/user/openpi05` on the same machine/GPU:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=checkpoints/pi05_piper/data_toiletButton_0609_quaternion/29999
```

Conclusion: GPU hardware is less likely to be the primary cause. Current priority shifts to repo-specific dependency environment, package versions, or import-time code differences in `/home/user/openpi05_quaternion`.

### JAX comparison

Working repo `/home/user/openpi05`:

- Python: `/home/user/openpi05/.venv/bin/python3`
- `jax 0.5.3`
- `numpy 1.26.4`
- `jax.devices()` returns `[CudaDevice(id=0), CudaDevice(id=1)]`

Failing repo `/home/user/openpi05_quaternion`:

- Python: `/media/Data/openpi05_quaternion/.venv/bin/python3`
- `jax 0.5.3`
- `numpy 1.26.4`
- output stops before `jax.devices()` result

Conclusion: import of `jax` and `numpy` succeeds, but GPU backend initialization during `jax.devices()` likely segfaults in the failing repo. Need compare `jaxlib`, CUDA plugin, and NVIDIA wheel versions, not just `jax`/`numpy`.

### Package list attempt

User ran `uv pip list | grep -E "jax|cuda|cudnn|cublas|nccl|nvidia"` in both repos. Both outputs showed identical package versions, but both also reported:

```text
Using Python 3.11.14 environment at: /home/user/miniforge3/envs/pi
```

This is not the same interpreter used by `uv run --python python3.11`, which previously printed repo-specific `.venv` paths. Therefore this package comparison did not inspect the crashing runtime environment.

### Root cause confirmed: LD_LIBRARY_PATH conflict

Site-packages comparison showed both repos have identical jax/nvidia versions, so not a package issue.

Environment inspection revealed:

```text
LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/local/cuda-12.4/lib64
```

The project uses CUDA 12.6 NVIDIA wheels, but `LD_LIBRARY_PATH` points to system CUDA 12.4 and ROS libraries, causing version mismatch during JAX GPU backend init.

Decisive test in failing repo:

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH= PYTHONFAULTHANDLER=1 uv run --python python3.11 python -X faulthandler -c "import jax; print('jax', jax.__version__); print(jax.devices())"
```

Output:

```text
jax 0.5.3
[CudaDevice(id=0)]
```

Exit code: 0. No crash.

Conclusion: root cause is `LD_LIBRARY_PATH` overriding wheel-bundled CUDA 12.6 libraries with system CUDA 12.4 / ROS libraries.

### Root cause confirmed: LD_LIBRARY_PATH conflict

Site-packages comparison showed both repos have identical jax/nvidia versions, so not a package issue.

Environment inspection revealed:

```text
LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/local/cuda-12.4/lib64
```

The project uses CUDA 12.6 NVIDIA wheels, but `LD_LIBRARY_PATH` points to system CUDA 12.4 and ROS libraries, causing version mismatch during JAX GPU backend init.

Decisive test in failing repo:

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH= PYTHONFAULTHANDLER=1 uv run --python python3.11 python -X faulthandler -c "import jax; print('jax', jax.__version__); print(jax.devices())"
```

Output:

```text
jax 0.5.3
[CudaDevice(id=0)]
```

Exit code: 0. No crash.

Conclusion: root cause is `LD_LIBRARY_PATH` overriding wheel-bundled CUDA 12.6 libraries with system CUDA 12.4 / ROS libraries.
