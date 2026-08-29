# Environment notes

`losscolumn doctor` reports what a given machine can measure and what it will
have to model. This file records the one environment repair that was needed on
the development machine, because the failure is generic to Anaconda on Windows
and the error message points nowhere useful.

---

## Triton fails to import under Anaconda on Windows

### Symptom

```
ImportError: DLL load failed while importing libtriton:
A dynamic link library (DLL) initialization routine failed.
```

Sometimes accompanied by `Windows fatal exception: access violation` printed by
`faulthandler` — a first-chance exception during the failed load, not a crash.

`pip list` shows the correct package. `triton-windows 3.2.0` is the right
pairing for torch 2.6, so the obvious diagnosis — wrong package — is wrong.

### Cause

`libtriton.pyd` is built with **MSVC 14.43**. Its Rich header records linker
build 35217:

```python
# the Rich header records the toolset that built each linked object
python -c "..."   # see the parser in the commit that added this file
# highest linker build: 35217   ->  MSVC 14.43 (VS 2022 17.13)
```

Anaconda ships its own Visual C++ redistributable next to `python.exe`:

| Location | Version |
|----------|---------|
| `anaconda3\vcruntime140.dll` | **14.27.29016** |
| `C:\Windows\System32\vcruntime140.dll` | 14.50.35719 |

`python311.dll` links `vcruntime140.dll`, and the application directory wins
the DLL search order, so Anaconda's **14.27** copy is loaded at interpreter
startup. Windows resolves subsequent imports by module base name, so when
`libtriton.pyd` loads it binds to the already-loaded 14.27 module rather than
to System32's 14.50 — and initialisation fails.

Confirm which copy a live process has:

```python
import ctypes
from ctypes import wintypes
k32 = ctypes.WinDLL("kernel32")
h = k32.GetModuleHandleW("VCRUNTIME140.dll")
buf = ctypes.create_unicode_buffer(1024)
k32.GetModuleFileNameW(h, buf, 1024)
print(buf.value)      # -> C:\Users\...\anaconda3\VCRUNTIME140.dll
```

Nothing is missing, which is why dependency walkers come back clean: all 16 of
`libtriton.pyd`'s imports resolve. The failure is entirely about which version of the
module gets bound.

### Fix

Replace Anaconda's bundled redistributable with the system one. The VC++
redistributable is backward compatible within the v140 family by design, so
14.50 satisfies everything built against 14.16–14.43.

```powershell
$root = "C:\Users\<you>\anaconda3"
$backup = Join-Path $root ("_vcruntime_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force $backup | Out-Null
Get-ChildItem "$root\*140*.dll" | ForEach-Object {
  $sysfile = Join-Path "C:\Windows\System32" $_.Name
  if ((Test-Path $sysfile) -and
      ((Get-Item $sysfile).VersionInfo.FileVersionRaw -gt $_.VersionInfo.FileVersionRaw)) {
    Copy-Item $_.FullName (Join-Path $backup $_.Name) -Force   # reversible
    Copy-Item $sysfile $_.FullName -Force
  }
}
```

Close every Python process first, or the copy fails with a sharing violation.
To revert, copy the backup folder's contents back.

### Why not `conda update`

`vs2015_runtime 14.44.35208` exists on `pkgs/main` and would be the tidier fix,
keeping conda's metadata truthful. It was not used here because this base
environment is already inconsistent — a mix of pip and conda installs, which is
normal for a working Anaconda base — and conda's solver wanted to reconcile
several hundred packages including torch, Spyder and Jupyter. Churning the
whole environment to update three DLLs is a bad trade.

If your base environment is clean, prefer:

```bash
conda install "vs2015_runtime>=14.44"
```

The trade-off to know about: after the manual fix, conda's metadata still says
14.27 while the files are 14.50, so a later `conda install` touching
`vs2015_runtime` may put the old ones back. Re-run the snippet if Triton
starts failing again after a conda operation.

### Verifying

```bash
losscolumn doctor          # "triton kernel  available"
pytest -q -m triton        # exercises the FlashAttention kernel on the device
```

---

## Compute capability and dtype

Triton needs sm_70 for `tl.dot` to lower to tensor cores; below that the kernel
compiles but runs on CUDA cores, and timing it measures a path nobody deploys.
`bfloat16` tensor cores arrive with Ampere (sm_80).

`supported_dtypes()` reports what the present device can actually run.
Thrust III's protocol registers both `float16` and `bfloat16`, so on a
pre-Ampere device the run **declares a deviation** against the sealed protocol
rather than quietly narrowing its grid — the artifact then carries the reason
(`sm_75 has no bf16 tensor cores`) next to the result it affects.

That is the mechanism working as intended: the hardware constraint is real, and
the standard's answer to a real constraint is to disclose it, not to hide it.

---

## Other environment facts worth knowing

**numba is broken here, and always was.** `Numba needs NumPy 1.24 or less`
against NumPy 1.26. Unrelated to the runtime repair, and unrelated to this
package, which does not use numba.

**`LC_DISABLE_TRITON=1`** skips the Triton probe entirely. Useful when a broken
install faults during import on a machine you would rather not repair.

**The serving engines are absent** on this machine, so Thrust II runs its
simulated backend and every artifact it produces is stamped
`evidence_class="simulated"` with a note saying what was modelled.
