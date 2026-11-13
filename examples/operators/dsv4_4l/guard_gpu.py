import json, re, subprocess, sys
from pathlib import Path

gpu = int(sys.argv[1])
p = subprocess.check_output(["rocm-smi", "--showpidgpus"], text=True)
current = None
used = []
for line in p.splitlines():
    m = re.match(r"PID (\d+) is using (\d+) DRM", line)
    if m:
        current = (int(m[1]), int(m[2]))
        continue
    if current and current[1] and re.fullmatch(r"[\d\s]+", line) and str(gpu) in line.split():
        used.append(current[0])
mem = json.loads(
    subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True)
)
amount = int(mem[f"card{gpu}"]["VRAM Total Used Memory (B)"])
assert not used, f"physical GPU{gpu} occupied by {used}"
assert amount < 1024**3, f"physical GPU{gpu} has unexplained resident VRAM: {amount}"
print("GPU_IDLE_GUARD_PASS", gpu, amount, flush=True)
