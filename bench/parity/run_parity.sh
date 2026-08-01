#!/bin/bash
# CUDA <-> Python parity gate. Run after ANY change to pd_controller.cuh or to
# bench/aim_opt.py: the tuning is only meaningful while the two agree exactly.
set -e
cd "$(dirname "$(readlink -f "$0")")"
nvcc -O2 -I../../inference_pc/needaimbot/cuda parity_test.cu -o parity_test
fail=0
echo "== default path (ego_frame=0, class_reject=1) =="
./parity_test 0 1 > cuda_out.txt && python3 compare.py || fail=1
echo
echo "== class_switch_reject OFF =="
./parity_test 0 0 > cuda_off.txt
python3 - <<'PY' || fail=1
import sys; sys.path.insert(0,'..')
from compare import py_run
rows=[tuple(map(float,l.split()[:2]))+tuple(map(int,l.split()[2:])) for l in open('cuda_off.txt')]
py=py_run(rows, cls_reject=False)
bad=[i for i in range(len(rows)) if (rows[i][3],rows[i][4])!=py[i]]
print("exact-match %d/%d"%(len(rows)-len(bad),len(rows)))
sys.exit(1 if bad else 0)
PY
[ $fail -eq 0 ] && echo && echo "ALL PARITY CHECKS PASSED" || { echo; echo "PARITY FAILED"; exit 1; }
