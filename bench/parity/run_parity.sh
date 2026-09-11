#!/bin/bash
# CUDA <-> Python parity gate. Run after ANY change to pd_controller.cuh or to
# bench/aim_opt.py: the tuning is only meaningful while the two agree exactly.
set -e
cd "$(dirname "$(readlink -f "$0")")"
# --use_fast_math matches the production build (inference_pc/CMakeLists.txt).
# Without it the gate was comparing a DIFFERENT compilation of the controller
# than the one that ships: fast math changes division, rsqrt and fma contraction,
# which is exactly the kind of last-bit difference an exact-match gate exists to
# catch.
nvcc -O2 --use_fast_math -I../../inference_pc/needaimbot/cuda parity_test.cu -o parity_test
fail=0
echo "== default path (class_reject=1) =="
./parity_test 1 > cuda_out.txt && python3 compare.py || fail=1
echo
echo "== class_switch_reject OFF =="
./parity_test 0 > cuda_off.txt
python3 - <<'PY' || fail=1
import sys; sys.path.insert(0,'..')
from compare import py_run
rows=[tuple(map(float,l.split()[:2]))+tuple(map(int,l.split()[2:])) for l in open('cuda_off.txt')]
py=py_run(rows, cls_reject=False)
bad=[i for i in range(len(rows)) if (rows[i][3],rows[i][4])!=py[i]]
print("exact-match %d/%d"%(len(rows)-len(bad),len(rows)))
sys.exit(1 if bad else 0)
PY
echo
echo "== output gate OFF (aim key up while inference keeps running warm) =="
./parity_test 1 1.0 0 > cuda_gate.txt
python3 - <<'PYG' || fail=1
import sys
rows=[l.split() for l in open('cuda_gate.txt')]
bad=[i for i,r in enumerate(rows) if (int(r[3]),int(r[4]))!=(0,0)]
print("  emitted-zero %d/%d"%(len(rows)-len(bad),len(rows)))
sys.exit(1 if bad else 0)
PYG

echo
echo "== aim_y_scale 0.1 / 2.0 (감쇠 경로와 재클램프 경로) =="
./parity_test 1 0.1 > cuda_y.txt
./parity_test 1 2.0 > cuda_y2.txt
python3 - <<'PYY' || fail=1
import sys; sys.path.insert(0,'..')
from compare import py_run
rows=[tuple(map(float,l.split()[:2]))+tuple(map(int,l.split()[2:])) for l in open('cuda_y.txt')]
bad_all=0; tot=0
for f,ys in (('cuda_y.txt',0.1),('cuda_y2.txt',2.0)):
    rows=[tuple(map(float,l.split()[:2]))+tuple(map(int,l.split()[2:])) for l in open(f)]
    py=py_run(rows, y_scale=ys)
    bad=[i for i in range(len(rows)) if (rows[i][3],rows[i][4])!=py[i]]
    print("  y_scale %.1f : exact-match %d/%d"%(ys,len(rows)-len(bad),len(rows)))
    bad_all+=len(bad); tot+=len(rows)
sys.exit(1 if bad_all else 0)
PYY
[ $fail -eq 0 ] && echo && echo "ALL PARITY CHECKS PASSED" || { echo; echo "PARITY FAILED"; exit 1; }
