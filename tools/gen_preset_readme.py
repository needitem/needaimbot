#!/usr/bin/env python3
"""프리셋 README 의 값 표를 프리셋 JSON 에서 재생성한다.

손으로 적은 표는 반드시 어긋난다 - 실제로 이 README 의 표는 한 세대 뒤처져 있었다
(softness 9.33/11.0, thumb 11.0/10.0, predict 2.94). 표는 데이터의 사본이므로
데이터에서 만들어야 한다. preset.sh 가 매 실행마다 이걸 호출한다.

  --check  재생성 결과와 파일이 다르면 diff 를 찍고 1 로 종료 (쓰지 않음)
  (기본)   AUTO 블록을 제자리에서 갱신
"""
import argparse
import io
import json
import os
import sys

BEGIN = "<!-- AUTO:BEGIN 이 블록은 tools/gen_preset_readme.py 가 생성한다. 직접 고치지 말 것 -->"
END = "<!-- AUTO:END -->"

# 캡처 크기에 종속돼 프리셋마다 갈리는 값 + 각 프리셋이 자기 값을 갖는 것
ROWS = [
    ("pre_capture_shapes", lambda d: "[[%d,%d]]" % tuple(d["pre_capture_shapes"][0])),
    ("aim_softness_x/y", lambda d: "%s / %s" % (d["aim_softness_x"], d["aim_softness_y"])),
    ("thumb_softness_x/y", lambda d: "%s / %s" % (d["thumb_aim_softness_x"],
                                                  d["thumb_aim_softness_y"])),
    ("lead_vgate", lambda d: d["lead_vgate"]),
    ("lead_err_gate", lambda d: d["lead_err_gate"]),
    ("conf_threshold", lambda d: d["conf_threshold"]),
    ("head_aim_point", lambda d: d["head_aim_point"]),
    ("body_aim_point", lambda d: d["body_aim_point"]),
]
# 두 프리셋이 반드시 같아야 하는 것 중, 표에 같이 보여 주면 유용한 것
SHARED = ["aim_kp_x", "aim_kp_y", "aim_kd_x", "aim_kd_y", "aim_max_step",
          "thumb_aim_kp_x", "thumb_aim_kp_y", "thumb_aim_kd_x", "thumb_aim_kd_y",
          "inflight_deadtime_frames", "deadtime_adaptive", "ff_gain", "ff_v_ema",
          "predict_frames", "oneeuro_min_cutoff", "oneeuro_beta", "aim_h_ema",
          "class_switch_reject", "head_deprioritized", "config_version"]


def build(presets_dir):
    a = json.load(open(os.path.join(presets_dir, "simple_config.320.json")))
    b = json.load(open(os.path.join(presets_dir, "simple_config.160.json")))
    out = [BEGIN, "", "### 프리셋마다 다른 값", "",
           "```", "  %-22s %-14s %s" % ("키", "320", "160"), "  " + "-" * 50]
    for k, f in ROWS:
        out.append("  %-22s %-14s %s" % (k, f(a), f(b)))
    out += ["```", "",
            "### 두 프리셋이 공유하는 값 (여기가 갈리면 `preset.sh show` 가 잡는다)", "",
            "```"]
    line = "  "
    for k in SHARED:
        cell = "%s=%s" % (k, a.get(k))
        if len(line) + len(cell) > 78:
            out.append(line.rstrip())
            line = "  "
        line += cell + "  "
    out += [line.rstrip(), "```", "", END]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="inference_pc/presets")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.presets, "README.md")
    s = io.open(path, encoding="utf-8").read()
    block = build(a.presets)
    if BEGIN in s and END in s:
        new = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        # 마커가 없으면 첫 제목 뒤에 새로 심는다
        i = s.index("\n\n") + 2 if "\n\n" in s else 0
        new = s[:i] + block + "\n\n" + s[i:]
    if new == s:
        if not a.check:
            print("  README 표 최신")
        return 0
    if a.check:
        sys.stderr.write("  [경고] presets/README.md 의 값 표가 프리셋과 다르다.\n"
                         "         tools/gen_preset_readme.py 로 갱신할 것.\n")
        return 1
    io.open(path, "w", encoding="utf-8").write(new)
    print("  README 표 재생성")
    return 0


if __name__ == "__main__":
    sys.exit(main())
