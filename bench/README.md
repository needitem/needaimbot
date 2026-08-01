# bench — 조준 튜닝 하네스

일회성 실험 드라이버는 남기지 않는다. 결과는 커밋 메시지와 코드 주석에 적고
스크립트는 지운다 — 안 그러면 "이게 아직 유효한가"를 매번 다시 판단해야 한다.
남아있는 것은 **재사용되는 라이브러리 / 게이트 / 실측 도구**뿐이다.

## 컨트롤러 포트 (실행 코드의 손유지 사본)

| 파일 | 무엇 |
|---|---|
| `aim_sim_ring.py` | `needaimbot/cuda/pd_controller.cuh` 의 line-for-line 이식 |
| `aim_sim_select.py` | `simple_postprocess.cu` 의 타겟 선택 파이프라인 이식 |
| `aim_opt.py` | 위에 실측 노이즈·데드타임·클래스전환을 얹은 **현행 튜닝 하네스** |
| `realjit.py` | 실측 데드타임 분포(로그정규, 중앙 1.10프레임)로 돌리는 평가기 |
| `holdout.py` | 탐색에 안 쓴 seed 로 재평가 (과적합 검증) |
| `adaptive_deadtime.py` | 프레임별 적응 데드타임 실험 인프라 (`deadtime_adaptive`) |
| `ctrl_zoo.py` / `ctrl_zoo_opt.py` | 기각된 대안 컨트롤러(PID/Holt/KF-CA/게인스케줄) + 탐색 스캐폴딩 |
| `aim_alt_designs.py` | 기각된 대안 아키텍처 기록 |

**포트는 반드시 원본과 같은 숫자를 내야 한다.** `pd_controller.cuh` 나 `aim_opt.py`
를 건드렸으면 `parity/run_parity.sh` 를 돌릴 것 — 두 사본이 갈라지면 튜닝 결과가
통째로 무효가 된다.

## 게이트

- `parity/run_parity.sh` — 실제 device 함수를 결정론적 입력열에 돌려 프레임 단위로
  파이썬 모델과 비교. 8% 확률로 head↔body 를 뒤집어 클래스 전환 경로도 밟는다.

## 실측 도구 (리그에서 수집한 데이터를 읽는다)

| 파일 | 무엇 |
|---|---|
| `calibrate.py` | calib CSV → 검출기 σ / 결측률 / 프레임 타이밍 |
| `deadtime_fit.py` | emit→visible 데드타임을 **소수 프레임**까지 (`calibration_step_px>0` 필요) |
| `crop_ab/` | 리그 프레임(`frames.npy`)으로 크롭·앵커·관측식 비교. `grab_frames.py` 로 재수집 |
| `plot_bottleneck.py` | `perf_stats.log` 병목 시각화 |
| `build_int8_engine.py` | INT8 엔진 빌드 (미사용 — 2026-08-01 기준 INT8 은 배제) |

`frames.npy` 와 원시 calib CSV 는 커밋하지 않는다(gitignore). 큐레이션한 표본만
`calib_rig_sample*.csv` 로 남긴다.

## 하네스를 쓸 때 걸린 함정 (전부 실제로 겪음)

- **탐색 seed 와 확정 seed 를 분리할 것.** 안 하면 노이즈 실현값에 맞춘 걸 실력으로
  착각한다. 제약 판정과 점수도 같은 seed 수로 재야 한다 — 다르면 현재 설정조차
  "제약 이탈" 로 나온다.
- **목적함수에 없는 것은 지켜지지 않는다.** 획득 속도를 제약으로만 걸었더니 근거리가
  4.2% 느려진 게 벌점 없이 통과했다.
- **측정 도구부터 검증할 것.** `deadtime_fit.py` 초기 구현에 정확히 -0.5 프레임 편향이
  있었다(차분 창 중심이 어긋남). 합성 정답으로 안 걸렀으면 그 값으로 설정을
  "고쳐서" 악화시켰을 것이다.
