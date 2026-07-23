# simple_config.json 레퍼런스

`simple_config.json`은 섹션(`_section_*` 구분 키)으로 나뉜다. 구분 키는 순수 시각용이고 로더가 무시한다(모르는 키는 전부 무시). 아래는 섹션별 각 값의 의미 · 기본값 · 튜닝 노트.

> 값을 바꾸고 저장하면 다음 실행부터 적용된다. 파일이 없거나 엔진 경로가 정규화되면 프로그램이 이 포맷으로 다시 써준다.

---

## ENGINE / NETWORK
| 키 | 기본 | 의미 |
|---|---|---|
| `engine_path` | — | TensorRT 엔진(.engine) 경로. 파일명의 `320`/`fp16`이 입력 해상도·정밀도. |
| `udp_port` | 5007 | game_pc가 캡처 프레임을 보내는 UDP 포트. game_pc 송신 포트와 일치해야 함. |
| `makcu_port` | /dev/ttyACM0 | MAKCU 마우스 디바이스 시리얼 포트. |
| `makcu_baudrate` | 4000000 | MAKCU 보드레이트. 펌웨어와 일치해야 함. |

## DETECTION
| 키 | 기본 | 의미 |
|---|---|---|
| `conf_threshold` | 0.2 | 이 confidence 미만 탐지는 버림. 높이면 헛것↓·놓침↑. |
| `head_class_id` | 1 | head 클래스 ID(모델 기준). 머리 우선 선택·조준점 계산에 사용. |
| `max_detections` | 100 | 프레임당 디코드할 최대 박스 수(상한). |
| `allowed_classes` | [0,1] | 이 클래스만 타겟 후보로 사용(0=body, 1=head 등). |
| `pre_capture_shapes` | [[320,320]] | 시작 시 미리 CUDA 그래프를 캡처할 입력 해상도들. 들어오는 프레임 크기와 맞으면 콜드스타트 없음. |

## AIM POINT
조준할 "지점"을 박스 안 어디로 둘지. 0=박스 위(top), 1=아래(bottom) 방향 비율(모델 좌표 규약).
| 키 | 기본 | 의미 |
|---|---|---|
| `head_aim_point` | 1.0 | head 박스에서의 세로 조준 지점. |
| `body_aim_point` | 0.15 | body 박스에서의 세로 조준 지점(위쪽=가슴/목). |
| `shoot_offset_x` | 0.0 | **발사 중에만** 적용되는 조준점 가로 이동(setpoint shift). |
| `shoot_offset_y` | 0.0 | 발사 중 세로 조준점 이동(반동 상쇄용 소량). |

## AIM CONTROLLER + CENTER FILTER
크로스헤어를 타겟으로 밀어붙이는 제어기. 우클릭용(`aim_*`)과 사이드버튼(thumb)용 게인이 분리돼 있다.

**PD 게인** — `kp`=비례(당기는 세기), `softness`=비선형 P의 부드러움(클수록 원거리에서 순함), `kd`=미분(오버슛/떨림 억제).
| 키 | 기본 | 의미 |
|---|---|---|
| `aim_kp_x` / `aim_kp_y` | 0.75 / 0.82 | 우클릭 조준 비례 게인(가로/세로). **`inflight_comp`가 켜져 있어야 이 값이 안전하다** — 보정 없이 이 게인이면 데드타임 링잉이 난다(보정 OFF의 안전값은 ~0.55). |
| `aim_softness_x` / `_y` | 9.0 / 8.0 | 비선형 P 소프트닝. 클수록 먼 거리에서 완만. |
| `aim_kd_x` / `_y` | 0.05 / 0.06 | 미분 게인. **주의: D항은 낡은 오차에 반응하므로 데드타임 하에선 감쇠가 아니라 링잉을 키운다**(실측 kd 0.18→0.10에서 오버슈트 0.5→0.3회). `inflight_comp`가 데드타임을 처리하므로 D는 거의 불필요 — 낮게 유지할 것. |
| `thumb_aim_kp_x`/`_y` | 0.6 / 0.62 | thumb(사이드버튼) 조준 비례 게인. |
| `thumb_aim_softness_x`/`_y` | 11.0 / 10.0 | thumb 소프트닝. |
| `thumb_aim_kd_x`/`_y` | 0.18 / 0.22 | thumb 미분 게인. |

**Target tracking / stickiness**
| 키 | 기본 | 의미 |
|---|---|---|
| `iou_stickiness_threshold` | 0.3 | 이전 프레임 타겟과 IoU가 이 이상이면 같은 타겟으로 유지(타겟 스위칭 억제). |
| `distance_stickiness_factor` | 0.0 | 거리 기반 스티키니스 가중(0=끔). 켜면 가까운 이전 타겟을 더 붙잡음. |
| `track_persistence_frames` | 3 | 탐지 끊겨도 이 프레임 수만큼 트랙 유지(coast 창). |
| `coast_enabled` | true | 탐지 공백 동안 마지막 속도로 활공(coast). |
| `coast_decay` | 0.85 | coast 속도 감쇠(1에 가까울수록 오래 미끄러짐). |
| `inflight_comp` | 1.0 | **데드타임 보정(Smith predictor).** 검출은 `inflight_deadtime_frames`만큼 낡았으므로, 그 사이 내보낸 이동은 아직 측정에 안 나타난다. 그걸 빼지 않으면 **같은 오차에 두 번 반응**해 오버슈트·링잉이 난다. 0=끔(레거시), 1=완전 보정. 실측: 스텝 오버슈트 12.7px→0.4px, 정착 7→4프레임. 이상하면 0으로 즉시 되돌릴 수 있다. |
| `inflight_deadtime_frames` | 1 | 보정에 쓸 emit→visible 지연(프레임). **리그에서 실측한 값을 넣을 것** — `bench/calibrate.py`의 STEP-RESPONSE 섹션(`calibration_step_px>0`으로 캡처). 과대 추정하면 과보정되어 오히려 진동한다. |
| `aim_max_step` | 25.0 | 프레임당 최대 이동(px). 급격한 튐 방지 상한. 0=무제한. 저신뢰 검출(conf~0.2)이 만드는 간헐적 수직 튐의 **크기 상한** 역할 — 낮출수록 튐이 잘리지만(30→23px, 15→15px) 큰 보정이 굼떠진다. 정상 추적 이동량은 수 px라 25에서도 클리핑되지 않는다. |

> `feedforward_gain` / `feedforward_vgate` / `predict_horizon`은 제거됨. 화면좌표
> 속도 추정이 에고모션(조준 자체의 카메라 회전)에 오염되어 정상상태 추적에서 0이
> 되므로, 이 신호 위의 피드포워드/예측은 램프 지연을 못 지우고 검출 노이즈만
> 증폭한다(sim 검증). 재도입하려면 지연 정렬된 에고 보정 속도가 선행돼야 한다.

**One Euro center filter** — 탐지 중심의 고주파 지터 억제. 속도 적응형 저역통과.
| 키 | 기본 | 의미 |
|---|---|---|
| `oneeuro_enabled` | true | 필터 on/off. 끄면 raw 중심 사용(더 즉각적이나 떨림↑). |
| `oneeuro_min_cutoff` | 0.1 | 최소 컷오프. 낮을수록 정지 시 부드럽지만 지연↑. |
| `oneeuro_beta` | 0.02 | 속도 적응 강도. 높이면 빠른 움직임에서 지연↓(덜 부드러움). |
| `oneeuro_dcutoff` | 0.5 | 미분 신호 컷오프. |

## FLICK (warped-replay)
`flick_enabled=false`면 이 섹션 전체 미사용.
| 키 | 기본 | 의미 |
|---|---|---|
| `flick_enabled` | false | 저장된 궤적 리플레이로 초기 플릭 생성. |
| `flick_replay_db_path` | flick_trajectories.json | 플릭 궤적 DB 경로. |
| `flick_distance_tolerance` | 0.15 | 목표 거리 매칭 허용 오차. |
| `flick_elastic_amp` / `flick_elastic_modes` | 0.03 / 3 | 궤적 탄성 변형 진폭/모드 수. |
| `flick_var_amp` | 0.06 | 궤적 랜덤 변형 진폭. |
| `flick_min_reach` | 5.0 | 플릭을 적용할 최소 거리(px). |

## RECOIL COMPENSATION
`no_recoil_enabled=false`면 미사용.
| 키 | 기본 | 의미 |
|---|---|---|
| `no_recoil_enabled` | false | 발사 중 반동 상쇄 tick 활성. |
| `recoil_comp_x` / `recoil_comp_y` | 0.0 / 3.0 | tick당 보정 이동(가로/세로 counts). |
| `recoil_tick_ms` | 8 | 보정 tick 주기(ms). |

## PIPELINE / LATENCY
| 키 | 기본 | 의미 |
|---|---|---|
| `max_inflight_frames` | 1 | GPU에 동시에 올릴 프레임 수. 1=최저 지연(오버슛 방지), ↑=처리량↑·지연↑. |
| `frame_credit_depth` | 2 | 프레임 크레딧 깊이(수신-제출 흐름 제어). |
| `mouse_min_interval_ms` | 0 | 마우스 이동 최소 간격(레이트 리밋). 0=무제한. |
| `direct_aim_move_in_callback` | true | GPU 콜백 스레드에서 바로 마우스 전송(지연↓). |
| `inference_keepwarm_ms` | 0 | **조준 안 할 땐 추론을 끄는데(전력 절약), 조준 뗀 뒤 이 ms만큼 더 돌려 warm 유지.** `0`=엄격(재조준마다 콜드 1사이클 지연), `>0`(예 300)=교전 중 재조준 즉시, `<0`=항상 켜짐(콜드 지연 0, 유휴 GPU 부하). |
| `udp_busy_spin` | false | 수신 소켓을 커널 대기 대신 **busy-poll**(MSG_DONTWAIT 스핀). 프레임 첫 패킷의 IRQ→웨이크업 지연(~5-15µs) 제거. 대가: 수신 코어(affinity_core_receive) 100% 점유. Linux 전용. |
| `idle_graph_precapture_enabled` | true | 유휴 시 미리 CUDA 그래프 캡처(첫 조준 콜드스타트 제거). |
| `idle_graph_precapture_interval_ms` | 100 | 유휴 프리캡처 시도 주기(ms). |

## SYSTEM (threads / CPU affinity)
| 키 | 기본 | 의미 |
|---|---|---|
| `realtime_threads_enabled` | true | 실시간 스케줄링 우선순위 사용. |
| `cpu_affinity_enabled` | true | 스레드를 특정 코어에 고정. |
| `affinity_core_main` | 5 | 메인(프레임 취득+제출) 루프 코어. |
| `affinity_core_receive` | 7 | UDP 수신 스레드 코어. |
| `affinity_core_callback` | 6 | GPU 완료 콜백(마우스 전송) 코어. |
| `affinity_core_sender` | 4 | 송신 스레드 코어. |

## DIAGNOSTICS (perf log / calibration / bench)
평상시엔 대부분 off. `perf_stats_enabled`가 마스터 스위치 역할.
| 키 | 기본 | 의미 |
|---|---|---|
| `perf_stats_enabled` | false | 성능 통계 수집(지연 퍼센타일 등). 켜면 calibration 로그도 자동 활성. |
| `perf_stats_interval_ms` | 1000 | 통계 출력 주기. |
| `perf_log_path` | perf_stats.log | 통계 로그 파일. |
| `perf_log_max_bytes` | 33554432 | 로그 회전 최대 크기(32MB). |
| `perf_log_truncate_on_start` | false | 시작 시 로그 비우기. |
| `stage_timing_enabled` | false | 파이프라인 단계별 세부 타이밍. |
| `force_aim_on` | false | **벤치마크 전용.** 조준키 없이 항상 조준 활성(측정용). 실사용 금지. |
| `calibration_log_path` | calib.csv | 캘리브레이션 CSV 경로(**어디에 쓸지**만 결정). 로깅 자체는 `perf_stats_enabled`가 켜져야 시작됨. `bench/calibrate.py`로 분석. |
| `calibration_step_px` | 0 | >0이면 스텝응답 dead-time 측정용 마우스 펄스 주입(±px). 조준 OFF·정지 타겟에서만. 0=끔. |
| `calibration_step_period_ms` | 250 | 스텝 주입 주기(ms). |
