# 캡처 해상도 프리셋 (2026-08-01)

<!-- AUTO:BEGIN 이 블록은 tools/gen_preset_readme.py 가 생성한다. 직접 고치지 말 것 -->

### 프리셋마다 다른 값

```
  키                      320            160
  --------------------------------------------------
  pre_capture_shapes     [[320,320]]    [[160,160]]
  aim_softness_x/y       8.6 / 6.57     17.2 / 13.14
  thumb_softness_x/y     4.66 / 9.77    9.32 / 19.54
  lead_vgate             8.6            17.2
  lead_err_gate          22.25          44.5
  conf_threshold         0.15           0.25
  head_aim_point         1.0            0.601
  body_aim_point         0.21           0.15
```

### 두 프리셋이 공유하는 값 (여기가 갈리면 `preset.sh show` 가 잡는다)

```
  aim_kp_x=0.445  aim_kp_y=0.406  aim_kd_x=0.052  aim_kd_y=0.037
  aim_max_step=11.37  thumb_aim_kp_x=0.709  thumb_aim_kp_y=0.408
  thumb_aim_kd_x=0.02  thumb_aim_kd_y=0.082  inflight_deadtime_frames=2.85
  deadtime_adaptive=True  ff_gain=2.924  ff_v_ema=0.137  predict_frames=5.47
  oneeuro_min_cutoff=0.049  oneeuro_beta=0.02  aim_h_ema=0.2
  class_switch_reject=True  head_deprioritized=True  config_version=2
```

<!-- AUTO:END -->

이 디렉터리가 프리셋의 **원본**이다. `tools/preset.sh {320|160|show} [--dev]` 가 여기서
활성 설정(`build_stable/` 또는 `build/`)으로 복사한다.

원래는 `build_stable/bin/Release/` 에 있었다. 그 디렉터리는 로컬 바이너리·로그 때문에
통째로 gitignore 라 **프리셋이 커밋에 실리지 않았고**, 튜닝을 해도 다른 설치 환경에는
배포되지 않았다. 프리셋은 빌드 산출물이 아니라 큐레이션된 설정이므로 여기 있어야 한다.

**클론한 뒤 한 번**: `tools/install_hooks.sh` — pre-commit 훅이 프리셋을 고친 커밋마다
이 표를 재생성해 함께 넣고, 두 프리셋이 규칙에서 벗어나면 커밋을 막는다. `preset.sh` 를
거치지 않고 JSON 을 직접 고쳐도 문서가 따라오게 하려면 이게 필요하다 — 훅은 클론에
딸려오지 않으므로(`.git/hooks` 는 추적 대상이 아니다) 한 번 실행해야 한다.

`preset.sh show` 는 두 프리셋이 규칙 — 캡처 종속 6개는 정확히 x2, 나머지는 동일 — 을
지키는지 검사하고 위반 시 비정상 종료한다. 한쪽만 튜닝하고 잊으면 `preset.sh` 한 번으로
그 튜닝이 조용히 되돌아가기 때문이다. 실제로 겪었다(18~22개 키가 어긋나 있었다).

실행: `../../needaimbot_stable.sh`

두 프리셋은 캡처 해상도와 그에 종속된 값만 다르다. 아래 표는
`tools/gen_preset_readme.py` 가 프리셋 JSON 에서 생성한다 - 손으로 적은 표는 반드시
어긋나기 때문이다(실제로 한 세대 뒤처져 있었다). `preset.sh` 가 매 실행마다 갱신한다.

2026-08-01 이 스냅샷에서 바뀐 것:

* 프레임별 적응 데드타임(deadtime_adaptive) + 그 여유를 속도로 환전한 게인.
  근거리 획득 -20.5%, 오버슈트 -14.7%, 반전피크 -6.6%, 에러 -1.3%.
  ⚠ 게인이 이 기능 ON 전제로 최적화됐다 - 끄면 게인도 함께 되돌릴 것.
* thumb 프로필 최초 튜닝: kd 0.18/0.22 -> 0.02/0.082. 에러 -7.5%, t4 -10.4%.
  그전까지 우클릭 프로필만 재최적화되고 thumb 은 초기값에 남아 있었다.
* body 조준점 관측식(aim_h_ema): 같은 지점을 더 조용하게. sigma_y -10.4%.
* 검출 공백 프레임에서 in-flight 링 전진(과보정 제거).
* ego_frame_filter 제거(적응 데드타임이 대체), flick 완전 제거.

  ** preset.sh show 가 이제 두 프리셋의 정합성을 직접 검사한다. **
  한쪽만 튜닝하고 잊으면 preset.sh 한 번으로 그 튜닝이 조용히 되돌아간다 -
  실제로 이 스냅샷 직전까지 18~22개 키가 어긋나 있었다.

2026-07-30 이 스냅샷에서 바뀐 것:

* 게인 결합 재최적화 (11개 값). sim 상 유지 계열 -2~5% (정지유지 -4.3%,
  좌우반전 최대이탈 -5.1%), 획득 40px +4.3% / 120px +6.5%. 핵심은 Y 를
  무르게 한 것 (kp_y 0.82->0.744, softness_y 8->11): Y 는 head<->body 앵커
  전환이 분산의 27~36% 를 차지하는 축이고 그 전환은 표적 이동이 아니라
  검출 튐이라 따라가면 손해다. 그 덕에 One Euro 를 더 조일 수 있게 됐다
  (mincut 0.3->0.225). 두 변경은 세트로만 작동한다 - 따로 넣으면 무의미하거나
  오버슈트가 3배 된다. 플레이 체감으로 "훨씬 좋다" 판정을 받아 승격.

* flick (warped-replay) 완전 제거. 소스/설정/문서/DB 전부. 바이너리 3.6MB
  -> 633KB (임베드 궤적 DB 가 용량 대부분이었다). 이제 신규 락 시 이동도
  전부 PD 컨트롤러가 만든다 - sim 의 획득 지표가 실제 거동과 일치한다.

* inference_keepwarm_ms 300 -> -1. 우클릭을 눌러야 추론이 시작되던 게이트를
  해제. 조준 이동은 여전히 우클릭/사이드2 를 눌러야 나간다.

왜 이렇게 갈리는가 (전부 측정 결과, 추정 아님):

* softness / lead 게이트
  movement_scale = 캡처/320 이므로 캡처가 작아지면 model 좌표의 오차·속도가
  그만큼 커진다. 그 공간의 절대값과 비교되는 파라미터는 같이 키워야 화면 기준
  거동이 보존된다. 안 맞추면 표적 근처에서 과도하게 공격적이 되어 흔들린다.

* conf_threshold — 스케일마다 최적이 다르다
  160: 표적이 크게 잡혀 conf 평균 0.82. 0.25 로 올려도 검출률 손실 0 이고
       sigmaX -17% (저conf 오검출만 걸러짐).
  320: conf 평균 0.69. 0.25 로 올리면 검출률 -7%p 인데 sigma 개선은 없다.
       진짜 표적이 걸러지는 것이므로 0.15 를 유지한다.

* body_aim_point — head/body 앵커 편향 보정. 이것도 스케일마다 다르다
  320: 편향 5.22 model px  -> +0.0591  (0.15 -> 0.21)
  160: 편향 14.69 model px -> +0.0899  (0.15 -> 0.24)
  보정 후 잔여 편향은 양쪽 모두 ~0. 남는 전환 튐(표준편차)은 검출 노이즈라
  파라미터로는 못 지운다.

측정 검출기 sigma (화면 px):
  320  X 5.89 / Y 3.01
  160  X 3.91 / Y 2.36     <- 정밀하지만 시야는 절반 (화면 가로 16.7% -> 8.3%)

반드시 game_pc 의 CaptureWidth/Height 도 같이 바꿀 것. 한쪽만 바꾸면 어긋난다.

네트워크(MTU 9000, rx-usecs 8 등)는 systemd 유닛 needaimbot-nettune.service 가
부팅마다 적용한다. 프레임이 안 들어오면 먼저 `systemctl status needaimbot-nettune`.
