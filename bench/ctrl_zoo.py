#!/usr/bin/env python3
"""이번 세션에서 아직 시도하지 않은 제어/필터 기법 일괄 검증.

이미 기각된 것(재시도 금지): 최소자승 궤적적합, 등속 칼만(CV), deadbeat 역모델,
통합 에고좌표계, 점프감지, 선형 P, D항 제거, dt-adaptive One Euro, 박스크기 적응,
head/body 융합, 축별 리드 강화, 픽셀정합 속도추정. 근거는 각 항목 커밋/주석 참조.

여기서 새로 보는 것:
  PID   적분항 추가 - 정상상태 편향(반동 등)을 지운다. 데드타임 하에서 헌팅 위험.
  KF-CA 등가속 칼만 - CV 는 기각됐지만 가속 모델은 미검증. 스트레이프 시작/정지에서
        이득이 있을 수 있다.
  HOLT  이중지수평활(level+trend) - 알파베타의 단순형. One Euro 대체 후보.
  SCHED 속도 기반 게인 스케줄 - 지금은 오차 크기로만 스케줄(nonlinear P)하고
        표적 속도는 안 본다. 정지 시 저게인(노이즈↓) / 추적 시 고게인(랙↓).
"""
import math

from aim_opt import OptCtrl, SC
from aim_sim_ring import nonlinear_p, clamp_max_step, emit_mouse_delta


class PIDCtrl(OptCtrl):
    """P+D 에 적분항. ki>0. 적분은 오차가 작을 때만 누적(윈드업 방지),
    그리고 리드가 이미 램프랙을 지우므로 잔여 정상편향만 노린다."""
    def reset(self):
        OptCtrl.reset(self); self.ix = self.iy = 0.0
    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        out = OptCtrl.step(self, rx, ry, cls_changed)
        ki = p.get("ki", 0.0)
        if ki <= 0.0:
            return out
        band = p.get("i_band", 20.0)
        ex, ey = self.prev_err_x, self.prev_err_y
        if abs(ex) < band:
            self.ix = max(-p.get("i_clamp", 15.0), min(p.get("i_clamp", 15.0), self.ix + ex))
        else:
            self.ix *= 0.9
        if abs(ey) < band:
            self.iy = max(-p.get("i_clamp", 15.0), min(p.get("i_clamp", 15.0), self.iy + ey))
        else:
            self.iy *= 0.9
        # 적분 기여분을 추가 emit 으로 (기본 경로를 건드리지 않기 위해)
        mx, my = ki * self.ix, ki * self.iy
        mx, my = clamp_max_step(mx, my, p["max_step"])
        dx, self.res_x = emit_mouse_delta(mx, self.res_x)
        dy, self.res_y = emit_mouse_delta(my, self.res_y)
        self._push(float(out[0] + dx), float(out[1] + dy))
        return out[0] + dx, out[1] + dy


class HoltCtrl(OptCtrl):
    """One Euro 대신 Holt 이중지수평활(level+trend)로 위치를 추정.
    trend 항이 램프를 따라가므로 저역통과의 램프 지연이 없다."""
    def reset(self):
        OptCtrl.reset(self); self.lx = self.ly = None; self.bx = self.by = 0.0
    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        a = p.get("holt_a", 0.5); b = p.get("holt_b", 0.1)
        if self.lx is None:
            self.lx, self.ly = rx, ry
        else:
            lx = a*rx + (1-a)*(self.lx + self.bx)
            self.bx = b*(lx - self.lx) + (1-b)*self.bx
            self.lx = lx
            ly = a*ry + (1-a)*(self.ly + self.by)
            self.by = b*(ly - self.ly) + (1-b)*self.by
            self.ly = ly
        # 평활 결과를 One Euro 자리에 주입: filt_* 를 덮어쓰고 필터를 끈다
        saved = p["oneeuro"]; p["oneeuro"] = False
        self.filt_x, self.filt_y = self.lx, self.ly
        out = OptCtrl.step(self, self.lx, self.ly, cls_changed)
        p["oneeuro"] = saved
        return out


class KFCACtrl(OptCtrl):
    """등가속(CA) 칼만으로 위치 추정. CV 는 이미 기각됐지만 가속 모델은 미검증."""
    def reset(self):
        OptCtrl.reset(self)
        self.sx = None
    def _kf(self, s, P, z, R, Q):
        # 상태 [pos, vel, acc], dt=1
        s[0] += s[1] + 0.5*s[2]; s[1] += s[2]
        for i in range(3):
            P[i][i] += Q*(3-i)
        S = P[0][0] + R
        K = [P[0][0]/S, P[1][0]/S, P[2][0]/S]
        r = z - s[0]
        for i in range(3):
            s[i] += K[i]*r
        for i in range(3):
            P[i][0] -= K[i]*P[0][0]
        return s[0]
    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        R = p.get("kf_R", 66.0); Q = p.get("kf_Q", 0.5)
        if self.sx is None:
            self.sx = [rx, 0., 0.]; self.sy = [ry, 0., 0.]
            self.Px = [[1e3, 0, 0], [0, 1e3, 0], [0, 0, 1e3]]
            self.Py = [[1e3, 0, 0], [0, 1e3, 0], [0, 0, 1e3]]
            fx, fy = rx, ry
        else:
            fx = self._kf(self.sx, self.Px, rx, R, Q)
            fy = self._kf(self.sy, self.Py, ry, R, Q)
        saved = p["oneeuro"]; p["oneeuro"] = False
        self.filt_x, self.filt_y = fx, fy
        out = OptCtrl.step(self, fx, fy, cls_changed)
        p["oneeuro"] = saved
        return out


class SchedCtrl(OptCtrl):
    """표적 속도로 kp 를 스케줄. 정지 시 저게인(노이즈 억제), 추적 시 고게인(랙 감소).
    지금의 nonlinear P 는 '오차 크기'로만 스케줄하고 '표적 속도'는 보지 않는다."""
    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        g = p.get("sched_gain", 0.0)
        if g <= 0.0:
            return OptCtrl.step(self, rx, ry, cls_changed)
        sp = math.hypot(self.vx, self.vy)
        k = 1.0 + g * (sp / (sp + p.get("sched_vref", 4.0)))
        kx, ky = p["kp_x"], p["kp_y"]
        p["kp_x"], p["kp_y"] = kx*k, ky*k
        out = OptCtrl.step(self, rx, ry, cls_changed)
        p["kp_x"], p["kp_y"] = kx, ky
        return out
