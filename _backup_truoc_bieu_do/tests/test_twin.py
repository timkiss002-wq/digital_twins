import random

from twin import DigitalTwin, PhysicalGates, TwinConfig, fluid_forecast


def run(tw, lam, seconds, fps=10, t0=0.0, w_obs=4.0, seed=1):
    """Sinh dòng xe Poisson(λ) và cho twin chạy theo từng khung."""
    rng = random.Random(seed)
    next_a = t0 + rng.expovariate(lam)
    t = t0
    while t < t0 + seconds:
        t = round(t + 1 / fps, 4)
        arr = []
        while next_a <= t:
            arr.append(next_a)
            next_a += rng.expovariate(lam)
        tw.step(t, arr, lam, w_obs, 20)
    return t


def test_physical_gates_fifo_and_queue():
    g = PhysicalGates(1, mu=1.0, seed=0)
    for a in (0.0, 0.1, 0.2):
        g.arrive(a)
    assert g.queue(0.25) == 3
    assert g.queue(1e6) == 0
    assert len(g.done) == 3


def test_more_gates_reduce_wait():
    waits = []
    for c in (1, 3):
        g = PhysicalGates(c, mu=0.5, seed=3)
        rng, t = random.Random(5), 0.0
        for _ in range(2000):
            t += rng.expovariate(0.45)
            g.arrive(t)
        waits.append(g.mean_wait(0, 1e9))
    assert waits[1] < waits[0]


def test_fluid_stable_vs_overloaded():
    assert fluid_forecast(0, lam=0.5, mu=0.5, c=2, horizon=200, dt=0.1) < 2       # ρ = 0.5
    assert fluid_forecast(0, lam=1.5, mu=0.5, c=2, horizon=60, dt=0.1) > 20       # ρ > 1


def test_sync_resets_virtual_and_tracks_fidelity():
    tw = DigitalTwin(TwinConfig(sync_s=2.0))
    run(tw, lam=0.6, seconds=45)
    s = tw.snapshot()
    assert tw.sync_count >= 14
    assert s["sync_mae"] is not None and s["sync_mae"] >= 0
    assert s["forecast_mae"] is not None
    assert s["mu_calibrated"]
    assert s["q_virtual"] >= 0
    assert any(e["dir"] == "T2A" for e in s["events"])


def test_overload_triggers_recommendation_then_closed_loop():
    cfg = TwinConfig(cooldown_s=5, actuation_delay_s=2, effect_window_s=8)
    tw = DigitalTwin(cfg)
    # W quan sát lớn với λ cao -> μ nhỏ -> ρ cao với 2 cổng
    t = run(tw, lam=1.0, seconds=20, w_obs=8.0)
    rec = tw.snapshot()["recommendation"]
    assert rec is not None and rec["from"] == 2 and rec["to"] == 3
    assert tw.snapshot()["forecast_alt"] is not None
    assert tw.respond(rec["id"], True)
    assert not tw.respond(rec["id"], True)             # không chấp nhận hai lần
    t = run(tw, lam=1.0, seconds=15, t0=t, w_obs=8.0, seed=2)
    s = tw.snapshot()
    assert s["gates"] == 3 and tw.physical.gates == 3
    assert s["commands"] and s["commands"][0][1:] == (2, 3)
    eff = [e["text"] for e in s["events"] if "Tác động" in e["text"]]
    assert eff and "nếu giữ 2 cổng" in eff[0]


def test_counterfactual_shows_benefit_of_extra_gate():
    tw = DigitalTwin(TwinConfig(cooldown_s=5, actuation_delay_s=1, effect_window_s=40))
    t = run(tw, lam=1.0, seconds=20, w_obs=8.0)
    tw.respond(tw.snapshot()["recommendation"]["id"], True)
    run(tw, lam=1.0, seconds=70, t0=t, w_obs=8.0, seed=4)
    txt = [e["text"] for e in tw.snapshot()["events"] if "Tác động" in e["text"]][0]
    assert "-" in txt.split("(")[-1]           # TG chờ giảm so với đối chứng


def test_forecast_depends_on_gate_count():
    from twin import fluid_forecast as ff
    assert ff(5, lam=0.9, mu=0.5, c=3, horizon=30, dt=0.25) < ff(5, lam=0.9, mu=0.5, c=2, horizon=30, dt=0.25)


def test_rejected_recommendation_keeps_gates():
    tw = DigitalTwin(TwinConfig(cooldown_s=5))
    run(tw, lam=1.0, seconds=20, w_obs=8.0)
    rec = tw.snapshot()["recommendation"]
    assert tw.respond(rec["id"], False)
    assert tw.snapshot()["gates"] == 2
    assert tw.snapshot()["recommendation"] is None


def test_no_recommendation_without_calibration():
    tw = DigitalTwin()
    t = 0.0
    for _ in range(300):
        t += 0.1
        tw.step(t, [t], 10.0, None, 0)                 # chưa có mẫu TG chờ
    assert tw.snapshot()["recommendation"] is None


def test_gate_state_shows_waiting_and_serving():
    from twin import PhysicalGates
    g = PhysicalGates(2, mu=0.5)
    for a, s in ((0.0, 10.0), (0.1, 10.0), (0.2, 10.0)):
        g.arrive(a, s)
    st = g.state(1.0)
    assert [x["gate"] for x in st["serving"]] == [1, 2]
    assert all(x["car"] for x in st["serving"])
    assert len(st["waiting"]) == 1 and st["waiting"][0]["id"] == 3
    g.set_gates(3, 1.0)
    assert [x["gate"] for x in g.state(1.0)["serving"]] == [1, 2, 3]
    assert g.state(100.0)["served_total"] == 3


def test_manual_gate_control_goes_through_closed_loop():
    tw = DigitalTwin(TwinConfig(actuation_delay_s=2, effect_window_s=8))
    t = run(tw, lam=0.5, seconds=10)
    assert tw.manual_set(4)[0] is False                  # luôn giữ 1 cổng Ra
    assert tw.manual_set(2)[0] is False                  # không đổi
    ok, _ = tw.manual_set(3)
    assert ok and tw.snapshot()["recommendation"]["manual"]
    assert tw.manual_set(1)[0] is False                  # lệnh trước chưa xong
    t = run(tw, lam=0.5, seconds=20, t0=t, seed=3)
    s = tw.snapshot()
    assert s["gates"] == 3 and tw.physical.gates == 3
    assert any("Lệnh thủ công" in e["text"] for e in s["events"])
    assert any("Tác động lệnh 2→3" in e["text"] for e in s["events"])
    assert tw.manual_set(1)[0] and run(tw, lam=0.5, seconds=5, t0=t) and tw.snapshot()["gates"] == 1


def test_manual_replaces_pending_recommendation():
    tw = DigitalTwin(TwinConfig(cooldown_s=5))
    run(tw, lam=1.0, seconds=20, w_obs=8.0)
    rec = tw.snapshot()["recommendation"]
    assert rec["status"] == "pending"
    assert tw.manual_set(1)[0]
    assert not tw.respond(rec["id"], True)
