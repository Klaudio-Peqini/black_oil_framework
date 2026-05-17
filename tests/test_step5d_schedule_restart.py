import numpy as np

from blackoil import (
    WellControlEvent,
    WellSchedule,
    ControlledWell,
    StateBlackOil,
    save_black_oil_restart,
    load_black_oil_restart,
    apply_black_oil_restart,
    day,
    bar,
)


class DummySimulator:
    def __init__(self):
        self.state = StateBlackOil(
            p=np.array([1.0, 2.0]),
            sw=np.array([0.2, 0.3]),
            x=np.array([100.0, 0.02]),
            is_saturated=np.array([False, True]),
        )
        self.cumulative_oil_produced = 1.0
        self.cumulative_water_produced = 2.0
        self.cumulative_free_gas_produced = 3.0
        self.cumulative_gas_component_produced = 4.0
        self.cumulative_oil_injected = 5.0
        self.cumulative_water_injected = 6.0
        self.cumulative_free_gas_injected = 7.0
        self.cumulative_gas_component_injected = 8.0


def test_schedule_replays_piecewise_controls():
    schedule = WellSchedule(
        events=[
            WellControlEvent(0.0, "P1", cell=3, well_index=2.0e-15, control="total_rate", target=-10.0 / day, min_bhp=120.0 * bar),
            WellControlEvent(2.0 * day, "P1", control="bhp", target=150.0 * bar),
            WellControlEvent(3.0 * day, "P1", control="shut"),
            WellControlEvent(1.0 * day, "I1", cell=0, well_index=1.0e-15, control="water_rate", target=5.0 / day, max_bhp=300.0 * bar),
        ]
    )
    wells0 = schedule.wells_at(0.5 * day)
    assert len(wells0) == 1
    assert wells0[0].name == "P1"
    assert wells0[0].control == "total_rate"
    wells1 = schedule.wells_at(2.5 * day)
    by_name = {w.name: w for w in wells1}
    assert by_name["P1"].control == "bhp"
    assert np.isclose(by_name["P1"].target, 150.0 * bar)
    assert by_name["I1"].control == "water_rate"
    wells2 = schedule.wells_at(3.5 * day)
    assert [w.name for w in wells2] == ["I1"]


def test_schedule_milestones_include_report_times():
    schedule = WellSchedule(
        events=[WellControlEvent(1.0 * day, "I1", cell=0, well_index=1.0e-15, control="water_rate", target=1.0 / day)],
        report_times=[0.25 * day, 1.5 * day],
    )
    milestones = schedule.milestone_times(0.0, 2.0 * day)
    days = [m / day for m in milestones]
    assert days == [0.25, 1.0, 1.5]


def test_restart_roundtrip(tmp_path):
    sim = DummySimulator()
    path = tmp_path / "restart.npz"
    save_black_oil_restart(path, sim, 12.5, metadata={"case": "unit"})
    data = load_black_oil_restart(path)
    assert data["time"] == 12.5
    assert data["metadata"]["case"] == "unit"
    assert np.allclose(data["state"].p, sim.state.p)
    assert np.array_equal(data["state"].is_saturated, sim.state.is_saturated)

    sim2 = DummySimulator()
    sim2.state.p[:] = 0.0
    t = apply_black_oil_restart(sim2, data)
    assert t == 12.5
    assert np.allclose(sim2.state.p, sim.state.p)
    assert sim2.cumulative_gas_component_injected == 8.0
