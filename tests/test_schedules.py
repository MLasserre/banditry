import pytest

from banditry.schedules import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
)


def test_constant_schedule_keeps_start_value():
    schedule = ConstantSchedule(initial_value=0.4)
    assert schedule.value(step=0) == pytest.approx(0.4)
    assert schedule.value(step=100) == pytest.approx(0.4)


def test_linear_decay_schedule_interpolates_to_floor():
    schedule = LinearDecaySchedule(initial_value=1.0, min_value=0.1, decay_steps=4)
    assert schedule.value(step=0) == pytest.approx(1.0)
    assert schedule.value(step=2) == pytest.approx(0.55)
    assert schedule.value(step=4) == pytest.approx(0.1)
    assert schedule.value(step=10) == pytest.approx(0.1)


def test_exponential_decay_schedule_respects_floor():
    schedule = ExponentialDecaySchedule(initial_value=1.0, min_value=0.1, decay=0.5)
    assert schedule.value(step=0) == pytest.approx(1.0)
    assert schedule.value(step=1) == pytest.approx(0.5)
    assert schedule.value(step=2) == pytest.approx(0.25)
    assert schedule.value(step=10) == pytest.approx(0.1)


def test_schedule_validations():
    with pytest.raises(ValueError):
        LinearDecaySchedule(initial_value=1.0, decay_steps=0)
    with pytest.raises(ValueError):
        ExponentialDecaySchedule(initial_value=1.0, decay=0.0)
    with pytest.raises(ValueError):
        ExponentialDecaySchedule(initial_value=1.0, decay=1.1)
    with pytest.raises(ValueError):
        LinearDecaySchedule(initial_value=1.0).value(step=-1)
    with pytest.raises(ValueError):
        ExponentialDecaySchedule(initial_value=1.0).value(step=-1)
