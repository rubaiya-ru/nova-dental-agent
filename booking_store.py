from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, date, time, timedelta
from pathlib import Path

STORE_PATH = Path(__file__).parent / "bookings.json"
_LOCK = threading.Lock()

# Monday=0 ... Sunday=6
HOURS: dict[int, tuple[time, time] | None] = {
    0: (time(8, 30), time(17, 30)),
    1: (time(8, 30), time(17, 30)),
    2: (time(8, 30), time(17, 30)),
    3: (time(8, 30), time(17, 30)),
    4: (time(8, 30), time(16, 0)),
    5: (time(9, 0), time(13, 0)),
    6: None,  # closed
}

EMERGENCY_TIMES = [time(11, 30), time(15, 30)]


def _load() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    with STORE_PATH.open("r") as f:
        return json.load(f)


def _save(records: list[dict]) -> None:
    with STORE_PATH.open("w") as f:
        json.dump(records, f, indent=2, default=str)


def append_record(record: dict) -> dict:
    with _LOCK:
        records = _load()
        records.append(record)
        _save(records)
    return record


def generate_reference() -> str:
    return f"PD-{datetime.now():%y%m%d}-{secrets.token_hex(2).upper()}"


def validate_slot(dt: datetime, service_minutes: int, *, urgent: bool = False) -> tuple[bool, str]:
    hours = HOURS.get(dt.weekday())
    if hours is None:
        return False, "closed_sunday"

    open_t, close_t = hours
    end_dt = dt + timedelta(minutes=service_minutes)
    if dt.time() < open_t or end_dt.time() > close_t or end_dt.date() != dt.date():
        return False, "out_of_hours"

    if not urgent and dt.time() in EMERGENCY_TIMES:
        return False, "emergency_slot_reserved"

    return True, "ok"


def _next_open_day(from_date: date, weekday_target: int) -> date:
    d = from_date
    for _ in range(14):
        d = d + timedelta(days=1)
        if HOURS.get(d.weekday()) is not None and d.weekday() == weekday_target:
            return d
    return from_date + timedelta(days=7)


def suggest_alternatives(dt: datetime) -> list[str]:
    sat = _next_open_day(dt.date(), weekday_target=5)
    mon = _next_open_day(dt.date(), weekday_target=0)
    return [
        f"Saturday {sat.isoformat()} morning (9:00am-1:00pm)",
        f"Monday {mon.isoformat()} from 8:30am",
    ]


def _slot_taken(candidate: datetime) -> bool:
    for record in _load():
        if record.get("type") == "booking" and record.get("datetime") == candidate.isoformat():
            return True
    return False


def next_emergency_slot(now: datetime) -> datetime:
    for day_offset in range(30):
        check_date = now.date() + timedelta(days=day_offset)
        hours = HOURS.get(check_date.weekday())
        if hours is None:
            continue
        open_t, close_t = hours
        for t in EMERGENCY_TIMES:
            if open_t <= t <= close_t:  # e.g. Sat closes 1pm, so 15:30 never offered Sat
                candidate = datetime.combine(check_date, t)
                if candidate > now and not _slot_taken(candidate):
                    return candidate
    raise RuntimeError("no emergency slot found in the next 30 days")