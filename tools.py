from __future__ import annotations

import datetime as dt

from livekit.agents import function_tool, RunContext

import booking_store as store

SERVICE_DURATIONS = {
    "check-up and clean": 45, "check-up": 45, "clean": 45,
    "filling": 45,
    "emergency": 30, "toothache": 30,
    "extraction consult": 30, "extraction": 30,
}


def _service_minutes(service: str) -> int:
    key = service.strip().lower()
    for name, minutes in SERVICE_DURATIONS.items():
        if name in key:
            return minutes
    return 45


@function_tool()
async def create_booking(
    context: RunContext,
    name: str,
    mobile: str,
    service: str,
    date_time: str,
    new_patient: bool,
) -> dict:
    """Create a routine (non-emergency) booking at Parkline Dental.

    Args:
        name: Caller's full name.
        mobile: Callback mobile number.
        service: One of "check-up and clean", "filling", "extraction consult".
            Never use this for emergencies - use book_emergency instead.
        date_time: Requested appointment date and time in ISO 8601,
            e.g. "2026-09-14T09:00:00".
        new_patient: True if this is a new patient, False if existing.
    """
    try:
        requested = dt.datetime.fromisoformat(date_time)
    except ValueError:
        return {"accepted": False, "reason": "invalid_datetime_format"}

    minutes = _service_minutes(service)
    ok, reason = store.validate_slot(requested, minutes, urgent=False)
    if not ok:
        payload: dict = {"accepted": False, "reason": reason}
        if reason in ("closed_sunday", "out_of_hours"):
            payload["suggested_alternatives"] = store.suggest_alternatives(requested)
        return payload

    record = {
        "type": "booking",
        "reference": store.generate_reference(),
        "name": name, "mobile": mobile, "service": service,
        "datetime": requested.isoformat(),
        "new_patient": new_patient, "urgent": False,
        "created_at": dt.datetime.now().isoformat(),
    }
    store.append_record(record)
    return {"accepted": True, **record}


@function_tool()
async def book_emergency(context: RunContext, name: str, mobile: str, symptom: str) -> dict:
    """Reserve the next free emergency slot (11:30 or 15:30) for a caller in pain.
    Only ever offers the next free slot - never negotiate a different time.

    Args:
        name: Caller's full name.
        mobile: Callback mobile number.
        symptom: Brief description, e.g. "throbbing tooth since last night".
    """
    slot = store.next_emergency_slot(dt.datetime.now())
    record = {
        "type": "booking",
        "reference": store.generate_reference(),
        "name": name, "mobile": mobile,
        "service": "Emergency / toothache",
        "datetime": slot.isoformat(),
        "new_patient": None, "urgent": True, "symptom": symptom,
        "created_at": dt.datetime.now().isoformat(),
    }
    store.append_record(record)
    return record


@function_tool()
async def take_message(context: RunContext, name: str, mobile: str, reason: str) -> dict:
    """Log a message for the practice team: fees, health funds, HICAPS, Medicare,
    complaints, or any clinical question Nova can't answer.
    """
    record = {
        "type": "message",
        "reference": store.generate_reference(),
        "name": name, "mobile": mobile, "reason": reason,
        "created_at": dt.datetime.now().isoformat(),
    }
    store.append_record(record)
    return record