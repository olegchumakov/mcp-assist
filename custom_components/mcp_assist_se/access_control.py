"""Per-person access decisions for voice and typed commands.

Pure data and checks. Home Assistant registries are read in access_hass.py.
Nothing here contacts the Speaker ID add-on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .const import (
    CONF_ACCESS_ADDON_URL,
    CONF_ACCESS_CACHED_PEOPLE,
    CONF_ACCESS_CLASSES,
    CONF_ACCESS_CONTROL_ENABLED,
    CONF_ACCESS_MANUAL_PEOPLE,
    CONF_ACCESS_MIN_CONFIDENCE,
    CONF_ACCESS_NONE_REPLY,
    CONF_ACCESS_RULES,
    CONF_ACCESS_SHARED_SECRET,
    CONF_ACCESS_UNKNOWN_RULE,
    CONF_ACCESS_USER_MAP,
    DEFAULT_ACCESS_CONTROL_ENABLED,
    DEFAULT_ACCESS_MIN_CONFIDENCE,
    DEFAULT_ACCESS_NONE_REPLY,
    DEFAULT_ACCESS_UNKNOWN_LEVEL,
)
from .memory_settings import as_bool
from .speaker_tag import NonceCache, verify_speaker_tag

LEVEL_NONE = "none"
LEVEL_SELECTED = "selected"
LEVEL_ALL = "all"
LEVELS = (LEVEL_NONE, LEVEL_SELECTED, LEVEL_ALL)

REFUSAL_TEXT = (
    "NOT ALLOWED\n"
    "This action is not allowed for the current person. "
    "Tell the user it is not allowed. Do not name devices they cannot see "
    "and do not try a wider target, a script, a scene, or another tool to get around this."
)

_TARGET_KEYS = ("entity_id", "area_id", "device_id", "label_id", "floor_id")
_BROAD_ENTITY_IDS = {"all", "none"}
_ENTITY_TOKEN = re.compile(r"[a-z_]+[.][a-z0-9_]+")
_SCRIPT_META_ACTIONS = {"turn_on", "turn_off", "toggle", "reload"}


@dataclass(frozen=True)
class DeviceClass:
    """A named set of entities, devices, areas, and labels."""

    id: str
    name: str
    entity_ids: frozenset[str] = frozenset()
    device_ids: frozenset[str] = frozenset()
    area_ids: frozenset[str] = frozenset()
    label_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PersonRule:
    """NONE, SELECTED classes, or ALL."""

    level: str = LEVEL_ALL
    class_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EntityRef:
    """One entity as the checker needs to see it."""

    entity_id: str
    domain: str
    device_id: str | None = None
    area_id: str | None = None
    area_name: str = ""
    label_ids: frozenset[str] = frozenset()
    label_names: frozenset[str] = frozenset()
    member_ids: tuple[str, ...] = ()
    name: str = ""
    friendly_name: str = ""


@dataclass(frozen=True)
class HomeSnapshot:
    """Entities and area→floor links used to resolve a service target."""

    entities: dict[str, EntityRef] = field(default_factory=dict)
    area_floors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AccessConfig:
    """Shared access-control settings. Disabled config changes nothing."""

    enabled: bool = False
    secret: str = ""
    addon_url: str = ""
    min_confidence: float = 0.0
    none_reply: str = ""
    classes: dict[str, DeviceClass] = field(default_factory=dict)
    rules: dict[str, PersonRule] = field(default_factory=dict)
    unknown_rule: PersonRule = field(default_factory=lambda: PersonRule(DEFAULT_ACCESS_UNKNOWN_LEVEL))
    user_map: dict[str, str] = field(default_factory=dict)
    manual_people: tuple[str, ...] = ()
    cached_people: tuple[str, ...] = ()


@dataclass(frozen=True)
class Grant:
    """What the current speaker may do. ``all_of`` means every part must allow it."""

    level: str = LEVEL_ALL
    class_ids: frozenset[str] = frozenset()
    all_of: tuple[Grant, ...] = ()

    def intersect(self, other: Grant) -> Grant:
        """Rights that both grants allow. Never wider than either one."""
        if _grant_blocks(self) or _grant_blocks(other):
            return Grant(LEVEL_NONE)
        if is_unrestricted(self):
            return other
        if is_unrestricted(other):
            return self
        parts: list[Grant] = []
        parts.extend(self.all_of or (self,))
        parts.extend(other.all_of or (other,))
        flat: list[Grant] = []
        for part in parts:
            if part.all_of:
                flat.extend(part.all_of)
            else:
                flat.append(Grant(part.level, part.class_ids))
        return Grant(LEVEL_SELECTED, frozenset(), tuple(flat))


@dataclass(frozen=True)
class TurnDecision:
    """What to do with one user utterance."""

    enabled: bool
    text: str
    grant: Grant | None
    blocked: bool
    reply: str
    verified: bool
    person: str


@dataclass(frozen=True)
class ActionDecision:
    """Whether a service call may run, and the entity ids to pass if it may."""

    allowed: bool
    passthrough: bool = False
    entity_ids: tuple[str, ...] = ()
    refusal: str = ""


def is_unrestricted(grant: Grant) -> bool:
    """ALL with no tighter part. Disabled mode does not build a grant."""
    if grant.all_of:
        return all(is_unrestricted(part) for part in grant.all_of)
    return grant.level == LEVEL_ALL


def _grant_blocks(grant: Grant) -> bool:
    if grant.all_of:
        return any(_grant_blocks(part) for part in grant.all_of)
    return grant.level == LEVEL_NONE


def should_fetch_people(enabled: bool, url: str | None) -> bool:
    """The people list is fetched only while access control is on and a URL is set."""
    return bool(enabled and str(url or "").strip())


def people_endpoint(url: str | None) -> str:
    """GET URL for the add-on's people list. Accepts a host root or the full path.

    Example: ``http://local-wyoming-speaker-id:8099/api/people``.
    """
    base = str(url or "").strip().rstrip("/")
    if base.endswith("/api/people"):
        return base
    return f"{base}/api/people"


def people_from_response(status: int, payload: Any) -> list[str] | None:
    """Names from a people-list response. 401 and every other non-200 is a miss."""
    if status != 200:
        return None
    return parse_people_payload(payload)


def parse_people_payload(payload: Any) -> list[str]:
    """Names from GET /api/people. Empty when the payload is not the contract."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("people")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("name")
        else:
            value = item
        if isinstance(value, str):
            names.append(value)
    return dedupe_people(names)


def normalize_person_name(name: str | None) -> str:
    """Collapse whitespace. Capitals stay as written, for display."""
    return " ".join(str(name or "").split()).strip()


def person_key(name: str | None) -> str:
    """Match key. Capitals and extra spaces do not count. Empty stays empty."""
    return normalize_person_name(name).casefold()


def dedupe_people(items) -> list[str]:
    """One display spelling per person. The first spelling is kept."""
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = normalize_person_name(item if isinstance(item, str) else "")
        key = person_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def split_names(text: str | None) -> list[str]:
    """Split a textarea of people on commas and new lines."""
    raw: list[str] = []
    for line in str(text or "").replace(",", "\n").splitlines():
        raw.append(line)
    return dedupe_people(raw)


def people_catalog(config: AccessConfig) -> list[str]:
    """Manual names, cached add-on names, and anyone who already has a rule."""
    return dedupe_people((*config.cached_people, *config.manual_people, *config.rules.keys()))


def _as_str_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item).strip() for item in value if str(item).strip())


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(split_names(value))
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dedupe_people(str(item) if item is not None else "" for item in value))


def _level(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in LEVELS:
        return text
    return default


def _rule_from(value: Any, default_level: str) -> PersonRule:
    if not isinstance(value, dict):
        return PersonRule(default_level)
    return PersonRule(
        level=_level(value.get("level"), default_level),
        class_ids=_as_str_set(value.get("classes") or value.get("class_ids")),
    )


def _class_from(value: Any) -> DeviceClass | None:
    if not isinstance(value, dict):
        return None
    class_id = str(value.get("id") or "").strip()
    name = " ".join(str(value.get("name") or "").split()).strip()
    if not class_id or not name:
        return None
    return DeviceClass(
        id=class_id,
        name=name,
        entity_ids=_as_str_set(value.get("entity_ids")),
        device_ids=_as_str_set(value.get("device_ids")),
        area_ids=_as_str_set(value.get("area_ids")),
        label_ids=_as_str_set(value.get("label_ids")),
    )


def load_access_config(source: dict[str, Any] | None) -> AccessConfig:
    """Build config from a system-entry dict. Missing keys leave the feature off."""
    data = source or {}
    try:
        enabled = as_bool(data.get(CONF_ACCESS_CONTROL_ENABLED), DEFAULT_ACCESS_CONTROL_ENABLED)
    except Exception:
        return AccessConfig(enabled=False)
    classes: dict[str, DeviceClass] = {}
    raw_classes = data.get(CONF_ACCESS_CLASSES) or []
    if isinstance(raw_classes, list):
        for item in raw_classes:
            parsed = _class_from(item)
            if parsed is not None:
                classes[parsed.id] = parsed
    rules: dict[str, PersonRule] = {}
    seen_people: set[str] = set()
    raw_rules = data.get(CONF_ACCESS_RULES) or {}
    if isinstance(raw_rules, dict):
        for name, rule in raw_rules.items():
            person = normalize_person_name(str(name))
            key = person_key(person)
            if not key or key in seen_people:
                continue
            seen_people.add(key)
            rules[person] = _rule_from(rule, LEVEL_ALL)
    try:
        minimum = float(data.get(CONF_ACCESS_MIN_CONFIDENCE, DEFAULT_ACCESS_MIN_CONFIDENCE) or 0)
    except (TypeError, ValueError):
        minimum = 0.0
    minimum = max(0.0, min(1.0, minimum))
    user_map: dict[str, str] = {}
    raw_map = data.get(CONF_ACCESS_USER_MAP) or {}
    if isinstance(raw_map, dict):
        for user_id, person in raw_map.items():
            uid = str(user_id).strip()
            name = normalize_person_name(person if isinstance(person, str) else "")
            if uid and name:
                user_map[uid] = name
    return AccessConfig(
        enabled=enabled,
        secret=str(data.get(CONF_ACCESS_SHARED_SECRET) or ""),
        addon_url=str(data.get(CONF_ACCESS_ADDON_URL) or "").strip(),
        min_confidence=minimum,
        none_reply=str(
            data.get(CONF_ACCESS_NONE_REPLY)
            if data.get(CONF_ACCESS_NONE_REPLY) is not None
            else DEFAULT_ACCESS_NONE_REPLY
        ),
        classes=classes,
        rules=rules,
        unknown_rule=_rule_from(data.get(CONF_ACCESS_UNKNOWN_RULE), DEFAULT_ACCESS_UNKNOWN_LEVEL),
        user_map=user_map,
        manual_people=_as_str_tuple(data.get(CONF_ACCESS_MANUAL_PEOPLE)),
        cached_people=_as_str_tuple(data.get(CONF_ACCESS_CACHED_PEOPLE)),
    )


def settings_updates(user_input: dict[str, Any]) -> dict[str, Any]:
    """Scalar settings from the first access-control form. Does not fetch anyone."""
    enabled = as_bool(user_input.get(CONF_ACCESS_CONTROL_ENABLED), False)
    try:
        minimum = float(user_input.get(CONF_ACCESS_MIN_CONFIDENCE, 0) or 0)
    except (TypeError, ValueError):
        minimum = 0.0
    return {
        CONF_ACCESS_CONTROL_ENABLED: enabled,
        CONF_ACCESS_SHARED_SECRET: str(user_input.get(CONF_ACCESS_SHARED_SECRET) or ""),
        CONF_ACCESS_ADDON_URL: str(user_input.get(CONF_ACCESS_ADDON_URL) or "").strip().rstrip("/"),
        CONF_ACCESS_MIN_CONFIDENCE: max(0.0, min(1.0, minimum)),
        CONF_ACCESS_NONE_REPLY: str(user_input.get(CONF_ACCESS_NONE_REPLY) or ""),
        CONF_ACCESS_MANUAL_PEOPLE: split_names(user_input.get(CONF_ACCESS_MANUAL_PEOPLE)),
    }


def class_matches(entity: EntityRef, device_class: DeviceClass) -> bool:
    """True when the entity, its device, its area, or a label was picked."""
    if entity.entity_id in device_class.entity_ids:
        return True
    if entity.device_id and entity.device_id in device_class.device_ids:
        return True
    if entity.area_id and entity.area_id in device_class.area_ids:
        return True
    if device_class.label_ids and (entity.label_ids & device_class.label_ids):
        return True
    return False


def entity_allowed(entity: EntityRef, grant: Grant, config: AccessConfig) -> bool:
    """Whether this speaker may see or control the entity."""
    if grant.all_of:
        return all(entity_allowed(entity, part, config) for part in grant.all_of)
    if grant.level == LEVEL_ALL:
        return True
    if grant.level == LEVEL_NONE:
        return False
    for class_id in grant.class_ids:
        device_class = config.classes.get(class_id)
        if device_class is not None and class_matches(entity, device_class):
            return True
    return False


def rule_for_person(rules: dict[str, PersonRule], person: str) -> PersonRule | None:
    """The stored rule for this name. Capitals and extra spaces do not matter."""
    key = person_key(person)
    if not key:
        return None
    for name, rule in rules.items():
        if person_key(name) == key:
            return rule
    return None


def grant_for_person(config: AccessConfig, person: str) -> Grant:
    """Rule for a recognized name. A missing name uses the unknown rule.

    ``олег`` is the same person as ``Олег``. A person with no stored rule is
    ALL, so turning the feature on before any rule is saved does not lock
    the home. SELECTED with no groups is NONE: nothing was chosen.
    """
    found = rule_for_person(config.rules, person)
    if found is not None:
        rule = found
    elif person_key(person):
        rule = PersonRule(LEVEL_ALL)
    else:
        rule = config.unknown_rule
    if rule.level == LEVEL_SELECTED and not rule.class_ids:
        return Grant(LEVEL_NONE)
    if rule.level == LEVEL_SELECTED:
        return Grant(LEVEL_SELECTED, frozenset(rule.class_ids))
    if rule.level == LEVEL_NONE:
        return Grant(LEVEL_NONE)
    return Grant(LEVEL_ALL)


def grant_to_payload(grant: Grant) -> dict[str, Any]:
    """JSON shape carried on the tool-call context for this turn."""
    return {
        "level": grant.level,
        "class_ids": sorted(grant.class_ids),
        "all_of": [grant_to_payload(part) for part in grant.all_of],
    }


def grant_from_payload(payload: Any, depth: int = 0) -> Grant:
    """Restore a grant. Garbage becomes NONE so a bad context cannot widen access."""
    if depth > 8 or not isinstance(payload, dict):
        return Grant(LEVEL_NONE)
    level = _level(payload.get("level"), LEVEL_NONE)
    class_ids = _as_str_set(payload.get("class_ids"))
    raw_parts = payload.get("all_of") or []
    parts: tuple[Grant, ...] = ()
    if isinstance(raw_parts, list) and raw_parts:
        parts = tuple(grant_from_payload(part, depth + 1) for part in raw_parts)
    return Grant(level, class_ids, parts)


def decide_turn(
    config: AccessConfig,
    text: str,
    *,
    user_id: str | None,
    previous: Grant | None,
    now: float,
    nonces: NonceCache | None = None,
) -> tuple[TurnDecision, Grant | None]:
    """Resolve the speaker for one utterance.

    Returns the decision and the grant to remember for this conversation id.
    The second value is None when access control is off: the caller must not
    replace a stored grant and must not parse the text.

    A verified tag or a typed Home Assistant user replaces the remembered
    grant. A follow-up with no valid identity keeps only the overlap with the
    unknown rule, so continuing the conversation cannot widen access.
    """
    original = text or ""
    if not config.enabled:
        return (
            TurnDecision(False, original, None, False, "", False, ""),
            None,
        )

    verdict = verify_speaker_tag(original, config.secret, now, nonces)
    verified = False
    person = ""
    if verdict.status == "valid":
        verified = True
        person = normalize_person_name(verdict.person)
        if config.min_confidence > 0 and verdict.confidence < config.min_confidence:
            person = ""
    elif verdict.status == "absent" and user_id:
        verified = True
        person = normalize_person_name(config.user_map.get(str(user_id), ""))

    if verified:
        grant = grant_for_person(config, person)
    else:
        unknown = grant_for_person(config, "")
        grant = unknown if previous is None else previous.intersect(unknown)

    blocked = _grant_blocks(grant)
    return (
        TurnDecision(
            enabled=True,
            text=verdict.clean_text,
            grant=grant,
            blocked=blocked,
            reply=config.none_reply if blocked else "",
            verified=verified,
            person=person if verified else "",
        ),
        grant,
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _walk_strings(value: Any, found: list[str]) -> None:
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _walk_strings(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_strings(item, found)


def contains_template(target: Any, data: Any) -> bool:
    """Home Assistant templates can expand to entities we did not check."""
    strings: list[str] = []
    _walk_strings(target, strings)
    _walk_strings(data, strings)
    return any("{{" in item or "{%" in item for item in strings)


def strip_targeting(data: dict[str, Any] | None) -> dict[str, Any]:
    """Drop target keys so a rewritten call cannot still name an area or device."""
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key not in _TARGET_KEYS}


def _domain_selected(service_domain: str, entity_domain: str) -> bool:
    """Area, device, label, and floor selectors hit this domain only.

    homeassistant.turn_on/off really can hit every entity in the area, so
    those calls have to be allowed for every entity there.
    """
    if service_domain == "homeassistant":
        return True
    return service_domain == entity_domain


def _normalize_entity_token(value: str, known: set[str]) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text in known or text.lower() in _BROAD_ENTITY_IDS:
        return [text]
    if "," in text:
        parts: list[str] = []
        for piece in text.split(","):
            parts.extend(_normalize_entity_token(piece, known))
        return parts
    return [text]


def _collect_selectors(source: Any) -> dict[str, list[str]]:
    bags = {key: [] for key in _TARGET_KEYS}
    if isinstance(source, dict):
        for key in _TARGET_KEYS:
            bags[key].extend(_as_list(source.get(key)))
    return bags


def _selector_entity_ids(
    selectors: dict[str, list[str]],
    snap: HomeSnapshot,
    service_domain: str,
    *,
    domain_filter: bool,
) -> list[str]:
    """Entities named by one selector dict.

    With ``domain_filter``, an area or device contributes only entities of the
    service domain (``homeassistant`` still contributes every entity). Without
    it, every entity in the area, device, label, or floor counts, which is
    what a script variable can reach.
    """
    known = set(snap.entities)
    resolved: list[str] = []

    def add(entity_id: str) -> None:
        if entity_id and entity_id not in resolved:
            resolved.append(entity_id)

    for raw in selectors["entity_id"]:
        for token in _normalize_entity_token(str(raw), known):
            if token.lower() in _BROAD_ENTITY_IDS:
                for entity in snap.entities.values():
                    if not domain_filter or _domain_selected(service_domain, entity.domain):
                        add(entity.entity_id)
                continue
            add(token)

    area_ids = set(selectors["area_id"])
    device_ids = set(selectors["device_id"])
    label_ids = set(selectors["label_id"])
    floor_ids = set(selectors["floor_id"])
    if not (area_ids or device_ids or label_ids or floor_ids):
        return resolved

    for entity in snap.entities.values():
        floor = snap.area_floors.get(entity.area_id or "")
        matched = (
            (entity.area_id in area_ids if entity.area_id else False)
            or (entity.device_id in device_ids if entity.device_id else False)
            or bool(label_ids and (entity.label_ids & label_ids))
            or bool(floor and floor in floor_ids)
        )
        if not matched:
            continue
        if domain_filter and not _domain_selected(service_domain, entity.domain):
            continue
        add(entity.entity_id)
    return resolved


def _mentioned_entity_ids(target: Any, data: Any, known: set[str]) -> list[str]:
    strings: list[str] = []
    _walk_strings(target, strings)
    _walk_strings(data, strings)
    found: list[str] = []
    for text in strings:
        if text in known:
            found.append(text)
            continue
        for token in _ENTITY_TOKEN.findall(text):
            if token in known and token not in found:
                found.append(token)
    return found


def _expand_members(entity_id: str, snap: HomeSnapshot, seen: set[str]) -> list[str]:
    if entity_id in seen:
        return []
    seen.add(entity_id)
    out = [entity_id]
    entity = snap.entities.get(entity_id)
    if entity is None:
        return out
    for member in entity.member_ids:
        out.extend(_expand_members(member, snap, seen))
    return out


def authorize_action(
    domain: str,
    service: str,
    target: Any,
    data: Any,
    grant: Grant,
    config: AccessConfig,
    snap: HomeSnapshot,
    *,
    broad_data: bool = False,
) -> ActionDecision:
    """Decide a service call.

    Unrestricted grants pass the original target through. A tighter grant must
    name entities that are all allowed. Area, device, label, and floor
    selectors are resolved to entity ids so Home Assistant cannot act on
    anything outside that list.

    ``broad_data`` is for scripts and automations, which receive ``data``
    unchanged and can use an area or label to touch every entity there.
    Templates, ``entity_id: all``, and calls with no resolvable entity are
    refused for a tighter grant. Group and scene members are checked too.
    """
    if is_unrestricted(grant):
        return ActionDecision(True, passthrough=True)
    if _grant_blocks(grant):
        return ActionDecision(False, refusal=REFUSAL_TEXT)
    if contains_template(target, data):
        return ActionDecision(False, refusal=REFUSAL_TEXT)

    service_domain = str(domain or "").strip()
    service_name = str(service or "").strip()
    call_ids = _selector_entity_ids(
        _collect_selectors(target), snap, service_domain, domain_filter=True
    )
    for entity_id in _selector_entity_ids(
        _collect_selectors(data),
        snap,
        service_domain,
        domain_filter=not broad_data,
    ):
        if entity_id not in call_ids:
            call_ids.append(entity_id)
    if service_domain == "script" and service_name and service_name not in _SCRIPT_META_ACTIONS:
        script_id = f"script.{service_name}"
        if script_id not in call_ids:
            call_ids.insert(0, script_id)

    if not call_ids:
        return ActionDecision(False, refusal=REFUSAL_TEXT)

    must_allow = list(call_ids)
    if broad_data:
        for entity_id in _selector_entity_ids(
            _collect_selectors(data), snap, service_domain, domain_filter=False
        ):
            if entity_id not in must_allow:
                must_allow.append(entity_id)
    for entity_id in _mentioned_entity_ids(target, data, set(snap.entities)):
        if entity_id not in must_allow:
            must_allow.append(entity_id)

    closure: list[str] = []
    seen: set[str] = set()
    for entity_id in must_allow:
        for member in _expand_members(entity_id, snap, seen):
            if member not in closure:
                closure.append(member)

    for entity_id in closure:
        entity = snap.entities.get(entity_id)
        if entity is None or not entity_allowed(entity, grant, config):
            return ActionDecision(False, refusal=REFUSAL_TEXT)
    return ActionDecision(True, passthrough=False, entity_ids=tuple(call_ids))


def entity_visible(entity_id: str, grant: Grant, config: AccessConfig, snap: HomeSnapshot) -> bool:
    """Read tools hide anything this grant cannot see."""
    if is_unrestricted(grant):
        return True
    entity = snap.entities.get(entity_id)
    if entity is None:
        return False
    return entity_allowed(entity, grant, config)


def filter_discovery_rows(
    entities: list[dict[str, Any]],
    grant: Grant,
    config: AccessConfig,
    snap: HomeSnapshot,
) -> list[dict[str, Any]]:
    """Drop discovery rows the speaker must not see. Summary counts follow."""
    if is_unrestricted(grant):
        return entities
    summary = None
    kept: list[dict[str, Any]] = []
    for row in entities:
        if not isinstance(row, dict):
            continue
        if row.get("entity_id") == "_summary":
            summary = dict(row)
            continue
        entity_id = str(row.get("entity_id") or "")
        if entity_id and entity_visible(entity_id, grant, config, snap):
            kept.append(row)
    if summary is not None:
        summary["total_found"] = len(kept)
        if not kept:
            return []
        return [summary, *kept]
    return kept


def visibility_sets(
    snap: HomeSnapshot, grant: Grant, config: AccessConfig
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Allowed entity ids, names, area names, and label names."""
    ids: set[str] = set()
    names: set[str] = set()
    areas: set[str] = set()
    labels: set[str] = set()
    for entity in snap.entities.values():
        if not entity_allowed(entity, grant, config):
            continue
        ids.add(entity.entity_id)
        if entity.name:
            names.add(entity.name)
        if entity.friendly_name:
            names.add(entity.friendly_name)
        if entity.area_name:
            areas.add(entity.area_name)
        labels.update(entity.label_names)
    return ids, names, areas, labels


def _keep_named(item: Any, allowed_ids: set[str], allowed_names: set[str]) -> bool:
    if isinstance(item, str):
        return item in allowed_names
    if isinstance(item, dict):
        entity_id = item.get("entity_id")
        if isinstance(entity_id, str) and entity_id in allowed_ids:
            return True
        script_id = item.get("id")
        if isinstance(script_id, str) and (
            f"script.{script_id}" in allowed_ids or f"automation.{script_id}" in allowed_ids
        ):
            return True
        name = item.get("name") or item.get("display_name")
        return isinstance(name, str) and name in allowed_names
    return False


def filter_index(
    index: dict[str, Any],
    grant: Grant,
    config: AccessConfig,
    snap: HomeSnapshot,
) -> dict[str, Any]:
    """Remove names of entities the speaker cannot see from the prompt index."""
    if is_unrestricted(grant) or not isinstance(index, dict):
        return index
    allowed_ids, allowed_names, allowed_areas, allowed_labels = visibility_sets(snap, grant, config)
    filtered: dict[str, Any] = {}
    for key, value in index.items():
        if key == "areas" and isinstance(value, list):
            filtered[key] = [
                area
                for area in value
                if isinstance(area, dict) and area.get("name") in allowed_areas
            ]
        elif key == "labels" and isinstance(value, list):
            filtered[key] = [
                label
                for label in value
                if isinstance(label, dict) and label.get("name") in allowed_labels
            ]
        elif key == "floors" and isinstance(value, list):
            floors = []
            for floor in value:
                if not isinstance(floor, dict):
                    continue
                area_names = [name for name in floor.get("areas") or [] if name in allowed_areas]
                if not area_names:
                    continue
                copied = dict(floor)
                copied["areas"] = area_names
                copied["area_count"] = len(area_names)
                floors.append(copied)
            filtered[key] = floors
        elif isinstance(value, list):
            filtered[key] = [
                item for item in value if _keep_named(item, allowed_ids, allowed_names)
            ]
        else:
            filtered[key] = value
    return filtered


def external_tool_allowed(grant: Grant | None, is_external: bool) -> bool:
    """External packages can call arbitrary services. Only ALL may run them."""
    if not is_external or grant is None:
        return True
    return is_unrestricted(grant)


def refusal_tool_result() -> dict[str, Any]:
    """Tool payload the model reads when a call is refused."""
    return {
        "content": [{"type": "text", "text": REFUSAL_TEXT}],
        "isError": True,
    }


def upsert_class(
    classes: list[dict[str, Any]],
    class_id: str | None,
    name: str,
    entity_ids: Iterable[str],
    device_ids: Iterable[str],
    area_ids: Iterable[str],
    label_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Insert or replace one device class. ``class_id`` None creates one."""
    import uuid

    cleaned_name = " ".join(name.split()).strip()
    stored = [dict(item) for item in classes if isinstance(item, dict)]
    new_id = class_id or uuid.uuid4().hex[:12]
    record = {
        "id": new_id,
        "name": cleaned_name,
        "entity_ids": sorted(_as_str_set(list(entity_ids))),
        "device_ids": sorted(_as_str_set(list(device_ids))),
        "area_ids": sorted(_as_str_set(list(area_ids))),
        "label_ids": sorted(_as_str_set(list(label_ids))),
    }
    replaced = False
    updated: list[dict[str, Any]] = []
    for item in stored:
        if str(item.get("id")) == new_id:
            updated.append(record)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(record)
    return updated


def delete_class(
    classes: list[dict[str, Any]],
    rules: dict[str, Any],
    unknown_rule: dict[str, Any] | None,
    class_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Remove a class and any references to it."""
    kept = [dict(item) for item in classes if isinstance(item, dict) and item.get("id") != class_id]

    def scrub(rule: Any) -> dict[str, Any]:
        if not isinstance(rule, dict):
            return {"level": LEVEL_ALL, "classes": []}
        class_ids = [item for item in _as_str_set(rule.get("classes")) if item != class_id]
        return {"level": _level(rule.get("level"), LEVEL_ALL), "classes": sorted(class_ids)}

    new_rules = {name: scrub(rule) for name, rule in (rules or {}).items()}
    return kept, new_rules, scrub(unknown_rule)
