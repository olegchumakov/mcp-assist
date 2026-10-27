"""Options-flow pages for voice access control.

The Speaker ID add-on is optional. The people list is requested only when
access control is on and a URL is filled in. A 401 or any other failure
leaves the manually typed names in place.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    AreaSelector,
    AreaSelectorConfig,
    DeviceSelector,
    DeviceSelectorConfig,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

try:
    from homeassistant.helpers.selector import LabelSelector, LabelSelectorConfig
except ImportError:  # pragma: no cover - cores without the label selector
    LabelSelector = None  # type: ignore[misc, assignment]
    LabelSelectorConfig = None  # type: ignore[misc, assignment]

from .access_control import (
    LEVEL_ALL,
    LEVEL_NONE,
    LEVEL_SELECTED,
    delete_class,
    load_access_config,
    normalize_person_name,
    people_catalog,
    people_endpoint,
    people_from_response,
    person_key,
    rule_for_person,
    settings_updates,
    should_fetch_people,
    split_names,
    upsert_class,
)
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
    CONF_ACCESS_USER_LINES,
    CONF_ACCESS_USER_MAP,
)

_LOGGER = logging.getLogger(__name__)

_UI = {
    "en": {
        "unknown_voice": "Unknown voice",
        "new_group": "New group",
        "back": "Back",
        "groups_for": "which groups",
        "groups_for_unknown": "Which groups an unknown voice may use",
    },
    "ru": {
        "unknown_voice": "Незнакомый голос",
        "new_group": "Новая группа",
        "back": "Назад",
        "groups_for": "какие группы",
        "groups_for_unknown": "Какие группы можно незнакомому голосу",
    },
}


def ui_text(language: str | None, key: str) -> str:
    """Short labels that are built at runtime and so are not translation keys."""
    lang = "ru" if str(language or "").lower().startswith("ru") else "en"
    return _UI[lang][key]


def flow_language(hass) -> str:
    config = getattr(hass, "config", None)
    return str(getattr(config, "language", None) or "en")


def parse_user_lines(text: str | None) -> dict[str, str]:
    """Lines of ``user id = person name`` for installs without the user list."""
    mapping: dict[str, str] = {}
    for line in str(text or "").splitlines():
        if "=" not in line:
            continue
        user_id, name = line.split("=", 1)
        user_id = user_id.strip()
        person = normalize_person_name(name)
        if user_id and person:
            mapping[user_id] = person
    return mapping


def format_user_lines(user_map: dict[str, str]) -> str:
    return "\n".join(f"{user_id} = {name}" for user_id, name in user_map.items())


async def fetch_addon_people(hass, url: str, secret: str) -> list[str] | None:
    """GET /api/people. None on 401, timeout, or any other failure. Never logs the secret."""
    if not should_fetch_people(True, url):
        return None
    endpoint = people_endpoint(url)
    try:
        session = async_get_clientsession(hass)
        timeout = aiohttp.ClientTimeout(total=3)
        async with session.get(
            endpoint,
            headers={"X-Speaker-Key": secret},
            timeout=timeout,
        ) as response:
            if response.status != 200:
                _LOGGER.info(
                    "Speaker list %s returned %s; typed names stay in use",
                    endpoint,
                    response.status,
                )
                return None
            payload = await response.json(content_type=None)
        return people_from_response(200, payload)
    except Exception:
        _LOGGER.info("Speaker list %s is unreachable; typed names stay in use", endpoint)
        return None


def _level_selector() -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=[LEVEL_NONE, LEVEL_SELECTED, LEVEL_ALL],
            mode=SelectSelectorMode.DROPDOWN,
            translation_key="access_level",
        )
    )


def _class_selector(classes: list[dict[str, Any]], selected: list[str]) -> SelectSelector:
    options = [
        {"value": item["id"], "label": item.get("name") or item["id"]}
        for item in classes
        if item.get("id")
    ]
    return SelectSelector(
        SelectSelectorConfig(
            options=options,
            multiple=True,
            mode=SelectSelectorMode.DROPDOWN,
        )
    )


def _label_selector():
    if LabelSelector is None:
        return TextSelector(TextSelectorConfig())
    return LabelSelector(LabelSelectorConfig(multiple=True))


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return split_names(value) if LabelSelector is None else [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


class AccessOptionsMixin:
    """Options steps appended after memory. Settings are stored on the system entry."""

    def _shared(self) -> dict[str, Any]:
        from .config_flow import _shared_config_source

        return _shared_config_source(self.hass)

    async def async_step_access(self, user_input=None):
        """Switch, optional secret, optional add-on URL, reply, typed names."""
        errors: dict[str, str] = {}
        current = self._shared()
        if user_input is not None:
            updates = settings_updates(user_input)
            if should_fetch_people(updates[CONF_ACCESS_CONTROL_ENABLED], updates[CONF_ACCESS_ADDON_URL]):
                fetched = await fetch_addon_people(
                    self.hass,
                    updates[CONF_ACCESS_ADDON_URL],
                    updates[CONF_ACCESS_SHARED_SECRET],
                )
                if fetched is not None:
                    updates[CONF_ACCESS_CACHED_PEOPLE] = fetched
            self._update_shared_settings(updates)
            return await self.async_step_access_menu()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_ACCESS_CONTROL_ENABLED,
                    default=bool(current.get(CONF_ACCESS_CONTROL_ENABLED, False)),
                ): bool,
                vol.Optional(
                    CONF_ACCESS_SHARED_SECRET,
                    default=str(current.get(CONF_ACCESS_SHARED_SECRET) or ""),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Optional(
                    CONF_ACCESS_ADDON_URL,
                    default=str(current.get(CONF_ACCESS_ADDON_URL) or ""),
                ): str,
                vol.Optional(
                    CONF_ACCESS_MIN_CONFIDENCE,
                    default=float(current.get(CONF_ACCESS_MIN_CONFIDENCE) or 0),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0,
                        max=1,
                        step=0.01,
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_ACCESS_NONE_REPLY,
                    default=str(current.get(CONF_ACCESS_NONE_REPLY) or ""),
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_ACCESS_MANUAL_PEOPLE,
                    default="\n".join(load_access_config(current).manual_people),
                ): TextSelector(TextSelectorConfig(multiline=True)),
            }
        )
        return self.async_show_form(step_id="access", data_schema=schema, errors=errors)

    async def async_step_access_menu(self, user_input=None):
        """Groups, people, and app users. Done returns to the profile save."""
        return self.async_show_menu(
            step_id="access_menu",
            menu_options=[
                "access_class_pick",
                "access_rules",
                "access_users",
                "access_finish",
            ],
        )

    async def async_step_access_finish(self, user_input=None):
        return self._finish_profile_options()

    async def async_step_access_class_pick(self, user_input=None):
        language = flow_language(self.hass)
        classes = list(self._shared().get(CONF_ACCESS_CLASSES) or [])
        options = [{"value": "__new__", "label": ui_text(language, "new_group")}]
        for item in classes:
            if isinstance(item, dict) and item.get("id"):
                options.append({"value": item["id"], "label": item.get("name") or item["id"]})
        options.append({"value": "__back__", "label": ui_text(language, "back")})
        if user_input is not None:
            choice = user_input.get("access_class_choice") or "__back__"
            if choice == "__back__":
                return await self.async_step_access_menu()
            self._editing_class_id = None if choice == "__new__" else choice
            return await self.async_step_access_class_edit()
        schema = vol.Schema(
            {
                vol.Required("access_class_choice", default="__new__"): SelectSelector(
                    SelectSelectorConfig(options=options, mode=SelectSelectorMode.DROPDOWN)
                )
            }
        )
        return self.async_show_form(step_id="access_class_pick", data_schema=schema)

    def _class_by_id(self, class_id: str | None) -> dict[str, Any]:
        for item in self._shared().get(CONF_ACCESS_CLASSES) or []:
            if isinstance(item, dict) and item.get("id") == class_id:
                return item
        return {}

    async def async_step_access_class_edit(self, user_input=None):
        errors: dict[str, str] = {}
        current = self._class_by_id(getattr(self, "_editing_class_id", None))
        classes = [item for item in (self._shared().get(CONF_ACCESS_CLASSES) or []) if isinstance(item, dict)]
        if user_input is not None:
            if user_input.get("access_class_delete") and current.get("id"):
                kept, rules, unknown = delete_class(
                    classes,
                    self._shared().get(CONF_ACCESS_RULES) or {},
                    self._shared().get(CONF_ACCESS_UNKNOWN_RULE),
                    current["id"],
                )
                self._update_shared_settings(
                    {
                        CONF_ACCESS_CLASSES: kept,
                        CONF_ACCESS_RULES: rules,
                        CONF_ACCESS_UNKNOWN_RULE: unknown,
                    }
                )
                return await self.async_step_access_class_pick()
            name = " ".join(str(user_input.get("access_class_name") or "").split()).strip()
            if not name:
                errors["access_class_name"] = "access_class_name"
            else:
                taken = any(
                    item.get("id") != current.get("id")
                    and str(item.get("name") or "").casefold() == name.casefold()
                    for item in classes
                )
                if taken:
                    errors["access_class_name"] = "access_class_name"
            if not errors:
                updated = upsert_class(
                    classes,
                    current.get("id"),
                    name,
                    _as_id_list(user_input.get("access_class_entities")),
                    _as_id_list(user_input.get("access_class_devices")),
                    _as_id_list(user_input.get("access_class_areas")),
                    _as_id_list(user_input.get("access_class_labels")),
                )
                self._update_shared_settings({CONF_ACCESS_CLASSES: updated})
                return await self.async_step_access_class_pick()

        schema_dict: dict[Any, Any] = {
            vol.Required("access_class_name", default=current.get("name", "")): str,
            vol.Optional(
                "access_class_entities",
                default=list(current.get("entity_ids") or []),
            ): EntitySelector(EntitySelectorConfig(multiple=True)),
            vol.Optional(
                "access_class_devices",
                default=list(current.get("device_ids") or []),
            ): DeviceSelector(DeviceSelectorConfig(multiple=True)),
            vol.Optional(
                "access_class_areas",
                default=list(current.get("area_ids") or []),
            ): AreaSelector(AreaSelectorConfig(multiple=True)),
            vol.Optional(
                "access_class_labels",
                default=list(current.get("label_ids") or []),
            ): _label_selector(),
        }
        if current.get("id"):
            schema_dict[vol.Optional("access_class_delete", default=False)] = bool
        return self.async_show_form(
            step_id="access_class_edit",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
        )

    def _rule_rows(self, config) -> list[dict[str, str]]:
        language = flow_language(self.hass)
        unknown = ui_text(language, "unknown_voice")
        groups_for = ui_text(language, "groups_for")
        groups_unknown = ui_text(language, "groups_for_unknown")
        used: set[str] = set()

        def unique(label: str) -> str:
            candidate = label
            number = 2
            while candidate in used:
                candidate = f"{label} ({number})"
                number += 1
            used.add(candidate)
            return candidate

        rows = [
            {
                "person": "",
                "level_key": unique(unknown),
                "class_key": unique(groups_unknown),
            }
        ]
        for name in people_catalog(config):
            rows.append(
                {
                    "person": name,
                    "level_key": unique(name),
                    "class_key": unique(f"{name} — {groups_for}"),
                }
            )
        return rows

    async def async_step_access_rules(self, user_input=None):
        """One row per person, plus the unknown voice, plus a box for new names."""
        errors: dict[str, str] = {}
        source = self._shared()
        config = load_access_config(source)
        rows = getattr(self, "_rule_rows_cache", None)
        if user_input is None or not rows:
            rows = self._rule_rows(config)
            self._rule_rows_cache = rows

        classes = [item for item in (source.get(CONF_ACCESS_CLASSES) or []) if isinstance(item, dict)]
        if user_input is not None:
            rules: dict[str, dict[str, Any]] = {}
            unknown_rule = {"level": LEVEL_ALL, "classes": []}
            for row in rows:
                level = str(user_input.get(row["level_key"]) or LEVEL_ALL)
                chosen = [item for item in _as_id_list(user_input.get(row["class_key"])) if item]
                if level == LEVEL_SELECTED and not chosen:
                    errors[row["class_key"] if classes else row["level_key"]] = "access_classes_required"
                record = {"level": level, "classes": chosen}
                if row["person"]:
                    rules[row["person"]] = record
                else:
                    unknown_rule = record
            added = split_names(user_input.get(CONF_ACCESS_MANUAL_PEOPLE))
            manual = list(config.manual_people)
            seen = {person_key(name) for name in manual}
            for name in added:
                key = person_key(name)
                if key and key not in seen:
                    manual.append(name)
                    seen.add(key)
            if not errors:
                self._update_shared_settings(
                    {
                        CONF_ACCESS_RULES: rules,
                        CONF_ACCESS_UNKNOWN_RULE: unknown_rule,
                        CONF_ACCESS_MANUAL_PEOPLE: manual,
                    }
                )
                if added:
                    self._rule_rows_cache = None
                    return await self.async_step_access_rules()
                self._rule_rows_cache = None
                return await self.async_step_access_menu()

        schema: dict[Any, Any] = {}
        for row in rows:
            rule = config.unknown_rule if not row["person"] else rule_for_person(config.rules, row["person"])
            level = rule.level if rule is not None else LEVEL_ALL
            chosen = sorted(rule.class_ids) if rule is not None else []
            schema[vol.Required(row["level_key"], default=level)] = _level_selector()
            if classes:
                schema[vol.Optional(row["class_key"], default=chosen)] = _class_selector(classes, chosen)
        schema[
            vol.Optional(CONF_ACCESS_MANUAL_PEOPLE, default="")
        ] = TextSelector(TextSelectorConfig(multiline=True))
        return self.async_show_form(
            step_id="access_rules",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async def _visible_users(self) -> list[Any]:
        auth = getattr(self.hass, "auth", None)
        getter = getattr(auth, "async_get_users", None)
        if getter is None:
            return []
        try:
            users = await getter()
        except Exception:
            _LOGGER.debug("Could not list Home Assistant users", exc_info=True)
            return []
        active = [user for user in users or [] if getattr(user, "is_active", True)]
        visible = [user for user in active if not getattr(user, "system_generated", False)]
        return visible or active

    async def async_step_access_users(self, user_input=None):
        """Map each Home Assistant user to a person. Unmapped users are unknown."""
        config = load_access_config(self._shared())
        users = await self._visible_users()
        language = flow_language(self.hass)
        people = people_catalog(config)
        person_options = [
            {"value": "__unknown__", "label": ui_text(language, "unknown_voice")},
            *[{"value": name, "label": name} for name in people],
        ]
        person_selector = SelectSelector(
            SelectSelectorConfig(options=person_options, mode=SelectSelectorMode.DROPDOWN)
        )

        if users:
            fields = getattr(self, "_user_fields", None)
            if user_input is None or not fields:
                used: set[str] = set()
                fields = {}
                for user in users:
                    label = getattr(user, "name", None) or getattr(user, "id", "")
                    candidate = str(label)
                    number = 2
                    while candidate in used:
                        candidate = f"{label} ({number})"
                        number += 1
                    used.add(candidate)
                    fields[candidate] = user.id
                self._user_fields = fields
            if user_input is not None:
                mapping = {}
                for label, user_id in fields.items():
                    chosen = user_input.get(label) or "__unknown__"
                    if chosen and chosen != "__unknown__":
                        mapping[user_id] = chosen
                self._update_shared_settings({CONF_ACCESS_USER_MAP: mapping})
                self._user_fields = None
                return await self.async_step_access_menu()
            schema: dict[Any, Any] = {}
            for label, user_id in fields.items():
                mapped = config.user_map.get(user_id) or ""
                current = "__unknown__"
                for item in person_options:
                    if item["value"] != "__unknown__" and person_key(item["value"]) == person_key(mapped):
                        current = item["value"]
                        break
                schema[vol.Required(label, default=current)] = person_selector
            return self.async_show_form(step_id="access_users", data_schema=vol.Schema(schema))

        if user_input is not None:
            self._update_shared_settings(
                {CONF_ACCESS_USER_MAP: parse_user_lines(user_input.get(CONF_ACCESS_USER_LINES))}
            )
            return await self.async_step_access_menu()
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_ACCESS_USER_LINES,
                    default=format_user_lines(config.user_map),
                ): TextSelector(TextSelectorConfig(multiline=True))
            }
        )
        return self.async_show_form(step_id="access_users", data_schema=schema)
