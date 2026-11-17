"""The planner's provider switch and the Groq planner adapter. No client is ever built for
real: init_chat_model is replaced, so nothing touches the network or needs a key."""

from typing import ClassVar

import pytest
from pydantic import ValidationError

from research_assistant.llm import providers
from research_assistant.llm.providers import (
    DEMO_SUBTASKS,
    _GroqPlanner,
    get_planner_llm,
    planner_provider_name,
)
from research_assistant.schemas.plan import Plan, PlanDraft


class FakeChatModel:
    """Stands in for a Groq chat model: records how structured output was requested."""

    draft: ClassVar[PlanDraft] = PlanDraft(subtasks=["What is X?", "What is Y?"])
    structured_calls: ClassVar[list] = []

    def with_structured_output(self, schema, **kwargs):
        FakeChatModel.structured_calls.append((schema, kwargs))
        return self

    def invoke(self, messages):
        return FakeChatModel.draft


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeChatModel.draft = PlanDraft(subtasks=["What is X?", "What is Y?"])
    FakeChatModel.structured_calls = []
    monkeypatch.delenv("PLANNER_PROVIDER", raising=False)
    created = []

    def fake_init_chat_model(**kwargs):
        created.append(kwargs)
        return FakeChatModel()

    monkeypatch.setattr(providers, "init_chat_model", fake_init_chat_model)
    return created


def test_the_planner_uses_groq_by_default(fakes):
    planner = get_planner_llm()
    assert isinstance(planner, _GroqPlanner)
    assert fakes[0]["model_provider"] == "groq"
    assert fakes[0]["model"] == providers.GROQ_PLANNER_MODEL
    assert fakes[0]["timeout"] == providers.PLANNER_TIMEOUT_SECONDS
    assert fakes[0]["max_retries"] == providers.PLANNER_MAX_RETRIES


def test_gemini_is_one_setting_away(fakes, monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", " Gemini ")
    planner = get_planner_llm()
    assert isinstance(planner, FakeChatModel)  # the plain chat model, no adapter
    assert fakes[0]["model_provider"] == "google-genai"
    assert fakes[0]["model"] == providers.PLANNER_MODEL


def test_an_unknown_provider_is_refused_with_the_valid_choices(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "openai")
    with pytest.raises(ValueError, match="'groq', 'gemini' or 'fixed'"):
        get_planner_llm()


def test_the_groq_planner_asks_for_descriptions_only_in_json_schema_mode():
    _GroqPlanner(FakeChatModel())
    assert FakeChatModel.structured_calls == [(PlanDraft, {"method": "json_schema"})]


def test_the_groq_planner_returns_a_plan_with_ids_made_by_the_code():
    planner = _GroqPlanner(FakeChatModel()).with_structured_output(Plan)
    plan = planner.invoke(["a message"])
    assert isinstance(plan, Plan)
    assert [s.description for s in plan.subtasks] == ["What is X?", "What is Y?"]
    assert len({s.id for s in plan.subtasks}) == 2  # distinct generated ids


def test_blank_descriptions_are_dropped_and_the_rest_trimmed():
    FakeChatModel.draft = PlanDraft(subtasks=["  What is X?  ", "", "   "])
    plan = _GroqPlanner(FakeChatModel()).invoke([])
    assert [s.description for s in plan.subtasks] == ["What is X?"]


def test_a_plan_with_no_subtasks_is_an_error_not_an_empty_plan():
    FakeChatModel.draft = PlanDraft(subtasks=[" "])
    with pytest.raises(ValidationError):
        _GroqPlanner(FakeChatModel()).invoke([])


def test_the_groq_planner_only_produces_a_plan():
    with pytest.raises(ValueError, match="only produces a Plan"):
        _GroqPlanner(FakeChatModel()).with_structured_output(PlanDraft)


def test_the_writer_gets_an_explicit_output_budget_and_low_reasoning_effort(fakes):
    providers.get_writer_llm()
    assert fakes[0]["model_provider"] == "groq"
    assert fakes[0]["max_tokens"] == providers.WRITER_MAX_TOKENS
    assert fakes[0]["reasoning_effort"] == providers.WRITER_REASONING_EFFORT
    assert fakes[0]["timeout"] == providers.WRITER_TIMEOUT_SECONDS


# --- the labelled demo mode ------------------------------------------------------


def test_the_fixed_planner_returns_the_demo_plan_and_builds_no_llm_client(
    fakes, monkeypatch
):
    monkeypatch.setenv("PLANNER_PROVIDER", "fixed")
    plan = get_planner_llm().with_structured_output(Plan).invoke(["ignored"])
    assert [s.description for s in plan.subtasks] == list(DEMO_SUBTASKS)
    assert fakes == []  # init_chat_model was never called: no key, no network


def test_every_plan_from_the_fixed_planner_has_new_ids(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "fixed")
    planner = get_planner_llm()
    first = planner.invoke([])
    second = planner.invoke([])
    assert {s.id for s in first.subtasks}.isdisjoint({s.id for s in second.subtasks})


def test_the_demo_plan_can_be_replaced_with_demo_plan(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "fixed")
    monkeypatch.setenv("DEMO_PLAN", " What is X? | What is Y?||  ")
    plan = get_planner_llm().invoke([])
    assert [s.description for s in plan.subtasks] == ["What is X?", "What is Y?"]


def test_the_fixed_planner_only_produces_a_plan(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "fixed")
    with pytest.raises(ValueError, match="only produces a Plan"):
        get_planner_llm().with_structured_output(PlanDraft)


def test_the_provider_name_is_normalised(monkeypatch):
    assert planner_provider_name() == "groq"
    monkeypatch.setenv("PLANNER_PROVIDER", " Fixed ")
    assert planner_provider_name() == "fixed"


def test_the_demo_plan_names_the_subject_in_every_subtask():
    # The rule of the planner prompt, applied to our own fixed plan.
    assert all("guitar" in subtask for subtask in DEMO_SUBTASKS)
