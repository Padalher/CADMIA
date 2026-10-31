"""Tests for cadmia.llm (config, registry, Anthropic provider, narrator) — no network."""
from __future__ import annotations

import io
import json

import pytest

from cadmia.llm import (
    LLMConfig, LLMError, LLMResponse, Provider, load_config, narrate_report,
)
from cadmia.llm.anthropic import AnthropicProvider
from cadmia.llm.registry import get_provider


# ─────────────────────────────────────────────
# config
# ─────────────────────────────────────────────

def test_load_config_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.provider == "anthropic"
    assert cfg.resolved_model() == "claude-sonnet-4-6"
    assert cfg.api_key_env() == "ANTHROPIC_API_KEY"

def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("provider: anthropic\nmodel: claude-haiku-4-5-20251001\nmax_tokens: 512\n")
    cfg = load_config(p)
    assert cfg.resolved_model() == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 512

def test_load_config_rejects_api_keys_in_file(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("provider: anthropic\napi_key: sk-secret\n")
    with pytest.raises(LLMError, match="environment variables"):
        load_config(p)

def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("provider: anthropic\ntemperatura: 0.7\n")
    with pytest.raises(LLMError, match="Unknown keys"):
        load_config(p)

def test_load_config_rejects_secret_keys(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("provider: anthropic\nauth_token: abc123\n")
    with pytest.raises(LLMError, match="environment variables"):
        load_config(p)

def test_load_config_allows_max_tokens(tmp_path):
    # "max_tokens" contains "token" but is a legitimate whitelisted key.
    p = tmp_path / "llm.yaml"
    p.write_text("provider: anthropic\nmax_tokens: 256\n")
    assert load_config(p).max_tokens == 256

def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LLMConfig().api_key()


# ─────────────────────────────────────────────
# registry
# ─────────────────────────────────────────────

def test_get_provider_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = get_provider(LLMConfig(provider="anthropic"))
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-4-6"

def test_get_provider_openai(monkeypatch):
    from cadmia.llm.openai_compat import OpenAICompatProvider
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = get_provider(LLMConfig(provider="openai"))
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.model == "gpt-5-mini"
    assert provider.base_url == "https://api.openai.com/v1"

def test_get_provider_openai_local_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider(LLMConfig(provider="openai", model="llama3",
                                      base_url="http://localhost:11434/v1"))
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "llama3"

def test_get_provider_openai_remote_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_provider(LLMConfig(provider="openai"))

def test_get_provider_gemini(monkeypatch):
    from cadmia.llm.gemini import GeminiProvider
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    provider = get_provider(LLMConfig(provider="gemini"))
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash"

def test_get_provider_unknown():
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        get_provider(LLMConfig(provider="mistral"))


# ─────────────────────────────────────────────
# Anthropic provider (mocked HTTP)
# ─────────────────────────────────────────────

class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

def test_anthropic_request_payload():
    prov = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test", max_tokens=99)
    req = prov.build_request("hola", system="sys")
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["max_tokens"] == 99
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "hola"}]
    assert req.get_header("X-api-key") == "sk-test"

def test_anthropic_complete_parses_response(monkeypatch):
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "Discusión: el sitio Ti domina."}],
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "stop_reason": "end_turn",
    }).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResponse(body))
    prov = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    resp = prov.complete("hola")
    assert resp.text == "Discusión: el sitio Ti domina."
    assert resp.provider == "anthropic"
    assert resp.output_tokens == 20

def test_openai_payload_and_response(monkeypatch):
    from cadmia.llm.openai_compat import OpenAICompatProvider
    prov = OpenAICompatProvider(model="gpt-5-mini", api_key="sk-test", max_tokens=77)
    payload = prov.build_payload("hola", system="sys")
    assert payload["messages"] == [{"role": "system", "content": "sys"},
                                   {"role": "user", "content": "hola"}]
    assert payload["max_tokens"] == 77

    body = json.dumps({
        "model": "gpt-5-mini",
        "choices": [{"message": {"content": "Texto."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResponse(body))
    resp = prov.complete("hola")
    assert resp.text == "Texto."
    assert resp.provider == "openai"
    assert resp.output_tokens == 3

def test_gemini_payload_and_response(monkeypatch):
    from cadmia.llm.gemini import GeminiProvider
    prov = GeminiProvider(model="gemini-2.5-flash", api_key="g-test", max_tokens=88)
    payload = prov.build_payload("hola", system="sys")
    assert payload["system_instruction"] == {"parts": [{"text": "sys"}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hola"}]}]
    assert payload["generationConfig"]["maxOutputTokens"] == 88

    body = json.dumps({
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{"content": {"parts": [{"text": "Texto "}, {"text": "Gemini."}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
    }).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResponse(body))
    resp = prov.complete("hola")
    assert resp.text == "Texto Gemini."
    assert resp.provider == "gemini"
    assert resp.output_tokens == 4

def test_gemini_empty_response_raises(monkeypatch):
    from cadmia.llm.gemini import GeminiProvider
    body = json.dumps({"candidates": [{"finishReason": "SAFETY"}]}).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResponse(body))
    prov = GeminiProvider(model="gemini-2.5-flash", api_key="g-test")
    with pytest.raises(LLMError, match="SAFETY"):
        prov.complete("hola")

def test_anthropic_empty_response_raises(monkeypatch):
    body = json.dumps({"content": [], "stop_reason": "max_tokens"}).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeHTTPResponse(body))
    prov = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
    with pytest.raises(LLMError, match="Empty response"):
        prov.complete("hola")


# ─────────────────────────────────────────────
# narrator
# ─────────────────────────────────────────────

class FakeProvider(Provider):
    name = "fake"
    def __init__(self):
        self.last_prompt = None
        self.last_system = None
    def complete(self, prompt, *, system=None, max_tokens=None):
        self.last_prompt = prompt
        self.last_system = system
        return LLMResponse(text="Borrador de discusión.", model="fake-1", provider="fake")

def test_narrate_report_uses_report_markdown():
    fake = FakeProvider()
    result = narrate_report("# Reporte\n\n| gap_eV | 4.0 |", language="es", provider=fake)
    assert "gap_eV" in fake.last_prompt
    assert "español" in fake.last_system
    assert result.text == "Borrador de discusión."
    assert "borrador generado por IA" in result.section
    assert "fake-1" in result.section

def test_narrate_report_english_system():
    fake = FakeProvider()
    result = narrate_report("# Report", language="en", provider=fake)
    assert "English" in fake.last_system
    assert "AI-generated draft" in result.section

def test_narrate_report_bad_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        narrate_report("# R", language="fr", provider=FakeProvider())


# ─────────────────────────────────────────────
# CLI: cadmia report --narrate
# ─────────────────────────────────────────────

def test_cli_report_narrate_appends_section(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main
    import cadmia.llm.narrator as narrator_mod

    monkeypatch.setattr(narrator_mod, "get_provider", lambda config=None: FakeProvider())

    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({
        "cluster": "Al2TiO5", "charge_scheme": "mulliken",
        "global_descriptors": {"gap_eV": 4.0},
        "sites": [],
    }))
    out = tmp_path / "rep.md"
    result = CliRunner().invoke(main, ["report", "--cdft", str(cdft),
                                       "-o", str(out), "--narrate"])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "borrador generado por IA" in text
    assert "Borrador de discusión." in text

# ─────────────────────────────────────────────
# NL → read-only SQL
# ─────────────────────────────────────────────

class FakeSQLProvider(Provider):
    name = "fake"
    def __init__(self, sql):
        self.sql = sql
        self.last_prompt = None
    def complete(self, prompt, *, system=None, max_tokens=None):
        self.last_prompt = prompt
        return LLMResponse(text=self.sql, model="fake-1", provider="fake")


def test_validate_readonly_sql_accepts_select_and_with():
    from cadmia.llm.sql import validate_readonly_sql
    assert validate_readonly_sql("SELECT 1;") == "SELECT 1"
    assert validate_readonly_sql("WITH t AS (SELECT 1 AS x) SELECT x FROM t").startswith("WITH")

def test_validate_readonly_sql_strips_fences():
    from cadmia.llm.sql import validate_readonly_sql
    assert validate_readonly_sql("```sql\nSELECT 42\n```") == "SELECT 42"

@pytest.mark.parametrize("bad", [
    "UPDATE stability SET e_rel_kcal = 0",
    "SELECT 1; DROP TABLE stability",
    "PRAGMA table_info(stability)",
    "DELETE FROM molecules",
    "",
    "SELECT 1; SELECT 2",
    # the table-valued pragma form evaded `\\bpragma\\b` (the `_` broke the boundary)
    "SELECT * FROM pragma_table_info('stability')",
    "SELECT load_extension('evil.so')",
])
def test_validate_readonly_sql_rejects(bad):
    from cadmia.llm.sql import validate_readonly_sql
    with pytest.raises(LLMError):
        validate_readonly_sql(bad)

@pytest.mark.parametrize("good", [
    "SELECT * FROM molecules WHERE name = 'a;b'",          # ';' inside a literal
    "SELECT replace(name, 'a', 'b') FROM molecules",       # replace() scalar function
    "SELECT name FROM stability -- has ; and DROP\nWHERE balanced = 1",  # comment noise
])
def test_validate_readonly_sql_accepts_legit_selects(good):
    from cadmia.llm.sql import validate_readonly_sql
    assert validate_readonly_sql(good).lower().startswith("select")


def _seed_db(tmp_path):
    from cadmia.db.session import init_db
    db = tmp_path / "ask.db"
    conn = init_db(db)
    conn.execute("INSERT INTO molecules (id, name, formula, n_atoms) "
                 "VALUES ('m1', 'Al2TiO5', 'Al2O5Ti', 8)")
    conn.execute("INSERT INTO stability (id, molecule_id, e_rel_kcal, balanced) "
                 "VALUES ('s1', 'm1', -146.169, 1)")
    conn.commit()
    conn.close()
    return db

def test_ask_database_executes_generated_select(tmp_path):
    from cadmia.llm.sql import ask_database
    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("SELECT name, e_rel_kcal FROM stability "
                           "JOIN molecules ON molecules.id = stability.molecule_id")
    result = ask_database("¿cuál es la estabilidad?", db, provider=fake)
    assert result.columns == ["name", "e_rel_kcal"]
    assert result.rows == [("Al2TiO5", -146.169)]
    assert "CREATE TABLE" in fake.last_prompt          # schema included in prompt
    assert "estabilidad" in fake.last_prompt

def test_ask_database_rejects_mutation(tmp_path):
    from cadmia.llm.sql import ask_database
    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("DELETE FROM stability")
    with pytest.raises(LLMError, match="SELECT"):
        ask_database("borra todo", db, provider=fake)

def test_ask_database_bad_sql_reports_error(tmp_path):
    from cadmia.llm.sql import ask_database
    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("SELECT nope FROM nothing")
    with pytest.raises(LLMError, match="failed to execute"):
        ask_database("x", db, provider=fake)

def test_ask_database_aborts_runaway_query(tmp_path):
    """A read-only SELECT can still be unbounded in cost; the deadline must stop it."""
    import time
    from cadmia.llm.sql import ask_database
    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("WITH RECURSIVE c(x) AS "
                           "(SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                           "SELECT count(*) FROM c")
    started = time.monotonic()
    with pytest.raises(LLMError, match="budget"):
        ask_database("cuenta sin fin", db, provider=fake, timeout_s=0.2)
    assert time.monotonic() - started < 10.0

def test_ask_database_timeout_does_not_disturb_normal_queries(tmp_path):
    from cadmia.llm.sql import ask_database
    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("SELECT name, e_rel_kcal FROM stability "
                           "JOIN molecules ON molecules.id = stability.molecule_id")
    result = ask_database("estabilidad", db, provider=fake, timeout_s=0.2)
    assert result.rows == [("Al2TiO5", -146.169)]

def test_cli_db_ask(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main
    import cadmia.llm.sql as sql_mod

    db = _seed_db(tmp_path)
    fake = FakeSQLProvider("SELECT name, e_rel_kcal FROM stability "
                           "JOIN molecules ON molecules.id = stability.molecule_id")
    monkeypatch.setattr(sql_mod, "get_provider", lambda config=None: fake)
    result = CliRunner().invoke(main, ["db", "ask", "estabilidad relativa",
                                       "--path", str(db)])
    assert result.exit_code == 0, result.output
    assert "SELECT name" in result.output
    assert "-146.169" in result.output

def test_cli_db_ask_fails_cleanly_without_key(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main
    import cadmia.llm.config as llm_config
    # Isolate from any real ~/.cadmia/llm.yaml so the default provider (anthropic) applies.
    monkeypatch.setattr(llm_config, "DEFAULT_CONFIG_PATH", tmp_path / "no_llm.yaml")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = _seed_db(tmp_path)
    result = CliRunner().invoke(main, ["db", "ask", "qué hay", "--path", str(db)])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_cli_report_narrate_fails_cleanly_without_key(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({
        "cluster": "x", "charge_scheme": "mulliken",
        "global_descriptors": {}, "sites": [],
    }))
    result = CliRunner().invoke(main, ["report", "--cdft", str(cdft),
                                       "-o", str(tmp_path / "r.md"), "--narrate"])
    assert result.exit_code != 0
    assert "--narrate failed" in result.output
