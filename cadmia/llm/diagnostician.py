"""Diagnostician — LLM fix suggestions for a Diagnosis.

The Diagnosis (symptoms, traces, gap, spin) is extracted deterministically by
cadmia.agents.diagnosis_agent; this module only asks the configured LLM for
prioritized convergence strategies. Suggestions are advisory: nothing is
executed, and the user validates every keyword before resubmitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cadmia.llm.config import LLMConfig
from cadmia.llm.provider import Provider
from cadmia.llm.registry import get_provider

_SYSTEM = {
    "es": (
        "Eres un experto en convergencia SCF de cálculos DFT con ORCA 6 y Gaussian 16, "
        "especializado en óxidos de metales de transición y nanocúmulos (estados casi "
        "degenerados, colapso variacional, contaminación de espín). Recibirás un diagnóstico "
        "JSON con síntomas extraídos de un archivo de salida.\n"
        "Responde en español con una lista priorizada (1., 2., …) de estrategias concretas "
        "para lograr la convergencia, con los keywords exactos del motor detectado "
        "(ORCA: SlowConv/VerySlowConv, MOREAD con %moinp desde un .gbw convergido, "
        "%scf con shift/damp, SOSCF/DIIS, guess alternativo, TRAH; "
        "Gaussian: scf=(xqc,maxcycle=N), guess=read, stable=opt, vshift).\n"
        "Reglas estrictas:\n"
        "- Básate solo en los síntomas del diagnóstico; si falta información, dilo.\n"
        "- Cambia lo mínimo: conserva funcional, base y dispersión del cálculo original.\n"
        "- Para cada estrategia explica en una frase por qué ataca el síntoma observado.\n"
        "- Son sugerencias a verificar antes de aplicarlas; no garantices resultados."
    ),
    "en": (
        "You are an expert in SCF convergence of DFT calculations with ORCA 6 and "
        "Gaussian 16, specialized in transition-metal oxides and nanoclusters "
        "(near-degenerate states, variational collapse, spin contamination). You will "
        "receive a JSON diagnosis with symptoms extracted from an output file.\n"
        "Reply in English with a prioritized list (1., 2., …) of concrete strategies to "
        "reach convergence, using the exact keywords of the detected engine "
        "(ORCA: SlowConv/VerySlowConv, MOREAD with %moinp from a converged .gbw, "
        "%scf shift/damp, SOSCF/DIIS, alternative guess, TRAH; "
        "Gaussian: scf=(xqc,maxcycle=N), guess=read, stable=opt, vshift).\n"
        "Strict rules:\n"
        "- Use only the symptoms in the diagnosis; if information is missing, say so.\n"
        "- Change as little as possible: keep the original functional, basis and dispersion.\n"
        "- For each strategy explain in one sentence why it addresses the observed symptom.\n"
        "- These are suggestions for the user to verify; do not guarantee results."
    ),
}


@dataclass
class SuggestionResult:
    text:     str
    model:    str
    provider: str


def suggest_fixes(
    diagnosis: Any,                   # cadmia.agents.diagnosis_agent.Diagnosis (or its JSON str)
    *,
    language: str = "es",
    provider: Optional[Provider] = None,
    config: Optional[LLMConfig] = None,
) -> SuggestionResult:
    """Ask the configured LLM for prioritized convergence strategies for *diagnosis*."""
    if language not in _SYSTEM:
        raise ValueError(f"Unsupported language '{language}' (available: es, en).")
    payload = diagnosis if isinstance(diagnosis, str) else diagnosis.to_json()
    provider = provider or get_provider(config)
    response = provider.complete(payload, system=_SYSTEM[language])
    return SuggestionResult(
        text=response.text.strip(),
        model=response.model,
        provider=response.provider,
    )
