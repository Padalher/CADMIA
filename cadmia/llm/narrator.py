"""Narrator — draft discussion text for a CADMIA report.

Takes the already-computed report (tables of CDFT descriptors, adsorption energetics,
stability, correlations) and asks the configured LLM for a discussion draft. The model
is instructed to use only the numbers present in the report; the output is clearly
marked as an AI draft for the user to revise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cadmia.llm.config import LLMConfig
from cadmia.llm.provider import LLMResponse, Provider
from cadmia.llm.registry import get_provider

_SYSTEM = {
    "es": (
        "Eres un químico computacional experto en catálisis heterogénea, DFT conceptual "
        "(funciones de Fukui, dureza, electrofilicidad) y energética de adsorción en "
        "nanocúmulos. Recibirás un reporte con tablas de resultados ya calculados. "
        "Redacta un borrador de DISCUSIÓN científica en español, en prosa académica, "
        "listo para revisar e integrar en un manuscrito científico.\n"
        "Reglas estrictas:\n"
        "- Usa únicamente los valores numéricos presentes en el reporte; no inventes ni "
        "extrapoles datos, referencias bibliográficas ni valores experimentales.\n"
        "- Si un dato falta, dilo explícitamente en lugar de suponerlo.\n"
        "- Conserva las unidades tal como aparecen (eV, kcal/mol, Å).\n"
        "- Estructura: reactividad intrínseca (CDFT), adsorción (incluye BSSE si está), "
        "estabilidad relativa si está presente, y cierre con la correlación sitio–energía."
    ),
    "en": (
        "You are a computational chemist expert in heterogeneous catalysis, conceptual DFT "
        "(Fukui functions, hardness, electrophilicity) and adsorption energetics on "
        "nanoclusters. You will receive a report with tables of already-computed results. "
        "Write a scientific DISCUSSION draft in English, in academic prose, ready to be "
        "revised and integrated into a scientific manuscript.\n"
        "Strict rules:\n"
        "- Use only the numeric values present in the report; do not invent or extrapolate "
        "data, bibliographic references or experimental values.\n"
        "- If a value is missing, say so explicitly instead of assuming it.\n"
        "- Keep units exactly as given (eV, kcal/mol, Å).\n"
        "- Structure: intrinsic reactivity (CDFT), adsorption (include BSSE if present), "
        "relative stability if present, closing with the site–energy correlation."
    ),
}

_HEADER = {
    "es": "## Discusión (borrador generado por IA — revisar antes de usar)",
    "en": "## Discussion (AI-generated draft — review before use)",
}


@dataclass
class NarrationResult:
    text:     str            # discussion body (no header)
    section:  str            # ready-to-append markdown section (header + body + footer)
    model:    str
    provider: str


def narrate_report(
    report: Any,                      # cadmia.agents.report_agent.Report (or its markdown str)
    *,
    language: str = "es",
    provider: Optional[Provider] = None,
    config: Optional[LLMConfig] = None,
) -> NarrationResult:
    """Generate a discussion draft for *report*; returns the text and an appendable section."""
    if language not in _SYSTEM:
        raise ValueError(f"Unsupported language '{language}' (available: es, en).")
    markdown = report if isinstance(report, str) else report.markdown
    provider = provider or get_provider(config)
    response: LLMResponse = provider.complete(markdown, system=_SYSTEM[language])
    footer = f"\n\n*({response.provider}: {response.model})*\n"
    section = f"\n{_HEADER[language]}\n\n{response.text.strip()}{footer}"
    return NarrationResult(
        text=response.text.strip(),
        section=section,
        model=response.model,
        provider=response.provider,
    )
