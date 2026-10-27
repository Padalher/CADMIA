<p align="center">
  <img src="docs/logo.png" alt="CADMIA" width="560">
</p>

<h3 align="center">Computational Agent-Driven Molecular Intelligence Architecture</h3>

<p align="center">

[English](README.md) · **Español**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23608477.svg)](https://doi.org/10.5281/zenodo.23608477)
[![CI](https://github.com/Padalher/CADMIA/actions/workflows/ci.yml/badge.svg)](https://github.com/Padalher/CADMIA/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](pyproject.toml)

</p>

Autor: Pablo Daniel Alvarez Hernandez, Universidad Juárez Autónoma de Tabasco (UJAT)

Las versiones siguen [Semantic Versioning](https://semver.org/lang/es/): antes de la 1.0.0, una versión
menor puede cambiar comandos o formatos de salida, y cada cambio así queda en el changelog. La 1.0.0
marcará una interfaz pública estable.

## ¿Qué es CADMIA?

CADMIA es una suite de línea de comandos para estudiar la reactividad y la adsorción en
nanocúmulos a partir de salidas de química cuántica (ORCA y Gaussian). Lee los cálculos, obtiene
los descriptores, prepara los cálculos siguientes y registra de dónde salió cada número. Un solo
comando, `cadmia`, da acceso a todas las herramientas.

Funciones principales:

- DFT conceptual: descriptores globales (I, A, η, μ, ω) por diferencias finitas o Koopmans, y
  funciones de Fukui condensadas y descriptor dual, coincidentes con Multiwfn.
- Adsorción: configuraciones iniciales de moléculas sonda pequeñas en los sitios más reactivos,
  pre-cribado GFN2-xTB y energética de adsorción (ΔE, ΔH, ΔG, deformación e interacción).
- Estabilidad: energías de mezcla y de enlace frente a los óxidos puros, series de tamaño,
  isómeros y poblaciones de Boltzmann.
- Estructura: números de coordinación (CN, ECN), distancias de enlace, anillos, poliedros de
  coordinación y simetría puntual.
- Estructura electrónica: DOS y PDOS, potencial electrostático sobre la superficie molecular,
  espectros IR y mapas volumétricos (densidad, Fukui, Δρ). El DOS, el PDOS, el IR y las gráficas
  estructurales salen como HTML interactivo o como figuras PNG, SVG y PDF de una o dos columnas
  (`--width`).
- Cálculos: preparación de entradas, cola local secuencial y envío a SLURM, con flujos
  reanudables.
- Procedencia: resultados guardados en una base de datos SQLite local con el hash del archivo, el
  comando exacto y la versión del programa.
- Ayuda bilingüe (`--help` en inglés, `--ayuda` en español) y un reporte de texto de columnas fijas
  con todos los módulos (`reports/cadmia.out`), con resumen Markdown y tablas LaTeX bajo pedido.

## Novedades de 0.3.1

Solo lo principal; el detalle completo está en
[CHANGELOG.md](https://github.com/Padalher/CADMIA/blob/main/CHANGELOG.md) (en inglés).

- **Un solo origen por módulo:** dentro de un proyecto de `cadmia init`, `structure report`, `cdft`,
  `analyze` y `stability` guardan sus datos y su reporte completo en `results/`; `cadmia report`
  los reúne en `reports/cadmia.out`, y `--report/-r ARCHIVO` solo reescribe el reporte de un
  módulo sin recalcular.
- **BSSE por counterpoise en ORCA:** `prepare bsse` escribe los cuatro puntos sencillos de
  Boys–Bernardi y `analyze --bsse` los lee; contrastado con `Counterpoise=2` de Gaussian en
  Al₂TiO₅ + CO₂ (1.49 frente a 1.52 kcal/mol).
- **Espín y carga a lo largo del flujo:** `geometry` escribe `charge=`/`mult=` (el O₂ es triplete),
  y `prescreen`, `rank`, `prepare`, `benchmark` y `pipeline --charge` los conservan.
- **Termoquímica:** ΔG con la corrección de σ y ΔG cuasi-RRHO en `analyze`; activación por
  flexión de sondas lineales (`--bend-threshold`, 10° por omisión).
- **Figuras estáticas:** `electronic dos|pdos|ir` y `structure plot` escriben PNG/SVG/PDF
  (`--format`, `--width single|double`). La DOS está en estados/eV con ambos espines.
- **Corridas más seguras:** no se mezclan resultados de cúmulos distintos en un reporte, una
  carpeta de pipeline ya usada exige `--resume` y cada salida se revisa con su propio motor.

## Lo que sigue

Planeado para la próxima versión (puede cambiar):

- **XYZ extendido:** `cdft`, `prescreen` y `rank` escribirán `.xyz` con la energía y su unidad, y
  los valores por átomo (cargas, Fukui, dual, ECN), legibles por visores como MOLE y por ASE.
- **Estilos de gráfica:** distintas formas para enlaces, ECN y DOS (barras, palitos, curvas).
- **Definiciones por módulo** en la guía; el `cadmia.out` remitirá a ellas.

## Requisitos

- Python 3.11 o posterior.
- Salidas de ORCA 6 o de Gaussian 16. ORCA solo hace falta para correr cálculos; el análisis
  funciona en cualquier equipo.
- Opcionales: `xtb` (pre-cribado), SLURM (envío de trabajos) y [MOLE](https://github.com/Padalher/MOLE)
  (`pip install mole-viewer`, visor rápido de conjuntos de estructuras).

## Inicio rápido

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Padalher/CADMIA/blob/main/examples/quickstart.ipynb)
corre los ejemplos en el navegador, sin instalar nada (cuaderno en inglés).

Desde una copia del repositorio, con las salidas de ejemplo que acompañan a las pruebas:

```bash
cadmia doctor                     # motores, dependencias y configuración
cadmia cdft --n tests/fixtures/1tial_cdft/N.out \
            --np1 tests/fixtures/1tial_cdft/N+1.out \
            --nm1 tests/fixtures/1tial_cdft/N-1.out     # descriptores + sitios reactivos ordenados
cadmia structure report tests/fixtures/1tial_orca/1tial_opt.xyz -o estructura.txt   # coordinación, enlaces, anillos
cadmia guide -es                  # la guía de uso completa
```

## Instalación, documentación y pruebas

La instalación y las pruebas son las mismas que en el [README en inglés](README.md#installation). La guía de uso en
español está en `docs/usage_guide.txt`, y cada comando documenta sus opciones con `--ayuda`.

## Cita y licencia

Si CADMIA se usa en un trabajo publicado, por favor cítalo como indica
[CITATION.cff](CITATION.cff). CADMIA es software libre, distribuido bajo la Licencia Pública General de GNU,
versión 3 o (a tu elección) cualquier versión posterior ([LICENSE](LICENSE)).

## Agradecimientos

CADMIA fue diseñado, dirigido y validado científicamente por Pablo Daniel Álvarez Hernández. Se
emplearon asistentes de código basados en IA para escribir y reestructurar partes del código, las
pruebas automatizadas y la documentación; cada cambio fue revisado y validado por el autor. Los
resultados numéricos provienen siempre de las salidas de química cuántica, nunca de un modelo de
lenguaje.

## Contacto

Errores y preguntas: [GitHub Issues](https://github.com/Padalher/CADMIA/issues).

Contacto directo: Pablo Daniel Alvarez Hernandez — pablo.alvarez@padalher.com — https://padalher.com
