"""Spanish-language help for the CADMIA CLI (--ayuda).

This module provides custom Click Command/Group subclasses that add a
``--ayuda`` flag to every command. The flag prints a
hand-written Spanish help page and exits, leaving the standard English
``--help`` untouched. CADMIA code is English-only; this is the single,
explicit exception for end-user help, mirroring the bilingual README.
"""
from __future__ import annotations

import click

# ─────────────────────────────────────────────
# Spanish help text, keyed by full command path
# ─────────────────────────────────────────────

SPANISH_HELP: dict[str, str] = {
    "init": """
cadmia init — Crea el andamiaje de un proyecto CADMIA.

Genera las carpetas del pipeline, un manifiesto y una base de datos local. Dentro del
proyecto, cada etapa (structure report, cdft, analyze, stability) guarda sola su resultado
en results/ — datos y reporte completo en un JSON — y `cadmia report` arma con ellos
reports/cadmia.out.

Argumento:
  DIRECTORIO            Carpeta del proyecto (por defecto: el directorio actual).

Opciones:
  --name NOMBRE         Nombre del proyecto (por defecto: nombre del directorio).
  --no-db               No crear la base de datos local .cadmia/cadmia.db.
  --overwrite           Sobrescribir un manifiesto/base de datos existentes.

Ejemplo:
  cadmia init mi_proyecto --name TiAl
""",

    "import": """
cadmia import — Importa salidas SCF existentes a la base de datos.

Por cada archivo detecta el motor, parsea geometría y energía, y registra
estructura + cálculo + procedencia (idempotente por hash de la geometría).
Reutiliza los parsers de cadmia/io; un archivo que falla no aborta el lote.

Argumento:
  ARCHIVOS              Salidas .out/.log (uno o varios).

Opciones:
  --engine MOTOR        gaussian | orca (autodetectado por archivo si se omite).
  --functional TEXTO    Registra este funcional en cada cálculo.
  --basis TEXTO         Registra esta base en cada cálculo.
  --dispersion TEXTO    Registra esta corrección de dispersión.
  --task TEXTO          Etiqueta de tarea (por defecto: scf).
  --db-path RUTA        Base de datos (por defecto: la base del propio proyecto, .cadmia/cadmia.db, si el directorio de trabajo está dentro de uno; si no, ~/.cadmia/cadmia.db).
  --no-table            Emite JSON en vez de una tabla.

Ejemplo:
  cadmia import salidas/*.out --functional PBE0 --basis def2-TZVP
""",

    "xyz": """
cadmia xyz — Extrae una geometría .xyz limpia desde una salida Gaussian/ORCA.

Toma la geometría de una salida y escribe un .xyz conservando el ORDEN ATÓMICO
del cálculo (justo lo que necesita el flujo CDFT → geometry). Evita el truco de
invocar Python a mano. El motor se detecta por el CONTENIDO del archivo, no por la
extensión: da igual que tu Gaussian y tu ORCA sean ambos .out.

Extrae la geometría CONVERGIDA (el último bloque cartesiano de la salida) y anota la
ENERGÍA final (Hartree) en la línea de comentario, como hace el .xyz nativo de ORCA;
también se muestra en pantalla al extraer. Un --comment explícito la sobrescribe.

Argumento:
  ARCHIVO               Salida Gaussian/ORCA (cualquier extensión; típicamente .out o .log).

Opciones:
  --engine MOTOR        gaussian | orca — fuerza el motor si la autodetección fallara.
  --output, -o ARCHIVO  .xyz de salida (por defecto: <archivo>.xyz junto al origen).
  --comment TEXTO       Línea de comentario del .xyz (línea 2); sustituye a la energía.

Ejemplo:
  cadmia xyz N.out -o cluster.xyz
  # luego: cadmia geometry --cdft cdft.json --xyz cluster.xyz --probe CO2 ...
  # (o directamente: cadmia geometry ... --xyz N.out)
""",

    "rank": """
cadmia rank — Ordena muchas salidas Gaussian/ORCA por energía → un solo .xyz.

Universal y agnóstico del motor: lee la energía electrónica final y la geometría de
cada salida, las ordena de MENOR a MAYOR energía y concatena las geometrías en un
único .xyz con la diferencia de energía (kcal/mol) en el comentario de cada bloque.
NO optimiza ni corre xTB: solo discrimina por energía resultados que YA calculaste
(p. ej. un cribado barato PBE/def2-SVP antes del alto nivel). De ese archivo ordenado
se construyen los inputs de los primeros (los de menor energía) como siempre.

Columna "Neg Freq" (frecuencias imaginarias), después de la energía absoluta:
  0    la salida trae frecuencias y NO hay imaginarias → mínimo verdadero.
  N>0  hay N frecuencias imaginarias → punto silla: descartar o reoptimizar.
  —    la salida NO trae frecuencias (p. ej. un Opt a secas). Ojo: "sin Freq" NO es
       lo mismo que "0 imaginarias"; para verificar el mínimo hay que calcular Freq
       AL MISMO nivel de teoría con que se optimizó (`cadmia prepare freq`).
El dato va en la tabla y en el sidecar JSON (n_imaginary), no en el comentario del .xyz.

Argumentos:
  SALIDAS...            Archivos de salida a ordenar (.out/.log; motor autodetectado).
                        Un argumento .xyz se interpreta como el nombre de salida.

Opciones:
  --output, -o ARCHIVO  .xyz único a escribir (por defecto: ranked_by_energy.xyz en el
                        directorio actual; si es un directorio, escribe
                        ranked_by_energy.xyz dentro).
  --window KCAL         Conserva en el .xyz solo las geometrías dentro de esta ΔE
                        (kcal/mol) de la de menor energía (por defecto 25; 0 = sin ventana).
                        El default es generoso a propósito: descarta outliers del cribado SVP
                        sin cortar confórmeros que podrían reordenarse en producción.
  --top N               Conserva a lo más N mínimos DISTINTOS (los de menor energía; por
                        defecto 15; 0 = sin tope). Se combina con --window (se escribe la
                        que cumpla AMBOS).
  --dedup-rmsd ANG      Agrupa las salidas que relajaron al MISMO mínimo (RMSD de Kabsch en
                        Å; por defecto 0.20; 0 desactiva). La superficie de energía embudona
                        configuraciones de partida distintas a un solo mínimo: sin esto,
                        --top cuenta archivos en vez de estructuras. Del grupo se escribe el
                        de menor energía; los duplicados quedan en el JSON con el
                        representante al que colapsaron.
  --cluster ARCHIVO     .xyz del cúmulo desnudo, optimizado AL MISMO NIVEL que estas salidas.
                        Agrupa los mínimos por el esqueleto al que relajó su cúmulo y da a cada
                        grupo su propio cero, su propia ventana y su propio tope, de modo que un
                        isómero del cúmulo más estable no se lleve el cero y descarte a los
                        aductos que conservaron el esqueleto de la referencia. Sin él, la
                        ventana es global (comportamiento de siempre).
                        OJO: para salidas DFT la referencia es el cúmulo optimizado con DFT, no
                        el relajado con xTB. Es el error fácil de cometer, y al revés que en
                        `cadmia prescreen`.

Por defecto rank ya recorta (window 25 / top 15) y el .xyz sale listo para `cadmia prepare`;
`--window 0 --top 0` escribe todas. El corte fino va al final, sobre el ΔG de producción, no
aquí.
Junto al .xyz se escribe un sidecar JSON con el mismo nombre (ranked.xyz → ranked.json):
lleva el ranking COMPLETO (con ΔE, energía absoluta, n_imaginary y si cada salida pasó el
recorte), aunque el .xyz solo tenga los ganadores.

El .xyz lleva solo lo indispensable en el comentario de cada bloque: ΔE (kcal/mol),
energía absoluta (Eh) y la procedencia. El conteo de frecuencias imaginarias es una
decisión de cribado: va en la tabla (y en el JSON), no en la geometría.

  --db / --no-db     Por defecto SÍ persiste. Persiste el destino de cada salida: energía final, a qué mínimo
                     distinto pertenece, su clase de esqueleto y si pasó la ventana.
  --db-path ARCHIVO  Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).

Ejemplos:
  cadmia rank *.out ranked.xyz
  cadmia rank *.out -o rank/co2.xyz
  # luego: cadmia prepare rank/co2.xyz ...  (inputs de los mejores para alto nivel)
""",

    "correlate": """
cadmia correlate — Correlaciona descriptores calculados con un observable.

Paso terminal de un estudio: ajusta una recta por mínimos cuadrados descriptor → observable
sobre los sistemas del estudio y reporta pendiente, ordenada, r, r² y un p-valor de dos colas.
En modo escaneo (--x all) prueba TODOS los descriptores disponibles contra un observable y los
ordena por |r|. NO calcula nada nuevo: consume los JSON que ya emiten `cadmia cdft` y
`cadmia analyze`, fusionados por identificador de sistema.

El observable puede ser experimental (columna de un CSV con -e) u OTRO descriptor calculado leído
del propio JSON (--y dG_ads, sin CSV). Con --cdft, un reporte CDFT de cúmulo se propaga a TODAS sus
configuraciones, de modo que los descriptores átomo-resueltos (dual@site, f_plus@site) se resuelven
en el sitio de enlace de cada aducto.

Opciones:
  --experimental, -e CSV   Tabla con una columna 'system' + columnas de observables. OPCIONAL:
                           omítela y usa --y <descriptor> para leer el observable del JSON.
  --descriptors, -d RUTA   JSON de descriptores: archivo, directorio o glob. Repetible.
  --cdft ARCHIVO           Reporte CDFT del cúmulo (--report-format json) que se propaga a cada
                           config → habilita dual@site / f_plus@site en el sitio de cada aducto.
  --x ALIAS|all            Descriptor: mu, eta, omega, omega_minus, omega_plus, N, ip, ea, gap,
                           e_ads, dG_ads, dH_ads, delta_n, e_stab, delta_gap, dq_probe, dq_site,
                           dr_site, o átomo-resuelto f_plus@site / dual@N. Si el reporte CDFT se
                           construyó desde los .engrad, también phi_norm@site, phi_along@site y
                           phi_dual_norm@site. 'all' escanea todos (por defecto).
  --y COL|ALIAS            Observable: columna del CSV con -e, o un alias de descriptor sin él.
  --abs-x                  Correlaciona |x| (p. ej. |ΔN|, como en la escala de Mayr).
  --log-x / --log-y        Correlaciona ln(x) / ln(y) (requiere valores > 0).
  --plot                   Escribe gráficas dispersión+recta (matplotlib; si falta, CSV de puntos).
  --format {txt,json,csv}  Formato de salida (por defecto txt).
  --output, -o ARCHIVO     Archivo de salida; las figuras van en ./figures junto a él.

Ejemplos:
  cadmia correlate -e mayr.csv -d study/ --y mayr_N --x omega_minus --plot
  cadmia correlate -d aductos/ --cdft cumulo_cdft.json --x dual@site --y dG_ads --plot
  cadmia correlate -d aductos/ --cdft cumulo_cdft.json --x phi_along@site --y dr_site
""",

    "bonds": """
cadmia bonds — Órdenes de enlace de Mayer desde una salida ORCA.

Requiere que la salida ORCA se haya corrido con `%output Print[P_Mayer] 1 end`
(Gaussian no imprime este análisis).

Argumento:
  ARCHIVO               Salida ORCA con el bloque de Mayer.

Opciones:
  --top N               Muestra solo los N enlaces más fuertes en pantalla (0 = todos).
  --output, -o ARCHIVO  Escribe el resultado a un archivo en vez de a la pantalla. La
                        extensión decide el formato: .json escribe JSON (el análisis
                        completo, sin recortar por --top); cualquier otra, la tabla.

Ejemplo:
  cadmia bonds 1tial_N_orca.out --top 10
""",

    "density": """
cadmia density — Genera cubos de densidad/HOMO/LUMO/MEP y mapas Fukui/dual/Δρ.

La generación de cubos usa orca_plot de ORCA (correr donde esté instalado ORCA); los mapas de
diferencia son aritmética de grids (sin motor). La rejilla se dimensiona por eje para un --step
(paso) uniforme, así los lóbulos salen suaves y no facetados; un mismo --step mantiene los cubos
de un juego Fukui/Δρ alineados voxel a voxel (necesario para restar).

El modo se deduce de las banderas que pasas (sin bandera de modo):
  • Cubos por especie:  --gbw X.gbw --kind density|homo|lumo|mep|all
  • Fukui/dual:         --gbw N.gbw --np1 anion.gbw --nm1 cation.gbw  (escribe f+, f-, f0, dual)
  • Deformación Δρ:     --adduct ad.gbw --cluster cl.gbw --probe pr.gbw

Opciones:
  --gbw ARCHIVO         .gbw de la especie (modo por especie / N en modo Fukui).
  --np1 ARCHIVO         .gbw del anión (N+1); con --nm1 deriva Fukui/dual.
  --nm1 ARCHIVO         .gbw del catión (N-1); con --np1 deriva Fukui/dual.
  --adduct/--cluster/--probe ARCHIVO   .gbw para Δρ (los tres → modo deformación). El cúmulo y la
                        sonda deben calcularse a la geometría del aducto y CON ÁTOMOS FANTASMA (los
                        del otro fragmento), para que los tres .gbw compartan la misma caja de
                        rejilla; si no, los cubos no quedan alineados y la resta se rechaza.
  --kind density|homo|lumo|mep|all     Cubo(s) a generar para --gbw (repetible).
  --homo-index N        Número de orbital HOMO (base 0); requerido para homo/lumo (LUMO = HOMO+1).
  --step BOHR           Paso de rejilla en bohr (uniforme por eje; por defecto 0.20). Un mismo
                        valor mantiene los cubos N/N±1 alineados.
  --orca-plot-path RUTA Ejecutable orca_plot (viene con ORCA).
  --output, -o DIR      Carpeta de salida de los cubos (por defecto: ./cubes).

Ejemplos:
  cadmia density --gbw N.gbw --np1 anion.gbw --nm1 cation.gbw -o cubes/
  cadmia density --gbw N.gbw --kind all --homo-index 41 -o cubes/
  cadmia density --adduct ad.gbw --cluster cl.gbw --probe co2.gbw -o cubes/
""",

    "molden": """
cadmia molden — Exporta funciones de onda de ORCA (.gbw) a formato Molden portátil.

El .gbw es la función de onda BINARIA y dependiente de versión de ORCA: solo ORCA y sus utilidades
lo leen. Las herramientas externas del análisis final (etapa char) leen Molden, no .gbw:
  NPA → JANPA (por defecto, gratis, vía Java) o NBO7 (opcional)
  AIM/QTAIM → Multiwfn (puntos críticos de enlace/anillo/caja, laplaciano, elipticidad)
Este comando pasa por orca_2mkl de ORCA para que las convenciones de las funciones f/g coincidan
con lo que esas herramientas esperan (sin re-correr ORCA, sin quedar atado a una versión de ORCA).

Nota: los cubos de `cadmia density` van por el otro lado (orca_plot lee el .gbw directo); el Molden
es solo para la rama NPA/AIM.

Uso:
  cadmia molden GBW... [opciones]
  GBW: uno o más .gbw (los de la caracterización final). En lote acepta comodines.

Opciones:
  --orca-2mkl-path RUTA  Ejecutable orca_2mkl (viene con ORCA).
  --output, -o DIR       Carpeta de salida de los .molden (por defecto: junto a cada .gbw).

Necesita orca_2mkl → correr en la máquina donde está instalado ORCA (o pasar --orca-2mkl-path).

Ejemplos:
  cadmia molden cluster.gbw
  cadmia molden *.gbw -o molden/
""",

    "prescreen": """
cadmia prescreen — Cribado rápido de configuraciones de adsorción con GFN2-xTB (antes del DFT).

Optimiza cada geometría de partida con xTB (barato), elimina las que relajan al MISMO
mínimo (duplicados por RMSD + energía), las ordena por energía xTB y conserva las de menor
energía dentro de una ventana. Las energías xTB solo sirven para ORDENAR y DEDUPLICAR: los
supervivientes DEBEN re-optimizarse en DFT (cadmia prepare/run). GFN2-xTB ya incluye D4, así
que el cribado y el DFT (D4) comparten la misma física de dispersión.

Requiere el binario `xtb` en el PATH (https://github.com/grimme-lab/xtb; libre, offline).

Argumento:
  CONFIGS               Directorio de .xyz, archivos sueltos o un glob (p. ej. ./configs/).

Opciones:
  --charge N            Carga total del aducto (por defecto: 0).
  --mult auto|N         Multiplicidad 2S+1. auto (por defecto): la mult= que geometry escribe
                        en el comentario de cada .xyz; si no hay, la menor por paridad.
  --gfn 0|1|2           Parametrización GFN (por defecto: 2 = GFN2-xTB con D4).
  --top N               Limita a N los supervivientes dentro de la ventana (0 = sin tope;
                        por defecto se conservan TODOS los que caen en --window).
  --window KCAL         Conserva configs dentro de KCAL/mol del mínimo (por defecto: 10).
  --rmsd Å              RMSD bajo el cual dos geometrías son el mismo mínimo (por defecto: 0.20).
  --min-keep N          Garantiza al menos N supervivientes DISTINTOS aunque los siguientes
                        mínimos queden fuera de la ventana, para que el DFT siempre tenga ≥N
                        candidatos que comparar (por defecto: 2; 0 = solo ventana). Limitado por
                        el nº de mínimos distintos hallados y por --top.
  --output, -o DIR      Carpeta para los .xyz optimizados supervivientes (por defecto: ./prescreened).
  --xtb-path RUTA       Nombre o ruta absoluta del ejecutable xtb.
  --jobs, -j N          Corridas de xtb simultáneas (por defecto: 0 = una por CPU disponible).
                        Una optimización xTB de un cúmulo pequeño no escala repartiéndola en
                        hilos; lo que sí escala es lanzar muchas a la vez, cada una con un hilo.
                        En SLURM se toma la máscara de afinidad, no el nº de núcleos del nodo.
  --threads N           Hilos OpenMP por corrida de xtb (por defecto: 1). Déjalo en 1:
                        algunas compilaciones (p. ej. apt 6.6.1 de Ubuntu) hacen SIGSEGV con >1.
  --no-sanity           Desactiva el filtro de sanidad. Por defecto se descartan los mínimos
                        NO físicos (átomos colapsados a <0.5 Å, o energía >250 kcal/mol por
                        debajo de la mediana = mínimo espurio de xTB) para que una geometría
                        rota no se vuelva la referencia de menor energía y arruine la ventana.
  --cluster ARCHIVO     .xyz del cúmulo desnudo, relajado AL MISMO NIVEL que este cribado.
                        Agrupa las configuraciones por el esqueleto al que relajó su cúmulo y
                        ordena cada grupo contra su propio cero, de modo que un isómero del
                        cúmulo más estable no pueda descartar a los aductos que conservaron el
                        esqueleto de la referencia. Sin él, la ventana es global.
                        OJO: la referencia debe venir del mismo nivel de teoría. El esqueleto
                        de un cúmulo puede diferir entre DFT y GFN2-xTB; cribar con xTB contra
                        una referencia DFT marca como isómeros a casi todas.
  --freeze              Mantiene rígido el cúmulo en xTB y solo relaja la sonda (requiere
                        --cluster, cuyos átomos deben abrir cada configuración). Por defecto
                        se relaja todo: un cribado congelado pierde los sitios que solo
                        enlazan si el cúmulo se acomoda. El DFT nunca va restringido.

Dentro de la carpeta -o se escribe SOLO un sidecar prescreen.json con el resultado
completo: energía xTB, ΔE y el motivo por el que cada configuración se conservó o se
descartó (lo que los .xyz supervivientes no pueden decir de las que no pasaron).

  --db / --no-db     Por defecto SÍ persiste. Persiste en la base de datos el destino de cada configuración en
                     esta etapa, enlazado con la configuración de partida que registró
                     `geometry`: energía, si convergió, de cuál resultó duplicada, su
                     clase de esqueleto y si sobrevivió a la ventana.
  --db-path ARCHIVO  Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).

Ejemplo:
  cadmia prescreen ./configs/ --window 20 -o ./prescreened/
  # conserva TODOS los que caen dentro de 20 kcal/mol + pool/candidates_by_energy.xyz (1 archivo)
  # luego: cadmia prepare ./prescreened/ ... && cadmia submit ... (re-opt en DFT con D4)
""",

    "main": """
CADMIA — Computational Agent-Driven Molecular Intelligence Architecture.

Plataforma de agentes para el estudio de reactividad y adsorción en
nanocúmulos mediante química cuántica (Gaussian 16, ORCA 6).

Comandos disponibles:

 Flujo (de principio a fin):
  pipeline    Corre el flujo COMPLETO encadenado y reanudable (Orchestrator).

 Reactividad y geometría:
  cdft        Calcula descriptores CDFT y rankea sitios reactivos.
  geometry    Genera geometrías de adsorción guiadas por el ranking CDFT.
  prescreen   Pre-cribado GFN2-xTB de configuraciones (dedup + ventana de energía).
  rank        Ordena salidas DFT por energía → un solo .xyz listo para prepare.

 Cálculo y HPC:
  prepare     Genera los inputs DFT (.inp/.gjf) desde .xyz u .out SIN ejecutarlos.
              Etapas: screen-opt | screen-freq | opt | freq | opt+freq | char | cdft | bsse.
  run         Corre cálculos EN ESTA MÁQUINA, en cola secuencial (workstation).
  submit      Envía cálculos DFT al HPC vía SLURM.
  monitor     Sondea el estado de los trabajos y descarga los resultados.
  profile     Gestiona perfiles de conexión HPC (add / list / show / ping).

 Análisis y reporte:
  validate    Valida archivos de salida DFT (terminación, SCF, espín, etc.).
  analyze     Energética de adsorción y activación post-cálculo.
  stability   Estabilidad relativa del óxido mixto vs. óxidos puros.
  stability-series  E_b, ΔE_mix, E_f (fragmentación), Δ²E e isómeros a lo largo de la serie.
  bde         Energía de disociación de enlace desde fragmentos.
  correlate   Regresión de descriptores contra datos experimentales o calculados (r²).
  report      Reporte integrador: reports/cadmia.out (texto); resumen Markdown con -o NOMBRE.md.

 Electrónica y estructura:
  electronic  Propiedades electrónicas: DOS/PDOS, IR, MEP.
  structure   Análisis estructural (ECN/coordinación) y gráficas 2D.
  density     Cubos de densidad/HOMO/LUMO/MEP y mapas Fukui/dual/Δρ (requiere orca_plot).
  molden      Exporta .gbw → .molden para NPA (JANPA) y AIM (Multiwfn).
  bonds       Órdenes de enlace de Mayer desde una salida ORCA (necesita Print[P_Mayer] 1).

 Datos y proyecto:
  init        Inicializa un proyecto CADMIA en el directorio actual.
  import      Importa salidas de cálculo a la base de datos del proyecto.
  db          Gestiona la base de datos local (init / status / ask).

 Utilidades:
  xyz         Extrae un .xyz limpio de una salida Gaussian/ORCA.
  doctor      Diagnóstico del entorno (dependencias, config, DB, binarios).
  guide       Muestra la guía de uso empaquetada (o su ruta).
  diagnose    Diagnostica cálculos problemáticos (síntomas + sugerencias LLM experimentales).
  literature  Busca bibliografía relacionada (Crossref, OpenAlex, Semantic Scholar).
  benchmark   Compara sistemáticamente Gaussian vs ORCA (generate / compare).

Ayuda detallada de cada comando:
  cadmia COMANDO --ayuda      (español)
  cadmia COMANDO --help       (inglés)
  cadmia --version            Versión instalada.
""",

    "pipeline": """
cadmia pipeline — Flujo COMPLETO encadenado y reanudable (Orchestrator Agent).

Ejecuta en orden de dependencia, partiendo de los tres estados SCF base:
  validate → cdft → geometry → run → analyze → stability → report

Es REANUDABLE: el avance se guarda en pipeline.json después de CADA etapa.
La etapa de cálculo (run) necesita un motor (ORCA/Gaussian); si no lo
encuentra se marca 'blocked' y el flujo se detiene LIMPIO. Así puedes correr
validate/cdft/geometry en cualquier máquina y reanudar el resto en la
workstation:

    cadmia pipeline --resume -o WORKDIR --engine-path /ruta/a/orca6

Opciones principales:
  --n / --np1 / --nm1   Los tres estados SCF base (requeridos salvo --resume).
  --cluster ETIQUETA    Nombre del cúmulo (p. ej. Al2TiO5).
  --probe SONDA         Sonda de adsorción: H2, CO2, NH3, CO, N2, O2, H2O o una
                        de ~/.cadmia/probes.yaml (def.: NH3).
  --ref ARCHIVO         Óxido puro de referencia para estabilidad (repetible).
  --probe-free ARCHIVO  Salida de la sonda aislada; sin ella la etapa analyze
                        se omite (la energética de adsorción la necesita).
                        Puede darse al reanudar si no estaba antes.
  --engine gaussian|orca  Motor de la etapa run (autodetectado de los outputs si se omite).
  --engine-path RUTA    Directorio de ORCA o binario g16 para la etapa run.
  --task/--functional/--basis/--dispersion   Nivel de teoría de los aductos.
  --charge N            Carga total del cúmulo (def.: 0); los aductos suman la de la sonda
                        (0 en todas las sondas incorporadas).
  --nprocs N / --mem-o MB   Recursos por trabajo en la etapa run.
  --bsse                Corrige por BSSE las energías de adsorción.
  --latex               Escribe también las tablas LaTeX del reporte.
  --until ETAPA         Detenerse después de esa etapa.
  --resume              Reanuda desde pipeline.json saltando lo ya hecho. Conserva el nivel de
                        teoría de la primera corrida (task, preset, funcional, base,
                        dispersión y carga) salvo que des alguna de esas opciones de nuevo.
  --dry-run             Muestra el plan de etapas sin ejecutar nada.
  --workdir, -o DIR     Directorio del manifiesto y de todas las salidas
                        (def.: ./pipeline).
  --preset {none,analysis,accurate}
                        Paquete de salidas/exactitud de la etapa de cálculo (def.: analysis).
                        accurate añade el bloque de producción (RIJCOSX, def2/J, TightSCF,
                        DefGrid3) sobre las propiedades de analysis.
  --timeout S           Límite de tiempo por trabajo en segundos (etapa de cálculo).
  --db / --no-db        Registra la corrida en la base (workflow_runs). Activo por defecto;
                        --db-path elige la base.

La geometría del cúmulo se extrae automáticamente del propio output N, así que
el orden atómico siempre coincide con los sitios CDFT (no hace falta cluster.xyz).

Ejemplos:
  # Tramo sin motor (en cualquier máquina), se detiene en run:
  cadmia pipeline --n N.out --np1 N+1.out --nm1 N-1.out \\
                  --cluster Al2TiO5 --probe NH3 \\
                  --ref TiO2.out --ref Al2O3.out -o ./corrida/ --db
  # En la máquina con ORCA instalado, retomar todo:
  cadmia pipeline --resume -o ./corrida/ --engine-path /opt/orca6 \\
                  --probe-free NH3.out --nprocs 12 --mem-o 3000 --db
  # Ver el plan sin ejecutar:
  cadmia pipeline --resume -o ./corrida/ --dry-run
""",

    "cdft": """
cadmia cdft — Descriptores CDFT y ranking de sitios reactivos.

Toma las salidas SCF de los estados N, N+1 y N-1 (Gaussian u ORCA) y calcula
funciones de Fukui condensadas, descriptores globales (η, μ, ω, gap, IP, EA)
y clasifica cada átomo como electrofílico, nucleofílico o ambifílico.

Opciones:
  --n ARCHIVO        Salida SCF con N electrones. Opcional solo dentro de un proyecto y
                     con --report (ver abajo); si no, requerido.
  --np1 ARCHIVO      Salida SCF con N+1 electrones, anión (ídem).
  --nm1 ARCHIVO      Salida SCF con N-1 electrones, catión (ídem).
  --cluster TEXTO    Etiqueta del cúmulo (por defecto: directorio padre).
  --engine {gaussian,orca}   Motor; autodetectado del contenido si se omite.
  --output, -o ARCHIVO       Escribe el JSON de salida a un archivo (sin -o, --no-table lo
                             imprime en pantalla).
  --report, -r ARCHIVO  Escribe además el reporte completo
                     (global + condensados por átomo, electro/nucleofilicidad
                     local, softnesses). Dentro de un proyecto el resultado se guarda
                     en results/cdft.json, y --report solo (sin --n/--np1/--nm1) lo
                     escribe desde ese archivo, sin recalcular.
  --report-format {txt,json}  Formato de --report (def.: txt).
  --tce-homo EV      Referencia HOMO(TCE) en eV para el índice de nucleofilicidad.
  --top N            Número de sitios a mostrar en la tabla (por defecto: 5).
  --no-table         Omite la tabla; emite solo JSON.
  --engrad-np1 ARCHIVO  .engrad del estado N+1, punto sencillo sobre la MISMA
                     geometría que N. Junto con --engrad-nm1 añade la función de
                     Fukui nuclear Φ_A: cómo cambia la fuerza sobre cada núcleo al
                     variar el número de electrones.
  --engrad-nm1 ARCHIVO  .engrad del estado N-1, sobre la misma geometría.
  --engrad-n ARCHIVO Opcional: .engrad del neutro. Añade Φ⁺ y Φ⁻ por separado y el
                     dual nuclear Φ⁽²⁾, que los dos estados cargados solos no dan.
  --project-normal   Proyecta además Φ sobre la normal saliente de cada átomo, que es la
                     dirección por la que se aproxima una sonda. Conserva el SIGNO, que la
                     magnitud descarta: un núcleo atraído hacia el interior del cúmulo es el
                     caso opuesto a uno empujado hacia fuera.
  --db / --no-db     Por defecto SÍ persiste. Persiste el resultado (molécula + cálculo + descriptores).
  --db-path ARCHIVO  Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).

Ejemplos:
  cadmia cdft --n N.out --np1 N+1.out --nm1 N-1.out -o cdft.json --db
  cadmia cdft --n N.out --np1 N+1.out --nm1 N-1.out \\
              --engrad-np1 anion.engrad --engrad-nm1 cation.engrad \\
              --engrad-n neutro.engrad -o cdft.json
""",

    "validate": """
cadmia validate — Control de calidad de salidas DFT.

Ejecuta seis verificaciones sobre cada archivo: terminación normal,
convergencia SCF, contaminación de espín ⟨S²⟩, frecuencias imaginarias,
sanidad geométrica (radios covalentes) y, opcionalmente, corrección BSSE.

Argumentos:
  ARCHIVOS...        Uno o más archivos de salida (.out/.log).

Opciones:
  --engine {gaussian,orca}   Motor; autodetectado si se omite.
  --bsse             Verifica la corrección BSSE (solo cálculos de adsorción).
  --output, -o ARCHIVO       Escribe el reporte JSON a un archivo.
  --no-table         Omite la tabla; emite solo JSON.
  --db / --no-db     Registra cada validación en run_log de la base de datos (activo por defecto).
  --db-path ARCHIVO  Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).

Devuelve código de salida 1 si alguna validación falla.

Ejemplo:
  cadmia validate salida.out --bsse
""",

    "geometry": """
cadmia geometry — Geometrías iniciales de adsorción.

Lee el JSON del agente CDFT y coloca la molécula sonda (ver el catálogo de sondas más abajo)
sobre los sitios mejor rankeados, usando radios covalentes para la
distancia inicial M–X (R_sitio + R_sonda + 0.5 Å) y barriendo orientaciones.

Opciones:
  --cdft ARCHIVO     JSON producido por `cadmia cdft -o` (requerido).
  --xyz ARCHIVO      Geometría del cúmulo: un .xyz O una salida Gaussian/ORCA
                     (cualquier extensión; el motor se autodetecta por el CONTENIDO,
                     no por .out/.log) — se extrae la geometría conservando el orden
                     atómico del SCF (requerido).
  --probe SONDA      Molécula sonda: H2, CO2, NH3, CO, N2, O2, H2O o una de
                     ~/.cadmia/probes.yaml (requerido).
  --probes-yaml ARCHIVO  Archivo de sondas propias (por defecto: ~/.cadmia/probes.yaml).
  --n-sites N        Número de sitios CDFT a usar; 0 = TODOS (por defecto: 0). Restringirlo
                     sesga el muestreo hacia los extremos del descriptor dual, que no es
                     donde se une toda sonda.
  --n-configs N      Orientaciones azimutales por sitio y modo (por defecto: 6). A polar=0
                     el azimut deja de ser dirección de aproximación y pasa a ser el giro
                     propio de la sonda: sólo se colapsa a una configuración si la sonda es
                     axial (H2 eta1, CO2 o_down).
  --polar GRADOS     Ángulos polares de aproximación (repetible; def.: 0 20 40 60).
  --output, -o DIR   Directorio donde se escriben los .xyz.
  --no-table         Omite la tabla; emite solo JSON.
  --cluster TEXTO    Etiqueta del cúmulo (por defecto: nombre del .xyz).
  --db / --no-db     Por defecto SÍ persiste. Persiste el cúmulo y las configuraciones de adsorción en la base de datos.
  --db-path RUTA     Base de datos para --db (por defecto: la base del propio proyecto, .cadmia/cadmia.db, si el directorio de trabajo está dentro de uno; si no, ~/.cadmia/cadmia.db).

Ejemplo:
  cadmia geometry --cdft cdft.json --xyz cluster.xyz --probe NH3 -o ./configs/
""",

    "analyze": """
cadmia analyze — Energética de adsorción y activación post-cálculo (Analysis Agent).

A partir de tres salidas validadas —adducto (cúmulo+sonda), cúmulo aislado y
sonda libre— calcula la energía de adsorción, la corrección BSSE (counterpoise),
las energías de deformación/interacción, la elongación del enlace de la sonda y
su activación, y contrasta el sitio de enlace con el ranking CDFT.

Una sonda cuenta como activada si un enlace se alarga ≥ 5 % respecto a la molécula libre
o si un ángulo que era lineal en ella (≥ 175°, como en el CO2) se dobla ≥ 10° en el
aducto. El JSON añade probe_angles (libre, aducto y Δ de cada ángulo, en grados) y
max_bend_deg (la mayor flexión de un ángulo lineal).

Además del veredicto de reorganización del cúmulo, el JSON incluye cuánto se movió
cada átomo del cúmulo al adsorber (`cluster_displacement_ang`, por átomo, tras
superposición óptima sobre el bloque de cúmulo) y el del sitio de enlace
(`delta_r_site_ang`), que `cadmia correlate` direcciona como `dr_site`.

Modo simple: --adduct UNO.out -o resultado.json.
Modo lote: --adduct-dir CARPETA/ -o SALIDA/ analiza todos los *.out de la carpeta
contra el mismo cúmulo y sonda, y escribe un JSON por aducto (índice por nombre).

Opciones:
  --adduct ARCHIVO        Salida del adducto cúmulo+sonda (modo simple).
  --adduct-dir CARPETA    Carpeta de salidas de aductos (modo lote): un JSON por *.out en -o.
  --cluster ARCHIVO       Salida del cúmulo aislado (requerido).
  --probe-free ARCHIVO    Salida de la sonda libre (requerido).
  --probe SONDA           Etiqueta de la sonda (p. ej. CO2, NH3; si se omite, se infiere de la fórmula).
  --bsse                  Lee el BSSE counterpoise del adducto y corrige E_ads.
  --cluster-deformed ARCHIVO  SP del cúmulo a la geometría del adducto (solo modo simple).
  --probe-deformed ARCHIVO    SP de la sonda a la geometría del adducto (solo modo simple).
  --cdft ARCHIVO          JSON de CDFTResult para contrastar el sitio de enlace.
  --config-index N        Índice de configuración del triplete (modo simple).
  --bend-threshold GRADOS Flexión de un ángulo lineal de la sonda que cuenta como
                          activación (def.: 10).
  --label TEXTO           Etiqueta del cúmulo.
  --output, -o RUTA       Modo simple: archivo JSON. Modo lote: carpeta de salida.
  --report, -r ARCHIVO    Escribe además el reporte de texto completo. Solo, dentro de un
                          proyecto, lo escribe desde results/analysis/ sin recalcular.
  --no-table              Omite la tabla; emite solo JSON.
  --db / --no-db [--db-path RUTA]   Persiste la(s) fila(s) de adsorción en la base de datos
                                    (activo por defecto).

Ejemplos:
  cadmia analyze --adduct n1_NH3.out --cluster n1.out --probe-free NH3.out --bsse
  cadmia analyze --adduct-dir ./adduct_out --cluster N.out --probe-free co2.out --probe CO2 -o analysis/
""",

    "diagnose": """
cadmia diagnose — Diagnóstico de cálculos DFT problemáticos.

Extrae síntomas de convergencia de un archivo de salida Gaussian/ORCA mediante
parseo determinista (terminación con error, SCF no convergido, oscilación de
energía, colapso variacional, gap HOMO-LUMO negativo o casi nulo, contaminación
de espín) y, si hay un LLM configurado, propone estrategias de convergencia
priorizadas con los keywords exactos del motor (p. ej. ORCA: SlowConv, MOREAD
desde un .gbw convergido).

Las sugerencias del LLM son solo texto a verificar: nada se ejecuta y los
síntomas se extraen siempre con parsers locales, con o sin LLM.

Argumentos:
  ARCHIVO               Salida .out/.log a diagnosticar.

Opciones:
  --engine MOTOR        gaussian | orca (autodetectado si se omite).
  --no-llm              Solo síntomas; omite las sugerencias del LLM (experimentales).
  --language es|en      Idioma de las sugerencias (por defecto: es).
  --output, -o ARCHIVO  Escribe el diagnóstico JSON.

Ejemplo:
  cadmia diagnose cation.out
""",

    "doctor": """
cadmia doctor — Diagnóstico del entorno CADMIA.

Verifica Python, dependencias requeridas (numpy, click, pyyaml), los
subpaquetes de CADMIA, la configuración (atomic_data.yaml), las dependencias
opcionales (rich, matplotlib, plotly, scipy, skimage), las herramientas
externas en el PATH (orca, g16, xtb, ssh) y el estado/esquema de la base
de datos.

Opciones:
  --db-path RUTA   Base de datos a revisar (por defecto: la base del propio proyecto, .cadmia/cadmia.db, si el directorio de trabajo está dentro de uno; si no, ~/.cadmia/cadmia.db).

Sale con código 1 si falla alguna verificación requerida; las advertencias
(funciones opcionales) no fallan.

Ejemplo:
  cadmia doctor
""",

    "stability": """
cadmia stability — Estabilidad relativa (de formación) del óxido mixto vs. puros.

Calcula E_rel = E(mixto) − Σ cᵢ·E(puroᵢ), donde los coeficientes equilibran la
fórmula del mixto contra las referencias (p. ej. Al2TiO5 = 1·Al2O3 + 1·TiO2).
E_rel negativa ⇒ el mixto es más estable que la suma de sus óxidos puros.
Las energías de referencia salen de tus propios .out al mismo nivel de teoría.
Por defecto E es E0 = E + ZPE. El JSON trae además E_rel en cada base
(E, E0, H, G, Gq) que tengan todas las salidas. Gq es G con la entropía
vibracional cuasi-RRHO de Grimme (ν0 = 100 cm⁻¹), la que ORCA aplica por defecto;
a las salidas de Gaussian se les corrige su G armónica.

Opciones:
  --mixed ARCHIVO     Salida del óxido mixto (requerido).
  --ref ARCHIVO       Salida de un óxido puro (repetible: --ref TiO2.out --ref Al2O3.out).
  --coeff N           Coeficiente estequiométrico por --ref (mismo orden): 1, 0.5 o 1/2.
                      Si se omite, se resuelve como fracción exacta: referencias del
                      doble de tamaño dan ½, ½ (reference_size = equal-size).
  --label TEXTO       Etiqueta del óxido mixto.
  --energy E0|E|H|G|Gq  Energía de todos los términos (por defecto E0). Si alguna salida
                      no tiene frecuencias, todas pasan a E y se avisa: nunca se resta
                      una E de una E0.
  --output, -o ARCHIVO  Escribe el JSON de StabilityResult.
  --report, -r ARCHIVO  Escribe además el reporte de texto completo. Solo, dentro de un
                        proyecto, lo escribe desde results/stability.json sin recalcular.
  --no-table          Omite la tabla; emite solo JSON.
  --db / --no-db [--db-path RUTA]  Persiste la estabilidad relativa en la base de datos
                                   (activo por defecto).

Ejemplo:
  cadmia stability --mixed Al2TiO5.out --ref TiO2.out --ref Al2O3.out
""",

    "stability-series": """
cadmia stability-series — Estabilidad dependiente del tamaño a lo largo de la serie.

Calcula, para una serie de cúmulos (unidad)n de una misma familia, mixta o pura:
  E_b      = (E[Sn] − Σ c·E[puros de tamaño 1])/n  (enlace por unidad frente a los puros)
  ΔE_mix(n) = E[Sn] − Σ c·E[puros del mismo n]
  ΔE_mix,eq = (E[Sn] − Σ c·E[puros de igual tamaño])/n
  E_b,fam  = (n·E[S1] − E[Sn])/n              (enlace por unidad frente al monómero propio)
  E_f      = E[S(n-1)] + E[S1] − E[Sn]        (energía de fragmentación)
  Δ²E      = E[S(n-1)] + E[S(n+1)] − 2·E[Sn]  (segunda diferencia de energía)
E_b y E_b,fam responden preguntas distintas y no deben mezclarse. Por defecto E
es E0 = E + ZPE. El número de unidades n se infiere de la
estequiometría (gcd de los conteos atómicos). Cada descriptor se emite solo
donde existen sus entradas. Varios --cluster con el mismo n son isómeros: se reportan todos, con ΔE0 y ΔG frente a sus mínimos, y el más bajo
en la base elegida representa a n.

Opciones:
  --cluster ARCHIVO   Salida de un cúmulo de la serie (repetible; n se infiere de cada uno).
  --ref ARCHIVO       Óxido puro de cualquier tamaño (repetible), emparejado por
                      estequiometría: tamaño 1 → E_b, mismo n → ΔE_mix(n),
                      igual tamaño → ΔE_mix,eq.
  --energy E0|E|H|G|Gq  Energía de todos los términos (por defecto E0); pasa a E con
                      aviso si alguna salida no tiene frecuencias.
  --pop               Poblaciones de Boltzmann de los isómeros de cada n, en E0 y en G (σ de
                      la geometría, g = 2 si es quiral, T de las salidas) y los
                      descriptores promediados con ellas. Indicativas: valen solo
                      para los isómeros dados.
  --output, -o ARCHIVO  Escribe el JSON de SeriesStabilityResult.
  --no-table          Omite la tabla; emite solo JSON.

Ejemplo:
  cadmia stability-series --cluster n1.out --cluster n2.out --cluster n3.out \\
                          --ref A1.out --ref B1.out --ref A2.out --ref B2.out -o serie.json
""",

    "bde": """
cadmia bde — Energía de disociación de enlace desde fragmentos.

Calcula BDE = Σ E(fragmento) − E(molécula) a partir de los .out de la molécula y
sus fragmentos (al MISMO nivel, geometrías relajadas). Positiva = energía para
romper el enlace (mayor ⇒ enlace más fuerte). Verifica la conservación de átomos
(Σ fragmentos == molécula) y avisa si no cuadra. Salida en kcal/mol y eV.

Opciones:
  --molecule ARCHIVO  Salida de la molécula (requerido).
  --fragment ARCHIVO  Salida de un fragmento (repetible; normalmente dos).
  --label TEXTO       Etiqueta de la molécula.
  --output, -o ARCHIVO  Escribe el JSON de BDEResult.
  --no-table          Omite la tabla; emite solo JSON.

Ejemplo:
  cadmia bde --molecule AB.out --fragment A.out --fragment B.out -o bde.json
""",

    "literature": """
cadmia literature — Descubrimiento de referencias desde APIs abiertas (Literature Agent).

Consulta Crossref, OpenAlex y Semantic Scholar (sin API key) por tema y/o por DOI,
normaliza los resultados, elimina duplicados por DOI, los ordena y los presenta en
Markdown, BibTeX o JSON. Los identificadores y números (DOI, año, citas) provienen
SIEMPRE de la API, nunca de un modelo de lenguaje; --narrate añade solo una nota de
relevancia en prosa (opcional, requiere LLM configurado).

Argumento:
  CONSULTA              Tema en texto libre (opcional si se usan --doi).

Opciones:
  --doi DOI             DOI explícito a recuperar (repetible).
  --source FUENTE       crossref | openalex | semanticscholar (repetible).
  --limit N             Máximo de referencias tras deduplicar/ordenar.
  --from-year AÑO       Solo trabajos publicados desde ese año.
  --sort CRITERIO       relevance | citations | year.
  --format FMT          md | bibtex | json.
  --narrate             Experimental. Nota de relevancia por LLM (texto, nunca números).
  --language es|en      Idioma de la nota --narrate.
  --output, -o ARCHIVO  Escribe a archivo en vez de stdout.

Ejemplos:
  cadmia literature "conceptual DFT heterogeneous catalysis" --from-year 2024 --sort citations
  cadmia literature --doi 10.1021/cr990029p --format bibtex
  cadmia literature "TiO2 cluster reactivity" --format json -o refs.json
  # luego: cadmia report --cdft cdft.json --literature refs.json
""",

    "report": """
cadmia report — El output de CADMIA: el reporte completo de todos los módulos (Report Agent).

Dentro de un proyecto (cadmia init), sin opciones de módulo, lee los resultados originales
que cada módulo guardó en results/ (las copias, cdft-2.json…, se dejan fuera y se listan) y
escribe reports/cadmia.out: cabecera (programa, versión, cita, nivel de teoría), el reporte
COMPLETO de cada módulo en su recuadro numerado (1 Structure, 2 CDFT, 3 Adsorption,
4 Stability; el número es fijo), las correlaciones entre módulos y el pie de terminación
normal. Es texto de columnas fijas, legible con cat.

Las opciones de módulo (--structure, --cdft, --analysis, --stability) toman uno o más JSON
seguidos, o 'all' para todos los de ese módulo en results/ (copias incluidas); entonces solo
se reportan los módulos dados:
  cadmia report --cdft results/cdft.json results/cdft-2.json --structure all Con
-o NOMBRE.out el output lleva otro nombre; con -o NOMBRE.md se escribe en cambio el resumen
Markdown (--simple: solo tablas), y --latex añade las tablas LaTeX.

Un resultado de results/ nunca se pisa con uno calculado desde otros archivos: las mismas
entradas actualizan su archivo y otras van a la siguiente copia (cdft-2.json, site5-2.json…);
ese nombre es el identificador persistente del resultado. Si results/ mezcla cúmulos distintos
(se compara la fórmula entre módulos), el reporte se detiene y nombra el archivo que no coincide. Las figuras
matplotlib se escriben solo si matplotlib está instalado; si no, se emite (x,y) como CSV.

Opciones:
  --structure ARCH…|all  JSON de structure report (uno o más, o all).
  --cdft ARCH…|all       JSON de cdft (uno o más, o all).
  --analysis ARCH…|all   JSON de analyze (uno o más, o all; uno por configuración).
  --analysis-dir CARPETA  Carpeta de JSON de AnalysisResult (agrega todos los *.json);
                        evita listarlos uno a uno con --analysis.
  --stability ARCH…|all JSON de stability (uno o más, o all).
  --benchmark ARCHIVO   JSON de BenchmarkResult.
  --literature ARCHIVO  JSON de LiteratureResult (cadmia literature --format json):
                        añade una sección de bibliografía al reporte.
  --dos ARCHIVO         Archivo DOS de `cadmia electronic dos` (repetible): se enlaza
                        en una sección "Density of states" del reporte.
  --pdos ARCHIVO        Archivo PDOS de `cadmia electronic pdos` (repetible): idem.
  --title TEXTO         Título del reporte.
  --output, -o ARCHIVO  Archivo de salida. Con sufijo .md, el resumen Markdown (por defecto:
                        reports/cadmia.out dentro de un proyecto, si no cadmia.out).
  --latex               Escribe también las tablas en LaTeX (.tex).
  --figures             Escribe figuras de correlación (matplotlib; CSV si falta).
  --engine gaussian|orca  Motor a registrar en la cabecera de información de la corrida.
  --functional TEXTO    Funcional a registrar en la cabecera (p. ej. PBE0).
  --basis TEXTO         Base a registrar en la cabecera (p. ej. def2-TZVP).
  --dispersion TEXTO    Corrección de dispersión a registrar (p. ej. D4).
  --simple              Reporte ligero: solo tablas de resultados, sin cabecera/pie de proceso.
  --narrate             Experimental. Añade una sección de discusión redactada por IA (borrador,
                        claramente marcado para revisión). Requiere configurar el LLM:
                        ~/.cadmia/llm.yaml con provider anthropic | openai | gemini
                        (openai acepta base_url para servidores locales tipo Ollama)
                        + la API key por variable de entorno (ANTHROPIC_API_KEY /
                        OPENAI_API_KEY / GEMINI_API_KEY). El LLM solo redacta texto
                        a partir de los números ya calculados; nunca los altera.
  --language es|en      Idioma del borrador de discusión (por defecto: es).

Ejemplos:
  cadmia report                                   (dentro del proyecto → reports/cadmia.out)
  cadmia report --cdft cdft.json --analysis a0.json --stability s.json -o reporte.md --latex
""",

    "prepare": """
cadmia prepare ETAPA GEOMETRÍAS — Genera los inputs DFT de una ETAPA SIN ejecutarlos.

Uso: cadmia prepare <etapa> <geometrías...> [opciones]

La ETAPA es el primer argumento y estandariza el input de cada fase del flujo. Las
geometrías van después (un archivo suelto, varios, un directorio o un glob). Escribe
un .inp por geometría, listo para revisar o enviar (`cadmia submit`/`run`).

Los dos motores usan .inp: ORCA recibe el nombre del archivo como argumento y Gaussian lee
por stdin, así que la extensión es una convención de CADMIA, no un requisito del motor. El
motor de un input ya hecho se deduce de su CONTENIDO (%chk/%mem → gaussian; !/%pal/%maxcore
→ orca), nunca de su nombre.

Qué acepta como geometría:
  .xyz        Geometría suelta. Si trae varios bloques concatenados (lo que escribe
              `cadmia rank`), genera UN input por bloque y conserva el nombre de origen.
  .out/.log   Output de ORCA/Gaussian: extrae la geometría final, así cada etapa
              encadena con el resultado de la anterior sin pasar por `cadmia xyz`.
              El motor del output se detecta solo (--engine solo elige qué se escribe).
  .inp/.gjf   Input ya hecho (de cualquier motor): se deja pasar tal cual, sin regenerarlo.
Cualquier otra extensión es un error, no un archivo ignorado en silencio.

Etapas:
  screen-opt   Pre-cribado: Opt en def2-SVP para ordenar candidatos por energía.
  screen-freq  Pre-cribado: Freq al MISMO nivel que screen-opt, para descartar puntos
               silla (frecuencias imaginarias) antes de las etapas caras.
  opt          Geometría final: Opt al nivel de producción (sin análisis).
  freq         Frecuencias sobre la geometría optimizada: mínimo + termoquímica.
  opt+freq     Geometría + frecuencias en un solo job (Opt Freq).
  char         Caracterización electrónica: single point con Hirshfeld/CHELPG/Mayer/PDOS.
  cdft         Trío N/N+1/N−1 (single points) para descriptores por diferencias finitas.
  bsse         Counterpoise (ORCA): cada fragmento del aducto con su base y con la base
               completa (átomos fantasma), en la geometría del aducto. Requiere --cluster
               y --probe-free; luego `cadmia analyze --bsse` lee los cuatro *_cp_*.out.

Por qué screen-freq existe y no se usa `freq` en el cribado: una Freq solo tiene sentido
sobre una geometría optimizada AL MISMO NIVEL, y el nivel incluye el umbral SCF y la malla,
no solo la base. La etapa `freq` es de producción (TightSCF DefGrid3); correrla sobre una
geometría de screen-opt (solo RIJCOSX) devuelve frecuencias imaginarias espurias pequeñas
y haría descartar aductos buenos. `screen-freq` se queda en el nivel barato.

La etapa es dueña de la tarea y del bloque de exactitud. No ejecuta nada ni necesita
motor instalado: solo escribe texto. Para personalizar cualquier otro eje, una bandera
explícita SOBRESCRIBE el valor de la etapa (base, funcional, salidas). El nivel se aplica
a TODAS las geometrías del lote — para cargas distintas por archivo, genera por separado.

Opciones (todas opcionales; sobrescriben la etapa):
  --engine MOTOR        orca | gaussian — motor para el que se escribe el input; ambos
                        salen como .inp (def.: orca).
  --functional/--basis/--dispersion   Nivel de teoría (def.: PBE0/def2-TZVP/d4;
                        las etapas screen-* usan def2-SVP salvo que pases --basis).
  --charge N            Carga total (def.: 0).
  --mult auto|N         Multiplicidad: 'auto' (la mult= del comentario del .xyz que escribe
                        geometry; si no hay, por paridad de electrones) o un entero.
  --mult-anion/--mult-cation   [etapa cdft] Multiplicidad del anión/catión ('auto' o N).
  --cluster ARCHIVO     [etapa bsse] Output del cúmulo aislado: su número de átomos parte
                        el aducto, y su carga y multiplicidad son las del fragmento.
  --probe-free ARCHIVO  [etapa bsse] Output de la sonda libre (carga y multiplicidad).
  --preset NOMBRE       none | cdft | analysis | accurate — sobrescribe la salida de la etapa.
  --nprocs N            Núcleos por trabajo (%pal / %nprocshared; def.: 8).
  --mem-o MB            %maxcore de ORCA en MB por núcleo (def.: 2000).
  --mem-g GB            %mem de Gaussian en GB (def.: 16).
  --extra-orca "TXT"    Palabras clave extra de ORCA (línea simple).
  --extra-gaussian "TXT"  Palabras clave extra de la ruta de Gaussian.
  --scf-maxiter N       Límite de iteraciones SCF (0 = default del motor). Ayuda a que
                        converjan los iones/estados de capa abierta.
  --geom-maxiter N      Límite de pasos de la optimización (0 = default del motor, max(50, 3N)
                        en ORCA). Súbelo cuando la optimización tiene que reacomodar el
                        sustrato.
  --chelpg-fine         Emite un bloque %chelpg fino en ORCA (solo si se pide CHELPG).
  --overwrite           Regenera el input aunque el archivo ya exista.
  --output, -o DIR      Directorio donde escribir los inputs (def.: actual).

Ejemplos:
  # pre-cribado de las configuraciones de adsorción, luego enviarlas a la cola:
  cadmia prepare screen-opt ./configs_nh3/*.xyz --nprocs 26 -o ./inputs_nh3/
  cadmia submit --jobs ./inputs_nh3/ --hpc cluster
  # Freq de cribado sobre los ganadores del rank (encadena con los .out):
  cadmia prepare screen-freq ./screen/*.out --nprocs 26 -o ./screenfreq/
  # o directamente desde el .xyz combinado que escribe rank (un input por bloque):
  cadmia prepare screen-freq ./ranked.xyz --nprocs 26 -o ./screenfreq/
  # caracterización final de los ganadores y trío CDFT del cúmulo:
  cadmia prepare char ./ganadores/*.out -o ./char/
  cadmia prepare cdft ./cumulo_opt.out  -o ./cdft/
  # personalizar: añadir salidas de análisis al pre-cribado:
  cadmia prepare screen-opt mol.xyz --preset analysis
""",

    "run": """
cadmia run — Cola local de cálculos DFT en esta máquina (workstation, sin SSH/SLURM).

Ejecuta los trabajos UNO A LA VEZ: cada cálculo usa los núcleos y memoria
declarados en su input (%pal/%maxcore en ORCA, %nprocshared/%mem en Gaussian),
así la máquina nunca se sobresuscribe. El avance se guarda en cadmia_jobs.json
después de cada trabajo: al re-lanzar el lote, los trabajos ya terminados y
validados se saltan (reanudación) y solo se calcula lo que falta. Valida cada
salida con el Validation Agent al terminar.

Acepta .inp/.gjf directamente, o los .xyz que escribe `cadmia geometry`
(se convierten a inputs con el nivel de teoría indicado).

Opciones principales:
  --jobs RUTA           Input, .xyz o directorio con ellos (requerido).
  --engine MOTOR        gaussian | orca (auto por extensión; .xyz → orca).
  --engine-path RUTA    Directorio de ORCA o binario g16 (def.: PATH).
                        ORCA paralelo REQUIERE la ruta absoluta — esta opción
                        la resuelve por ti.
  --task TAREA          sp | opt | freq | opt+freq (def.: opt; para .xyz).
  --functional/--basis/--dispersion   Nivel de teoría (def.: PBE0/def2-TZVP/d4).
  --charge N            Carga para inputs generados de .xyz.
  --mult auto|N         Multiplicidad: 'auto' (la mult= del comentario del .xyz que escribe
                        geometry; si no hay, por paridad de electrones) o un entero.
  --preset NOMBRE       none | cdft | analysis | accurate — salidas a pedir (def.: analysis:
                        Hirshfeld+CHELPG + Mayer + poblaciones por MO para PDOS).
  --nprocs N            Núcleos por trabajo (def.: 8).
  --mem-o MB            %maxcore de ORCA en MB por núcleo (def.: 2000).
  --mem-g GB            %mem de Gaussian en GB (def.: 16).
  --extra-orca "TXT"    Palabras clave extra de ORCA (línea simple).
  --timeout S           Límite de tiempo por trabajo en segundos (mata el árbol
                        de procesos completo al vencer, sin dejar rangos MPI huérfanos).
  --overwrite           Recalcula todo, ignorando salidas ya terminadas (sin él, reanuda).
  --output, -o DIR      Directorio de inputs generados, salidas y cadmia_jobs.json.
  --db / --no-db        Persiste los trabajos terminados en la base de datos (activo por
                        defecto); --db-path elige la base.
  --no-tidy             Deja los archivos accesorios en su sitio (por defecto pasan a aux/).

Ejemplos:
  # correr las configuraciones NH3 generadas por cadmia geometry:
  cadmia run --jobs ./configs_nh3/ --task opt --nprocs 12 --mem-o 3000 \\
             --engine-path /opt/orca6 -o ./adsorcion_nh3/ --db
  # un solo input ya escrito:
  cadmia run --jobs N.inp --engine-path /opt/orca6
""",

    "submit": """
cadmia submit — Envío de cálculos al HPC vía SLURM.

Sube los archivos de entrada (.gjf / .inp), genera el script sbatch,
los envía y registra el estado en `cadmia_jobs.json`. Con --wait espera
a que terminen, descarga las salidas y las valida automáticamente.

Opciones:
  --jobs RUTA        Archivo .gjf/.inp o directorio con ellos (requerido).
  --hpc NOMBRE       Perfil HPC a usar (ver `cadmia profile list`) (requerido).
  --engine {gaussian,orca}   Forzar motor; autodetectado por extensión si se omite.
  --nprocs N         Núcleos por trabajo (por defecto: 8).
  --mem-g GB         Memoria de Gaussian en GB (por defecto: 16).
  --mem-o MB         Memoria por núcleo de ORCA en MB (por defecto: 2000).
  --output, -o DIR   Directorio local para salidas y estado (por defecto: .).
  --wait             Bloquea hasta que terminen; luego descarga y valida.
  --db / --no-db     Por defecto SÍ persiste. Persiste los trabajos enviados como registros de cálculo en la base de datos.
  --db-path RUTA     Base de datos para --db (por defecto: la base del propio proyecto, .cadmia/cadmia.db, si el directorio de trabajo está dentro de uno; si no, ~/.cadmia/cadmia.db).

Ejemplo:
  cadmia submit --jobs ./inputs/ --hpc hpc1 --wait -o ./outputs/
""",

    "monitor": """
cadmia monitor — Sondeo de trabajos y descarga de resultados.

Reanuda el seguimiento de los trabajos registrados por `cadmia submit`.
Consulta SLURM (squeue/sacct y, si no hay accounting, el marcador CADMIA_DONE
del .out), descarga las salidas completadas y las valida.

Opciones:
  --jobs-file ARCHIVO   Ruta a cadmia_jobs.json (por defecto: ./cadmia_jobs.json).
                        Alias: --jobs (igual que en submit/run).
  --hpc NOMBRE          Perfil HPC (requerido).
  --output, -o DIR      Directorio local para descargas (por defecto: .).
  --once                Sondea una sola vez y termina (útil en scripts).
  --interval SEGUNDOS   Segundos entre ciclos de sondeo (por defecto: 60).

Ejemplo:
  cadmia monitor --jobs-file ./outputs/cadmia_jobs.json --hpc hpc1
""",

    "profile": """
cadmia profile — Gestión de perfiles de ejecución (SLURM local o HPC por SSH).

Un perfil puede ser LOCAL (SLURM en esta misma workstation, sin SSH; `add --local`)
o REMOTO (HPC por SSH). submit/monitor eligen el backend según el perfil.

Subcomandos:
  add        Crea y guarda un perfil en ~/.cadmia/profiles/<nombre>.yaml.
  list       Lista los perfiles guardados.
  show       Muestra los detalles de un perfil (indica el modo: local | remoto SSH).
  ping       Prueba la disponibilidad: sbatch local (perfil --local) o SSH (remoto).

Ayuda en español de cada subcomando:
  cadmia profile SUBCOMANDO --ayuda
""",

    "profile add": """
cadmia profile add — Guardar un perfil de conexión HPC.

Crea ~/.cadmia/profiles/<nombre>.yaml con los datos de conexión y scheduler.

Opciones:
  --name NOMBRE      Identificador del perfil (ej.: hpc1) (requerido).
  --local            Perfil LOCAL: SLURM en esta workstation, sin SSH. Con --local,
                     --host/--user no son necesarios y los módulos quedan vacíos por defecto.
  --host HOST        Nombre o IP del servidor SSH (requerido si no es --local).
  --user USUARIO     Usuario SSH (requerido si no es --local).
  --work-dir RUTA    Directorio scratch (requerido, local o remoto).
  --partition NOMBRE Partición SLURM (por defecto: general).
  --account CUENTA   Cuenta SLURM.
  --ssh-key RUTA     Ruta a la llave privada SSH.
  --gaussian-module MOD   Módulo a cargar para Gaussian (def.: gaussian/16).
  --orca-module MOD       Módulo a cargar para ORCA (def.: orca/6).
  --orca-path RUTA        Ruta absoluta al binario de ORCA.
  --max-time HH:MM:SS     Límite de tiempo SLURM (def.: 24:00:00).
  --port PUERTO      Puerto SSH (por defecto: 22).

Ejemplo:
  cadmia profile add --name hpc1 --host hpc.ejemplo.mx --user usuario \\
      --work-dir /scratch/usuario/cadmia --partition compute
""",

    "profile list": """
cadmia profile list — Listar los perfiles HPC guardados.

Sin opciones. Muestra los nombres de los perfiles en ~/.cadmia/profiles/.
""",

    "profile show": """
cadmia profile show — Mostrar los detalles de un perfil HPC.

Argumentos:
  NOMBRE     Nombre del perfil a mostrar.

Ejemplo:
  cadmia profile show hpc1
""",

    "profile ping": """
cadmia profile ping — Probar la disponibilidad de un perfil.

Para un perfil --local comprueba que `sbatch` esté en el PATH; para un perfil
remoto prueba la conectividad SSH.

Argumentos:
  NOMBRE     Nombre del perfil a probar.

Devuelve código de salida 1 si la conexión/disponibilidad falla.

Ejemplo:
  cadmia profile ping hpc1
""",

    "benchmark": """
cadmia benchmark — Comparación sistemática Gaussian vs ORCA.

Subcomandos:
  generate   Genera entradas .gjf y .inp idénticas desde la misma geometría.
  compare    Compara salidas ya calculadas (ΔE, RMSD, frecuencias, tiempos).

Ayuda en español de cada subcomando:
  cadmia benchmark SUBCOMANDO --ayuda
""",

    "benchmark generate": """
cadmia benchmark generate — Generar entradas Gaussian y ORCA.

Escribe un .gjf y un .inp con el mismo nivel de teoría y la misma geometría,
listos para enviar a ambos motores.

Opciones:
  --xyz ARCHIVO      Archivo .xyz del cúmulo (requerido).
  --functional FUNC  Funcional DFT (por defecto: PBE0).
  --basis BASE       Conjunto de base (por defecto: def2-TZVP).
  --dispersion {d3bj,d4,none}   Corrección de dispersión (def.: d3bj).
  --task {sp,opt,freq,opt+freq} Tipo de cálculo (def.: opt).
  --charge CARGA     Carga molecular total (def.: 0).
  --mult auto|N      Multiplicidad: 'auto' (la mult= del comentario del .xyz; si no hay, por
                     paridad de electrones) o un entero (def.: auto).
  --nprocs N         Núcleos solicitados en la entrada (def.: 8).
  --mem-g GB         Memoria de Gaussian en GB (def.: 16).
  --mem-o MB         Memoria por núcleo de ORCA en MB (def.: 2000).
  --output, -o DIR   Directorio de salida para .gjf y .inp (requerido).
  --cluster TEXTO    Etiqueta del cúmulo.

Ejemplo:
  cadmia benchmark generate --xyz n1.xyz --functional PBE0 --task opt+freq -o ./inputs/
""",

    "benchmark compare": """
cadmia benchmark compare — Comparar salidas Gaussian y ORCA ya calculadas.

Valida ambas salidas y calcula la diferencia de energía (ΔE en kcal/mol),
el RMSD de geometrías (Kabsch), el RMSD de los primeros 6 modos vibracionales
y la razón de tiempos ORCA/Gaussian, con una recomendación de motor.

Opciones:
  --gaussian ARCHIVO Salida de Gaussian .log/.out (requerido).
  --orca ARCHIVO     Salida de ORCA .out (requerido).
  --cluster TEXTO    Etiqueta del cúmulo.
  --output, -o ARCHIVO       Escribe el reporte JSON.
  --markdown ARCHIVO Escribe la tabla comparativa en Markdown.
  --no-table         Omite la tabla; emite solo JSON.
  --db / --no-db     Por defecto SÍ persiste. Persiste ambos cálculos y la comparación en la base de datos.
  --db-path RUTA     Base de datos para --db (por defecto: la base del propio proyecto, .cadmia/cadmia.db, si el directorio de trabajo está dentro de uno; si no, ~/.cadmia/cadmia.db).

Devuelve código de salida 1 si los motores no coinciden (|ΔE| > 0.1 kcal/mol
o RMSD > 0.01 Å).

Ejemplo:
  cadmia benchmark compare --gaussian g.log --orca o.out --markdown reporte.md
""",

    "db": """
cadmia db — Gestión de la base de datos local de CADMIA.

Subcomandos:
  init       Crea la base de datos e inicializa el esquema.
  status     Muestra el conteo de registros por tabla.
  ask        Consulta en lenguaje natural (experimental; el LLM genera un SELECT de solo lectura).

Ejemplo:
  cadmia db init
""",

    "db init": """
cadmia db init — Crear la base de datos e inicializar el esquema.

Opciones:
  --path ARCHIVO     Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).

Ejemplo:
  cadmia db init
""",

    "db status": """
cadmia db status — Mostrar el conteo de registros por tabla.

Opciones:
  --path ARCHIVO     Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).
""",

    "db ask": """
cadmia db ask — Consultar la base de datos en lenguaje natural (experimental).

El LLM configurado (~/.cadmia/llm.yaml + variable de entorno de la API key)
traduce la pregunta a UNA consulta SELECT. La consulta se valida (solo lectura,
una sola sentencia) y se ejecuta en una conexión de solo lectura: el modelo
nunca puede modificar la base ni inventar números — todos los valores mostrados
salen de SQLite.

Argumento:
  PREGUNTA           Pregunta en lenguaje natural (entre comillas).

Opciones:
  --path ARCHIVO     Ruta de la base (def.: la del propio proyecto si el directorio de
                     trabajo cae dentro de uno, si no ~/.cadmia/cadmia.db).
  --limit N          Máximo de filas a mostrar (def.: 50).
  --sql-only         Muestra el SQL generado sin ejecutarlo.

Ejemplos:
  cadmia db ask "¿cuál es la estabilidad relativa de cada óxido mixto?"
  cadmia db ask "lista los cálculos ORCA con su energía" --limit 10
  cadmia db ask "qué descriptores tiene el Al2TiO5" --sql-only
""",

    "guide": """
cadmia guide — Muestra la guía de uso empaquetada (o su ruta).

Opciones:
  -es                Guía en español (por defecto).
  -en                Guía en inglés.
  --show             Imprime la guía completa en pantalla (en vez de la ruta).

Ejemplos:
  cadmia guide                 # ruta de la guía en español
  cadmia guide --show          # imprime la guía en español
  cadmia guide -en --show      # imprime la guía en inglés
""",

    "paths": """
cadmia paths — Cómo muestran las rutas los reportes de texto (.txt, cadmia.out).

Uso:
  cadmia paths               # muestra el modo actual
  cadmia paths relative      # rutas relativas al proyecto (por defecto)
  cadmia paths absolute      # rutas completas, con aviso

Con relative, un reporte compartido no lleva el nombre de usuario ni las carpetas
de la máquina. Con absolute, cadmia lo avisa al cambiar y en cada cadmia report.
Los JSON de results/ y la base de datos guardan siempre la ruta absoluta y el hash.
El modo se guarda en ~/.cadmia/config.yaml (report_paths).
""",

    "electronic": """
cadmia electronic — Propiedades electrónicas.

Subcomandos:
  dos     DOS total por ensanchamiento gaussiano de los eigenvalores.
  pdos    DOS proyectada por elemento (requiere Print[P_OrbPopMO_L] 1 en ORCA).
  ir      Espectro IR desde las intensidades de los modos normales (Freq de ORCA).
  mep     Análisis del potencial electrostático molecular desde un .cube.

Ayuda detallada: cadmia electronic SUBCOMANDO --ayuda
""",

    "electronic mep": """
cadmia electronic mep — Análisis del potencial electrostático molecular (MEP).

Lee un .cube de MEP y, con --density, lo evalúa sobre la isosuperficie de
densidad (cruces de las aristas de la rejilla, ponderados por área): extremos
Vs,min (nucleófilo) y Vs,max (electrófilo) con su posición y átomo más cercano,
áreas positiva/negativa, σ²±, σ²_tot, balance ν, Π y el índice de polaridad
molecular (MPI). Coincide con la función 12 de Multiwfn a menos de 0.52 %.

Argumento:
  ARCHIVO               .cube del MEP (en u.a., Hartree/e).

Opciones:
  --density ARCHIVO     .cube de densidad electrónica para definir la superficie
                        molecular (opcional).
  --isovalue VALOR      Isovalor de densidad que define la superficie (e/bohr³;
                        por defecto: 0.001).
  --output, -o ARCHIVO  Escribe el reporte a un archivo (por defecto: stdout).

Ejemplo:
  cadmia electronic mep mep.cube --density density.cube -o mep_report.txt
""",

    "electronic ir": """
cadmia electronic ir — Espectro IR desde las intensidades de los modos normales (salida Freq de ORCA).

Argumento:
  ARCHIVO            Salida ORCA de frecuencias (con intensidades IR).

Opciones:
  --scale FLOAT      Factor empírico de escalado de frecuencias (p. ej. 0.96 para DFT híbrido; def.: 1.0).
  --fwhm FLOAT       Ancho a media altura de los picos en cm⁻¹ (def.: 16.0).
  --lineshape lorentzian|gaussian   Perfil de ensanchamiento (def.: lorentzian).
  --points N         Puntos de la malla (def.: 4000).
  -o, --output ARCH  Escribe el espectro (.dat/.csv/.json/.html/.png/.svg/.pdf).
  --format FMT       dat | csv | json | html | png | svg | pdf (html = gráfica interactiva;
                     png/svg/pdf = figura estática con matplotlib). Si se omite: el sufijo
                     de -o; si no, dat.
  --width ANCHO      single | double: ancho de la figura estática, 3.4 in (una columna) o
                     6.5 in (def.: single).

La figura estática dibuja la curva en el eje izquierdo (unidades arbitrarias) y los modos
activos como palitos en su propio eje derecho, en km/mol.

Ejemplo:
  cadmia electronic ir 1tial_freq.out --scale 0.96 -o ir.html --format html
""",

    "electronic dos": """
cadmia electronic dos — DOS total desde los eigenvalores orbitales.

Argumento:
  SALIDA             Archivo de salida Gaussian u ORCA.

Opciones:
  --sigma EV         Ancho del ensanchamiento gaussiano en eV (def.: 0.1).
  --points N         Mínimo de puntos de la malla de energía; se usan más para un paso de σ/4 (def.: 2000).
  -o, --output ARCH  Escribe la curva (.dat/.csv/.json/.html/.png/.svg/.pdf).
  --format FMT       dat | csv | json | html | png | svg | pdf (html = gráfica interactiva;
                     png/svg/pdf = figura estática con matplotlib). Si se omite: el sufijo
                     de -o; si no, dat.
  --width ANCHO      single | double: ancho de la figura estática, 3.4 in (una columna) o
                     6.5 in (def.: single).

El DOS está en estados/eV y suma AMBOS espines: en capa cerrada cada orbital espacial cuenta 2
(la integral es el nº de orbitales por 2). En capa abierta suma los canales alfa y beta, que
también guarda por separado. --sigma debe ser positivo.
""",

    "electronic pdos": """
cadmia electronic pdos — DOS proyectada por elemento.

Requiere que la salida ORCA tenga el bloque de poblaciones por MO
(%output Print[P_OrbPopMO_L] 1 end).

Argumento:
  SALIDA             Archivo ORCA con poblaciones Löwdin por MO.

Opciones:
  --sigma EV         Ancho del ensanchamiento gaussiano en eV (def.: 0.1).
  --points N         Mínimo de puntos de la malla de energía; se usan más para un paso de σ/4 (def.: 2000).
  -o, --output ARCH  Escribe las curvas (.dat/.csv/.json/.html/.png/.svg/.pdf).
  --format FMT       dat | csv | json | html | png | svg | pdf: html = gráfica interactiva
                     autocontenida con una curva por elemento + total; png/svg/pdf = figura
                     estática con matplotlib. Si se omite: el sufijo de -o; si no, dat.
  --width ANCHO      single | double: ancho de la figura estática, 3.4 in (una columna) o
                     6.5 in (def.: single).

Como el DOS, el PDOS está en estados/eV con ambos espines (un orbital de capa cerrada cuenta 2).
""",

    "structure": """
cadmia structure — Análisis estructural.

Funciona con CUALQUIER nanocúmulo (bi/tri-metálico, óxido, mixto): la
detección de enlaces usa radios covalentes/metálicos con respaldo de tabla
periódica completa.

Subcomandos:
  report      Enlaces, CN/ECN de Hoppe, poliedros, formas continuas,
              conectividad, ángulos, anillos y forma global → TXT/CSV/JSON.
  compare     Compara estructuras: ¿el mismo isómero? ¿cuánto se movieron?
  plot        Gráficas interactivas (HTML) o figuras estáticas PNG/SVG/PDF (extra [full]).

Opción:
  --report, -r ARCHIVO  Dentro de un proyecto, escribe el reporte estructural completo
                        desde results/structure/, sin recalcular.

Ayuda detallada: cadmia structure SUBCOMANDO --ayuda
""",

    "structure compare": """
cadmia structure compare — ¿Es la misma estructura, y cuánto se ha movido?

Argumentos:
  REFERENCIA         Estructura .xyz contra la que se compara.
  CANDIDATAS...      Una o más .xyz a comparar (admite globs).

Opciones:
  --json             Emite la comparación en JSON en vez de en tabla.

Responde dos preguntas que un RMSD por sí solo no separa.

  ESQUELETO    Dice si la conectividad es la misma, es decir si se ha hecho o
               roto algún enlace. Es una firma de la topología, invariante a
               rotación, traslación y al orden en que estén escritos los átomos.

  DESVIACIÓN   Dice cuánto se han movido los átomos tras superponer de forma
               óptima. Se dan la máxima y la RMSD, ambas del mismo
               emparejamiento, que es POR ELEMENTO y no por posición.

Una estructura que sólo se ha doblado tiene el mismo esqueleto y una desviación
grande. Una que se ha reorganizado tiene otro esqueleto. La distancia sola no
las distingue, porque una deformación mueve los átomos tanto como una
reorganización: es la diferencia entre un sustrato que se adapta y uno que se
reconstruye.

Las estructuras de composición distinta se declaran no comparables, en vez de
recibir un número sin sentido. Y se avisa cuando algún contacto queda tan cerca
del corte de enlace que el veredicto del esqueleto no es seguro.

Ejemplos:
  cadmia structure compare cumulo_inicial.xyz optimizadas/*.xyz
  cadmia structure compare ref.xyz cand.xyz --json
""",

    "structure report": """
cadmia structure report — Análisis estructural completo.

Secciones del reporte: enlaces y estadística por par · CN y ECN de Hoppe
autoconsistente · poliedros de coordinación con su medida de forma continua
(CShM) · conectividad vértice/arista/cara entre poliedros · ángulos por triplete
y ángulos puente M-O-M · anillos del subgrafo metal-ligando · forma global por
el tensor de giro · coordinación relativa a la media del elemento.

OJO con dos columnas que parecen la misma: CN cuenta TODOS los vecinos, metales
incluidos; CN_L cuenta sólo los ligandos, porque un poliedro de coordinación es
metal-ligando por definición. Un Ti unido a 4 O y 2 Al tiene CN=6 y CN_L=4.
(El nombre antiguo `cadmia structure structure` sigue funcionando como alias.)

Argumento:
  ENTRADA            Archivo(s), directorio o glob: .xyz, .cube, salidas
                     Gaussian/ORCA, o XYZ concatenado .txt. Para
                     ecn-trend/ecn-heatmap pasa un reporte/tabla estructural.

Opciones:
  -o, --output ARCH  Archivo del reporte (relativo a la carpeta actual). Sin -o, va a
                     structural_report.<formato> en la carpeta analizada.
  --format FMT       txt | csv | json (si se omite, usa el sufijo del archivo).
  --config YAML      YAML propio (masas, cortes, parámetros ECN).
  --details          En reportes TXT por lote, añade secciones por estructura.
""",

    "structure plot": """
cadmia structure plot — Gráficas interactivas (HTML, plotly) o figuras estáticas para publicar
(PNG/SVG/PDF, matplotlib) de bonds, distances, ecn-trend y ecn-heatmap. Requiere [full].

Argumento:
  ENTRADA            Archivo(s)/directorio/glob, o un reporte/tabla ECN para
                     ecn-trend/ecn-heatmap.

Opciones:
  -o, --output RUTA  Archivo (una gráfica) o directorio (lote). Sin --format, el sufijo
                     decide: .html interactiva; .png/.svg/.pdf estática. En lote -o es siempre una carpeta: un -o
                     con sufijo de archivo (figs.png) es un error.
  --format FMT       html | png | svg | pdf. Si se omite: el sufijo de -o; si no, html. En
                     lote es la forma de pedir figuras estáticas (--format png -o figs/).
  --width ANCHO      single | double: ancho de la figura estática, 3.4 in (una columna) o
                     6.5 in (def.: single).
  --kind TIPO        bonds (def.) | distances | ecn-trend | ecn-heatmap. Solo 2D: las
                     vistas 3D de una estructura quedan para un visor molecular.
  --title TEXTO      Título opcional.
  --config YAML      YAML propio (masas, cortes, parámetros ECN).
  --ecn-decimals N   Decimales de las etiquetas ECN redondeadas (def.: 1).
  --ecn-label EST    raw | rounded | both (def.: both → "2.985 (3.0)"); celdas del
                     heatmap y texto emergente de la tendencia interactiva.

Las gráficas ECN descubren los pares presentes dinámicamente (cualquier
sistema); Ti-O/Al-O/Ti-Ti/Al-Al/Ti-Al es solo el orden preferente.
""",

}


# ─────────────────────────────────────────────
# Click integration
# ─────────────────────────────────────────────

def _command_key(ctx: click.Context) -> str:
    """Build the SPANISH_HELP key from the context chain (root program excluded)."""
    parts: list[str] = []
    cur: click.Context | None = ctx
    while cur is not None and cur.parent is not None:
        if cur.info_name:
            parts.append(cur.info_name)
        cur = cur.parent
    parts.reverse()
    return " ".join(parts) if parts else "main"


def _show_spanish_help(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    key = _command_key(ctx)
    text = SPANISH_HELP.get(key)
    if text is None:
        click.echo(
            f"(No hay ayuda en español para '{key}'. "
            f"Usa --help para la versión en inglés.)"
        )
    else:
        click.echo(text.strip("\n"))
    ctx.exit()


def _ayuda_option() -> click.Option:
    return click.Option(
        ["--ayuda"],
        is_flag=True,
        expose_value=False,
        is_eager=True,
        callback=_show_spanish_help,
        help="Mostrar la ayuda en español y salir.",
    )


class EsCommand(click.Command):
    """Click command that also offers --ayuda."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params.append(_ayuda_option())


# Top-level command grouping for `cadmia --help` (keeps the flat 27-command list scannable).
# Any command not listed here still appears, under "Other".
COMMAND_GROUPS: list[tuple[str, list[str]]] = [
    ("Workflow (end to end)", ["pipeline"]),
    ("Reactivity & geometry", ["cdft", "geometry", "prescreen", "rank"]),
    ("Calculation & HPC", ["prepare", "run", "submit", "monitor", "profile"]),
    ("Analysis & reporting", ["validate", "analyze", "stability", "stability-series", "bde", "correlate", "report"]),
    ("Electronic & structural", ["electronic", "structure", "density", "molden", "bonds"]),
    ("Data & project", ["init", "import", "db"]),
    ("Utilities", ["xyz", "doctor", "paths", "guide", "diagnose", "literature", "benchmark"]),
]


class EsGroup(click.Group):
    """Click group whose commands and subgroups all offer --ayuda."""

    command_class = EsCommand

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params.append(_ayuda_option())

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Top-level help lists commands grouped by category; nested groups use the default."""
        if ctx.parent is not None:        # only the root `cadmia` group is grouped
            return super().format_commands(ctx, formatter)

        available = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is not None and not getattr(cmd, "hidden", False):
                available[name] = cmd

        limit = formatter.width - 6 - max((len(n) for n in available), default=0)
        seen: set[str] = set()
        for title, names in COMMAND_GROUPS:
            rows = [(n, available[n].get_short_help_str(limit)) for n in names if n in available]
            seen.update(n for n in names if n in available)
            if rows:
                with formatter.section(title):
                    formatter.write_dl(rows)
        rest = [(n, available[n].get_short_help_str(limit)) for n in available if n not in seen]
        if rest:
            with formatter.section("Other"):
                formatter.write_dl(rest)


# Nested groups created via @group.group() inherit the same class.
EsGroup.group_class = EsGroup
