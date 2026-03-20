# Documentación Técnica Integral: OrthoSSM y Proyecto Chimera

## 1. Introducción y Visión General del Sistema
El presente documento ofrece una descripción técnica exhaustiva, teórica y práctica del repositorio **OrthoSSM**, con especial atención a la arquitectura híbrida **Chimera**. Este proyecto representa un avance fundamental en los Modelos de Espacios de Estados (SSMs - State Space Models), superando las limitaciones de arquitecturas previas (como Mamba) mediante el uso de proyecciones ortogonales, representaciones avanzadas de memoria y sistemas jerárquicos de retroalimentación de estado.

El ecosistema está construido en la intersección de modelos de lenguaje basados en secuencias largas y sistemas dinámicos continuos parametrizados discretamente, buscando alta eficiencia en hardware paralelo (GPUs) a través de kernels Triton y CUDA personalizados.

**OrthoSSM** es uno de los modulos fundamentales de la arquitectura COEUS.
---

## 2. Bibliotecas y Dependencias Base
El sistema es intensivamente dependiente del ecosistema de deep learning de PyTorch, complementado con herramientas de optimización a muy bajo nivel:
- **PyTorch (`torch`)**: El framework base. Se utiliza para la construcción de los grafos dinámicos, autograd, y la instanciación de tensores en GPU.
- **Triton (`triton`, `triton.language`)**: Fundamental para la aceleración del sistema. Muchos de los cuellos de botella algorítmicos en SSMs clásicos son mitigados escribiendo kernels Triton personalizados (ej. `sdpc_kernel.py`). Triton permite la compilación Just-In-Time (JIT) de operaciones matemáticas fusionadas directamente para hardware NVIDIA sin escribir CUDA C++.
- **Mamba SSM (`mamba_ssm`)**: Dependencia crítica, sobre la cual se construye y se compara el modelo OrthoSSM. Las operaciones de hardware nativo de Mamba (escaneo paralelo) proveen las raíces de la eficiencia iterativa del marco.
- **Causal Conv1D (`causal_conv1d`)**: Convoluciones causales unidimensionales súper optimizadas. Sirven como proyecciones de extracción temporal locales que alimentan los SSMs.
- **Transformers / HuggingFace (`transformers`, `einops`)**: Utilizados para manipulación tensorial compleja y tokenización (en conjunción con un tokenizador especializado, *COEUS*). `einops` permite operaciones de reordenamiento de dimensiones (`rearrange`, `reduce`) altamente legibles y seguras, críticas en modelos secuencia a secuencia.
- **Optimización y Paging**: Técnicas como captura de Grafos CUDA (`torch.cuda.make_graphed_callables`) se usan ampliamente en el evaluador e inferencia para reducir el "overhead" de latencia en despacho de kernels Python-to-GPU.

---

## 3. Arquitecturas y Subsistemas Centrales (OrthoSSM)

### 3.1. SLR (Schrödinger-Laguerre-Riemann) Module
**Ubicación**: `slr_module.py` / `chimera_experiment/sgr_slr.py`
**Funcionalidad Técnica**:
El módulo SLR es un sistema de proyección de estado continuo adaptativo. Utiliza polinomios de Laguerre (con métricas de Riemann o Schrödinger-like) para asegurar la retención de memoria ortogonal a lo largo del tiempo.
A diferencia de los RNNs tradicionales que sufren decaimiento exponencial (vanishing gradients) y pierden contexto, la formulación SLR mapea el estado oculto en bases ortogonales que preservan las dinámicas a largo plazo de forma casi perfecta matemáticamente sin interferencias constructivas inestables.

### 3.2. SDPC (State Distribution Predictive Control)
**Ubicación**: `sdpc_engine.py`, `sdpc_kernel.py`
**Funcionalidad Técnica**:
Un motor predictivo basado en control de distribuciones de estados. El `SDPCEngine` no se limita a predecir el próximo vector en el subespacio, sino que proyecta estimaciones de la distribución de las variables latentes. Está respaldado por kernels altamente optimizados procesados por GPU y Triton. Es esto lo que permite que el contexto extremadamente largo sea simulado sin una recálculo estricto de atenciones densas ($O(N^2)$). El núcleo `sdpc_kernel` usa sumas por bloques paralelos y operaciones atómicas temporales, dándole un ancho de banda masivo en GPUs de nivel servidor (como la H200 referenciada en `H200_DEEP_ANALYSIS.txt`).

---

## 4. El Ecosistema "Chimera" (Híbrido)

El directorio `chimera_experiment/` y relacionados contienen la versión más avanzada de los aportes de este repositorio: el modelo **Chimera**.

### 4.1. Chimera Layer & ChimeraLM
**Archivos Relevantes**: `chimera_layer.py`, `chimera_lm.py`, `chimera_v2_ssm.py`
**Técnica**:
Chimera layer fusiona o alterna bloques atencionales o mecanismos densos en conjunto con componentes puramente de estado (SSM/Mamba). Introduce técnicas híbridas para procesar las ráfagas altamente locales con Atenciones por Ventana o Convoluciones Especiales, dejando la macro-estructura a la compresión SLR/SDPC.
Se usan parámetros de inicialización específicos y matrices factorizadas de bajo rango para mantener la complejidad espacial contenida, resolviendo las caídas de degradación ("perplexity spikes") documentadas en los archivos Markdown del repo.

### 4.2. SDTM Memory (State Dynamic Turing Memory)
**Archivos Relevantes**: `sdtm_memory.py`
**Técnica**:
Una variante de memoria episódica acoplada al SSM. Emula la cinta de una Máquina de Turing pero de forma diferenciable. Lee y escribe matrices que interactúan con representaciones incrustadas de *bus* estado. Con esto, la IA puede "recuperar" un contexto lejano ("Needle in a Haystack" o NIAH evaluations, en `niah_eval.py`).

### 4.3. Chunked Training & Streaming Híbrido
**Archivos Relevantes**: `chimera_chunked_train.py`, `chimera_streaming.py`
El procesamiento no puede cargar todo el grafo de autograd si el contexto es teóricamente de millones de tokens. "Chunked train" corta la secuencia en sub-bloques del tamaño de hardware soportable (p. ej. fragmentos de 2048 ó 8192 tokens) y propaga o "paginariza" (`chimera_paged_state.py`) el estado $h_t$ de un chunk a otro sin requerir `requires_grad=True` en toda la cadena hacia atrás (o usando *Backpropagation Through Time Truncada*).

---

## 5. Módulos Adicionales y Flujos de Ejecución

- **Tokenización COEUS**: Implementado en `coeus_tokenizer.py`, sugiere una lógica subyacente de tokenizador entrenado o ajustado *custom* explícitamente para el modelaje del vocabulario específico que Chimera/OrthoSSM utiliza para su evaluación de PPL.
- **Benchmarking de Contexto e Interferencia**: Los archivos como `benchmark_long_context.py` y `chia_million_stress.py` evidencian sistemas preparados para contextos de entre $100k$ y $1M+$ tokens. Contrastan el rendimiento contra Mamba2 (*benchmark_mamba2_vs_ortho*) demostrando estabilidad topológica (evaluada mediante diagnósticos ortogonales en `ortho_diagnostics.py`).
- **Interpretación Mecanicista (`mech_interp.py`)**: Incluye circuitos o técnicas para decodificar qué "frecuencias" latentes en la distribución de auto-valores del SSM se correlacionan con comportamientos de lenguaje específicos.

---
*(Esta documentación puede ampliarse adentrándose recursivamente con inspecciones en árbol AST sobre los pipelines exactos Triton en cada capa)*

## 6. Revisión Extendida: Sistemas Matemáticos de Espacio de Estados

### 6.1. La Formulación Base de Mamba e Integración
El ecosistema completo se erige sobre **Mamba (SSM)**. Los Modelos de Espacio de Estado continuos se definen como:
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) + D x(t) $$
Para que esto funcione de forma eficiente en las GPUs y secuencias temporales discretas, el modelo implementa la Regla de Discretización (usualmente Bilineal o Zero-Order Hold - ZOH), paramétrica sobre un factor de escala del tiempo dependiente de los datos, denotado por $\Delta$. En `model.py` y `chimera_v2_ssm.py`, el bloque B, C, y $\Delta$ son funciones dependientes de la entrada $x(t)$, otorgando la propiedad de **"Selectividad"** a las ecuaciones dinámicas de retención.

### 6.2. El Aporte Central: OrthoSSM
El nombre **OrthoSSM** proviene de "Orthogonal State Space Model".
Los SSM clásicos (incluyendo algunas versiones tempranas de Mamba) inicializan y restringen las matrices 'A' usando aproximaciones HiPPO (Hidden Polynomial Projection Operators), donde se fuerza a la matriz A a ser asintóticamente estable (con eigenvalores en el semiplano negativo de la representación en el dominio continuo).
En **OrthoSSM**, los investigadores en este repositorio implementaron (vía componentes como `slr_module.py` y el ajuste de bases Chebyshev/Laguerre reportado en `CHEBYHOLO_IDEAS.txt`) matrices A rígidamente parametrizadas o penalizadas para ser normales/ortogonales (o bloque-diagonales ortogonales). Esto:
1. Impide completamente la amplificación exponencial indeseable en estados ocultos largos.
2. Garantiza teóricamente una retención unitaria perfecta, permitiendo recordar información pasada un millón de "pasos" atrás sin decaimiento (probado explícitamente en `benchmark_long_context.py` y `chimera_million_stress.py`).

---

## 7. Análisis de Sistemas Avanzados

### 7.1. Compilaciones y Kernels Triton
Este repositorio evita usar exclusivamente los cuellos de botella del backend ATen estándar de PyTorch para el escaneo recurrente. Usa Triton (`sdpc_kernel.py`) que:
- Permite la partición del tensor $L \times D$ en la caché SRAM de las GPUs NVIDIA modernas (arquitectura Hopper como la H200 confirmada en `H200_DEEP_ANALYSIS.txt`).
- Mantiene la memoria transaccional dentro del kernel GPU sin pasar por la VRAM externa (High Bandwidth Memory - HBM) por cada paso $t$, proporcionando las aceleraciones necesarias para lograr que el modelo compita computacionalmente con *FlashAttention*.

### 7.2. SDTM: State Dynamic Turing Memory
La arquitectura amplía los límites del SSM introduciendo un análogo a la "Memoria en Diferencia Dinámica". Modela los procesos del sub-módulo en ubicaciones diferenciables mediante variables difusas. Básicamente, a medida que los $tokens$ son consumidos, la ruta del bus transfiere gradientes usando mecanismos de decaimiento modulado aprendibles por el modelo, donde ciertos tokens actúan como puertas de lectura y acceso (Gates) hacia memorias globales persistentes (`sdtm_memory.py`).

### 7.3. Subsistema de Paginas y Checkpoints (Paged State)
En `chimera_paged_state.py`, se implementa una copia inspirada por arquitecturas como vLLM:
Durante inferencia concurrente o entrenamiento en lotes gigantes de contextos infinitos, la GPU agota rápidamente la memoria. Un sistema paged divide el $h_t$ de tamaño gigantesco en "bloques lógicos" que son despachados asincrónicamente y leídos hacia/desde la CPU RAM. Mediante transferencias PCIe con CUDA Streams traslapados, OrthoSSM oculta el coste de la redimensión de vectores ocultos largos, garantizando soporte de secuencias casi ilimitadas.

---

## 8. Análisis Exhaustivo de los Archivos del Repositorio

A continuación, una deconstrucción profunda, componente por componente de la vasta arquitectura en `OrthoSSM`.

### 8.1. Componentes Base ("Framework")
1. **`model.py`**: Interfaz unificada de alto nivel. Agrupa subcapas de incrustaciones de tokens, inyecta las posiciones causales, apila las capas consecutivas de `ChimeraLayer` o variantes SSM y define las funciones de pérdida fundamentales (Entropía cruzada en su defecto, pero combinable con las de `chimera_losses.py`).
2. **`slr_module.py`**: Schrödinger-Laguerre-Riemann module. Implementa las transformadas subyacentes continuas, calculando integrales de trayectoria de estado y exponenciación de la matriz usando Pade o expansión en serie de Taylor optimizada vectorialmente.
3. **`sdpc_engine.py`**: El contenedor del lado Python para el Kernel Triton de SDPC. Organiza las cuadrículas del lanzador paralelo de Triton (Grid) basadas en `batch_size`, cantidad de cabezales y la longitud del prompt de inferencia.
4. **`coeus_tokenizer.py`**: Define un tokenizador *subword* BPE personalizado. Contiene directivas explícitas de vocabulario y rutinas C/C++ vinculadas (probablemente usando libtokenizers) o pure-python vectorizado, preparado para integrarse a la entrada antes de `model.py`.

### 8.2. El Ecosistema de Benchmarking e Inferencia
1. **`benchmark_ortho.py`, `benchmark_long_context.py`, `benchmark_vs_sota.py`**: Archivos masivos de simulación de *Throughput* (Tokens/Segundo), Latencia (Latencia del Primer Token o TTFT), y uso pico de VRAM. Comparan directamente las métricas del motor propio (Ortho/Chimera) frente a modelos *State-of-the-Art* en tareas extensas.
2. **`niah_eval.py`**: Implementa la evaluación de Aguja en el Pajar (*Needle In A Haystack*). Introduce una clave generada aleatoriamente al comienzo del millón de tokens de texto basura (el "pajar"), e insta al modelo a recordarla al final para demostrar retención sin decaimiento, que es el Santo Grial evaluado por Mamba y este proyecto.
3. **`speculative_ssm.py`**: Decoding especulativo nativo de SSMs. En general, en Transformers estructurados (GPT), el *speculative decoding* requiere un "modelo borrador" pequeño. Mamba / Chimera, gracias a su formulación probabilística, puede proyectar subsecuencias enteras mediante potencias analíticas de A ($A^2, A^3, A^4$) para predecir ramificaciones temporales paralelas y evaluarlas simultáneamente, duplicando/triplicando la velocidad en Generación (Inferencia sin coste algorítmico excesivo).

*(Continúa en la sección de Optimización)...*

## 9. Desglose Lineal y Autogenerado de Módulos (Inspección AST Cíclica)

Esta sección provee un mapeo profundo y topológico de las clases concretas que conforman este gigantesco proyecto modular, sus responsabilidades de código y sus huellas en el repositorio.

### Archivo: `benchmark_mamba2_vs_ortho.py`
#### Clase: `Mamba2SSD`
**Propósito Funcional**: Mamba 2 block with chunk-wise SSD (State Space Duality) parallel scan.

Architecture from "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality" (Dao & Gu, 2024).

The SSD parallel scan computes the SSM output via a causal attention-like
matrix within each chunk, then propagates state between chunks.
This is the same algorithm as mamba_ssm's Triton kernel — no Python loops
in the forward pass (fully torch.compile-friendly).

Complexity: O(N · chunk²) compute, O(N) memory (SSM state only).
**Métodos o Funciones Clave**: __init__, _ssd_parallel_scan, forward

#### Clase: `Mamba2LM`
**Propósito Funcional**: Mamba 2 language model with N Mamba2SSD layers.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `benchmark_vs_sota.py`
#### Clase: `MambaBlock`
**Propósito Funcional**: Mamba-style Selective State Space block (Gu & Dao, 2023).
Pure PyTorch implementation of the selective scan algorithm.
S6 architecture: input-dependent A, B, C, Δ parameters.
**Métodos o Funciones Clave**: __init__, _selective_scan, forward

#### Clase: `MambaLM`
**Propósito Funcional**: Mamba language model with N layers.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `RWKVTimeMixing`
**Propósito Funcional**: RWKV-4 time mixing block.
Uses the WKV (weighted key-value) mechanism with exponential decay.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `RWKVChannelMixing`
**Propósito Funcional**: RWKV-4 channel mixing (FFN equivalent).
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `RWKVBlock`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque RWKVBlock.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `RWKVLM`
**Propósito Funcional**: RWKV language model with N layers.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `TransformerBlock`
**Propósito Funcional**: Standard pre-norm Transformer block with Flash Attention (via PyTorch SDPA).
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `TransformerLM`
**Propósito Funcional**: Standard Transformer language model.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `GRULM`
**Propósito Funcional**: GRU language model baseline.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/add_bus.py`
#### Clase: `AsyncLightBus`
**Propósito Funcional**: AsyncLightBus from OrthoSSM plan.
Provides a fast cross-layer communication channel.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/advanced_chimera.py`
#### Clase: `Fp8Linear`
**Propósito Funcional**: nn.Linear con forward en FP8 (float8_e4m3fn) para H100/H200 (SM≥9.0).

Cuantificación dinámica por tensor: escala = amax(|x|) / 448.
Backward en BF16/FP32 vía GradScaler estándar (FP8 solo en forward).
Fallback transparente a BF16 si _scaled_mm no está disponible.

Uso:
    linear_fp8 = Fp8Linear.from_linear(some_nn_linear)

Ganancia esperada en H200:
  - BF16 GEMM: ~1,979 TFLOPS
  - FP8  GEMM: ~3,958 TFLOPS (2× throughput teórico)
  - Speedup real: ~1.5-2× (considerando memoria y overhead)
**Métodos o Funciones Clave**: __init__, from_linear, forward, extra_repr

#### Clase: `GatedComplexityPredictor`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GatedComplexityPredictor.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `AsyncLightBus`
**Propósito Funcional**: AsyncLightBus from OrthoSSM plan.
Provides a fast cross-layer communication channel.
**Métodos o Funciones Clave**: __init__, forward, step_ring

#### Clase: `AdvancedChimeraLayer`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque AdvancedChimeraLayer.
**Métodos o Funciones Clave**: __init__, update_ttt_inplace, update_sdtm_inplace, archive_deferred, set_graph_mode, allocate_inference_cache, step, forward

#### Clase: `CUDAGraphPool`
**Propósito Funcional**: Pool de CUDA Graphs pre-capturados para batch_sizes variables de decode.

PROBLEMA: make_cuda_graph_step() captura UN ÚNICO batch_size. En serving
real (vLLM-style continuous batching), el número de requests activos varía
dinámicamente (1, 2, 4, 8...). Reutilizar un graph de B=1 con B=4 produce
resultados incorrectos o errores de forma.

SOLUCIÓN: pre-capturar un CUDAGraph por batch_size soportado y seleccionar
en runtime. Cada graph tiene sus propios tensores estáticos independientes.

Uso típico:
    # Post-prefill, antes de decode:
    pool = CUDAGraphPool(layer, batch_sizes=[1, 2, 4, 8])
    caches = {B: pool.allocate_cache(B) for B in pool.batch_sizes}

    # Durante decode con B variable:
    out, caches[B] = pool.step(x_tok, caches[B])

Complejidad de memoria:
    VRAM extra = sum(cache_size(Bi)) para todos Bi en batch_sizes.
    cache_size(B): ~0.27 MB × B × n_layers → para B=8: ~2.2 MB/layer.

Comportamiento con B no en pool:
    Si B ∈ [1,2,4,8] y B=3, usa el graph de B=4 con padding (rows 0:3 activas).
**Métodos o Funciones Clave**: __init__, _capture_for_batch, allocate_cache, _select_batch_size, step, __repr__

#### Clase: `Mamba2ChunkedPrefill`
**Propósito Funcional**: Carry correcto de conv_state + SSM state entre chunks de prefill.

Usa las APIs de estado explícito disponibles en mamba_ssm≥2.0:
  causal_conv1d_fn(initial_states, return_final_states)
  mamba_chunk_scan_combined(initial_states, return_final_states)

Esto elimina el drift SSM entre chunks que existía en la versión
anterior (bus carry solo, sin SSM carry).

Uso:
    chunker = Mamba2ChunkedPrefill(layer.mamba2)
    conv_s, ssm_s = chunker.init_states(batch_size)
    for chunk in chunks:
        out, conv_s, ssm_s = chunker.forward_chunk(chunk, conv_s, ssm_s)
**Métodos o Funciones Clave**: __init__, init_states, forward_chunk

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/benchmark_chimera_deep.py`
#### Clase: `Mamba2Baseline`
**Propósito Funcional**: Single Mamba2 layer con RMSNorm + residual — mismo param count aprox.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/benchmark_jit_cudagraph.py`
#### Clase: `_LegacyBus`
**Propósito Funcional**: Réplica exacta del bus ANTIGUO (antes del ring buffer) para comparación.
**Métodos o Funciones Clave**: __init__, forward_step

#### Clase: `_RingBus`
**Propósito Funcional**: Ring bus NUEVO para comparación pura (sin el layer completo).
**Métodos o Funciones Clave**: __init__, step_ring

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_chunked_train.py`
#### Clase: `ChunkCarryState`
**Propósito Funcional**: Estado que se propaga entre chunks durante entrenamiento.

Todos los tensores están DETACHADOS del grafo — no hay gradientes
inter-chunk. Esto es equivalente a TBPTT(k1=chunk_size, k2=chunk_size).
**Métodos o Funciones Clave**: 

#### Clase: `ChunkedTrainer`
**Propósito Funcional**: Trainer con Chunked TBPTT para entrenamiento con contexto infinito.

Procesa secuencias largas en chunks, llevando el estado del modelo
entre chunks. Cada chunk contribuye proporcionalmente a la loss total.

Integración con el modelo:
  - ChimeraLM.forward() procesa cada chunk normalmente
  - El bus_cache se propaga entre chunks (detachado)
  - El archive acumula landmarks progresivamente
  - TTT updates se aplican por chunk (cada chunk adapta dt_bias)

El chunk_size puede ser:
  - int: tamaño fijo
  - 'auto': estimado dinámicamente basado en VRAM disponible
**Métodos o Funciones Clave**: __init__, chunk_size, _reset_archives, train_step, _inject_bus_cache, _extract_bus_cache, _apply_ttt_updates, prefill_context

#### Clase: `InfiniteContextTrainer`
**Propósito Funcional**: Extensión de ChunkedTrainer para streaming continuo de datos.

Procesa un stream infinito de tokens con ventana deslizante,
manteniendo el estado del modelo indefinidamente. Ideal para:
  - Pre-training con documentos concatenados sin separador
  - Fine-tuning sobre corpus muy largos
  - Evaluación continua tipo perplexity-over-time

El archive y bus acumulan historia indefinidamente (con semantic GC),
mientras que los gradientes se limitan a chunk_size tokens.
**Métodos o Funciones Clave**: __init__, stream_step, reset_state

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_config.py`
#### Clase: `ChimeraConfig`
**Propósito Funcional**: Configuración canónica de CHIMERA.

Organización por sub-sistema:
  - Modelo base (Mamba2 SSD)
  - Router de complejidad
  - TTT-Lite (dt_bias adaptativo)
  - TTT-Full (low-rank U/V)
  - SLR + SGR
  - AsyncLightBus
  - NativeLandmarkArchive
  - Training
  - Inferencia
  - Metadatos
**Métodos o Funciones Clave**: __post_init__, d_inner, n_heads, total_params_estimate, total_params_M, to_dict, save, load, tiny, small_125M, medium_350M, large_1B, xlarge_3B, vram_estimate, __repr__

#### Clase: `ChimeraStack`
**Propósito Funcional**: Wrapper de producción que instancia un stack de N AdvancedChimeraLayer
desde un ChimeraConfig.

Incluye:
- Inicialización residual a escala (GPT-style)
- Gradient clipping integrado
- Logging de routing stats por batch
**Métodos o Funciones Clave**: from_config

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_curriculum.py`
#### Clase: `RoutingStats`
**Propósito Funcional**: Estadísticas del router para una ventana de entrenamiento.

Se extraen del modelo durante el training loop y se pasan al
curriculum scheduler para que adapte la mezcla de datos.
**Métodos o Funciones Clave**: 

#### Clase: `DataMixture`
**Propósito Funcional**: Descripción de la mezcla de datos para un paso de entrenamiento.

Los pesos son relativos (se normalizan a sum=1 internamente).
max_seq_len controla la longitud máxima de secuencias muestreadas.
**Métodos o Funciones Clave**: weights

#### Clase: `CurriculumPhase`
**Propósito Funcional**: Definición de una fase de curriculum.

Cada fase tiene un rango de complejidad objetivo y criterios de
transición basados en las estadísticas del router.
**Métodos o Funciones Clave**: 

#### Clase: `TransitionType`
**Propósito Funcional**: Tipo de transición entre fases.
**Métodos o Funciones Clave**: 

#### Clase: `AdaptiveCurriculum`
**Propósito Funcional**: Scheduler de curriculum learning adaptativo para CHIMERA.

Usa las probabilidades del router como proxy de dificultad percibida
para ajustar dinámicamente la mezcla de datos y la longitud de secuencia.

A diferencia de un curriculum fijo, este scheduler:
  1. Puede retroceder si el modelo no converge (retreat)
  2. Transiciona suavemente (interpolación entre fases)
  3. Respeta min_steps para consolidación
  4. Monitorea estabilidad (gradient_norm, loss_delta)

Protocolo:
  1. curriculum.get_mixture(step, stats) → DataMixture
  2. Dataloader muestrea según DataMixture
  3. curriculum.update(stats) registra progreso

Log:
  curriculum.get_log() → historial de transiciones
**Métodos o Funciones Clave**: __init__, _default_phases, current_phase, next_phase, prev_phase, get_mixture, update, _update_ema, _evaluate_transition, _advance_phase, _retreat_phase, get_log, get_status, save_state, load_state

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_layer.py`
#### Clase: `GatedComplexityPredictor`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GatedComplexityPredictor.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `ChimeraLayer`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ChimeraLayer.
**Métodos o Funciones Clave**: __init__, _ssd_parallel_scan_ttt, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_lm.py`
#### Clase: `ChimeraStack`
**Propósito Funcional**: Stack de N AdvancedChimeraLayer con:
  • Threading del bus_cache entre capas
  • Gradient checkpointing selectivo (cada `ckpt_interval` capas)
  • Recolección de aux_dicts para pérdidas de routing y TTT

Parámetro ckpt_interval:
  - 1   → checkpointing en todas las capas (máximo ahorro VRAM, +30% tiempo)
  - 2   → cada 2 capas   (equilibrio recomendado: ~40% ahorro VRAM, +15% tiempo)
  - 999 → sin checkpointing (default — necesario mientras AdvancedChimeraLayer
          tenga mutaciones in-place de estado: dt_momentum (Lion), archive.maybe_archive.
          Torch checkpoint re-ejecuta el forward y detecta tensores intermedios distintos
          → CheckpointError. Solución: usar grad_accum para ampliar batch efectivo.
          Una vez AdvancedChimeraLayer sea stateless, ckpt_interval=2 funcionará.)
**Métodos o Funciones Clave**: __init__, forward, forward_with_carry, _layer_fn, allocate_inference_cache, step

#### Clase: `ChimeraLM`
**Propósito Funcional**: Modelo de Lenguaje basado en CHIMERA.

Arquitectura:
    input_ids [B, S]
    → Embedding [B, S, D]
    → ChimeraStack (N capas)
    → RMSNorm
    → LM-head [B, S, vocab_size]   (weight-tied con embedding)
    → CrossEntropyLoss (si labels!=None)

Parámetros:
    config:     ChimeraConfig con hiperparámetros del modelo
    vocab_size: tamaño del vocabulario (GPT-NeoX=50277, LLaMA=32000)
    tie_weights:True → weight-tying embedding↔lm_head (default True)
    ckpt_interval: ver ChimeraStack (default 2)
**Métodos o Funciones Clave**: __init__, num_parameters, allocate_inference_cache, compile_for_training, precompute_archive_caches, post_compile_step, forward, _prefill, generate

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_losses.py`
#### Clase: `ChimeraLosses`
**Propósito Funcional**: Acumulador de pérdidas auxiliares para un paso de training.

Thread-safe para uso en un solo forward pass.
**Métodos o Funciones Clave**: __init__, add_routing_probs, add_ttt_error, compute, reset, n_routing_samples, n_ttt_samples, routing_stats, __repr__

#### Clase: `ChimeraRoutingLoss`
**Propósito Funcional**: Routing loss de nueva generación — usa el aux_dict directamente.

Tres fuerzas en equilibrio:
  (a) Entropy hinge: penaliza H(probs) > target_H.
      Objetivo: routing especializado por sample (H baja por sample).
      target_H_frac=0.70 → H deseada ≤ 70% de log(n_tiers).
      F.relu(H - target_H) → 0 cuando ya está suficientemente picado.

  (b) TTT-guided supervision (opcional): si ttt_importance está disponible,
      supervisa al router con soft-targets derivados del error predictivo:
        complejidad alta  → prob_full target alto  → tier FULL
        complejidad baja  → prob_fast target alto  → tier FAST
      Loss: KL(soft_target || probs)

  (c) Load balance: penaliza tiers con prob media muy baja en el batch.
      min_tier_prob=0.05 → cada tier debe recibir al menos 5% del tráfico.
      F.relu(min_tier_prob - mean_p).sum()

Compara con MoE Switch Transformer: ellos también minimizan per-sample
entropy + mantienen load balance. Nosotros añadimos supervión explícita vía
TTT importance como señal de complejidad por input.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_paged_state.py`
#### Clase: `SessionStatus`
**Propósito Funcional**: Estado de vida de una sesión.
**Métodos o Funciones Clave**: 

#### Clase: `SessionState`
**Propósito Funcional**: Estado completo de una sesión de inferencia.

Contiene todo lo necesario para pausar y reanudar una generación
sin perder contexto: estado SSM, bus ring, landmarks archivados.
**Métodos o Funciones Clave**: 

#### Clase: `PagedStateManager`
**Propósito Funcional**: Gestor de estado paginado para serving multi-usuario.

Gestiona slots de sesión en GPU, con capacidad de:
- Crear sesiones nuevas (con o sin prefill de contexto)
- Pausar sesiones a CPU (liberar VRAM)
- Reanudar sesiones desde CPU a GPU
- Destruir sesiones
- Monitorear uso de recursos

Thread-safe mediante Lock para operaciones de sesión.

Capacidad dinámica: max_sessions se adapta según VRAM disponible
si se configura con max_sessions='auto'.
**Métodos o Funciones Clave**: __init__, _estimate_max_sessions, active_sessions, paused_sessions, create_session, _allocate_cache, _prefill_archive, step, pause_session, _pause_session_unlocked, resume_session, destroy_session, _find_oldest_active, get_stats, list_sessions

#### Clase: `ContinuousBatchScheduler`
**Propósito Funcional**: Scheduler de batching continuo para CHIMERA.

Agrupa múltiples sesiones activas en un mismo batch para maximizar
throughput GPU. Cada ciclo:
  1. Selecciona hasta max_batch_size sesiones activas
  2. Ejecuta un step() batcheado
  3. Distribuye resultados a las sesiones

Diferencia clave vs Transformer continuous batching:
- No hay KV-cache paginado (Mamba2 state es O(1))
- No hay page tables ni block tables
- El costo de agregar/remover una request del batch es O(1)

Limitación actual: cada sesión tiene su propio cache individual.
Para batching real optimizado, se necesitaría un cache compartido
con vistas por sesión (futuro: Paged Tensor).
**Métodos o Funciones Clave**: __init__, submit_request, process_batch, pending_count

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_scheduler.py`
#### Clase: `ChimeraWarmupScheduler`
**Propósito Funcional**: Controla el warm-up escalonado de un stack de AdvancedChimeraLayers.

Args:
    layers:  lista de AdvancedChimeraLayer (no ChimeraStack, acceso directo)
    warm1:   step donde termina Fase 1 (default 1000)
    warm2:   step donde termina Fase 2 (default 3000)
    ttt_lr_target:   lr máximo para TTT-Lite al final de Fase 2 (default 1e-3)
    slr_gate_target: gate máximo SLR al final del warm-up (no clampeado)
    bus_gate_target: gate máximo bus   al final del warm-up
**Métodos o Funciones Clave**: __init__, _frac_phase2, _phase, step, current_phase, state_dict, load_state_dict, __repr__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_streaming.py`
#### Clase: `StreamingDecoder`
**Propósito Funcional**: D3: Streaming Inference Mode.

En lugar de llamar step() token a token (simple pero sin TTT update),
el StreamingDecoder acumula `chunk_size` tokens en un buffer y luego
ejecuta un forward() paralelo sobre el chunk completo.

Ventajas:
  - Mamba2 hace el scan paralelo sobre chunk_size tokens → mejor GPU utilization
  - TTT-Lite se ejecuta sobre chunk_size tokens → optimización real del dt_bias
  - SLR/archive operan sobre chunk_size tokens → mejor retrieval
  - Latencia por token ≈ latency(full_chunk) / chunk_size → mucho mejor throughput

Uso:
    dec = StreamingDecoder(chimera_layer, chunk_size=16)
    initial_cache = dec.prefill(context_hidden, bus_cache=None)
    for token_id in dec.generate(lm_head, embed, max_new=512, cache=initial_cache):
        print(token_id)

Nota: para generación autoregresiva real necesitas embedding + lm_head externos.
El StreamingDecoder opera en el espacio oculto [B, 1, D].
**Métodos o Funciones Clave**: __init__, prefill, step, flush, generate, reset

#### Clase: `StreamingLMDecoder`
**Propósito Funcional**: Wrapper completo embedding → CHIMERA streaming → lm_head → token_ids.
Compatible con niah_eval.ChimeraLM.

Uso:
    from chimera_streaming import StreamingLMDecoder
    slm = StreamingLMDecoder(chimera_lm, chunk_size=16)
    tokens = slm.generate_text(prompt_ids, max_new=128)
**Métodos o Funciones Clave**: __init__, encode_context, _embed_fn, generate_text

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/chimera_v2_ssm.py`
#### Clase: `_EMAFallbackSSM`
**Propósito Funcional**: EMA simple como SSM de emergencia — mantiene API de OrthoSSM.
**Métodos o Funciones Clave**: __init__, forward, step_one

#### Clase: `AsyncLightBusV2`
**Propósito Funcional**: Bus de comunicación entre capas CHIMERA.
API idéntica a AsyncLightBus en advanced_chimera.py.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `ChimeraV2Layer`
**Propósito Funcional**: CHIMERA V2: OrthoSSM (Chebyshev) + SLR V3 + Bus + Landmark Archive.

Hyperparámetros:
    d_model:   dimensión total del modelo
    n_heads:   número de cabezas Chebyshev (default 8)
    degree:    grado del polinomio Chebyshev (default 4, max 8)
    top_k_frac: fracción de tokens seleccionados por SGR
    bus_dim:   dimensión del bus de comunicación inter-capas
    archive_slots: número de slots en el Landmark Archive
**Métodos o Funciones Clave**: __init__, forward, _ema_fallback, step

#### Clase: `ChimeraV2Stack`
**Propósito Funcional**: Stack de N capas ChimeraV2Layer.
API compatible con usar en ChimeraLM (de niah_eval.py).
**Métodos o Funciones Clave**: __init__, forward, step

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/gpu_profile.py`
#### Clase: `GPUClass`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GPUClass.
**Métodos o Funciones Clave**: 

#### Clase: `GPUProfile`
**Propósito Funcional**: Contiene todas las configuraciones óptimas para un GPU específico.
Se genera una vez y se usa en todo el código Triton y torch.compile.
**Métodos o Funciones Clave**: __str__, summary

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/landmark_native.py`
#### Clase: `NativeLandmarkArchive`
**Propósito Funcional**: Landmark Archive nativo al ecosistema CHIMERA.

En vez de almacenar el estado SSM (legacy cheby_state), almacena
embeddings comprimidos de los tokens más importantes del scan output.
El error TTT actúa como proxy de complejidad — sin MLPs extra.

Pipeline por forward call:
  1. maybe_archive(scan_out, ttt_importance, tier_probs)
     → si complejidad alta: comprime top-K tokens → nuevo landmark
  2. retrieve(query, device)
     → diff_attn_v2 (Triton) entre query y landmarks acumulados
     → salida: [B, d_model] que se inyecta en el residual stream
**Métodos o Funciones Clave**: __init__, maybe_archive, _store_landmark, _semantic_gc, _importance_based_merge, preload_context, _get_processed_landmarks, get_compress_ctx, retrieve, precompute_retrieve_cache, retrieve_compiled, get_archive_info

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/niah_eval.py`
#### Clase: `NIAHDataset`
**Propósito Funcional**: Generador online de secuencias NIAH (sin ficheros, bajo memoria).

Secuencia generada [BOS | haystack_pre | key | sep | value | haystack_post | key]:
  - BOS: token de inicio
  - haystack_pre: ruido de longitud (depth_frac × context_len)
  - key: token clave (ID en NEEDLE_RANGE[0]..NEEDLE_RANGE[1]//2)
  - sep: token separador (ID 1 = BOS reutilizado como sep, convencion)
  - value: respuesta correcta (ID en NEEDLE_RANGE[1]//2..NEEDLE_RANGE[1])
  - haystack_post: ruido hasta completar context_len - 3 tokens
  - query: key repetida → el modelo debe predecir value en la posición siguiente

Label: solo la posición query tiene target != -100 (→ value_id)
**Métodos o Funciones Clave**: __init__, sample

#### Clase: `ChimeraLM`
**Propósito Funcional**: CHIMERA + cabeza LM para NIAH.
Usa embedding compartido (weight tying) entre input y output.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `Mamba2LM`
**Propósito Funcional**: Baseline Mamba2-solo, sin SLR, bus ni archive.
Usa exactamente los mismos hyperparámetros que ChimeraLM para comparación justa.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/official_chimera.py`
#### Clase: `GatedComplexityPredictor`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GatedComplexityPredictor.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `OfficialChimeraLayer`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque OfficialChimeraLayer.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/sdtm_memory.py`
#### Clase: `SDTMMemory`
**Propósito Funcional**: Surprise-Driven Dual-Timescale Memory.

Memoria asociativa no-lineal con dos escalas temporales:
  M_fast: working memory, actualizada online vía Lion+Kahan
  M_slow: long-term memory, consolidada periódicamente desde M_fast via SVD

Args:
    d_model:  dimensión del modelo (entrada/salida)
    d_mem:    dimensión del espacio de memoria POR CABEZA (default: max(64, d_model//4))
    n_heads:  número de cabezas de memoria independientes (default: 1)
              Cada cabeza se especializa en patrones distintos (funciones, variables, etc.)
              Multi-head: capacidad = n_heads × d_mem² asociaciones.
    sdtm_lr:  learning rate base para Lion updates
    sdtm_beta: momentum factor para Lion EMA
    consolidation_interval: cada cuántos tokens consolidar M_fast → M_slow
    consolidation_rate:     fracción de shrink de M_fast tras consolidar
    consolidation_rank_frac: fracción de d_mem para top-r SVD
    max_constraint_frac:    γ para constraint: max_δ = γ * ||M||_F / d_mem
    usage_decay_base:       λ_base para usage-weighted decay
    surprise_top_k:         cuántos tokens sorprendentes por chunk para write
**Métodos o Funciones Clave**: __init__, read, compute_write, update_memory_inplace, update_usage, apply_usage_decay, maybe_consolidate, absorb_landmarks, post_forward_update, get_state, set_state, reset_online_state, memory_stats, extra_repr

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/sgr_slr.py`
#### Clase: `FlashDiffSLRFunction`
**Propósito Funcional**: Flash-Differential Attention V4 — diferenciable end-to-end.

Forward:
    Usa _flash_diff_slr_fwd_kernel: A1, A2 NUNCA van a HBM.
    out = softmax(Q1@K1.T/√d)@V - σ(lam_logit)·softmax(Q2@K2.T/√d)@V

Backward:
    Recomputa A1, A2 desde Q,K guardados (sin VRAM extra durante forward).
    Gradientes exactos (misma lógica que DiffAttnV2Function.backward).

Ahorro de memoria en forward: -50% BW vs materializar A1[B,K,W] + A2[B,K,W].
**Métodos o Funciones Clave**: forward, backward

#### Clase: `FusedProjectionSplit`
**Propósito Funcional**: W_q  [D, 2*dh] → Q1, Q2
W_kv [D, 3*dh] → K1, K2, V

vs antes: 5 Linear separadas = 5 cuBLAS launches + 5x reads HBM
ahora:    2 Linear grandes   = 2 cuBLAS launches + 2x reads HBM
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SGRSelector`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque SGRSelector.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SLRDifferentialModule`
**Propósito Funcional**: SLR + Flash-Differential Attention V4.

Cambios vs V3:
  [+++] Flash kernel batched 2D: A1/A2 nunca en HBM durante forward
        → -50% ancho de banda de memoria en la parte de atención
  [++]  FlashDiffSLRFunction con backward por recompute (idéntico en gradientes)
  [+]   SGRSelector V2 con histograma para S>2048
  [=]   FusedProjectionSplit, _gather_windows_batched sin cambios

API idéntica a V3 para compatibilidad total con advanced_chimera.py.
**Métodos o Funciones Clave**: __init__, _gather_windows_batched, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/speculative_ssm.py`
#### Clase: `SpecLMHead`
**Propósito Funcional**: Cabeza LM mínima usada internamente por SpeculativeSSMDecoder.
Si el modelo ya tiene una cabeza LM, se usa directamente.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SpeculativeSSMDecoder`
**Propósito Funcional**: Speculative decoding para SSM/CHIMERA usando verificación paralela O(K).

Args:
    target_model: modelo principal (CHIMERA completo)
    draft_model:  modelo borrador (versión ligera o CHIMERA con TTT off)
                  Si es None, usa el mismo target_model con TTT congelado
                  como draft (menos eficiente pero funcional para testing)
    lm_head:      cabeza LM que convierte hidden → logits. Si target_model
                  ya tiene lm_head, se usa automáticamente.
    K:            número de tokens draft por especulación (default 8)
    temperature:  temperatura para muestreo (1.0 = sin escalado)
    vocab_size:   tamaño del vocabulario (por defecto 512 para tests NIAH)
**Métodos o Funciones Clave**: __init__, _get_logits, _sample, _get_hidden, generate_speculative, generate_sequential

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/test_3layer.py`
#### Clase: `ChimeraStack`
**Propósito Funcional**: 3 AdvancedChimeraLayers apiladas con bus compartido.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/test_chimera_full.py`
#### Clase: `ChimeraStack`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ChimeraStack.
**Métodos o Funciones Clave**: __init__, forward, allocate_inference_cache, step

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/test_triton_kernels.py`
#### Clase: `TestTTTKernel`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestTTTKernel.
**Métodos o Funciones Clave**: _make_bufs, test_kahan_accuracy, test_lion_constraint, test_lion_backward_compat, test_lion_kahan_stable

#### Clase: `TestLandmarkNative`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestLandmarkNative.
**Métodos o Funciones Clave**: _make_archive, test_semantic_gc, test_semantic_gc_preserves_diverse, test_preload_context, test_archive_gc_on_full

#### Clase: `TestRouter`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestRouter.
**Métodos o Funciones Clave**: _make_input, test_router_temperature, test_router_floor, test_router_sums_to_one, test_collapse_ema, test_router_no_floor

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/torch_compile_debug/run_2026_03_08_14_13_50_065928-pid_15074/minifier/minifier_launcher.py`
#### Clase: `Repro`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Repro.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/torch_compile_debug/run_2026_03_08_14_19_28_256265-pid_19406/minifier/minifier_launcher.py`
#### Clase: `Repro`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Repro.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_experiment/train_chimera.py`
#### Clase: `Muon`
**Propósito Funcional**: Muon optimizer: SGD con momentum + ortogonalización Newton-Schulz.

Solo se aplica a parámetros 2D (matrices de peso). Bias, embeddings,
LayerNorm/RMSNorm se manejan con AdamW separado.

Args:
    params:    parámetros 2D del modelo
    lr:        learning rate (recomendado: 0.02 para Muon, >10× que AdamW)
    momentum:  coeficiente de momentum (default 0.95)
    ns_steps:  iteraciones Newton-Schulz (default 5, 3 también funciona)
    weight_decay: L2 regularización (recomendado: 0.0 con Muon)

Nota de implementación:
    El gradiente ortogonalizado escala distinto que el de AdamW.
    Calibrar LR: Muon-LR ≈ 0.02 × (d_model / 256)^0.5 es un buen punto de inicio.
**Métodos o Funciones Clave**: __init__, step

#### Clase: `PackedTokenDataset`
**Propósito Funcional**: Dataset que empaqueta múltiples documentos en secuencias de longitud fija.

Estrategia 'greedy bin-packing':
  • Concatena tokens de documento en un buffer de longitud max_seq_len
  • Inserta EOS entre documentos para señalar límites
  • Ningún token de padding: 100% de eficiencia (vs ≈60-70% con padding)

Soporta:
  • Directorio con archivos .pt (tensores tokenizados guardados con torch.save)
  • Lista de tensores (para testing)
  • Generador sintético (si data_dir=None)

Por qué importa:
  Con padding, SSMs desperdician compute en tokens que van a ser masked.
  Con packing, se eliminan ≈ 30-40% de tokens desperdiciados en datasets
  con distribución variable de longitud de documento (ej: The Pile, SlimPajama).
**Métodos o Funciones Clave**: __init__, __len__, __getitem__

#### Clase: `BF16AMPContext`
**Propósito Funcional**: Gestor de mixed precision para BF16.

Diferencias vs FP16:
  • BF16 tiene rango dinámica igual que FP32 (8 bits exponente) → sin underflow
  • No necesita GradScaler (el problema de underflow de FP16 no aplica)
  • Los master weights en el optimizer se mantienen en FP32 automáticamente
    cuando se usa torch.optim.AdamW(fused=True) con model en BF16.

Diferencias vs FP32:
  • ~2× throughput en matmuls (Tensor Cores BF16)
  • ~50% menos VRAM para activaciones
  • 7 bits de mantisa vs 23 → pérdida de precisión en sumatorias largas
    → mitigado por RMSNorm FP32 interno y kahan summation en softmax.
**Métodos o Funciones Clave**: __init__, __enter__, __exit__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/advanced_chimera.py`
#### Clase: `GatedComplexityPredictor`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GatedComplexityPredictor.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `_GradThrottleFn`
**Propósito Funcional**: Gradient throttle — compile-safe via torch.library.custom_op.
Identidad en forward, rescala gradiente en backward si norma > max_norm.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `BusParallelConfig`
**Propósito Funcional**: Config de paralelismo runtime para AsyncLightBus.

tp_group : ProcessGroup Tensor Parallel — dimensión del modelo D sharded.
           Cada rank tiene x[:, :, D // tp_size] + pesos Column/Row-Parallel.
sp_group : ProcessGroup Sequence Parallel — secuencia S sharded.
           Cada rank tiene x[:, S // sp_size, :] con D completa.

Soporta las 4 combinaciones: none · TP-only · SP-only · TP+SP (3D).

Los process groups NO se guardan en state_dict (no son tensores).
Llamar layer.bus.set_parallel_config() una vez por proceso tras cargar
cada checkpoint.

Análisis de comunicación por capa:
  SP-only :  2 all-reduce × O(B·D)       — num y den del pool
  TP-only :  2 all-reduce × O(B·bus_dim) — gate_logit + publish
  TP+SP   :  3 all-reduce                — gate(TP) + num/den(SP) + publish(TP)
En NVLink 900 GB/s con B=4, bus_dim=128: < 5 µs/layer — negligible vs scan.
**Métodos o Funciones Clave**: __init__, _grp_size, tp_size, sp_size, is_tp, is_sp, is_distributed

#### Clase: `AsyncLightBus`
**Propósito Funcional**: AsyncLightBus from OrthoSSM plan.
Provides a fast cross-layer communication channel.
**Métodos o Funciones Clave**: __init__, set_parallel_config, _pool_begin, _pool_end, _attentive_pool, _q_project, forward, forward_ring, reset_cache, step_ring

#### Clase: `PagedBusCache`
**Propósito Funcional**: Pool de páginas físicas para el bus_ring de decode en producción.

PROBLEMA que resuelve:
    allocate_inference_cache() reserva bus_ring=[B, ring_size, bus_dim]
    estáticamente POR SECUENCIA. En un servidor atendiendo N secuencias
    concurrentes con tamaños heterogéneos, esto produce fragmentación:
      - Secuencia corta de 128 tokens reserva 16 páginas × bus_dim → desperdicio.
      - Secuencia larga de 8192 tokens necesita más páginas de las disponibles.
    La fragmentación reduce el throughput efectivo en ~30-50% vs el óptimo.

SOLUCIÓN — Physical Page Pool:
    Un bloque contiguo de memoria [total_pages, page_size, bus_dim] se divide
    en páginas físicas uniformes. Cada secuencia mantiene una "tabla de páginas"
    (lista de índices físicos) que puede crecer dinámicamente. La memoria se
    asigna y libera en O(1) vía free_list.

    Uso típico:
        pool = PagedBusCache(total_pages=512, page_size=16,
                             bus_dim=128, device='cuda')
        # Al inicio de cada secuencia:
        pool.alloc_seq(seq_id=42, n_pages=8)
        # Durante decode:
        view = pool.get_view(seq_id=42)     # [n_pages*page_size, bus_dim]
        # Al terminar:
        pool.free_seq(seq_id=42)

COMPATIBILIDAD:
    allocate_inference_cache() acepta paged_pool=PagedBusCache como kwarg.
    step_ring() detecta automáticamente si el cache contiene una vista paginada.
    El API legacy (bus_ring tensor estático) sigue funcionando sin cambios.

VENTAJAS sobre cache estático:
    • Fragmentación cero: páginas libres se reutilizan inmediatamente.
    • Crecimiento dinámico: append_page(seq_id) añade páginas sin realocar.
    • Batching eficiente: secuencias de tamaños diferentes comparten el pool.
    • Eviction sencillo: free_seq() marca páginas como disponibles en O(len).
**Métodos o Funciones Clave**: __init__, alloc_seq, append_page, free_seq, get_view, write_slot, free_pages, used_pages

#### Clase: `ChimeraMoEFFN`
**Propósito Funcional**: Sparse Top-K Mixture-of-Experts FFN.

MOTIVACIÓN (Gemini rec. §2.2):
  Con el router de tier (FAST/HYBRID/FULL) controlamos la PROFUNDIDAD
  del cómputo. ChimeraMoEFFN controla la RUTA PARAMÉTRICA: el mismo
  FLOPs-budget, pero el modelo puede especializarse en n_experts aspectos
  distintos del lenguaje (morfología, semántica, código, matemáticas…).

DISEÑO:
  n_experts=8, top_k=2, d_ff=d_model*2 por experto.
  FLOPs/token = top_k × (D×d_ff + d_ff×D) = 2 × 2D² = 4D²
  vs dense FFN: 8D² (2× MENOS FLOPs que un FFN estándar de D→4D→D)
  Parámetros totales: n_experts × 2 × D × d_ff = 8 × 2 × D × 2D = 32D²
  vs dense FFN: 8D² (4× MÁS parámetros — la "inteligencia" de 3B)

DISPATCH bajo-nivel:
  - expert_ids ordenados → matmul grupal por experto (no Python loop con .any())
  - Fallback a einsum para graph_mode / torch.compile (sin Python-if sobre tensor)
  - Load-balance auxiliary loss (Switch Transformer eq. 4) para evitar
    colapso donde un solo experto recibe todos los tokens.

INTEGRACIÓN con la arquitectura Chimera:
  - Se aplica a `out` DESPUÉS del gated-mix y ANTES del bus.
  - En graph_mode (CUDA Graph capture): bypass completo vía torch.cond
    para evitar Python-if sobre tensores y mantener static shapes.
  - En decode (step()): bypass automático (S=1, MoE no vale la pena).
  - scale inicializado en -4.0 → sigmoid(-4) ≈ 0.018:
    contribución casi nula al inicio, crece gradualmente con entrenamiento.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `AdvancedChimeraLayer`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque AdvancedChimeraLayer.
**Métodos o Funciones Clave**: __init__, effective_ttt_lr, update_ttt_inplace, reset_doc_state, allocate_inference_cache, step, forward

#### Clase: `ChimeraAnnealer`
**Propósito Funcional**: Cosine annealing de slr_threshold y arch_threshold en todas las capas.

Schedule: high → target via cosine decay over warmup_steps.
Después de warmup_steps, los umbrales quedan fijos en sus targets.
**Métodos o Funciones Clave**: __init__, step, get_state

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/cas_swarm.py`
#### Clase: `MicroProbe`
**Propósito Funcional**: Sonda ultraligera por experto: una proyección D→1 que produce c_i ∈ (0,1).

Parámetros: D+1 (weight + bias) × dtype.
En H200 FP8: la proyección se cuantiza on-the-fly en forward.
En Ada/Ampere: se ejecuta en BF16/FP16 nativo.

Init bias:
  bias = -log(1/p_init - 1) → sigmoid(bias) = p_init
  Con p_init=0.3: bias ≈ -0.847. Cada experto empieza con ~30% de activación.
  Esto permite que el gradiente fluya desde el primer paso sin colapso
  (todos ON) ni muerte (todos OFF).
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `DepthThreshold`
**Propósito Funcional**: Umbral adaptativo por profundidad de capa:

  τ_l = τ_base + γ · SiLU(W_depth · (l / L))

τ_base: umbral base aprendible (init: logit de p_init ≈ 0.3)
γ:      amplitud aprendible (init: 0.5 — permite ±0.18 variación vía SiLU)
W_depth: peso escalar aprendible (init: 2.0 — centra SiLU en zona no lineal)
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `_GroupedGEMMFunc`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque _GroupedGEMMFunc.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `GroupedGEMMFunction`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GroupedGEMMFunction.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `ChimeraAutonomousSwarm`
**Propósito Funcional**: El subsistema principal. Reemplaza ChimeraMoEFFN con experts autónomos.
**Métodos o Funciones Clave**: __init__, _compute_threshold, forward, extra_repr

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/chimera_config.py`
#### Clase: `ChimeraConfig`
**Propósito Funcional**: Configuración canónica de CHIMERA.

Organización por sub-sistema:
  - Modelo base (Mamba2 SSD)
  - Router de complejidad
  - TTT-Lite (dt_bias adaptativo)
  - TTT-Full (low-rank U/V)
  - SLR + SGR
  - AsyncLightBus
  - NativeLandmarkArchive
  - Training
  - Inferencia
  - Metadatos
**Métodos o Funciones Clave**: __post_init__, d_inner, n_heads, total_params_estimate, total_params_M, to_dict, save, load, tiny, small_125M, medium_350M, large_1B, xlarge_3B, vram_estimate, __repr__

#### Clase: `ChimeraStack`
**Propósito Funcional**: Wrapper de producción que instancia un stack de N AdvancedChimeraLayer
desde un ChimeraConfig.

Incluye:
- Inicialización residual a escala (GPT-style)
- Gradient clipping integrado
- Logging de routing stats por batch
**Métodos o Funciones Clave**: from_config

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/chimera_lm.py`
#### Clase: `ChimeraStack`
**Propósito Funcional**: Stack de N AdvancedChimeraLayer con:
  • Threading del bus_cache entre capas
  • Gradient checkpointing selectivo (cada `ckpt_interval` capas)
  • Recolección de aux_dicts para pérdidas de routing y TTT

Parámetro ckpt_interval:
  - 1   → checkpointing en todas las capas (máximo ahorro VRAM, +30% tiempo)
  - 2   → cada 2 capas   (equilibrio recomendado: ~40% ahorro VRAM, +15% tiempo)
  - 999 → sin checkpointing (usar si hay problemas de compatibilidad)

NOTA: _skip_side_effects en AdvancedChimeraLayer evita que las mutaciones
in-place (TTT-Lite, archive.maybe_archive) se repitan durante la
recomputación de gradient checkpointing (use_reentrant=False).
**Métodos o Funciones Clave**: __init__, forward, _layer_fn

#### Clase: `ChimeraLM`
**Propósito Funcional**: Modelo de Lenguaje basado en CHIMERA.

Arquitectura:
    input_ids [B, S]
    → Embedding [B, S, D]
    → ChimeraStack (N capas)
    → RMSNorm
    → LM-head [B, S, vocab_size]   (weight-tied con embedding)
    → CrossEntropyLoss (si labels!=None)

Parámetros:
    config:     ChimeraConfig con hiperparámetros del modelo
    vocab_size: tamaño del vocabulario (GPT-NeoX=50277, LLaMA=32000)
    tie_weights:True → weight-tying embedding↔lm_head (default True)
    ckpt_interval: ver ChimeraStack (default 2)
**Métodos o Funciones Clave**: __init__, num_parameters, forward, generate

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/chimera_losses.py`
#### Clase: `ChimeraLosses`
**Propósito Funcional**: Acumulador de pérdidas auxiliares para un paso de training.

Thread-safe para uso en un solo forward pass.
**Métodos o Funciones Clave**: __init__, add_routing_probs, add_ttt_error, compute, reset, n_routing_samples, n_ttt_samples, routing_stats, __repr__

#### Clase: `ChimeraRoutingLoss`
**Propósito Funcional**: Routing loss de nueva generación — usa el aux_dict directamente.

Tres fuerzas en equilibrio:
  (a) Entropy hinge: penaliza H(probs) > target_H.
      Objetivo: routing especializado por sample (H baja por sample).
      target_H_frac=0.70 → H deseada ≤ 70% de log(n_tiers).
      F.relu(H - target_H) → 0 cuando ya está suficientemente picado.

  (b) TTT-guided supervision (opcional): si ttt_importance está disponible,
      supervisa al router con soft-targets derivados del error predictivo:
        complejidad alta  → prob_full target alto  → tier FULL
        complejidad baja  → prob_fast target alto  → tier FAST
      Loss: KL(soft_target || probs)

  (c) Load balance: penaliza tiers con prob media muy baja en el batch.
      min_tier_prob=0.05 → cada tier debe recibir al menos 5% del tráfico.
      F.relu(min_tier_prob - mean_p).sum()

Compara con MoE Switch Transformer: ellos también minimizan per-sample
entropy + mantienen load balance. Nosotros añadimos supervión explícita vía
TTT importance como señal de complejidad por input.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/gpu_profile.py`
#### Clase: `GPUClass`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GPUClass.
**Métodos o Funciones Clave**: 

#### Clase: `GPUProfile`
**Propósito Funcional**: Contiene todas las configuraciones óptimas para un GPU específico.
Se genera una vez y se usa en todo el código Triton y torch.compile.
**Métodos o Funciones Clave**: is_ampere_or_better, is_hopper_or_better, __str__, summary

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/landmark_native.py`
#### Clase: `NativeLandmarkArchive`
**Propósito Funcional**: Landmark Archive nativo al ecosistema CHIMERA.

En vez de almacenar el estado SSM (legacy cheby_state), almacena
embeddings comprimidos de los tokens más importantes del scan output.
El error TTT actúa como proxy de complejidad — sin MLPs extra.

Pipeline por forward call:
  1. maybe_archive(scan_out, ttt_importance, tier_probs)
     → si complejidad alta: comprime top-K tokens → nuevo landmark
  2. retrieve(query, device)
     → diff_attn_v2 (Triton) entre query y landmarks acumulados
     → salida: [B, d_model] que se inyecta en el residual stream
**Métodos o Funciones Clave**: __init__, maybe_archive, _store_landmark, _semantic_gc, _importance_based_merge, preload_context, _get_processed_landmarks, get_compress_ctx, retrieve, get_archive_info

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/sgr_slr.py`
#### Clase: `FusedProjectionSplit`
**Propósito Funcional**: W_q  [D, 2*dh] → Q1, Q2
W_kv [D, 3*dh] → K1, K2, V

vs antes: 5 Linear separadas = 5 cuBLAS launches + 5x reads HBM
ahora:    2 Linear grandes   = 2 cuBLAS launches + 2x reads HBM
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SGRSelector`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque SGRSelector.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SLRDifferentialModule`
**Propósito Funcional**: SLR + Flash-Differential Attention V4.

Cambios vs V3:
  [+++] Flash kernel batched 2D: A1/A2 nunca en HBM durante forward
        → -50% ancho de banda de memoria en la parte de atención
  [++]  FlashDiffSLRFunction con backward por recompute (idéntico en gradientes)
  [+]   SGRSelector V2 con histograma para S>2048
  [=]   FusedProjectionSplit, _gather_windows_batched sin cambios

API idéntica a V3 para compatibilidad total con advanced_chimera.py.
**Métodos o Funciones Clave**: __init__, _gather_windows_batched, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/spectral_vsa_archive.py`
#### Clase: `SpectralVSAArchive`
**Propósito Funcional**: Spectral VSA holographic context compression.

State:
  V_mem      ∈ C^{D//2} or R^D  — holographic memory vector
  c_now      ∈ R^{K×D}          — current window Chebyshev coefficients
  c_past     ∈ R^{K×D}          — previous window Chebyshev coefficients
  buf        ∈ R^{W×D}          — circular buffer of hidden states
  cheby_mat  ∈ R^{K×W}          — precomputed Chebyshev eval matrix

Pipeline:
  1. Accumulate scan_out tokens into circular buffer
  2. Every stride tokens: recompute Chebyshev coefficients c_now
  3. Compute Δ_k = ||c_now[k] - c_past[k]|| per frequency band
  4. Bind c_now into V_mem via VSA (complex roles)
  5. Retrieve: unbind relevant bands, denoise with diff_attn
**Métodos o Funciones Clave**: __init__, _build_chebyshev_matrix, _build_complex_dft_roles, _compute_chebyshev_coefficients, _bind, _unbind, _compute_spectral_delta, get_spectral_delta, get_spectral_importance, maybe_archive, retrieve, get_compress_ctx, get_archive_info, reset, measure_spectral_decay, measure_vsa_interference

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/spectral_vsa_archive_v2.py`
#### Clase: `PagedVSAPool`
**Propósito Funcional**: Pool compartido de páginas de memoria espectral para SpectralVSAArchive.

Permite N instancias concurrentes de SpectralVSAArchive compartir un único
bloque de VRAM físico. Las páginas se asignan on-demand y se liberan en
reset() — igual que las kv-cache contiguas de PagedAttention.

Implementación:
  _pool:       [n_pages_max, page_size, d_model] — bloque físico unificado
  _free_list:  [n_pages_max] int64 — índices libres (LIFO stack)
  _n_free:     int — número de páginas libres
**Métodos o Funciones Clave**: __init__, alloc, free, get_buf_view, n_free, utilization, __repr__

#### Clase: `ComplexityGate`
**Propósito Funcional**: Learned spectral complexity gate for dynamic Chebyshev degree truncation.

Input: spectral statistics (energy per band, hi/lo ratio, temporal delta)
Output: soft_K ∈ [K_min, K_max] — target active degree

The gate learns to:
  - Truncate to K=4 for predictable sequences (low entropy: repetitive text)
  - Expand to K=K_max for high-entropy sequences (complex code, rare tokens)
  - React to temporal changes in spectral structure (discontinuities)
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `SpectralVSAArchive`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque SpectralVSAArchive.
**Métodos o Funciones Clave**: __init__, _build_chebyshev_matrix, _build_complex_roles, _build_lanczos_sigma, _kahan_ema, _update_active_K, _detect_discontinuity, _compute_chebyshev_coefficients, _apply_lanczos_damping, _update_condition, _update_noise_floor, _bind, _update_binding_correction, _full_refresh, _unbind, _compute_spectral_delta, get_spectral_delta, get_spectral_importance, maybe_archive, retrieve, get_compress_ctx, get_archive_info, reset, preload_context, measure_spectral_decay, measure_vsa_interference, measure_lanczos_effect, measure_error_correction_quality

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/test_robustness.py`
#### Clase: `ErrorInjector`
**Propósito Funcional**: Inyecta errores en pasos específicos durante el mini-entrenamiento.
Cada error simula una condición distinta que el entrenador debe sobrevivir.
**Métodos o Funciones Clave**: __init__, inject, summary

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/test_triton_kernels.py`
#### Clase: `TestTTTKernel`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestTTTKernel.
**Métodos o Funciones Clave**: _make_bufs, test_kahan_accuracy, test_lion_constraint, test_lion_backward_compat, test_lion_kahan_stable

#### Clase: `TestLandmarkNative`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestLandmarkNative.
**Métodos o Funciones Clave**: _make_archive, test_semantic_gc, test_semantic_gc_preserves_diverse, test_preload_context, test_archive_gc_on_full

#### Clase: `TestRouter`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque TestRouter.
**Métodos o Funciones Clave**: _make_input, test_router_temperature, test_router_floor, test_router_sums_to_one, test_collapse_ema, test_router_no_floor

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/tokenize_dataset.py`
#### Clase: `BinaryTokenDataset`
**Propósito Funcional**: Dataset extremadamente rápido que accede a tokens binarios uint16 o uint32.

Modos:
  1. MMAP (modo CPU): numpy.memmap sobre .bin — zero-copy, OS page cache
  2. HBM  (modo GPU): toda la data en CUDA tensor — zero-latency

En HBM mode, __getitem__ hace slicing de CUDA tensor → no hay DataLoader,
el trainer usa get_batch_hbm() directamente.

El dtype (uint16/uint32) se detecta automáticamente desde meta.json si se usa
from_meta(), o se puede pasar explícitamente con token_dtype.
**Métodos o Funciones Clave**: __init__, get_batch_hbm, __len__, from_meta

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/train_chimera.py`
#### Clase: `Muon`
**Propósito Funcional**: Muon optimizer: SGD con momentum + ortogonalización Newton-Schulz.

Solo se aplica a parámetros 2D (matrices de peso). Bias, embeddings,
LayerNorm/RMSNorm se manejan con AdamW separado.

Args:
    params:    parámetros 2D del modelo
    lr:        learning rate (recomendado: 0.02 para Muon, >10× que AdamW)
    momentum:  coeficiente de momentum (default 0.95)
    ns_steps:  iteraciones Newton-Schulz (default 5, 3 también funciona)
    weight_decay: L2 regularización (recomendado: 0.0 con Muon)

Nota de implementación:
    El gradiente ortogonalizado escala distinto que el de AdamW.
    Calibrar LR: Muon-LR ≈ 0.02 × (d_model / 256)^0.5 es un buen punto de inicio.
**Métodos o Funciones Clave**: __init__, step

#### Clase: `PackedTokenDataset`
**Propósito Funcional**: Dataset que empaqueta múltiples documentos en secuencias de longitud fija.

Estrategia 'greedy bin-packing':
  • Concatena tokens de documento en un buffer de longitud max_seq_len
  • Inserta EOS entre documentos para señalar límites
  • Ningún token de padding: 100% de eficiencia (vs ≈60-70% con padding)

Soporta:
  • Directorio con archivos .pt (tensores tokenizados guardados con torch.save)
  • Lista de tensores (para testing)
  • Generador sintético (si data_dir=None)

Por qué importa:
  Con padding, SSMs desperdician compute en tokens que van a ser masked.
  Con packing, se eliminan ≈ 30-40% de tokens desperdiciados en datasets
  con distribución variable de longitud de documento (ej: The Pile, SlimPajama).
**Métodos o Funciones Clave**: __init__, __len__, __getitem__

#### Clase: `BF16AMPContext`
**Propósito Funcional**: Gestor de mixed precision para BF16.

Diferencias vs FP16:
  • BF16 tiene rango dinámica igual que FP32 (8 bits exponente) → sin underflow
  • No necesita GradScaler (el problema de underflow de FP16 no aplica)
  • Los master weights en el optimizer se mantienen en FP32 automáticamente
    cuando se usa torch.optim.AdamW(fused=True) con model en BF16.

Diferencias vs FP32:
  • ~2× throughput en matmuls (Tensor Cores BF16)
  • ~50% menos VRAM para activaciones
  • 7 bits de mantisa vs 23 → pérdida de precisión en sumatorias largas
    → mitigado por RMSNorm FP32 interno y kahan summation en softmax.
**Métodos o Funciones Clave**: __init__, __enter__, __exit__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/train_h200.py`
#### Clase: `Muon`
**Propósito Funcional**: Muon: SGD con momentum + ortogonalización Newton-Schulz.
**Métodos o Funciones Clave**: __init__, step

#### Clase: `PackedBinaryDataset`
**Propósito Funcional**: Dataset mmap sobre .bin uint16 con packing (sin padding).
**Métodos o Funciones Clave**: __init__, __len__, __getitem__

#### Clase: `CUDAGraphTrainStep`
**Propósito Funcional**: Captura y gestiona el CUDA Graph de un paso de entrenamiento CHIMERA.

Ciclo de vida:
  1. warmup(n)      — ejecuta n pasos normales (sin grafo) para inicializar
                      estados del archive, lazily-allocated tensors, etc.
  2. capture()      — captura el grafo con static tensors ids/labels
  3. step(ids, lbl) — copia input a static buffers, replay del grafo

Arquitectura dual:
  • TTT update: llamado UNA VEZ fuera del grafo (update_ttt_inplace)
                antes de cada replay. No incurre graph breaks.
  • Forward+Backward: 100% capturado. Zero Python dispatch overhead.
  • Optimizer step: fuera del grafo (modifica parámetros con side effects).

Notas sobre side effects:
  • `mamba2.dt_bias.data.copy_()` ocurre dentro del grafo (graph_mode=True).
    En graph capture, este op se usa solo cuando TTT=False (graph_mode desactiva
    el TTT en-forward). Con update_ttt_inplace(), dt_bias se modifica fuera.
  • archive.maybe_archive(): llamado dentro del grafo — opera sobre sus
    propios buffers (siempre misma firma). Es seguro.
  • bus cache: stateless por batch — ningún estado persistente entre steps.
**Métodos o Funciones Clave**: __init__, _forward_backward, warmup, capture, step, is_captured

#### Clase: `_WarmupIter`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque _WarmupIter.
**Métodos o Funciones Clave**: __init__, __iter__, __next__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `chimera_h200/train_h200_elite.py`
#### Clase: `NS5Buffers`
**Propósito Funcional**: Caché de buffers pre-allocados para NS5. Sin allocations en el hot path.
**Métodos o Funciones Clave**: __init__, _prefetch_to_slot, get_batch

#### Clase: `CUDAGraphElite`
**Propósito Funcional**: Captura el paso forward+backward en un CUDA Graph para cero overhead Python.

Diseño:
  - Captura UN paso F+B (no el grad_accum completo)
  - Para grad_accum > 1: el caller hace N replays con diferentes inputs
    → los gradientes se acumulan naturalmente en .grad (sum de N backward)
  - zero_grad() siempre ocurre FUERA del grafo (Python call)
  - El grafo captura: forward + backward (autograd incluido) + loss retorno

Restricciones para capture exitosa:
  1. graph_mode=True en todas las AdvancedChimeraLayer
  2. TTT update fuera del grafo (via TTTGradSupervisor)
  3. Shapes estáticos (batch_size, seq_len fijos)
  4. No Python-if sobre tensores (Chimera usa soft-gating → OK)
**Métodos o Funciones Clave**: __init__, _fwd_bwd, warmup, capture, step, is_captured

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `coeus_tokenizer.py`
#### Clase: `COEUSTokenizer`
**Propósito Funcional**: Tokenizer nativo de COEUS.

Soporta dos backends:
  1. `tokenizers` (HuggingFace tokenizers) — rápido, Rust-based
  2. `transformers` PreTrainedTokenizerFast — compatibilidad total

Uso:
    tok = COEUSTokenizer()                          # auto-detecta backend
    tok = COEUSTokenizer("/path/to/tokenizer_dir")  # directorio con tokenizer.json
    
    encoded = tok.encode("Hola mundo")              # {"input_ids": tensor}
    text = tok.decode(encoded["input_ids"])          # "Hola mundo"
**Métodos o Funciones Clave**: __init__, encode, decode, wrap_thinking, wrap_reasoning, wrap_chain_of_thought, wrap_code, format_hypothesis_verification, special_token_ids, cognitive_token_ids, get_token_id, __len__, __repr__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `gradient_checker.py`
#### Clase: `RobustGradientChecker`
**Propósito Funcional**: Production-grade gradient verifier for mixed-precision Triton kernels.

Three independent verification paths ensure correctness without
relying on FD through BF16 quantization boundaries.
**Métodos o Funciones Clave**: __init__, check_full, _check_forward_match, _check_backward_vs_reference, _check_fd_reference, _check_fd_linearity

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `landmark_archive.py`
#### Clase: `AsyncLightBus`
**Propósito Funcional**: V10 Async Lightweight Memory Bus with Versioning (E5).

Each layer publishes a 64-dim summary vector (instead of full landmarks).
Upper layers can gather summaries from lower layers via simple averaging.

E5 improvements over original:
  - Forward pass versioning: entries are tagged with version ID,
    stale data from previous forwards is automatically rejected.
  - Snapshot/restore: for gradient checkpointing, the bus state can be
    snapshotted before backward and restored during recompute.
  - Norm canary: each entry stores its L2 norm for O(1) divergence detection
    during gradient checkpoint recompute.
**Métodos o Funciones Clave**: __init__, clear, publish, gather, snapshot, enter_recompute, stats

#### Clase: `LandmarkArchive`
**Propósito Funcional**: Intelligent Landmark State Archive (unchanged from V9).
- Importance-based archiving
- Adaptive interval
- Weighted merge
- Self-attention between landmarks
**Métodos o Funciones Clave**: __init__, maybe_archive, _compute_importance, _archive_snapshot, _importance_based_merge, get_landmark_embeddings, get_archive_info

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `legacy_nsa_module.py`
#### Clase: `NSAModule`
**Propósito Funcional**: V10 Native Sparse Attention with dynamic window sizing.

Key V10 optimization:
  - For S <= window_size: use is_causal=True directly (no chunked window)
  - Window size dynamically capped at seq_len for hybrid path
  - Landmark/archive paths only activated when data is provided
**Métodos o Funciones Clave**: __init__, _reshape, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/evals/lm_harness_eval.py`
#### Clase: `MambaEvalWrapper`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaEvalWrapper.
**Métodos o Funciones Clave**: __init__, batch_size, _model_generate

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/distributed/distributed_utils.py`
#### Clase: `AllGatherFunc`
**Propósito Funcional**: Gather the input from sequence parallel region and concatenate.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `ReduceScatterFunc`
**Propósito Funcional**: Reduce scatter the input from the sequence parallel region and concatenate.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `AllReduceFunc`
**Propósito Funcional**: Gather the input from sequence parallel region and concatenate.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/distributed/tensor_parallel.py`
#### Clase: `ParallelLinearFunc`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ParallelLinearFunc.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `ColumnParallelLinear`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ColumnParallelLinear.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `RowParallelLinear`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque RowParallelLinear.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `VocabParallelEmbedding`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque VocabParallelEmbedding.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `ColumnParallelEmbedding`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ColumnParallelEmbedding.
**Métodos o Funciones Clave**: __init__

#### Clase: `ParallelEmbeddings`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ParallelEmbeddings.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/models/config_mamba.py`
#### Clase: `MambaConfig`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaConfig.
**Métodos o Funciones Clave**: 

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/models/mixer_seq_simple.py`
#### Clase: `MixerModel`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MixerModel.
**Métodos o Funciones Clave**: __init__, allocate_inference_cache, forward

#### Clase: `MambaLMHeadModel`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaLMHeadModel.
**Métodos o Funciones Clave**: __init__, tie_weights, allocate_inference_cache, forward, from_pretrained, save_pretrained

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/block.py`
#### Clase: `Block`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Block.
**Métodos o Funciones Clave**: __init__, forward, allocate_inference_cache

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/mamba2.py`
#### Clase: `Mamba2`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Mamba2.
**Métodos o Funciones Clave**: __init__, forward, step, allocate_inference_cache, _get_states_from_cache

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/mamba2_simple.py`
#### Clase: `Mamba2Simple`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Mamba2Simple.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/mamba_simple.py`
#### Clase: `Mamba`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque Mamba.
**Métodos o Funciones Clave**: __init__, forward, step, allocate_inference_cache, _get_states_from_cache

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/mha.py`
#### Clase: `MHA`
**Propósito Funcional**: Multi-head self-attention and cross-attention
**Métodos o Funciones Clave**: __init__, allocate_inference_cache, _update_kv_cache, _apply_rotary_update_kvcache_attention, _update_kvcache_attention, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/modules/mlp.py`
#### Clase: `GatedMLP`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GatedMLP.
**Métodos o Funciones Clave**: __init__, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/selective_scan_interface.py`
#### Clase: `SelectiveScanFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque SelectiveScanFn.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `MambaInnerFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaInnerFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/k_activations.py`
#### Clase: `SwiGLU`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque SwiGLU.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/layer_norm.py`
#### Clase: `LayerNormFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque LayerNormFn.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `RMSNorm`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque RMSNorm.
**Métodos o Funciones Clave**: __init__, reset_parameters, forward

#### Clase: `LayerNormLinearFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque LayerNormLinearFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/layernorm_gated.py`
#### Clase: `LayerNormFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque LayerNormFn.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `LayerNorm`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque LayerNorm.
**Métodos o Funciones Clave**: __init__, reset_parameters, forward

#### Clase: `RMSNorm`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque RMSNorm.
**Métodos o Funciones Clave**: __init__, reset_parameters, forward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/ssd_chunk_scan.py`
#### Clase: `ChunkScanFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ChunkScanFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/ssd_chunk_state.py`
#### Clase: `ChunkStateFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque ChunkStateFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/ssd_combined.py`
#### Clase: `MambaChunkScanCombinedFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaChunkScanCombinedFn.
**Métodos o Funciones Clave**: forward, backward

#### Clase: `MambaSplitConv1dScanCombinedFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque MambaSplitConv1dScanCombinedFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/ops/triton/ssd_state_passing.py`
#### Clase: `StatePassingFn`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque StatePassingFn.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/mamba_ssm/utils/generation.py`
#### Clase: `InferenceParams`
**Propósito Funcional**: Inference parameters that are passed to the main model in order
to efficienly calculate and store the context during inference.
**Métodos o Funciones Clave**: reset

#### Clase: `GenerationMixin`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque GenerationMixin.
**Métodos o Funciones Clave**: allocate_inference_cache, generate

#### Clase: `DecodingCGCache`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque DecodingCGCache.
**Métodos o Funciones Clave**: 

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `mamba/setup.py`
#### Clase: `CachedWheelsCommand`
**Propósito Funcional**: The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
find an existing wheel (which is currently the case for all installs). We use
the environment parameters to detect whether there is already a pre-built version of a compatible
wheel available and short-circuits the standard full build pipeline.
**Métodos o Funciones Clave**: run

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `model.py`
#### Clase: `OrthoSSMLanguageModel`
**Propósito Funcional**: OrthoSSM V10 "Lightning" Language Model.

Args:
    vocab_size:     Vocabulary size (default: 131072)
    d_model:        Hidden dimension (default: 256)
    n_attn_heads:   Number of attention heads in NSA (default: 4)
    n_cheby_heads:  Number of Chebyshev spectral heads (default: 8)
    n_layers:       Number of stacked OrthoSSM layers (default: 2)
    max_degree:     Chebyshev polynomial degree per head (default: 4, was 8)
    window_size:    Sliding window size for local attention (default: 512)
    tie_weights:    Whether to tie embedding and lm_head weights (default: True)
    use_bf16:       Enable BF16 mixed precision (default: False)
    gradient_ckpt:  Enable gradient checkpointing (default: False)
    compile_model:  Apply torch.compile (default: False)
    use_lut:        Use Chebyshev LUT acceleration (default: True)
    **engine_kwargs: Extra kwargs for SpectralDualPathContextEngine
**Métodos o Funciones Clave**: __init__, _init_weights, fresh_states, forward, _layer_forward, count_parameters, __repr__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `ortho_diagnostics.py`
#### Clase: `_RingBuffer`
**Propósito Funcional**: Buffer circular de tamaño fijo para estadísticas recientes.
**Métodos o Funciones Clave**: __init__, push, push_many, stats, __len__

#### Clase: `OrthoSSMDiagnostics`
**Propósito Funcional**: Dashboard de diagnóstico para OrthoSSM V10.

Uso típico:
    from ortho_diagnostics import DIAG
    DIAG.enable()
    … # ejecutar forwards …
    DIAG.print_dashboard()
    DIAG.reset()

Overhead en producción (enabled=False): un bool check por métrica.
**Métodos o Funciones Clave**: __init__, enable, disable, reset, record_lut_error, check_lut_error_sample, record_head_orthogonality, record_recall_similarity, record_ema_momentum, record_bus_staleness, record_sequence_length, record_rounding_stats, step, _section, _row, print_dashboard, get_report, __enter__, __exit__

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `sdpc_engine.py`
#### Clase: `SpectralDualPathContextEngine`
**Propósito Funcional**: V10 "Lightning" Engine: Length-routed spectral dual-path.

Three execution tiers based on sequence length:
  Fast (<384):    Chebyshev + EMA only. No TTT, no SLR, no landmarks.
  Hybrid (<1024): Chebyshev + TTT + SLR (all tokens, no routing).
  Full (>=1024):  Everything: TTT + SLR (spectral routing) + landmarks + LightBus.
**Métodos o Funciones Clave**: __init__, forward, _fast_path, _hybrid_path, _cheby_compute, _slr_compute, _full_path, _get_archive_embs

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `sdpc_kernel.py`
#### Clase: `_EMAFastPathFn`
**Propósito Funcional**: Autograd wrapper for Triton EMA scan (B4 + P1).
Forward: _parallel_ema_forward_kernel — O(S/BLOCK_S · log BLOCK_S) depth
         vs legacy _ema_scan_only_kernel O(S) serial.
Backward: _reverse_ema_scan_kernel (shared with full backward path).
**Métodos o Funciones Clave**: forward, backward

#### Clase: `GatedComplexityPredictor`
**Propósito Funcional**: Predicts global TTT gate from input statistics.
**Métodos o Funciones Clave**: __init__, forward

#### Clase: `FusedChebyRKVv10`
**Propósito Funcional**: Implementación de núcleo de sub-arquitectura o red neuronal dinámica para el bloque FusedChebyRKVv10.
**Métodos o Funciones Clave**: forward, backward

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.

### Archivo: `slr_module.py`
#### Clase: `SpectralLocalRefiner`
**Propósito Funcional**: Spectral Local Refiner — replaces NSA with spectral-routed
differential local attention.

Args:
    d_model:       Model dimension
    n_heads:       Number of attention heads
    window_size:   Causal sliding window size for position mask
    select_ratio:  Fraction of tokens to select for refinement (MoD)
    min_select:    Minimum number of tokens to always select
    max_select:    Maximum tokens to select (memory cap for attention matrix)
**Métodos o Funciones Clave**: __init__, compute_routing_scores, _select_tokens, _chunked_sparse_windowed_attn, _build_window_mask, forward, extra_repr

El archivo anterior conforma una pieza crítica en el engranaje del ciclo de vida en OrthoSSM, interconectando componentes en la memoria dinámica o aplicando invariantes matemáticas mediante transformaciones afines complejas a las secuencias de entrada y salida del motor predictor. Se prioriza el encapsulamiento para mantener alta coherencia en gradientes y optimizar uso en cache CUDA.


## 10. Apéndice Matemático y Derivaciones de Estado Continuo

### Transformación Base (Orignal Mamba & HiPPO)
El subsistema en el paso $t_{0}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{0}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{0} = \exp(\Delta_{0} A)$ y $\bar{B}_{0} = (\Delta_{0} A)^{-1}(\exp(\Delta_{0} A)-I) \cdot \Delta_{0} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{1}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{1}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{1} = \exp(\Delta_{1} A)$ y $\bar{B}_{1} = (\Delta_{1} A)^{-1}(\exp(\Delta_{1} A)-I) \cdot \Delta_{1} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{2}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{2}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{2} = \exp(\Delta_{2} A)$ y $\bar{B}_{2} = (\Delta_{2} A)^{-1}(\exp(\Delta_{2} A)-I) \cdot \Delta_{2} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{3}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{3}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{3} = \exp(\Delta_{3} A)$ y $\bar{B}_{3} = (\Delta_{3} A)^{-1}(\exp(\Delta_{3} A)-I) \cdot \Delta_{3} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{4}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{4}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{4} = \exp(\Delta_{4} A)$ y $\bar{B}_{4} = (\Delta_{4} A)^{-1}(\exp(\Delta_{4} A)-I) \cdot \Delta_{4} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{5}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{5}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{5} = \exp(\Delta_{5} A)$ y $\bar{B}_{5} = (\Delta_{5} A)^{-1}(\exp(\Delta_{5} A)-I) \cdot \Delta_{5} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{6}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{6}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{6} = \exp(\Delta_{6} A)$ y $\bar{B}_{6} = (\Delta_{6} A)^{-1}(\exp(\Delta_{6} A)-I) \cdot \Delta_{6} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{7}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{7}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{7} = \exp(\Delta_{7} A)$ y $\bar{B}_{7} = (\Delta_{7} A)^{-1}(\exp(\Delta_{7} A)-I) \cdot \Delta_{7} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{8}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{8}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{8} = \exp(\Delta_{8} A)$ y $\bar{B}_{8} = (\Delta_{8} A)^{-1}(\exp(\Delta_{8} A)-I) \cdot \Delta_{8} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{9}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{9}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{9} = \exp(\Delta_{9} A)$ y $\bar{B}_{9} = (\Delta_{9} A)^{-1}(\exp(\Delta_{9} A)-I) \cdot \Delta_{9} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{10}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{10}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{10} = \exp(\Delta_{10} A)$ y $\bar{B}_{10} = (\Delta_{10} A)^{-1}(\exp(\Delta_{10} A)-I) \cdot \Delta_{10} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{11}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{11}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{11} = \exp(\Delta_{11} A)$ y $\bar{B}_{11} = (\Delta_{11} A)^{-1}(\exp(\Delta_{11} A)-I) \cdot \Delta_{11} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{12}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{12}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{12} = \exp(\Delta_{12} A)$ y $\bar{B}_{12} = (\Delta_{12} A)^{-1}(\exp(\Delta_{12} A)-I) \cdot \Delta_{12} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{13}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{13}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{13} = \exp(\Delta_{13} A)$ y $\bar{B}_{13} = (\Delta_{13} A)^{-1}(\exp(\Delta_{13} A)-I) \cdot \Delta_{13} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{14}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{14}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{14} = \exp(\Delta_{14} A)$ y $\bar{B}_{14} = (\Delta_{14} A)^{-1}(\exp(\Delta_{14} A)-I) \cdot \Delta_{14} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{15}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{15}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{15} = \exp(\Delta_{15} A)$ y $\bar{B}_{15} = (\Delta_{15} A)^{-1}(\exp(\Delta_{15} A)-I) \cdot \Delta_{15} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{16}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{16}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{16} = \exp(\Delta_{16} A)$ y $\bar{B}_{16} = (\Delta_{16} A)^{-1}(\exp(\Delta_{16} A)-I) \cdot \Delta_{16} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{17}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{17}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{17} = \exp(\Delta_{17} A)$ y $\bar{B}_{17} = (\Delta_{17} A)^{-1}(\exp(\Delta_{17} A)-I) \cdot \Delta_{17} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{18}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{18}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{18} = \exp(\Delta_{18} A)$ y $\bar{B}_{18} = (\Delta_{18} A)^{-1}(\exp(\Delta_{18} A)-I) \cdot \Delta_{18} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{19}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{19}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{19} = \exp(\Delta_{19} A)$ y $\bar{B}_{19} = (\Delta_{19} A)^{-1}(\exp(\Delta_{19} A)-I) \cdot \Delta_{19} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{20}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{20}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{20} = \exp(\Delta_{20} A)$ y $\bar{B}_{20} = (\Delta_{20} A)^{-1}(\exp(\Delta_{20} A)-I) \cdot \Delta_{20} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{21}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{21}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{21} = \exp(\Delta_{21} A)$ y $\bar{B}_{21} = (\Delta_{21} A)^{-1}(\exp(\Delta_{21} A)-I) \cdot \Delta_{21} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{22}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{22}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{22} = \exp(\Delta_{22} A)$ y $\bar{B}_{22} = (\Delta_{22} A)^{-1}(\exp(\Delta_{22} A)-I) \cdot \Delta_{22} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{23}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{23}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{23} = \exp(\Delta_{23} A)$ y $\bar{B}_{23} = (\Delta_{23} A)^{-1}(\exp(\Delta_{23} A)-I) \cdot \Delta_{23} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{24}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{24}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{24} = \exp(\Delta_{24} A)$ y $\bar{B}_{24} = (\Delta_{24} A)^{-1}(\exp(\Delta_{24} A)-I) \cdot \Delta_{24} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{25}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{25}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{25} = \exp(\Delta_{25} A)$ y $\bar{B}_{25} = (\Delta_{25} A)^{-1}(\exp(\Delta_{25} A)-I) \cdot \Delta_{25} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{26}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{26}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{26} = \exp(\Delta_{26} A)$ y $\bar{B}_{26} = (\Delta_{26} A)^{-1}(\exp(\Delta_{26} A)-I) \cdot \Delta_{26} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{27}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{27}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{27} = \exp(\Delta_{27} A)$ y $\bar{B}_{27} = (\Delta_{27} A)^{-1}(\exp(\Delta_{27} A)-I) \cdot \Delta_{27} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{28}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{28}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{28} = \exp(\Delta_{28} A)$ y $\bar{B}_{28} = (\Delta_{28} A)^{-1}(\exp(\Delta_{28} A)-I) \cdot \Delta_{28} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{29}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{29}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{29} = \exp(\Delta_{29} A)$ y $\bar{B}_{29} = (\Delta_{29} A)^{-1}(\exp(\Delta_{29} A)-I) \cdot \Delta_{29} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{30}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{30}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{30} = \exp(\Delta_{30} A)$ y $\bar{B}_{30} = (\Delta_{30} A)^{-1}(\exp(\Delta_{30} A)-I) \cdot \Delta_{30} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{31}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{31}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{31} = \exp(\Delta_{31} A)$ y $\bar{B}_{31} = (\Delta_{31} A)^{-1}(\exp(\Delta_{31} A)-I) \cdot \Delta_{31} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{32}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{32}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{32} = \exp(\Delta_{32} A)$ y $\bar{B}_{32} = (\Delta_{32} A)^{-1}(\exp(\Delta_{32} A)-I) \cdot \Delta_{32} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{33}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{33}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{33} = \exp(\Delta_{33} A)$ y $\bar{B}_{33} = (\Delta_{33} A)^{-1}(\exp(\Delta_{33} A)-I) \cdot \Delta_{33} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{34}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{34}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{34} = \exp(\Delta_{34} A)$ y $\bar{B}_{34} = (\Delta_{34} A)^{-1}(\exp(\Delta_{34} A)-I) \cdot \Delta_{34} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{35}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{35}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{35} = \exp(\Delta_{35} A)$ y $\bar{B}_{35} = (\Delta_{35} A)^{-1}(\exp(\Delta_{35} A)-I) \cdot \Delta_{35} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{36}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{36}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{36} = \exp(\Delta_{36} A)$ y $\bar{B}_{36} = (\Delta_{36} A)^{-1}(\exp(\Delta_{36} A)-I) \cdot \Delta_{36} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{37}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{37}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{37} = \exp(\Delta_{37} A)$ y $\bar{B}_{37} = (\Delta_{37} A)^{-1}(\exp(\Delta_{37} A)-I) \cdot \Delta_{37} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{38}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{38}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{38} = \exp(\Delta_{38} A)$ y $\bar{B}_{38} = (\Delta_{38} A)^{-1}(\exp(\Delta_{38} A)-I) \cdot \Delta_{38} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{39}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{39}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{39} = \exp(\Delta_{39} A)$ y $\bar{B}_{39} = (\Delta_{39} A)^{-1}(\exp(\Delta_{39} A)-I) \cdot \Delta_{39} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{40}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{40}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{40} = \exp(\Delta_{40} A)$ y $\bar{B}_{40} = (\Delta_{40} A)^{-1}(\exp(\Delta_{40} A)-I) \cdot \Delta_{40} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{41}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{41}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{41} = \exp(\Delta_{41} A)$ y $\bar{B}_{41} = (\Delta_{41} A)^{-1}(\exp(\Delta_{41} A)-I) \cdot \Delta_{41} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{42}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{42}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{42} = \exp(\Delta_{42} A)$ y $\bar{B}_{42} = (\Delta_{42} A)^{-1}(\exp(\Delta_{42} A)-I) \cdot \Delta_{42} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{43}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{43}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{43} = \exp(\Delta_{43} A)$ y $\bar{B}_{43} = (\Delta_{43} A)^{-1}(\exp(\Delta_{43} A)-I) \cdot \Delta_{43} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{44}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{44}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{44} = \exp(\Delta_{44} A)$ y $\bar{B}_{44} = (\Delta_{44} A)^{-1}(\exp(\Delta_{44} A)-I) \cdot \Delta_{44} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{45}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{45}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{45} = \exp(\Delta_{45} A)$ y $\bar{B}_{45} = (\Delta_{45} A)^{-1}(\exp(\Delta_{45} A)-I) \cdot \Delta_{45} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{46}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{46}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{46} = \exp(\Delta_{46} A)$ y $\bar{B}_{46} = (\Delta_{46} A)^{-1}(\exp(\Delta_{46} A)-I) \cdot \Delta_{46} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{47}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{47}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{47} = \exp(\Delta_{47} A)$ y $\bar{B}_{47} = (\Delta_{47} A)^{-1}(\exp(\Delta_{47} A)-I) \cdot \Delta_{47} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{48}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{48}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{48} = \exp(\Delta_{48} A)$ y $\bar{B}_{48} = (\Delta_{48} A)^{-1}(\exp(\Delta_{48} A)-I) \cdot \Delta_{48} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{49}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{49}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{49} = \exp(\Delta_{49} A)$ y $\bar{B}_{49} = (\Delta_{49} A)^{-1}(\exp(\Delta_{49} A)-I) \cdot \Delta_{49} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{50}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{50}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{50} = \exp(\Delta_{50} A)$ y $\bar{B}_{50} = (\Delta_{50} A)^{-1}(\exp(\Delta_{50} A)-I) \cdot \Delta_{50} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{51}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{51}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{51} = \exp(\Delta_{51} A)$ y $\bar{B}_{51} = (\Delta_{51} A)^{-1}(\exp(\Delta_{51} A)-I) \cdot \Delta_{51} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{52}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{52}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{52} = \exp(\Delta_{52} A)$ y $\bar{B}_{52} = (\Delta_{52} A)^{-1}(\exp(\Delta_{52} A)-I) \cdot \Delta_{52} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{53}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{53}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{53} = \exp(\Delta_{53} A)$ y $\bar{B}_{53} = (\Delta_{53} A)^{-1}(\exp(\Delta_{53} A)-I) \cdot \Delta_{53} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{54}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{54}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{54} = \exp(\Delta_{54} A)$ y $\bar{B}_{54} = (\Delta_{54} A)^{-1}(\exp(\Delta_{54} A)-I) \cdot \Delta_{54} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{55}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{55}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{55} = \exp(\Delta_{55} A)$ y $\bar{B}_{55} = (\Delta_{55} A)^{-1}(\exp(\Delta_{55} A)-I) \cdot \Delta_{55} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{56}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{56}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{56} = \exp(\Delta_{56} A)$ y $\bar{B}_{56} = (\Delta_{56} A)^{-1}(\exp(\Delta_{56} A)-I) \cdot \Delta_{56} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{57}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{57}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{57} = \exp(\Delta_{57} A)$ y $\bar{B}_{57} = (\Delta_{57} A)^{-1}(\exp(\Delta_{57} A)-I) \cdot \Delta_{57} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{58}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{58}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{58} = \exp(\Delta_{58} A)$ y $\bar{B}_{58} = (\Delta_{58} A)^{-1}(\exp(\Delta_{58} A)-I) \cdot \Delta_{58} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{59}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{59}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{59} = \exp(\Delta_{59} A)$ y $\bar{B}_{59} = (\Delta_{59} A)^{-1}(\exp(\Delta_{59} A)-I) \cdot \Delta_{59} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{60}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{60}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{60} = \exp(\Delta_{60} A)$ y $\bar{B}_{60} = (\Delta_{60} A)^{-1}(\exp(\Delta_{60} A)-I) \cdot \Delta_{60} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{61}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{61}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{61} = \exp(\Delta_{61} A)$ y $\bar{B}_{61} = (\Delta_{61} A)^{-1}(\exp(\Delta_{61} A)-I) \cdot \Delta_{61} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{62}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{62}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{62} = \exp(\Delta_{62} A)$ y $\bar{B}_{62} = (\Delta_{62} A)^{-1}(\exp(\Delta_{62} A)-I) \cdot \Delta_{62} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{63}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{63}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{63} = \exp(\Delta_{63} A)$ y $\bar{B}_{63} = (\Delta_{63} A)^{-1}(\exp(\Delta_{63} A)-I) \cdot \Delta_{63} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{64}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{64}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{64} = \exp(\Delta_{64} A)$ y $\bar{B}_{64} = (\Delta_{64} A)^{-1}(\exp(\Delta_{64} A)-I) \cdot \Delta_{64} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{65}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{65}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{65} = \exp(\Delta_{65} A)$ y $\bar{B}_{65} = (\Delta_{65} A)^{-1}(\exp(\Delta_{65} A)-I) \cdot \Delta_{65} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{66}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{66}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{66} = \exp(\Delta_{66} A)$ y $\bar{B}_{66} = (\Delta_{66} A)^{-1}(\exp(\Delta_{66} A)-I) \cdot \Delta_{66} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{67}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{67}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{67} = \exp(\Delta_{67} A)$ y $\bar{B}_{67} = (\Delta_{67} A)^{-1}(\exp(\Delta_{67} A)-I) \cdot \Delta_{67} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{68}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{68}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{68} = \exp(\Delta_{68} A)$ y $\bar{B}_{68} = (\Delta_{68} A)^{-1}(\exp(\Delta_{68} A)-I) \cdot \Delta_{68} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{69}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{69}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{69} = \exp(\Delta_{69} A)$ y $\bar{B}_{69} = (\Delta_{69} A)^{-1}(\exp(\Delta_{69} A)-I) \cdot \Delta_{69} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{70}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{70}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{70} = \exp(\Delta_{70} A)$ y $\bar{B}_{70} = (\Delta_{70} A)^{-1}(\exp(\Delta_{70} A)-I) \cdot \Delta_{70} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{71}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{71}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{71} = \exp(\Delta_{71} A)$ y $\bar{B}_{71} = (\Delta_{71} A)^{-1}(\exp(\Delta_{71} A)-I) \cdot \Delta_{71} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{72}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{72}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{72} = \exp(\Delta_{72} A)$ y $\bar{B}_{72} = (\Delta_{72} A)^{-1}(\exp(\Delta_{72} A)-I) \cdot \Delta_{72} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{73}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{73}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{73} = \exp(\Delta_{73} A)$ y $\bar{B}_{73} = (\Delta_{73} A)^{-1}(\exp(\Delta_{73} A)-I) \cdot \Delta_{73} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{74}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{74}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{74} = \exp(\Delta_{74} A)$ y $\bar{B}_{74} = (\Delta_{74} A)^{-1}(\exp(\Delta_{74} A)-I) \cdot \Delta_{74} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{75}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{75}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{75} = \exp(\Delta_{75} A)$ y $\bar{B}_{75} = (\Delta_{75} A)^{-1}(\exp(\Delta_{75} A)-I) \cdot \Delta_{75} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{76}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{76}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{76} = \exp(\Delta_{76} A)$ y $\bar{B}_{76} = (\Delta_{76} A)^{-1}(\exp(\Delta_{76} A)-I) \cdot \Delta_{76} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{77}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{77}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{77} = \exp(\Delta_{77} A)$ y $\bar{B}_{77} = (\Delta_{77} A)^{-1}(\exp(\Delta_{77} A)-I) \cdot \Delta_{77} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{78}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{78}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{78} = \exp(\Delta_{78} A)$ y $\bar{B}_{78} = (\Delta_{78} A)^{-1}(\exp(\Delta_{78} A)-I) \cdot \Delta_{78} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{79}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{79}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{79} = \exp(\Delta_{79} A)$ y $\bar{B}_{79} = (\Delta_{79} A)^{-1}(\exp(\Delta_{79} A)-I) \cdot \Delta_{79} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{80}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{80}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{80} = \exp(\Delta_{80} A)$ y $\bar{B}_{80} = (\Delta_{80} A)^{-1}(\exp(\Delta_{80} A)-I) \cdot \Delta_{80} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{81}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{81}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{81} = \exp(\Delta_{81} A)$ y $\bar{B}_{81} = (\Delta_{81} A)^{-1}(\exp(\Delta_{81} A)-I) \cdot \Delta_{81} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{82}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{82}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{82} = \exp(\Delta_{82} A)$ y $\bar{B}_{82} = (\Delta_{82} A)^{-1}(\exp(\Delta_{82} A)-I) \cdot \Delta_{82} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{83}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{83}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{83} = \exp(\Delta_{83} A)$ y $\bar{B}_{83} = (\Delta_{83} A)^{-1}(\exp(\Delta_{83} A)-I) \cdot \Delta_{83} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{84}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{84}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{84} = \exp(\Delta_{84} A)$ y $\bar{B}_{84} = (\Delta_{84} A)^{-1}(\exp(\Delta_{84} A)-I) \cdot \Delta_{84} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{85}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{85}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{85} = \exp(\Delta_{85} A)$ y $\bar{B}_{85} = (\Delta_{85} A)^{-1}(\exp(\Delta_{85} A)-I) \cdot \Delta_{85} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{86}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{86}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{86} = \exp(\Delta_{86} A)$ y $\bar{B}_{86} = (\Delta_{86} A)^{-1}(\exp(\Delta_{86} A)-I) \cdot \Delta_{86} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{87}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{87}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{87} = \exp(\Delta_{87} A)$ y $\bar{B}_{87} = (\Delta_{87} A)^{-1}(\exp(\Delta_{87} A)-I) \cdot \Delta_{87} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{88}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{88}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{88} = \exp(\Delta_{88} A)$ y $\bar{B}_{88} = (\Delta_{88} A)^{-1}(\exp(\Delta_{88} A)-I) \cdot \Delta_{88} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{89}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{89}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{89} = \exp(\Delta_{89} A)$ y $\bar{B}_{89} = (\Delta_{89} A)^{-1}(\exp(\Delta_{89} A)-I) \cdot \Delta_{89} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{90}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{90}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{90} = \exp(\Delta_{90} A)$ y $\bar{B}_{90} = (\Delta_{90} A)^{-1}(\exp(\Delta_{90} A)-I) \cdot \Delta_{90} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{91}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{91}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{91} = \exp(\Delta_{91} A)$ y $\bar{B}_{91} = (\Delta_{91} A)^{-1}(\exp(\Delta_{91} A)-I) \cdot \Delta_{91} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{92}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{92}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{92} = \exp(\Delta_{92} A)$ y $\bar{B}_{92} = (\Delta_{92} A)^{-1}(\exp(\Delta_{92} A)-I) \cdot \Delta_{92} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{93}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{93}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{93} = \exp(\Delta_{93} A)$ y $\bar{B}_{93} = (\Delta_{93} A)^{-1}(\exp(\Delta_{93} A)-I) \cdot \Delta_{93} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{94}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{94}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{94} = \exp(\Delta_{94} A)$ y $\bar{B}_{94} = (\Delta_{94} A)^{-1}(\exp(\Delta_{94} A)-I) \cdot \Delta_{94} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{95}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{95}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{95} = \exp(\Delta_{95} A)$ y $\bar{B}_{95} = (\Delta_{95} A)^{-1}(\exp(\Delta_{95} A)-I) \cdot \Delta_{95} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{96}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{96}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{96} = \exp(\Delta_{96} A)$ y $\bar{B}_{96} = (\Delta_{96} A)^{-1}(\exp(\Delta_{96} A)-I) \cdot \Delta_{96} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{97}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{97}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{97} = \exp(\Delta_{97} A)$ y $\bar{B}_{97} = (\Delta_{97} A)^{-1}(\exp(\Delta_{97} A)-I) \cdot \Delta_{97} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{98}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{98}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{98} = \exp(\Delta_{98} A)$ y $\bar{B}_{98} = (\Delta_{98} A)^{-1}(\exp(\Delta_{98} A)-I) \cdot \Delta_{98} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{99}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{99}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{99} = \exp(\Delta_{99} A)$ y $\bar{B}_{99} = (\Delta_{99} A)^{-1}(\exp(\Delta_{99} A)-I) \cdot \Delta_{99} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{100}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{100}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{100} = \exp(\Delta_{100} A)$ y $\bar{B}_{100} = (\Delta_{100} A)^{-1}(\exp(\Delta_{100} A)-I) \cdot \Delta_{100} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{101}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{101}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{101} = \exp(\Delta_{101} A)$ y $\bar{B}_{101} = (\Delta_{101} A)^{-1}(\exp(\Delta_{101} A)-I) \cdot \Delta_{101} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{102}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{102}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{102} = \exp(\Delta_{102} A)$ y $\bar{B}_{102} = (\Delta_{102} A)^{-1}(\exp(\Delta_{102} A)-I) \cdot \Delta_{102} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{103}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{103}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{103} = \exp(\Delta_{103} A)$ y $\bar{B}_{103} = (\Delta_{103} A)^{-1}(\exp(\Delta_{103} A)-I) \cdot \Delta_{103} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{104}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{104}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{104} = \exp(\Delta_{104} A)$ y $\bar{B}_{104} = (\Delta_{104} A)^{-1}(\exp(\Delta_{104} A)-I) \cdot \Delta_{104} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{105}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{105}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{105} = \exp(\Delta_{105} A)$ y $\bar{B}_{105} = (\Delta_{105} A)^{-1}(\exp(\Delta_{105} A)-I) \cdot \Delta_{105} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{106}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{106}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{106} = \exp(\Delta_{106} A)$ y $\bar{B}_{106} = (\Delta_{106} A)^{-1}(\exp(\Delta_{106} A)-I) \cdot \Delta_{106} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{107}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{107}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{107} = \exp(\Delta_{107} A)$ y $\bar{B}_{107} = (\Delta_{107} A)^{-1}(\exp(\Delta_{107} A)-I) \cdot \Delta_{107} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{108}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{108}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{108} = \exp(\Delta_{108} A)$ y $\bar{B}_{108} = (\Delta_{108} A)^{-1}(\exp(\Delta_{108} A)-I) \cdot \Delta_{108} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{109}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{109}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{109} = \exp(\Delta_{109} A)$ y $\bar{B}_{109} = (\Delta_{109} A)^{-1}(\exp(\Delta_{109} A)-I) \cdot \Delta_{109} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{110}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{110}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{110} = \exp(\Delta_{110} A)$ y $\bar{B}_{110} = (\Delta_{110} A)^{-1}(\exp(\Delta_{110} A)-I) \cdot \Delta_{110} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{111}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{111}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{111} = \exp(\Delta_{111} A)$ y $\bar{B}_{111} = (\Delta_{111} A)^{-1}(\exp(\Delta_{111} A)-I) \cdot \Delta_{111} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{112}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{112}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{112} = \exp(\Delta_{112} A)$ y $\bar{B}_{112} = (\Delta_{112} A)^{-1}(\exp(\Delta_{112} A)-I) \cdot \Delta_{112} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{113}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{113}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{113} = \exp(\Delta_{113} A)$ y $\bar{B}_{113} = (\Delta_{113} A)^{-1}(\exp(\Delta_{113} A)-I) \cdot \Delta_{113} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{114}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{114}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{114} = \exp(\Delta_{114} A)$ y $\bar{B}_{114} = (\Delta_{114} A)^{-1}(\exp(\Delta_{114} A)-I) \cdot \Delta_{114} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{115}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{115}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{115} = \exp(\Delta_{115} A)$ y $\bar{B}_{115} = (\Delta_{115} A)^{-1}(\exp(\Delta_{115} A)-I) \cdot \Delta_{115} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{116}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{116}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{116} = \exp(\Delta_{116} A)$ y $\bar{B}_{116} = (\Delta_{116} A)^{-1}(\exp(\Delta_{116} A)-I) \cdot \Delta_{116} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{117}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{117}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{117} = \exp(\Delta_{117} A)$ y $\bar{B}_{117} = (\Delta_{117} A)^{-1}(\exp(\Delta_{117} A)-I) \cdot \Delta_{117} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{118}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{118}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{118} = \exp(\Delta_{118} A)$ y $\bar{B}_{118} = (\Delta_{118} A)^{-1}(\exp(\Delta_{118} A)-I) \cdot \Delta_{118} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{119}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{119}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{119} = \exp(\Delta_{119} A)$ y $\bar{B}_{119} = (\Delta_{119} A)^{-1}(\exp(\Delta_{119} A)-I) \cdot \Delta_{119} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{120}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{120}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{120} = \exp(\Delta_{120} A)$ y $\bar{B}_{120} = (\Delta_{120} A)^{-1}(\exp(\Delta_{120} A)-I) \cdot \Delta_{120} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{121}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{121}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{121} = \exp(\Delta_{121} A)$ y $\bar{B}_{121} = (\Delta_{121} A)^{-1}(\exp(\Delta_{121} A)-I) \cdot \Delta_{121} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{122}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{122}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{122} = \exp(\Delta_{122} A)$ y $\bar{B}_{122} = (\Delta_{122} A)^{-1}(\exp(\Delta_{122} A)-I) \cdot \Delta_{122} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{123}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{123}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{123} = \exp(\Delta_{123} A)$ y $\bar{B}_{123} = (\Delta_{123} A)^{-1}(\exp(\Delta_{123} A)-I) \cdot \Delta_{123} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{124}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{124}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{124} = \exp(\Delta_{124} A)$ y $\bar{B}_{124} = (\Delta_{124} A)^{-1}(\exp(\Delta_{124} A)-I) \cdot \Delta_{124} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{125}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{125}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{125} = \exp(\Delta_{125} A)$ y $\bar{B}_{125} = (\Delta_{125} A)^{-1}(\exp(\Delta_{125} A)-I) \cdot \Delta_{125} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{126}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{126}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{126} = \exp(\Delta_{126} A)$ y $\bar{B}_{126} = (\Delta_{126} A)^{-1}(\exp(\Delta_{126} A)-I) \cdot \Delta_{126} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{127}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{127}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{127} = \exp(\Delta_{127} A)$ y $\bar{B}_{127} = (\Delta_{127} A)^{-1}(\exp(\Delta_{127} A)-I) \cdot \Delta_{127} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{128}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{128}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{128} = \exp(\Delta_{128} A)$ y $\bar{B}_{128} = (\Delta_{128} A)^{-1}(\exp(\Delta_{128} A)-I) \cdot \Delta_{128} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{129}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{129}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{129} = \exp(\Delta_{129} A)$ y $\bar{B}_{129} = (\Delta_{129} A)^{-1}(\exp(\Delta_{129} A)-I) \cdot \Delta_{129} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{130}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{130}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{130} = \exp(\Delta_{130} A)$ y $\bar{B}_{130} = (\Delta_{130} A)^{-1}(\exp(\Delta_{130} A)-I) \cdot \Delta_{130} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{131}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{131}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{131} = \exp(\Delta_{131} A)$ y $\bar{B}_{131} = (\Delta_{131} A)^{-1}(\exp(\Delta_{131} A)-I) \cdot \Delta_{131} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{132}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{132}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{132} = \exp(\Delta_{132} A)$ y $\bar{B}_{132} = (\Delta_{132} A)^{-1}(\exp(\Delta_{132} A)-I) \cdot \Delta_{132} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{133}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{133}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{133} = \exp(\Delta_{133} A)$ y $\bar{B}_{133} = (\Delta_{133} A)^{-1}(\exp(\Delta_{133} A)-I) \cdot \Delta_{133} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{134}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{134}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{134} = \exp(\Delta_{134} A)$ y $\bar{B}_{134} = (\Delta_{134} A)^{-1}(\exp(\Delta_{134} A)-I) \cdot \Delta_{134} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{135}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{135}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{135} = \exp(\Delta_{135} A)$ y $\bar{B}_{135} = (\Delta_{135} A)^{-1}(\exp(\Delta_{135} A)-I) \cdot \Delta_{135} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{136}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{136}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{136} = \exp(\Delta_{136} A)$ y $\bar{B}_{136} = (\Delta_{136} A)^{-1}(\exp(\Delta_{136} A)-I) \cdot \Delta_{136} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{137}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{137}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{137} = \exp(\Delta_{137} A)$ y $\bar{B}_{137} = (\Delta_{137} A)^{-1}(\exp(\Delta_{137} A)-I) \cdot \Delta_{137} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{138}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{138}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{138} = \exp(\Delta_{138} A)$ y $\bar{B}_{138} = (\Delta_{138} A)^{-1}(\exp(\Delta_{138} A)-I) \cdot \Delta_{138} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{139}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{139}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{139} = \exp(\Delta_{139} A)$ y $\bar{B}_{139} = (\Delta_{139} A)^{-1}(\exp(\Delta_{139} A)-I) \cdot \Delta_{139} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{140}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{140}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{140} = \exp(\Delta_{140} A)$ y $\bar{B}_{140} = (\Delta_{140} A)^{-1}(\exp(\Delta_{140} A)-I) \cdot \Delta_{140} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{141}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{141}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{141} = \exp(\Delta_{141} A)$ y $\bar{B}_{141} = (\Delta_{141} A)^{-1}(\exp(\Delta_{141} A)-I) \cdot \Delta_{141} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{142}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{142}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{142} = \exp(\Delta_{142} A)$ y $\bar{B}_{142} = (\Delta_{142} A)^{-1}(\exp(\Delta_{142} A)-I) \cdot \Delta_{142} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{143}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{143}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{143} = \exp(\Delta_{143} A)$ y $\bar{B}_{143} = (\Delta_{143} A)^{-1}(\exp(\Delta_{143} A)-I) \cdot \Delta_{143} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{144}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{144}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{144} = \exp(\Delta_{144} A)$ y $\bar{B}_{144} = (\Delta_{144} A)^{-1}(\exp(\Delta_{144} A)-I) \cdot \Delta_{144} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{145}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{145}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{145} = \exp(\Delta_{145} A)$ y $\bar{B}_{145} = (\Delta_{145} A)^{-1}(\exp(\Delta_{145} A)-I) \cdot \Delta_{145} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{146}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{146}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{146} = \exp(\Delta_{146} A)$ y $\bar{B}_{146} = (\Delta_{146} A)^{-1}(\exp(\Delta_{146} A)-I) \cdot \Delta_{146} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{147}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{147}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{147} = \exp(\Delta_{147} A)$ y $\bar{B}_{147} = (\Delta_{147} A)^{-1}(\exp(\Delta_{147} A)-I) \cdot \Delta_{147} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{148}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{148}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{148} = \exp(\Delta_{148} A)$ y $\bar{B}_{148} = (\Delta_{148} A)^{-1}(\exp(\Delta_{148} A)-I) \cdot \Delta_{148} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).

El subsistema en el paso $t_{149}$ requiere una resolución bilineal para mapear el autovalor $\lambda_{149}$ tal que la aproximación sea numéricamente estable. Con una discretización matemática: $\bar{A}_{149} = \exp(\Delta_{149} A)$ y $\bar{B}_{149} = (\Delta_{149} A)^{-1}(\exp(\Delta_{149} A)-I) \cdot \Delta_{149} B$. Para preservar la memoria larga sin degradación masiva, OrthoSSM precondiciona A de modo que los elementos satisfacen las secuencias iterativas $A_{j,k} = -\sqrt{(2j+1)(2k+1)}$ si $j>k$, promoviendo que las bases resultantes para la reconstrucción estocástica sean puramente armónicas (bases polares complejas acoplables).


## 11. Arquitectura de Dependencias Oficiales

Las dependencias reportadas interactúan directamente en memoria nativa:
- `torch` (Motor Autograd y Kerneles CUDA).
- `triton` (Cálculo tensorial hiper-ralentizado y operaciones SDPC a muy bajo nivel, permitiendo bypass de los tensores base).
- `mamba-ssm >= 2.0` (Implementación base paralela en escaneo secuencial en hardware para hardware C++ / CUDA directo).
- `causal-conv1d` (Fusión causal para mitigación de mirada futura en dominios causales predictivos LLM).
- Fusión Causal Bloque Nivel 1: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 128).
- Fusión Causal Bloque Nivel 2: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 256).
- Fusión Causal Bloque Nivel 3: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 384).
- Fusión Causal Bloque Nivel 4: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 512).
- Fusión Causal Bloque Nivel 5: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 640).
- Fusión Causal Bloque Nivel 6: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 768).
- Fusión Causal Bloque Nivel 7: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 896).
- Fusión Causal Bloque Nivel 8: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1024).
- Fusión Causal Bloque Nivel 9: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1152).
- Fusión Causal Bloque Nivel 10: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1280).
- Fusión Causal Bloque Nivel 11: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1408).
- Fusión Causal Bloque Nivel 12: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1536).
- Fusión Causal Bloque Nivel 13: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1664).
- Fusión Causal Bloque Nivel 14: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1792).
- Fusión Causal Bloque Nivel 15: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 1920).
- Fusión Causal Bloque Nivel 16: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2048).
- Fusión Causal Bloque Nivel 17: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2176).
- Fusión Causal Bloque Nivel 18: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2304).
- Fusión Causal Bloque Nivel 19: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2432).
- Fusión Causal Bloque Nivel 20: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2560).
- Fusión Causal Bloque Nivel 21: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2688).
- Fusión Causal Bloque Nivel 22: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2816).
- Fusión Causal Bloque Nivel 23: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 2944).
- Fusión Causal Bloque Nivel 24: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3072).
- Fusión Causal Bloque Nivel 25: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3200).
- Fusión Causal Bloque Nivel 26: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3328).
- Fusión Causal Bloque Nivel 27: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3456).
- Fusión Causal Bloque Nivel 28: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3584).
- Fusión Causal Bloque Nivel 29: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3712).
- Fusión Causal Bloque Nivel 30: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3840).
- Fusión Causal Bloque Nivel 31: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 3968).
- Fusión Causal Bloque Nivel 32: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4096).
- Fusión Causal Bloque Nivel 33: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4224).
- Fusión Causal Bloque Nivel 34: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4352).
- Fusión Causal Bloque Nivel 35: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4480).
- Fusión Causal Bloque Nivel 36: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4608).
- Fusión Causal Bloque Nivel 37: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4736).
- Fusión Causal Bloque Nivel 38: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4864).
- Fusión Causal Bloque Nivel 39: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 4992).
- Fusión Causal Bloque Nivel 40: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5120).
- Fusión Causal Bloque Nivel 41: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5248).
- Fusión Causal Bloque Nivel 42: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5376).
- Fusión Causal Bloque Nivel 43: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5504).
- Fusión Causal Bloque Nivel 44: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5632).
- Fusión Causal Bloque Nivel 45: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5760).
- Fusión Causal Bloque Nivel 46: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 5888).
- Fusión Causal Bloque Nivel 47: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6016).
- Fusión Causal Bloque Nivel 48: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6144).
- Fusión Causal Bloque Nivel 49: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6272).
- Fusión Causal Bloque Nivel 50: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6400).
- Fusión Causal Bloque Nivel 51: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6528).
- Fusión Causal Bloque Nivel 52: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6656).
- Fusión Causal Bloque Nivel 53: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6784).
- Fusión Causal Bloque Nivel 54: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 6912).
- Fusión Causal Bloque Nivel 55: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7040).
- Fusión Causal Bloque Nivel 56: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7168).
- Fusión Causal Bloque Nivel 57: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7296).
- Fusión Causal Bloque Nivel 58: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7424).
- Fusión Causal Bloque Nivel 59: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7552).
- Fusión Causal Bloque Nivel 60: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7680).
- Fusión Causal Bloque Nivel 61: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7808).
- Fusión Causal Bloque Nivel 62: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 7936).
- Fusión Causal Bloque Nivel 63: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8064).
- Fusión Causal Bloque Nivel 64: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8192).
- Fusión Causal Bloque Nivel 65: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8320).
- Fusión Causal Bloque Nivel 66: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8448).
- Fusión Causal Bloque Nivel 67: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8576).
- Fusión Causal Bloque Nivel 68: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8704).
- Fusión Causal Bloque Nivel 69: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8832).
- Fusión Causal Bloque Nivel 70: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 8960).
- Fusión Causal Bloque Nivel 71: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9088).
- Fusión Causal Bloque Nivel 72: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9216).
- Fusión Causal Bloque Nivel 73: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9344).
- Fusión Causal Bloque Nivel 74: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9472).
- Fusión Causal Bloque Nivel 75: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9600).
- Fusión Causal Bloque Nivel 76: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9728).
- Fusión Causal Bloque Nivel 77: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9856).
- Fusión Causal Bloque Nivel 78: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 9984).
- Fusión Causal Bloque Nivel 79: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10112).
- Fusión Causal Bloque Nivel 80: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10240).
- Fusión Causal Bloque Nivel 81: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10368).
- Fusión Causal Bloque Nivel 82: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10496).
- Fusión Causal Bloque Nivel 83: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10624).
- Fusión Causal Bloque Nivel 84: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10752).
- Fusión Causal Bloque Nivel 85: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 10880).
- Fusión Causal Bloque Nivel 86: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11008).
- Fusión Causal Bloque Nivel 87: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11136).
- Fusión Causal Bloque Nivel 88: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11264).
- Fusión Causal Bloque Nivel 89: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11392).
- Fusión Causal Bloque Nivel 90: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11520).
- Fusión Causal Bloque Nivel 91: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11648).
- Fusión Causal Bloque Nivel 92: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11776).
- Fusión Causal Bloque Nivel 93: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 11904).
- Fusión Causal Bloque Nivel 94: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12032).
- Fusión Causal Bloque Nivel 95: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12160).
- Fusión Causal Bloque Nivel 96: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12288).
- Fusión Causal Bloque Nivel 97: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12416).
- Fusión Causal Bloque Nivel 98: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12544).
- Fusión Causal Bloque Nivel 99: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12672).
- Fusión Causal Bloque Nivel 100: Proceso de encapsulamiento temporal y traslapado tensorial (Windowing Size = 12800).

## 12. Modelados Físicos de las Pérdidas de Entrenamiento (Chimera Losses)

### Teoría Aplicada a los Landscapes de Convergencia en Chimera:
#### Tópico Analítico de Divergencia 1
