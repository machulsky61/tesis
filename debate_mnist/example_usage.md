# 🧪 Guía de Uso: Script de Automatización de Experimentos

## Ejecutar el Script

```bash
python run_experiments.py
```

## Ejemplo de Sesión Interactiva

### 1. Banner Inicial
```
================================================================================
🧪 COMPLETE DEBATE EXPERIMENTS AUTOMATION SCRIPT 🧪
================================================================================
Script completo para:
• Entrenar modelos juez
• Ejecutar experimentos de debates
• Generar visualizaciones configurables
• Análisis de resultados
```

### 2. Fase 1: Entrenamiento de Jueces

**Pregunta:** `¿Entrenar modelos juez? (y/n) [n]:`

Si respondes **`y`**:
```
================================================================================
🎓 ENTRENAMIENTO DE MODELOS JUEZ
================================================================================

Configuraciones de juez disponibles:
  28: Resolución 28x28, 6 píxeles - ✅ Existe
  16: Resolución 16x16, 6 píxeles - ❌ No existe
  28_4px: Resolución 28x28, 4 píxeles - ❌ No existe
  16_4px: Resolución 16x16, 4 píxeles - ❌ No existe

¿Qué modelos juez quieres entrenar?

Entrenar 16 (Resolución 16x16, 6 píxeles) (y/n) [n]: y
Épocas para 16 [64]: 32

Entrenar 28_4px (Resolución 28x28, 4 píxeles) (y/n) [n]: y
Épocas para 28_4px [64]: 64

¿Entrenar un juez con configuración personalizada? (y/n) [n]: n

📋 Se entrenarán 2 modelos juez:
  - 16: Resolución 16x16, 6 píxeles (32 épocas, ~16 minutos)
  - 28_4px: Resolución 28x28, 4 píxeles (64 épocas, ~32 minutos)

⏰ Tiempo estimado total de entrenamiento: 48 minutos

¿Proceder con el entrenamiento? (y/n) [n]: y
```

### 3. Fase 2: Configuración de Visualizaciones

**Pregunta:** `¿Habilitar visualizaciones? (y/n) [n]:`

Si respondes **`y`**:
```
================================================================================
🎨 CONFIGURACIÓN DE VISUALIZACIONES
================================================================================

Tipos de visualización:
• ¿Guardar imágenes coloreadas del debate? (--save_colored_debate) (y/n) [n]: y
• ¿Guardar metadatos completos? (--save_metadata) (y/n) [n]: y

📊 EXPERIMENTOS DE VISUALIZACIÓN PERSONALIZADOS
Estos son experimentos específicos para generar visualizaciones de alta calidad.

¿Crear experimentos de visualización personalizados? (y/n) [n]: y

• Experimento de alta precisión (1 imagen, muchos rollouts) (y/n) [n]: y
Tipo de agente (greedy/mcts/mixed) [mcts]: mcts
Número de rollouts [2000]: 3000

• Experimento de comparación (misma semilla, diferentes agentes) (y/n) [n]: y
Semilla fija [42]: 123
Número de imágenes [10]: 5
Rollouts para MCTS [1000]: 1500

• Experimento de escalabilidad de rollouts (misma imagen, diferentes rollouts) (y/n) [n]: y
Semilla fija [42]: 456
Selecciona valores de rollouts:
  - 50 rollouts (y/n) [n]: y
  - 100 rollouts (y/n) [n]: y
  - 500 rollouts (y/n) [n]: y
  - 1000 rollouts (y/n) [n]: y

📋 Se crearon 7 experimentos de visualización personalizados.
```

### 4. Fase 3: Configuración Global

```
================================================================================
📋 CONFIGURACIÓN GLOBAL DE EXPERIMENTOS
================================================================================

Modelos juez disponibles: 28, 16, 28_4px
Nombre del modelo juez (28/16/28_4px) [28]: 28_4px

Usando juez '28_4px' con resolución 28x28

Número de imágenes para experimentos Greedy [1000]: 500
Número de imágenes para experimentos MCTS [500]: 300
Número de rollouts para MCTS [500]: 800
```

### 5. Selección de Experimentos

```
================================================================================
🎯 SELECCIÓN DE EXPERIMENTOS
================================================================================

1️⃣ EXPERIMENTOS SIMÉTRICOS (mismo tipo de agente)

¿Ejecutar experimentos Greedy vs Greedy? (y/n) [n]: y
  - Greedy baseline (sin precommit, liar primero) (y/n) [n]: y
  - Greedy con precommit (y/n) [n]: y
  - Greedy con honest primero (y/n) [n]: n
  - Greedy con precommit + honest primero (y/n) [n]: n

¿Ejecutar experimentos MCTS vs MCTS? (y/n) [n]: y
  - MCTS baseline (y/n) [n]: y
  - MCTS con precommit (y/n) [n]: y

2️⃣ EXPERIMENTOS ASIMÉTRICOS (MCTS vs Greedy)

¿Ejecutar experimentos MCTS Honesto vs Greedy Mentiroso? (y/n) [n]: y
  - MCTS honest vs Greedy liar baseline (y/n) [n]: y
  - MCTS honest vs Greedy liar con precommit (y/n) [n]: y

¿Ejecutar experimentos Greedy Honesto vs MCTS Mentiroso? (y/n) [n]: n

3️⃣ EXPERIMENTOS ESPECIALES

¿Ejecutar experimentos con diferentes números de rollouts MCTS? (y/n) [n]: y
  - MCTS con 100 rollouts (y/n) [n]: y
  - MCTS con 200 rollouts (y/n) [n]: y
  - MCTS con 1000 rollouts (y/n) [n]: n

4️⃣ EXPERIMENTOS CON VISUALIZACIÓN

¿Crear muestras visuales de cada tipo de agente? (y/n) [n]: y
  - Visualización greedy (y/n) [n]: y
  - Visualización mcts (y/n) [n]: y
  - Visualización mixed_mcts_honest (y/n) [n]: y
```

### 6. Resumen Final

```
================================================================================
📊 RESUMEN DE EXPERIMENTOS SELECCIONADOS
================================================================================
Total de experimentos: 15
Tiempo estimado total: 2.5 horas

Experimentos a ejecutar:
 1. Greedy vs Greedy - baseline
 2. Greedy vs Greedy - precommit
 3. MCTS vs MCTS - baseline
 4. MCTS vs MCTS - precommit
 5. MCTS Honest vs Greedy Liar - baseline
 6. MCTS Honest vs Greedy Liar - precommit
 7. MCTS con 100 rollouts
 8. MCTS con 200 rollouts
 9. Visualización greedy
10. Visualización mcts
11. Visualización mixed_mcts_honest
12. Alta precisión MCTS 3000 rollouts
13. Comparación Greedy seed123
14. Comparación MCTS seed123
15. Escalabilidad MCTS 50 rollouts

⏰ Tiempo estimado de ejecución: 2.5 horas

¿Proceder con la ejecución de todos los experimentos? (y/n) [n]: y
```

### 7. Ejecución

```
================================================================================
🚀 INICIANDO EJECUCIÓN DE EXPERIMENTOS
================================================================================

🔄 Progreso: 1/15 (6.7%)
============================================================
🚀 Ejecutando: Greedy vs Greedy - baseline
Comando: python run_debate.py --judge_name 28_4px --resolution 28 --agent_type greedy --n_images 500 --note greedy_baseline
============================================================
✅ Completado en 45 segundos

🔄 Progreso: 2/15 (13.3%)
...
```

### 8. Resultados Finales

```
================================================================================
🏁 EXPERIMENTOS COMPLETADOS
================================================================================
✅ Exitosos: 15
❌ Fallidos: 0
⏱️  Tiempo total: 2.3 horas
📊 Resultados guardados en:
   - outputs/debates.csv (debates simétricos)
   - outputs/debates_asimetricos.csv (debates asimétricos)
   - outputs/debate_*/ (visualizaciones)

🎉 ¡Experimentos completados! Revisa los archivos CSV para analizar los resultados.
```

## Casos de Uso Específicos

### Solo Entrenar Jueces
```bash
python run_experiments.py
# Responder 'y' solo a entrenamiento, 'n' a todo lo demás
```

### Solo Visualizaciones de Alta Precisión
```bash
python run_experiments.py
# 'n' a entrenamiento
# 'y' a visualizaciones → 'y' a experimentos personalizados → 'y' a alta precisión
# 'n' a experimentos normales
```

### Experimentos Rápidos para Testing
```bash
python run_experiments.py
# Usar números pequeños de imágenes (10-50)
# Solo experimentos Greedy (más rápidos)
```

## Archivos de Salida

- **`outputs/debates.csv`**: Debates simétricos (mismo tipo de agente)
- **`outputs/debates_asimetricos.csv`**: Debates mixtos (MCTS vs Greedy)
- **`outputs/debate_*/`**: Carpetas con visualizaciones coloreadas
- **`outputs/judges.csv`**: Registro de entrenamientos de jueces
