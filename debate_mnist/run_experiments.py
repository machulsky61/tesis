#!/usr/bin/env python3
"""
Script interactivo completo para experimentos de debates automatizados.
Incluye entrenamiento de jueces, experimentos y visualizaciones configurables.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def print_banner():
    print("=" * 80)
    print("🧪 COMPLETE DEBATE EXPERIMENTS AUTOMATION SCRIPT 🧪")
    print("=" * 80)
    print("Script completo para:")
    print("• Entrenar modelos juez")
    print("• Ejecutar experimentos de debates")
    print("• Generar visualizaciones configurables")
    print("• Análisis de resultados\n")

def get_user_input(prompt, default=None, options=None):
    """Obtiene input del usuario con validación."""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if options and user_input.lower() not in [opt.lower() for opt in options]:
            print(f"Opción inválida. Opciones disponibles: {', '.join(options)}")
            continue
            
        return user_input

def get_yes_no(prompt, default="n"):
    """Obtiene respuesta sí/no del usuario."""
    response = get_user_input(f"{prompt} (y/n)", default, ["y", "n", "yes", "no"])
    return response.lower() in ["y", "yes"]

def estimate_time(experiment_type, n_images, rollouts=None):
    """Estima el tiempo de ejecución de un experimento."""
    if experiment_type == "greedy":
        return n_images * 0.1  # ~0.1 segundos por imagen
    elif experiment_type == "mcts":
        base_time = n_images * 0.5  # tiempo base
        rollout_factor = (rollouts or 500) / 500
        return base_time * rollout_factor
    elif experiment_type == "mixed":
        return n_images * 0.3  # tiempo intermedio
    return n_images * 0.2

def format_time(seconds):
    """Formatea tiempo en horas, minutos y segundos."""
    if seconds < 60:
        return f"{seconds:.0f} segundos"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutos"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def check_judge_exists(judge_name):
    """Verifica si existe un modelo juez."""
    model_path = f"models/{judge_name}.pth"
    return os.path.exists(model_path)

def train_judge_models():
    """Maneja el entrenamiento de modelos juez."""
    print("\n" + "="*80)
    print("🎓 ENTRENAMIENTO DE MODELOS JUEZ")
    print("="*80)
    
    judges_to_train = []
    
    # Configuraciones estándar
    standard_configs = [
        ("28", 28, 6, "Resolución 28x28, 6 píxeles"),
        ("16", 16, 6, "Resolución 16x16, 6 píxeles"), 
        ("28_4px", 28, 4, "Resolución 28x28, 4 píxeles"),
        ("16_4px", 16, 4, "Resolución 16x16, 4 píxeles"),
    ]
    
    print("Configuraciones de juez disponibles:")
    for judge_name, resolution, k, description in standard_configs:
        exists = "✅ Existe" if check_judge_exists(judge_name) else "❌ No existe"
        print(f"  {judge_name}: {description} - {exists}")
    
    print("\n¿Qué modelos juez quieres entrenar?")
    
    for judge_name, resolution, k, description in standard_configs:
        if not check_judge_exists(judge_name):
            if get_yes_no(f"Entrenar {judge_name} ({description})"):
                epochs = int(get_user_input(f"Épocas para {judge_name}", "64"))
                judges_to_train.append((judge_name, resolution, k, epochs, description))
        else:
            if get_yes_no(f"Re-entrenar {judge_name} ({description}) [Ya existe]"):
                epochs = int(get_user_input(f"Épocas para {judge_name}", "64"))
                judges_to_train.append((judge_name, resolution, k, epochs, description))
    
    # Opción para configuración personalizada
    if get_yes_no("¿Entrenar un juez con configuración personalizada?"):
        custom_name = get_user_input("Nombre del juez personalizado")
        custom_resolution = int(get_user_input("Resolución", "28"))
        custom_k = int(get_user_input("Número de píxeles", "6"))
        custom_epochs = int(get_user_input("Épocas", "64"))
        judges_to_train.append((custom_name, custom_resolution, custom_k, custom_epochs, "Personalizado"))
    
    if not judges_to_train:
        print("No se seleccionaron jueces para entrenar.")
        return []
    
    print(f"\n📋 Se entrenarán {len(judges_to_train)} modelos juez:")
    total_training_time = 0
    
    for judge_name, resolution, k, epochs, description in judges_to_train:
        training_time = epochs * 30  # Estimación: 30 segundos por época
        total_training_time += training_time
        print(f"  - {judge_name}: {description} ({epochs} épocas, ~{format_time(training_time)})")
    
    print(f"\n⏰ Tiempo estimado total de entrenamiento: {format_time(total_training_time)}")
    
    if get_yes_no("¿Proceder con el entrenamiento?"):
        trained_judges = []
        for judge_name, resolution, k, epochs, description in judges_to_train:
            cmd = f"python train_judge.py --judge_name {judge_name} --resolution {resolution} --k {k} --epochs {epochs}"
            desc = f"Entrenando juez {judge_name} ({description})"
            
            if run_command(cmd, desc):
                trained_judges.append(judge_name)
            else:
                print(f"❌ Error entrenando {judge_name}")
                if not get_yes_no("¿Continuar con los entrenamientos restantes?"):
                    break
        
        return trained_judges
    
    return []

def configure_visualizations():
    """Configura opciones detalladas de visualización."""
    print("\n" + "="*80)
    print("🎨 CONFIGURACIÓN DE VISUALIZACIONES")
    print("="*80)
    
    viz_config = {
        'enabled': False,
        'save_colored_debate': False,
        'save_metadata': False,
        'custom_experiments': []
    }
    
    if not get_yes_no("¿Habilitar visualizaciones?"):
        return viz_config
    
    viz_config['enabled'] = True
    
    print("\nTipos de visualización:")
    viz_config['save_colored_debate'] = get_yes_no("• ¿Guardar imágenes coloreadas del debate? (--save_colored_debate)")
    viz_config['save_metadata'] = get_yes_no("• ¿Guardar metadatos completos? (--save_metadata)")
    
    # Experimentos de visualización personalizados
    print("\n📊 EXPERIMENTOS DE VISUALIZACIÓN PERSONALIZADOS")
    print("Estos son experimentos específicos para generar visualizaciones de alta calidad.")
    
    if get_yes_no("¿Crear experimentos de visualización personalizados?"):
        
        # Experimento de alta precisión (pocas imágenes, muchos rollouts)
        if get_yes_no("• Experimento de alta precisión (1 imagen, muchos rollouts)"):
            agent_type = get_user_input("Tipo de agente (greedy/mcts/mixed)", "mcts", ["greedy", "mcts", "mixed"])
            
            if agent_type == "mcts":
                rollouts = int(get_user_input("Número de rollouts", "2000"))
                cmd = f"--agent_type mcts --rollouts {rollouts} --n_images 1"
                note = f"high_precision_mcts_{rollouts}_rollouts"
            elif agent_type == "mixed":
                honest_type = get_user_input("Agente honesto (greedy/mcts)", "mcts", ["greedy", "mcts"])
                rollouts = int(get_user_input("Número de rollouts", "2000"))
                cmd = f"--mixed_agents --honest_agent {honest_type} --rollouts {rollouts} --n_images 1"
                note = f"high_precision_mixed_{honest_type}_{rollouts}_rollouts"
            else:  # greedy
                cmd = f"--agent_type greedy --n_images 1"
                note = "high_precision_greedy"
            
            viz_config['custom_experiments'].append((cmd, note, "Alta precisión"))
        
        # Experimento de comparación directa
        if get_yes_no("• Experimento de comparación (misma semilla, diferentes agentes)"):
            seed = int(get_user_input("Semilla fija", "42"))
            n_images = int(get_user_input("Número de imágenes", "10"))
            rollouts = int(get_user_input("Rollouts para MCTS", "1000"))
            
            # Agregar múltiples experimentos con la misma semilla
            experiments = [
                (f"--agent_type greedy --seed {seed} --n_images {n_images}", f"comparison_greedy_seed{seed}", "Comparación Greedy"),
                (f"--agent_type mcts --rollouts {rollouts} --seed {seed} --n_images {n_images}", f"comparison_mcts_seed{seed}", "Comparación MCTS"),
                (f"--mixed_agents --honest_agent mcts --rollouts {rollouts} --seed {seed} --n_images {n_images}", f"comparison_mixed_mcts_honest_seed{seed}", "Comparación Mixed MCTS Honest"),
                (f"--mixed_agents --honest_agent greedy --rollouts {rollouts} --seed {seed} --n_images {n_images}", f"comparison_mixed_greedy_honest_seed{seed}", "Comparación Mixed Greedy Honest")
            ]
            
            for cmd, note, desc in experiments:
                viz_config['custom_experiments'].append((cmd, note, desc))
        
        # Experimento de escalabilidad de rollouts
        if get_yes_no("• Experimento de escalabilidad de rollouts (misma imagen, diferentes rollouts)"):
            seed = int(get_user_input("Semilla fija", "42"))
            rollout_values = [50, 100, 200, 500, 1000, 2000]
            
            print("Selecciona valores de rollouts:")
            selected_rollouts = []
            for rollouts in rollout_values:
                if get_yes_no(f"  - {rollouts} rollouts"):
                    selected_rollouts.append(rollouts)
            
            for rollouts in selected_rollouts:
                cmd = f"--agent_type mcts --rollouts {rollouts} --seed {seed} --n_images 3"
                note = f"rollout_scaling_{rollouts}"
                desc = f"Escalabilidad MCTS {rollouts} rollouts"
                viz_config['custom_experiments'].append((cmd, note, desc))
        
        # Experimento de análisis de precommit
        if get_yes_no("• Experimento de análisis de precommit"):
            n_images = int(get_user_input("Número de imágenes", "20"))
            
            precommit_experiments = [
                (f"--agent_type greedy --n_images {n_images}", f"precommit_analysis_greedy_no", "Greedy sin precommit"),
                (f"--agent_type greedy --precommit --n_images {n_images}", f"precommit_analysis_greedy_yes", "Greedy con precommit"),
                (f"--mixed_agents --honest_agent mcts --rollouts 500 --n_images {n_images}", f"precommit_analysis_mixed_no", "Mixed sin precommit"),
                (f"--mixed_agents --honest_agent mcts --rollouts 500 --precommit --n_images {n_images}", f"precommit_analysis_mixed_yes", "Mixed con precommit")
            ]
            
            for cmd, note, desc in precommit_experiments:
                viz_config['custom_experiments'].append((cmd, note, desc))
    
    if viz_config['custom_experiments']:
        print(f"\n📋 Se crearon {len(viz_config['custom_experiments'])} experimentos de visualización personalizados.")
    
    return viz_config

def run_command(cmd, description):
    """Ejecuta un comando y muestra el progreso."""
    print(f"\n{'='*60}")
    print(f"🚀 Ejecutando: {description}")
    print(f"Comando: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Completado en {format_time(elapsed)}")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Error después de {format_time(elapsed)}: {e}")
        return False

def main():
    print_banner()
    
    # === FASE 1: ENTRENAMIENTO DE JUECES ===
    trained_judges = []
    if get_yes_no("¿Entrenar modelos juez?"):
        trained_judges = train_judge_models()
        if trained_judges:
            print(f"✅ Se entrenaron exitosamente {len(trained_judges)} modelos juez: {', '.join(trained_judges)}")
        else:
            print("⚠️ No se entrenó ningún modelo juez.")
    
    # === FASE 2: CONFIGURACIÓN DE VISUALIZACIONES ===
    viz_config = configure_visualizations()
    
    # === FASE 3: CONFIGURACIÓN GLOBAL DE EXPERIMENTOS ===
    print("\n" + "="*80)
    print("📋 CONFIGURACIÓN GLOBAL DE EXPERIMENTOS")
    print("="*80)
    
    # Seleccionar juez para experimentos
    available_judges = []
    for judge_candidate in ["28", "16", "28_4px", "16_4px"] + trained_judges:
        if check_judge_exists(judge_candidate):
            available_judges.append(judge_candidate)
    
    if not available_judges:
        print("❌ No hay modelos juez disponibles. Ejecuta primero el entrenamiento.")
        return
    
    print(f"Modelos juez disponibles: {', '.join(available_judges)}")
    judge_name = get_user_input(f"Nombre del modelo juez ({'/'.join(available_judges)})", available_judges[0])
    
    if judge_name not in available_judges:
        print(f"❌ Modelo juez '{judge_name}' no encontrado.")
        return
    
    # Determinar resolución automáticamente
    if "16" in judge_name:
        resolution = 16
    else:
        resolution = 28
    
    print(f"Usando juez '{judge_name}' con resolución {resolution}x{resolution}")
    
    # Parámetros por defecto
    default_images_greedy = int(get_user_input("Número de imágenes para experimentos Greedy", "1000"))
    default_images_mcts = int(get_user_input("Número de imágenes para experimentos MCTS", "500"))
    default_rollouts = int(get_user_input("Número de rollouts para MCTS", "500"))
    
    print("\n" + "="*80)
    print("🎯 SELECCIÓN DE EXPERIMENTOS")
    print("="*80)
    
    experiments = []
    total_estimated_time = 0
    
    # === 1. EXPERIMENTOS SIMÉTRICOS ===
    print("\n1️⃣ EXPERIMENTOS SIMÉTRICOS (mismo tipo de agente)")
    
    if get_yes_no("¿Ejecutar experimentos Greedy vs Greedy?"):
        greedy_variants = []
        if get_yes_no("  - Greedy baseline (sin precommit, liar primero)"):
            greedy_variants.append("baseline")
        if get_yes_no("  - Greedy con precommit"):
            greedy_variants.append("precommit")
        if get_yes_no("  - Greedy con honest primero"):
            greedy_variants.append("honest_first")
        if get_yes_no("  - Greedy con precommit + honest primero"):
            greedy_variants.append("precommit_honest_first")
        
        for variant in greedy_variants:
            cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type greedy --n_images {default_images_greedy}"
            description = f"Greedy vs Greedy - {variant}"
            note = f"greedy_{variant}"
            
            if "precommit" in variant:
                cmd += " --precommit"
            if "honest_first" in variant:
                cmd += " --starts honest"
            
            cmd += f" --note {note}"
            
            experiments.append((cmd, description))
            total_estimated_time += estimate_time("greedy", default_images_greedy)
    
    if get_yes_no("¿Ejecutar experimentos MCTS vs MCTS?"):
        mcts_variants = []
        if get_yes_no("  - MCTS baseline"):
            mcts_variants.append("baseline")
        if get_yes_no("  - MCTS con precommit"):
            mcts_variants.append("precommit")
        if get_yes_no("  - MCTS con honest primero"):
            mcts_variants.append("honest_first")
        if get_yes_no("  - MCTS con precommit + honest primero"):
            mcts_variants.append("precommit_honest_first")
        
        for variant in mcts_variants:
            cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type mcts --rollouts {default_rollouts} --n_images {default_images_mcts}"
            description = f"MCTS vs MCTS - {variant}"
            note = f"mcts_{variant}"
            
            if "precommit" in variant:
                cmd += " --precommit"
            if "honest_first" in variant:
                cmd += " --starts honest"
            
            cmd += f" --note {note}"
            
            experiments.append((cmd, description))
            total_estimated_time += estimate_time("mcts", default_images_mcts, default_rollouts)
    
    # === 2. EXPERIMENTOS ASIMÉTRICOS ===
    print("\n2️⃣ EXPERIMENTOS ASIMÉTRICOS (MCTS vs Greedy)")
    
    if get_yes_no("¿Ejecutar experimentos MCTS Honesto vs Greedy Mentiroso?"):
        async_variants = []
        if get_yes_no("  - MCTS honest vs Greedy liar baseline"):
            async_variants.append("baseline")
        if get_yes_no("  - MCTS honest vs Greedy liar con precommit"):
            async_variants.append("precommit")
        if get_yes_no("  - MCTS honest vs Greedy liar (honest primero)"):
            async_variants.append("honest_first")
        if get_yes_no("  - MCTS honest vs Greedy liar (precommit + honest primero)"):
            async_variants.append("precommit_honest_first")
        
        for variant in async_variants:
            cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --mixed_agents --honest_agent mcts --rollouts {default_rollouts} --n_images {default_images_mcts}"
            description = f"MCTS Honest vs Greedy Liar - {variant}"
            note = f"mcts_honest_vs_greedy_liar_{variant}"
            
            if "precommit" in variant:
                cmd += " --precommit"
            if "honest_first" in variant:
                cmd += " --starts honest"
            
            cmd += f" --note {note}"
            
            experiments.append((cmd, description))
            total_estimated_time += estimate_time("mixed", default_images_mcts)
    
    if get_yes_no("¿Ejecutar experimentos Greedy Honesto vs MCTS Mentiroso?"):
        async_variants_2 = []
        if get_yes_no("  - Greedy honest vs MCTS liar baseline"):
            async_variants_2.append("baseline")
        if get_yes_no("  - Greedy honest vs MCTS liar con precommit"):
            async_variants_2.append("precommit")
        if get_yes_no("  - Greedy honest vs MCTS liar (honest primero)"):
            async_variants_2.append("honest_first")
        if get_yes_no("  - Greedy honest vs MCTS liar (precommit + honest primero)"):
            async_variants_2.append("precommit_honest_first")
        
        for variant in async_variants_2:
            cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --mixed_agents --honest_agent greedy --rollouts {default_rollouts} --n_images {default_images_mcts}"
            description = f"Greedy Honest vs MCTS Liar - {variant}"
            note = f"greedy_honest_vs_mcts_liar_{variant}"
            
            if "precommit" in variant:
                cmd += " --precommit"
            if "honest_first" in variant:
                cmd += " --starts honest"
            
            cmd += f" --note {note}"
            
            experiments.append((cmd, description))
            total_estimated_time += estimate_time("mixed", default_images_mcts)
    
    # === 3. EXPERIMENTOS ESPECIALES ===
    print("\n3️⃣ EXPERIMENTOS ESPECIALES")
    
    if get_yes_no("¿Ejecutar experimentos con diferentes números de rollouts MCTS?"):
        rollout_values = [100, 200, 1000]
        for rollouts in rollout_values:
            if get_yes_no(f"  - MCTS con {rollouts} rollouts"):
                n_imgs = max(100, default_images_mcts // 2)  # Menos imágenes para rollouts altos
                cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type mcts --rollouts {rollouts} --n_images {n_imgs} --note mcts_{rollouts}_rollouts"
                description = f"MCTS con {rollouts} rollouts"
                experiments.append((cmd, description))
                total_estimated_time += estimate_time("mcts", n_imgs, rollouts)
    
    if get_yes_no("¿Ejecutar experimentos con diferentes semillas (reproducibilidad)?"):
        seeds = [42, 123, 456, 789]
        base_config = get_user_input("Configuración base para semillas (greedy/mcts/mixed)", "greedy", ["greedy", "mcts", "mixed"])
        
        for seed in seeds:
            if get_yes_no(f"  - Semilla {seed}"):
                if base_config == "greedy":
                    cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type greedy --n_images 500 --seed {seed} --note greedy_seed{seed}"
                    description = f"Greedy con semilla {seed}"
                    total_estimated_time += estimate_time("greedy", 500)
                elif base_config == "mcts":
                    cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type mcts --rollouts {default_rollouts} --n_images 300 --seed {seed} --note mcts_seed{seed}"
                    description = f"MCTS con semilla {seed}"
                    total_estimated_time += estimate_time("mcts", 300, default_rollouts)
                else:  # mixed
                    cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --mixed_agents --honest_agent mcts --rollouts {default_rollouts} --n_images 300 --seed {seed} --note mixed_seed{seed}"
                    description = f"Mixed con semilla {seed}"
                    total_estimated_time += estimate_time("mixed", 300)
                
                experiments.append((cmd, description))
    
    # === 4. EXPERIMENTOS VISUALES ===
    if viz_config['enabled']:
        print("\n4️⃣ EXPERIMENTOS CON VISUALIZACIÓN")
        
        # Agregar flags de visualización según configuración
        viz_flags = ""
        if viz_config['save_colored_debate']:
            viz_flags += " --save_colored_debate"
        if viz_config['save_metadata']:
            viz_flags += " --save_metadata"
        
        # Experimentos de visualización estándar
        if get_yes_no("¿Crear muestras visuales de cada tipo de agente?"):
            visual_experiments = [
                ("greedy", f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type greedy --n_images 50{viz_flags} --note greedy_visual"),
                ("mcts", f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --agent_type mcts --rollouts 200 --n_images 20{viz_flags} --note mcts_visual"),
                ("mixed_mcts_honest", f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --mixed_agents --honest_agent mcts --rollouts 200 --n_images 30{viz_flags} --note mixed_mcts_honest_visual"),
                ("mixed_greedy_honest", f"python run_debate.py --judge_name {judge_name} --resolution {resolution} --mixed_agents --honest_agent greedy --rollouts 200 --n_images 30{viz_flags} --note mixed_greedy_honest_visual")
            ]
            
            for exp_type, cmd_template in visual_experiments:
                if get_yes_no(f"  - Visualización {exp_type}"):
                    description = f"Visualización {exp_type}"
                    experiments.append((cmd_template, description))
                    total_estimated_time += estimate_time("greedy", 30)  # Estimación conservadora
        
        # Experimentos de visualización personalizados
        for cmd, note, description in viz_config['custom_experiments']:
            full_cmd = f"python run_debate.py --judge_name {judge_name} --resolution {resolution} {cmd}{viz_flags} --note {note}"
            experiments.append((full_cmd, description))
            
            # Estimar tiempo basado en el tipo de experimento
            if "mcts" in cmd:
                rollouts_match = [int(s) for s in cmd.split() if s.isdigit()]
                rollouts = rollouts_match[0] if rollouts_match else 500
                n_images = 1 if "n_images 1" in cmd else 10
                total_estimated_time += estimate_time("mcts", n_images, rollouts)
            elif "mixed" in cmd:
                n_images = 1 if "n_images 1" in cmd else 10
                total_estimated_time += estimate_time("mixed", n_images)
            else:
                n_images = 1 if "n_images 1" in cmd else 10
                total_estimated_time += estimate_time("greedy", n_images)
    
    # === RESUMEN Y CONFIRMACIÓN ===
    print("\n" + "="*80)
    print("📊 RESUMEN DE EXPERIMENTOS SELECCIONADOS")
    print("="*80)
    print(f"Total de experimentos: {len(experiments)}")
    print(f"Tiempo estimado total: {format_time(total_estimated_time)}")
    print("\nExperimentos a ejecutar:")
    
    for i, (cmd, desc) in enumerate(experiments, 1):
        print(f"{i:2d}. {desc}")
    
    if not experiments:
        print("❌ No se seleccionaron experimentos para ejecutar.")
        return
    
    print(f"\n⏰ Tiempo estimado de ejecución: {format_time(total_estimated_time)}")
    
    if not get_yes_no("\n¿Proceder con la ejecución de todos los experimentos?"):
        print("❌ Ejecución cancelada por el usuario.")
        return
    
    # === EJECUCIÓN ===
    print("\n" + "="*80)
    print("🚀 INICIANDO EJECUCIÓN DE EXPERIMENTOS")
    print("="*80)
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, (cmd, description) in enumerate(experiments, 1):
        print(f"\n🔄 Progreso: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
        
        if run_command(cmd, description):
            successful += 1
        else:
            failed += 1
            if not get_yes_no("❓ ¿Continuar con los experimentos restantes?"):
                break
    
    # === RESULTADOS FINALES ===
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("🏁 EXPERIMENTOS COMPLETADOS")
    print("="*80)
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    print(f"⏱️  Tiempo total: {format_time(total_time)}")
    print(f"📊 Resultados guardados en:")
    print(f"   - outputs/debates.csv (debates simétricos)")
    print(f"   - outputs/debates_asimetricos.csv (debates asimétricos)")
    if viz_config['enabled']:
        print(f"   - outputs/debate_*/ (visualizaciones)")
    
    print(f"\n🎉 ¡Experimentos completados! Revisa los archivos CSV para analizar los resultados.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n❌ Ejecución interrumpida por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Error inesperado: {e}")
        sys.exit(1)