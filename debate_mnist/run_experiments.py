#!/usr/bin/env python3
"""
Script interactivo mejorado para experimentos de debates automatizados.
Versión 2.0 con configuración dinámica y más opciones.
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from collections import defaultdict

class ExperimentManager:
    def __init__(self):
        self.experiments = []
        self.available_judges = []
        self.config = {
            'judge_name': None,
            'resolution': None,
            'k': 6,  # píxeles por defecto
            'default_images_greedy': 1000,
            'default_images_mcts': 500,
            'default_rollouts': 500,
            'thr': 0.0,
            'viz_enabled': False,
            'viz_colored': False,
            'viz_metadata': False
        }
        
    def print_banner(self):
        print("=" * 80)
        print("🧪 AI SAFETY DEBATE - EXPERIMENT AUTOMATION v2.0 🧪")
        print("=" * 80)
        print("Sistema mejorado con:")
        print("• Configuración dinámica de píxeles (k)")
        print("• Rollouts personalizables")
        print("• Gestión avanzada de experimentos")
        print("• Templates y configuraciones guardables\n")

    def get_input(self, prompt, default=None, options=None, input_type=str):
        """Obtiene input del usuario con validación mejorada."""
        while True:
            if default is not None:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if options and user_input.lower() not in [opt.lower() for opt in options]:
                print(f"❌ Opción inválida. Opciones: {', '.join(options)}")
                continue
            
            try:
                if input_type == int:
                    return int(user_input)
                elif input_type == float:
                    return float(user_input)
                return user_input
            except ValueError:
                print(f"❌ Por favor ingresa un {input_type.__name__} válido")

    def get_yes_no(self, prompt, default="n"):
        """Obtiene respuesta sí/no del usuario."""
        response = self.get_input(f"{prompt} (y/n)", default, ["y", "n", "yes", "no"])
        return response.lower() in ["y", "yes"]

    def check_judge_exists(self, judge_name):
        """Verifica si existe un modelo juez."""
        return os.path.exists(f"models/{judge_name}.pth")

    def detect_judge_params(self, judge_name):
        """Detecta parámetros del juez basándose en el nombre."""
        # Resolución
        if "16" in judge_name:
            resolution = 16
        else:
            resolution = 28
        
        # Píxeles (k)
        if "_4px" in judge_name:
            k = 4
        elif "_3px" in judge_name:
            k = 3
        elif "_5px" in judge_name:
            k = 5
        elif "_8px" in judge_name:
            k = 8
        elif "_10px" in judge_name:
            k = 10
        else:
            k = 6  # default
            
        return resolution, k

    def scan_available_judges(self):
        """Escanea todos los jueces disponibles en models/"""
        self.available_judges = []
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith(".pth"):
                    judge_name = file[:-4]  # remover .pth
                    self.available_judges.append(judge_name)
        
        # Ordenar: primero los estándar, luego los custom
        standard = ["28", "16", "28_4px", "16_4px"]
        self.available_judges.sort(key=lambda x: (x not in standard, x))

    def show_main_menu(self):
        """Muestra el menú principal mejorado."""
        while True:
            print("\n" + "="*80)
            print("📋 MENÚ PRINCIPAL")
            print("="*80)
            print("1. 🎓 Entrenar nuevos modelos juez")
            print("2. 🔬 Configurar y ejecutar experimentos")
            print("3. 📊 Ver resultados anteriores")
            print("4. 💾 Cargar configuración guardada")
            print("5. 📝 Crear template de experimentos")
            print("6. ❌ Salir")
            
            choice = self.get_input("Selecciona una opción", "2", ["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                self.train_judges()
            elif choice == "2":
                self.configure_experiments()
            elif choice == "3":
                self.show_previous_results()
            elif choice == "4":
                self.load_configuration()
            elif choice == "5":
                self.create_template()
            elif choice == "6":
                print("👋 ¡Hasta luego!")
                sys.exit(0)

    def train_judges(self):
        """Maneja el entrenamiento de modelos juez con configuración mejorada."""
        print("\n" + "="*80)
        print("🎓 ENTRENAMIENTO DE MODELOS JUEZ")
        print("="*80)
        
        # Configuraciones estándar expandidas
        standard_configs = [
            ("28", 28, 6, "Resolución 28x28, 6 píxeles"),
            ("16", 16, 6, "Resolución 16x16, 6 píxeles"),
            ("28_4px", 28, 4, "Resolución 28x28, 4 píxeles"),
            ("16_4px", 16, 4, "Resolución 16x16, 4 píxeles"),
            ("28_3px", 28, 3, "Resolución 28x28, 3 píxeles"),
            ("28_5px", 28, 5, "Resolución 28x28, 5 píxeles"),
            ("28_8px", 28, 8, "Resolución 28x28, 8 píxeles"),
            ("16_3px", 16, 3, "Resolución 16x16, 3 píxeles"),
            ("16_8px", 16, 8, "Resolución 16x16, 8 píxeles"),
        ]
        
        print("\n📊 Estado de modelos juez:")
        print("-" * 60)
        for i, (judge_name, resolution, k, description) in enumerate(standard_configs, 1):
            status = "✅" if self.check_judge_exists(judge_name) else "❌"
            print(f"{i:2d}. {status} {judge_name:<12} - {description}")
        
        print(f"\n{len(standard_configs)+1}. 🔧 Configuración personalizada")
        print(f"{len(standard_configs)+2}. ↩️  Volver al menú principal")
        
        # Selección múltiple
        judges_to_train = []
        while True:
            choice = self.get_input("\nSelecciona opción (o 'done' para terminar)", 
                                  str(len(standard_configs)+2))
            
            if choice.lower() == 'done' or choice == str(len(standard_configs)+2):
                break
                
            if choice == str(len(standard_configs)+1):
                # Configuración personalizada
                custom_name = self.get_input("Nombre del juez personalizado")
                custom_resolution = self.get_input("Resolución", "28", input_type=int)
                custom_k = self.get_input("Número de píxeles (k)", "6", input_type=int)
                custom_epochs = self.get_input("Épocas", "64", input_type=int)
                judges_to_train.append((custom_name, custom_resolution, custom_k, 
                                      custom_epochs, "Personalizado"))
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(standard_configs):
                        judge_name, resolution, k, description = standard_configs[idx]
                        epochs = self.get_input(f"Épocas para {judge_name}", "64", input_type=int)
                        judges_to_train.append((judge_name, resolution, k, epochs, description))
                except:
                    print("❌ Opción inválida")
        
        if judges_to_train:
            self._execute_training(judges_to_train)

    def _execute_training(self, judges_to_train):
        """Ejecuta el entrenamiento de los jueces seleccionados."""
        print(f"\n📋 Se entrenarán {len(judges_to_train)} modelos:")
        for judge_name, _, _, epochs, desc in judges_to_train:
            print(f"  • {judge_name}: {desc} ({epochs} épocas)")
        
        if self.get_yes_no("\n¿Proceder con el entrenamiento?"):
            for judge_name, resolution, k, epochs, description in judges_to_train:
                cmd = f"python train_judge.py --judge_name {judge_name} --resolution {resolution} --k {k} --epochs {epochs}"
                if self.run_command(cmd, f"Entrenando {judge_name}"):
                    print(f"✅ {judge_name} entrenado exitosamente")
                    self.available_judges.append(judge_name)
                else:
                    print(f"❌ Error entrenando {judge_name}")
                    if not self.get_yes_no("¿Continuar con los demás?"):
                        break

    def configure_experiments(self):
        """Configuración principal de experimentos con menú mejorado."""
        print("\n" + "="*80)
        print("🔬 CONFIGURACIÓN DE EXPERIMENTOS")
        print("="*80)
        
        # Escanear jueces disponibles
        self.scan_available_judges()
        
        if not self.available_judges:
            print("❌ No hay modelos juez disponibles. Entrena uno primero.")
            return
        
        # Seleccionar juez
        print("\n📊 Modelos juez disponibles:")
        for i, judge in enumerate(self.available_judges, 1):
            res, k = self.detect_judge_params(judge)
            print(f"{i:2d}. {judge} (res: {res}x{res}, k: {k})")
        
        idx = self.get_input("Selecciona juez (número)", "1", input_type=int) - 1
        self.config['judge_name'] = self.available_judges[idx]
        self.config['resolution'], self.config['k'] = self.detect_judge_params(self.config['judge_name'])
        
        # Permitir override de k
        if self.get_yes_no(f"\n¿Usar k={self.config['k']} detectado del juez?", "y"):
            pass
        else:
            self.config['k'] = self.get_input("Número de píxeles (k)", "6", input_type=int)
        
        # Configurar parámetros globales
        print("\n⚙️  PARÁMETROS GLOBALES")
        print("-" * 40)
        self.config['thr'] = self.get_input("Threshold (thr)", "0.0", input_type=float)
        self.config['default_images_greedy'] = self.get_input(
            "Imágenes por defecto para Greedy", "1000", input_type=int)
        self.config['default_images_mcts'] = self.get_input(
            "Imágenes por defecto para MCTS", "500", input_type=int)
        self.config['default_rollouts'] = self.get_input(
            "Rollouts por defecto para MCTS", "500", input_type=int)
        
        # Configurar visualizaciones
        self._configure_visualizations()
        
        # Menú de experimentos
        self._experiment_selection_menu()

    def _configure_visualizations(self):
        """Configura las opciones de visualización."""
        print("\n🎨 CONFIGURACIÓN DE VISUALIZACIONES")
        print("-" * 40)
        
        self.config['viz_enabled'] = self.get_yes_no("¿Habilitar visualizaciones?")
        if self.config['viz_enabled']:
            self.config['viz_colored'] = self.get_yes_no(
                "  • ¿Guardar debates coloreados?")
            self.config['viz_metadata'] = self.get_yes_no(
                "  • ¿Guardar metadatos completos?")

    def _experiment_selection_menu(self):
        """Menú mejorado para selección de experimentos."""
        while True:
            print("\n" + "="*80)
            print("📋 SELECCIÓN DE EXPERIMENTOS")
            print("="*80)
            print(f"Juez: {self.config['judge_name']} | k: {self.config['k']} | ")
            print(f"Experimentos en cola: {len(self.experiments)}")
            print("-" * 80)
            
            print("1. 🔄 Experimentos Simétricos (mismo tipo de agente)")
            print("2. ⚔️  Experimentos Asimétricos (MCTS vs Greedy)")
            print("3. 🎯 Experimentos de Ablación (variar k)")
            print("4. 📈 Curvas de Escalabilidad (rollouts)")
            print("5. 🎲 Análisis de Robustez (múltiples semillas)")
            print("6. 🔬 Experimentos Personalizados")
            print("7. 📊 Ver cola de experimentos")
            print("8. 🚀 Ejecutar experimentos")
            print("9. 💾 Guardar configuración")
            print("0. ↩️  Volver al menú principal")
            
            choice = self.get_input("Selecciona opción", "8")
            
            if choice == "1":
                self._add_symmetric_experiments()
            elif choice == "2":
                self._add_asymmetric_experiments()
            elif choice == "3":
                self._add_ablation_experiments()
            elif choice == "4":
                self._add_scalability_experiments()
            elif choice == "5":
                self._add_robustness_experiments()
            elif choice == "6":
                self._add_custom_experiments()
            elif choice == "7":
                self._show_experiment_queue()
            elif choice == "8":
                self._execute_experiments()
                break
            elif choice == "9":
                self._save_configuration()
            elif choice == "0":
                break

    def _add_symmetric_experiments(self):
        """Añade experimentos simétricos a la cola."""
        print("\n🔄 EXPERIMENTOS SIMÉTRICOS")
        print("-" * 40)
        
        agent_type = self.get_input("Tipo de agente (greedy/mcts)", "greedy", ["greedy", "mcts"])
        
        variants = []
        if self.get_yes_no("• Baseline (liar primero, sin precommit)"):
            variants.append(("baseline", "", ""))
        if self.get_yes_no("• Con precommit"):
            variants.append(("precommit", " --precommit", "precommit"))
        if self.get_yes_no("• Honest primero"):
            variants.append(("honest_first", " --starts honest", "honest_first"))
        if self.get_yes_no("• Precommit + Honest primero"):
            variants.append(("precommit_honest", " --precommit --starts honest", "precommit_honest_first"))
        
        if not variants:
            return
        
        # Configuración específica
        n_images = self.get_input("Número de imágenes", 
                                self.config[f'default_images_{agent_type}'], input_type=int)
        
        rollouts = ""
        if agent_type == "mcts":
            r = self.get_input("Rollouts", str(self.config['default_rollouts']), input_type=int)
            rollouts = f" --rollouts {r}"
            
        for variant_name, flags, note in variants:
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {self.config['k']} "
                   f"--thr {self.config['thr']} --agent_type {agent_type} "
                   f"--n_images {n_images}{rollouts}{flags} "
                   f"--note {agent_type}_{note}")
            
            desc = f"{agent_type.upper()} vs {agent_type.upper()} - {variant_name}"
            self.experiments.append((cmd, desc))
            
        print(f"✅ Añadidos {len(variants)} experimentos simétricos")

    def _add_asymmetric_experiments(self):
        """Añade experimentos asimétricos a la cola."""
        print("\n⚔️ EXPERIMENTOS ASIMÉTRICOS")
        print("-" * 40)
        
        honest_agent = self.get_input("Agente honesto (greedy/mcts)", "mcts", ["greedy", "mcts"])
        liar_agent = "greedy" if honest_agent == "mcts" else "mcts"
        
        variants = []
        if self.get_yes_no("• Baseline"):
            variants.append(("baseline", "", ""))
        if self.get_yes_no("• Con precommit"):
            variants.append(("precommit", " --precommit", "precommit"))
        if self.get_yes_no("• Honest primero"):
            variants.append(("honest_first", " --starts honest", "honest_first"))
        if self.get_yes_no("• Precommit + Honest primero"):
            variants.append(("precommit_honest", " --precommit --starts honest", "precommit_honest_first"))
        
        if not variants:
            return
            
        n_images = self.get_input("Número de imágenes", 
                                str(self.config['default_images_mcts']), input_type=int)
        rollouts = self.get_input("Rollouts para MCTS", 
                                str(self.config['default_rollouts']), input_type=int)
        
        for variant_name, flags, note in variants:
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {self.config['k']} "
                   f"--thr {self.config['thr']} --mixed_agents --honest_agent {honest_agent} "
                   f"--rollouts {rollouts} --n_images {n_images}{flags} "
                   f"--note {honest_agent}_honest_vs_{liar_agent}_liar_{note}")
            
            desc = f"{honest_agent.upper()} Honest vs {liar_agent.upper()} Liar - {variant_name}"
            self.experiments.append((cmd, desc))
            
        print(f"✅ Añadidos {len(variants)} experimentos asimétricos")

    def _add_ablation_experiments(self):
        """Añade experimentos de ablación variando k."""
        print("\n🎯 EXPERIMENTOS DE ABLACIÓN (VARIAR K)")
        print("-" * 40)
        
        k_values = []
        print("Selecciona valores de k a probar:")
        for k in [3, 4, 5, 6, 7, 8, 10, 12]:
            if self.get_yes_no(f"  • k = {k}"):
                k_values.append(k)
        
        if not k_values:
            return
            
        agent_type = self.get_input("Tipo de agente (greedy/mcts/mixed)", "greedy", 
                                  ["greedy", "mcts", "mixed"])
        n_images = self.get_input("Imágenes por valor de k", "500", input_type=int)
        
        if agent_type in ["mcts", "mixed"]:
            rollouts = self.get_input("Rollouts", "200", input_type=int)
        
        precommit = self.get_yes_no("¿Usar precommit?")
        
        for k in k_values:
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {k} "
                   f"--thr {self.config['thr']} --n_images {n_images}")
            
            if agent_type == "mixed":
                honest = self.get_input("Agente honesto para mixed (greedy/mcts)", 
                                      "mcts", ["greedy", "mcts"])
                cmd += f" --mixed_agents --honest_agent {honest} --rollouts {rollouts}"
                desc = f"Ablación k={k} - Mixed ({honest} honest)"
            elif agent_type == "mcts":
                cmd += f" --agent_type mcts --rollouts {rollouts}"
                desc = f"Ablación k={k} - MCTS"
            else:
                cmd += f" --agent_type greedy"
                desc = f"Ablación k={k} - Greedy"
            
            if precommit:
                cmd += " --precommit"
                desc += " con precommit"
                
            cmd += f" --note ablation_k{k}_{agent_type}"
            self.experiments.append((cmd, desc))
            
        print(f"✅ Añadidos {len(k_values)} experimentos de ablación")

    def _add_scalability_experiments(self):
        """Añade experimentos de escalabilidad de rollouts."""
        print("\n📈 CURVAS DE ESCALABILIDAD (ROLLOUTS)")
        print("-" * 40)
        
        print("Configurar serie de rollouts:")
        print("1. Serie logarítmica (50, 100, 200, 500, 1000, 2000, 5000)")
        print("2. Serie lineal (100, 200, 300, ..., 1000)")
        print("3. Serie personalizada")
        
        choice = self.get_input("Selecciona", "1", ["1", "2", "3"])
        
        if choice == "1":
            rollout_values = [50, 100, 200, 500, 1000, 2000, 5000]
        elif choice == "2":
            rollout_values = list(range(100, 1100, 100))
        else:
            rollout_values = []
            print("Ingresa valores de rollouts (escribe 'done' para terminar):")
            while True:
                val = input("Rollouts: ").strip()
                if val.lower() == 'done':
                    break
                try:
                    rollout_values.append(int(val))
                except:
                    print("❌ Valor inválido")
        
        # Filtrar según recursos computacionales
        if self.get_yes_no("¿Filtrar valores altos (>2000) por costo computacional?", "y"):
            original_len = len(rollout_values)
            rollout_values = [r for r in rollout_values if r <= 2000]
            if original_len > len(rollout_values):
                print(f"⚠️  Filtrados {original_len - len(rollout_values)} valores > 2000")
        
        n_images = self.get_input("Imágenes por punto", "100", input_type=int)
        fixed_seed = self.get_yes_no("¿Usar semilla fija para comparación?", "y")
        if fixed_seed:
            seed = self.get_input("Semilla", "42", input_type=int)
        
        use_mixed = self.get_yes_no("¿Incluir agentes mixtos además de MCTS puro?")
        
        for rollouts in rollout_values:
            # MCTS puro
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {self.config['k']} "
                   f"--thr {self.config['thr']} --agent_type mcts --rollouts {rollouts} "
                   f"--n_images {n_images}")
            
            if fixed_seed:
                cmd += f" --seed {seed}"
                
            cmd += f" --note scalability_mcts_r{rollouts}"
            self.experiments.append((cmd, f"Escalabilidad MCTS - {rollouts} rollouts"))
            
            # Mixed si se solicita
            if use_mixed:
                cmd_mixed = cmd.replace("--agent_type mcts", "--mixed_agents --honest_agent mcts")
                cmd_mixed = cmd_mixed.replace("scalability_mcts", "scalability_mixed")
                self.experiments.append((cmd_mixed, f"Escalabilidad Mixed - {rollouts} rollouts"))
                
        print(f"✅ Añadidos {len(rollout_values) * (2 if use_mixed else 1)} experimentos")

    def _add_robustness_experiments(self):
        """Añade experimentos de robustez con múltiples semillas."""
        print("\n🎲 ANÁLISIS DE ROBUSTEZ (MÚLTIPLES SEMILLAS)")
        print("-" * 40)
        
        seeds = []
        print("Configurar semillas:")
        print("1. Serie estándar (42, 123, 456, 789, 1337)")
        print("2. Serie personalizada")
        
        choice = self.get_input("Selecciona", "1", ["1", "2"])
        if choice == "1":
            seeds = [42, 123, 456, 789, 1337]
        else:
            n_seeds = self.get_input("Número de semillas", "5", input_type=int)
            for i in range(n_seeds):
                seed = self.get_input(f"Semilla {i+1}", str(42 + i*100), input_type=int)
                seeds.append(seed)
        
        # Configuración base
        agent_type = self.get_input("Tipo de agente (greedy/mcts/mixed)", "greedy", 
                                  ["greedy", "mcts", "mixed"])
        n_images = self.get_input("Imágenes por semilla", "200", input_type=int)
        
        base_cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                    f"--resolution {self.config['resolution']} --k {self.config['k']} "
                    f"--thr {self.config['thr']} --n_images {n_images}")
        
        if agent_type == "greedy":
            base_cmd += " --agent_type greedy"
        elif agent_type == "mcts":
            rollouts = self.get_input("Rollouts", "500", input_type=int)
            base_cmd += f" --agent_type mcts --rollouts {rollouts}"
        else:  # mixed
            honest = self.get_input("Agente honesto", "mcts", ["greedy", "mcts"])
            rollouts = self.get_input("Rollouts", "500", input_type=int)
            base_cmd += f" --mixed_agents --honest_agent {honest} --rollouts {rollouts}"
        
        for seed in seeds:
            cmd = base_cmd + f" --seed {seed} --note robustness_{agent_type}_s{seed}"
            self.experiments.append((cmd, f"Robustez {agent_type} - Semilla {seed}"))
            
        print(f"✅ Añadidos {len(seeds)} experimentos de robustez")

    def _add_custom_experiments(self):
        """Añade experimentos completamente personalizados."""
        print("\n🔬 EXPERIMENTO PERSONALIZADO")
        print("-" * 40)
        
        # Construir comando paso a paso
        cmd = f"python run_debate.py --judge_name {self.config['judge_name']}"
        cmd += f" --resolution {self.config['resolution']}"
        
        # k personalizado
        k = self.get_input("Píxeles (k)", str(self.config['k']), input_type=int)
        cmd += f" --k {k}"
        
        # threshold
        thr = self.get_input("Threshold", str(self.config['thr']), input_type=float)
        cmd += f" --thr {thr}"
        
        # Tipo de agente
        print("\nTipo de debate:")
        print("1. Greedy vs Greedy")
        print("2. MCTS vs MCTS")
        print("3. Mixed (diferentes tipos)")
        
        debate_type = self.get_input("Selecciona", "1", ["1", "2", "3"])
        
        if debate_type == "1":
            cmd += " --agent_type greedy"
            agent_desc = "Greedy"
        elif debate_type == "2":
            rollouts = self.get_input("Rollouts", "500", input_type=int)
            cmd += f" --agent_type mcts --rollouts {rollouts}"
            agent_desc = f"MCTS ({rollouts}r)"
        else:
            honest = self.get_input("Agente honesto (greedy/mcts)", "mcts", ["greedy", "mcts"])
            rollouts = self.get_input("Rollouts para MCTS", "500", input_type=int)
            cmd += f" --mixed_agents --honest_agent {honest} --rollouts {rollouts}"
            agent_desc = f"Mixed ({honest} honest)"
        
        # Otros parámetros
        n_images = self.get_input("Número de imágenes", "100", input_type=int)
        cmd += f" --n_images {n_images}"
        
        if self.get_yes_no("¿Usar precommit?"):
            cmd += " --precommit"
            
        if self.get_yes_no("¿Honest empieza primero?"):
            cmd += " --starts honest"
        
        seed = self.get_input("Semilla (dejar vacío para default)", "")
        if seed:
            cmd += f" --seed {seed}"
            
        note = self.get_input("Nota descriptiva", f"custom_{agent_desc}")
        cmd += f" --note {note}"
        
        # Visualizaciones
        if self.config['viz_enabled']:
            if self.config['viz_colored']:
                cmd += " --save_colored_debate"
            if self.config['viz_metadata']:
                cmd += " --save_metadata"
        
        desc = self.get_input("Descripción del experimento", f"Custom - {agent_desc}")
        
        self.experiments.append((cmd, desc))
        print("✅ Experimento personalizado añadido")

    def _show_experiment_queue(self):
        """Muestra y permite gestionar la cola de experimentos."""
        if not self.experiments:
            print("\n❌ No hay experimentos en la cola")
            return
            
        while True:
            print("\n" + "="*80)
            print(f"📊 COLA DE EXPERIMENTOS ({len(self.experiments)} total)")
            print("="*80)
            
            for i, (cmd, desc) in enumerate(self.experiments, 1):
                print(f"{i:3d}. {desc}")
                if self.get_yes_no("     ¿Ver comando completo?", "n"):
                    print(f"     {cmd}")
            
            print("\nOpciones:")
            print("1. Eliminar experimento")
            print("2. Reordenar experimentos")
            print("3. Duplicar experimento")
            print("4. Editar experimento")
            print("5. Limpiar toda la cola")
            print("0. Volver")
            
            choice = self.get_input("Selecciona", "0")
            
            if choice == "0":
                break
            elif choice == "1":
                idx = self.get_input("Número a eliminar", input_type=int) - 1
                if 0 <= idx < len(self.experiments):
                    removed = self.experiments.pop(idx)
                    print(f"✅ Eliminado: {removed[1]}")
            elif choice == "2":
                idx1 = self.get_input("Mover experimento #", input_type=int) - 1
                idx2 = self.get_input("A posición #", input_type=int) - 1
                if 0 <= idx1 < len(self.experiments) and 0 <= idx2 < len(self.experiments):
                    self.experiments.insert(idx2, self.experiments.pop(idx1))
                    print("✅ Reordenado")
            elif choice == "3":
                idx = self.get_input("Número a duplicar", input_type=int) - 1
                if 0 <= idx < len(self.experiments):
                    self.experiments.append(self.experiments[idx])
                    print("✅ Duplicado")
            elif choice == "4":
                idx = self.get_input("Número a editar", input_type=int) - 1
                if 0 <= idx < len(self.experiments):
                    cmd, desc = self.experiments[idx]
                    print(f"Comando actual: {cmd}")
                    new_cmd = input("Nuevo comando (Enter para mantener): ").strip()
                    if new_cmd:
                        cmd = new_cmd
                    new_desc = input(f"Nueva descripción [{desc}]: ").strip()
                    if new_desc:
                        desc = new_desc
                    self.experiments[idx] = (cmd, desc)
                    print("✅ Editado")
            elif choice == "5":
                if self.get_yes_no("¿Seguro que quieres limpiar toda la cola?"):
                    self.experiments = []
                    print("✅ Cola limpiada")

    def _execute_experiments(self):
        """Ejecuta todos los experimentos en la cola."""
        if not self.experiments:
            print("\n❌ No hay experimentos para ejecutar")
            return
            
        print("\n" + "="*80)
        print("🚀 RESUMEN DE EJECUCIÓN")
        print("="*80)
        print(f"Total de experimentos: {len(self.experiments)}")
        
        # Advertir sobre experimentos costosos
        costly_experiments = []
        for cmd, desc in self.experiments:
            if "--rollouts" in cmd:
                rollouts = int(cmd.split("--rollouts")[1].split()[0])
                if rollouts > 1000:
                    costly_experiments.append((desc, rollouts))
        
        if costly_experiments:
            print("\n⚠️  ADVERTENCIA: Los siguientes experimentos son computacionalmente costosos:")
            for desc, rollouts in costly_experiments:
                print(f"   • {desc} ({rollouts} rollouts)")
        
        if not self.get_yes_no("\n¿Proceder con la ejecución?"):
            return
            
        # Ejecución
        print("\n" + "="*80)
        print("🔄 EJECUTANDO EXPERIMENTOS")
        print("="*80)
        
        successful = 0
        failed = 0
        
        for i, (cmd, desc) in enumerate(self.experiments, 1):
            print(f"\n📍 Experimento {i}/{len(self.experiments)} ({i/len(self.experiments)*100:.0f}%)")
            
            if self.run_command(cmd, desc):
                successful += 1
            else:
                failed += 1
                if not self.get_yes_no("❌ Error. ¿Continuar con los siguientes?"):
                    break
        
        # Resumen final
        print("\n" + "="*80)
        print("🏁 EJECUCIÓN COMPLETADA")
        print("="*80)
        print(f"✅ Exitosos: {successful}")
        print(f"❌ Fallidos: {failed}")
        print(f"📊 Resultados guardados en:")
        print(f"   - outputs/debates.csv")
        print(f"   - outputs/debates_asimetricos.csv")
        if self.config['viz_enabled']:
            print(f"   - outputs/debate_*/ (visualizaciones)")

    def run_command(self, cmd, description):
        """Ejecuta un comando y muestra el progreso."""
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"📝 {cmd}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Completado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            return False

    def show_previous_results(self):
        """Muestra resumen de resultados anteriores."""
        print("\n" + "="*80)
        print("📊 RESULTADOS ANTERIORES")
        print("="*80)
        
        # Verificar archivos CSV
        files_to_check = [
            ("outputs/debates.csv", "Debates simétricos"),
            ("outputs/debates_asimetricos.csv", "Debates asimétricos"),
            ("outputs/judges.csv", "Jueces entrenados")
        ]
        
        for filepath, desc in files_to_check:
            if os.path.exists(filepath):
                print(f"\n📄 {desc} ({filepath}):")
                # Mostrar últimas 5 líneas
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Saltar header
                            print(f"   Total de registros: {len(lines)-1}")
                            print("   Últimos registros:")
                            for line in lines[-5:]:
                                if line.strip() and not line.startswith("timestamp"):
                                    parts = line.strip().split(',')
                                    if "judges.csv" in filepath:
                                        print(f"   - {parts[1]}: acc={parts[10]}, k={parts[9]}")
                                    else:
                                        acc_idx = -2 if "asimetricos" not in filepath else -2
                                        print(f"   - {parts[1]}: acc={parts[acc_idx]}")
                except Exception as e:
                    print(f"   Error leyendo archivo: {e}")
            else:
                print(f"\n❌ No encontrado: {filepath}")

    def _save_configuration(self):
        """Guarda la configuración actual de experimentos."""
        print("\n💾 GUARDAR CONFIGURACIÓN")
        filename = self.get_input("Nombre del archivo", "config_experimentos")
        if not filename.endswith('.json'):
            filename += '.json'
            
        config_data = {
            "config": self.config,
            "experiments": self.experiments,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs("configs", exist_ok=True)
        filepath = f"configs/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"✅ Configuración guardada en {filepath}")

    def load_configuration(self):
        """Carga una configuración guardada."""
        print("\n📂 CARGAR CONFIGURACIÓN")
        
        if not os.path.exists("configs"):
            print("❌ No hay configuraciones guardadas")
            return
            
        configs = [f for f in os.listdir("configs") if f.endswith('.json')]
        if not configs:
            print("❌ No hay configuraciones guardadas")
            return
            
        print("Configuraciones disponibles:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config}")
            
        idx = self.get_input("Selecciona configuración", "1", input_type=int) - 1
        if 0 <= idx < len(configs):
            filepath = f"configs/{configs[idx]}"
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.config = data["config"]
            self.experiments = data["experiments"]
            print(f"✅ Cargada configuración de {data['timestamp']}")
            print(f"   - Juez: {self.config['judge_name']}")
            print(f"   - Experimentos: {len(self.experiments)}")

    def create_template(self):
        """Crea templates predefinidos de experimentos."""
        print("\n📝 CREAR TEMPLATE DE EXPERIMENTOS")
        print("-" * 40)
        
        print("Templates disponibles:")
        print("1. 📚 Paper Replication - Configuración del paper original")
        print("2. 🚀 Quick Test - Prueba rápida con pocos recursos")
        print("3. 🔬 Full Analysis - Análisis exhaustivo")
        print("4. 📊 Benchmarking - Comparación de agentes")
        print("5. 🎯 Ablation Study - Estudio de ablación completo")
        
        choice = self.get_input("Selecciona template", "1", ["1", "2", "3", "4", "5"])
        
        # Limpiar experimentos actuales
        self.experiments = []
        
        # Configurar según template
        if choice == "1":
            self._create_paper_template()
        elif choice == "2":
            self._create_quick_template()
        elif choice == "3":
            self._create_full_template()
        elif choice == "4":
            self._create_benchmark_template()
        elif choice == "5":
            self._create_ablation_template()
            
        print(f"\n✅ Template creado con {len(self.experiments)} experimentos")
        if self.get_yes_no("¿Guardar este template?"):
            self._save_configuration()

    def _create_paper_template(self):
        """Template basado en el paper original."""
        print("\n📚 Configurando Paper Replication Template...")
        
        # Configurar juez por defecto
        if "28" in self.available_judges:
            self.config['judge_name'] = "28"
            self.config['resolution'] = 28
            self.config['k'] = 6
        
        base_cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                    f"--resolution {self.config['resolution']} --k 6 --thr 0.0")
        
        # Experimentos del paper
        experiments = [
            # Greedy baseline
            (f"{base_cmd} --agent_type greedy --n_images 1000 --note paper_greedy_baseline",
             "Paper - Greedy baseline"),
            # Greedy precommit
            (f"{base_cmd} --agent_type greedy --n_images 1000 --precommit --note paper_greedy_precommit",
             "Paper - Greedy precommit"),
            # MCTS diferentes rollouts
            (f"{base_cmd} --agent_type mcts --rollouts 100 --n_images 500 --note paper_mcts_100",
             "Paper - MCTS 100 rollouts"),
            (f"{base_cmd} --agent_type mcts --rollouts 500 --n_images 500 --note paper_mcts_500",
             "Paper - MCTS 500 rollouts"),
            (f"{base_cmd} --agent_type mcts --rollouts 1000 --n_images 300 --note paper_mcts_1000",
             "Paper - MCTS 1000 rollouts"),
        ]
        
        self.experiments.extend(experiments)

    def _create_quick_template(self):
        """Template para pruebas rápidas."""
        print("\n🚀 Configurando Quick Test Template...")
        
        base_cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                    f"--resolution {self.config['resolution']} --k {self.config['k']} "
                    f"--thr {self.config['thr']}")
        
        experiments = [
            (f"{base_cmd} --agent_type greedy --n_images 100 --note quick_greedy",
             "Quick - Greedy test"),
            (f"{base_cmd} --agent_type mcts --rollouts 50 --n_images 50 --note quick_mcts",
             "Quick - MCTS test"),
            (f"{base_cmd} --mixed_agents --honest_agent mcts --rollouts 50 --n_images 50 --note quick_mixed",
             "Quick - Mixed test"),
        ]
        
        self.experiments.extend(experiments)

    def _create_full_template(self):
        """Template para análisis exhaustivo."""
        print("\n🔬 Configurando Full Analysis Template...")
        
        base_cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                    f"--resolution {self.config['resolution']} --k {self.config['k']} "
                    f"--thr {self.config['thr']}")
        
        # Todas las combinaciones posibles
        for agent in ["greedy", "mcts"]:
            for precommit in ["", " --precommit"]:
                for starts in ["liar", "honest"]:
                    n_img = 1000 if agent == "greedy" else 500
                    rollouts = " --rollouts 500" if agent == "mcts" else ""
                    
                    note = f"full_{agent}"
                    if precommit:
                        note += "_precommit"
                    note += f"_{starts}first"
                    
                    cmd = (f"{base_cmd} --agent_type {agent}{rollouts} --n_images {n_img} "
                           f"--starts {starts}{precommit} --note {note}")
                    
                    desc = f"Full - {agent.upper()}"
                    if precommit:
                        desc += " + precommit"
                    desc += f" ({starts} first)"
                    
                    self.experiments.append((cmd, desc))

    def _create_benchmark_template(self):
        """Template para benchmarking de agentes."""
        print("\n📊 Configurando Benchmarking Template...")
        
        base_cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                    f"--resolution {self.config['resolution']} --k {self.config['k']} "
                    f"--thr {self.config['thr']} --seed 42")  # Semilla fija
        
        # Comparación directa con mismas condiciones
        experiments = [
            # Greedy vs Greedy
            (f"{base_cmd} --agent_type greedy --n_images 500 --note bench_greedy",
             "Benchmark - Greedy vs Greedy"),
            # MCTS vs MCTS (diferentes rollouts)
            (f"{base_cmd} --agent_type mcts --rollouts 100 --n_images 300 --note bench_mcts_100",
             "Benchmark - MCTS 100r"),
            (f"{base_cmd} --agent_type mcts --rollouts 500 --n_images 300 --note bench_mcts_500",
             "Benchmark - MCTS 500r"),
            (f"{base_cmd} --agent_type mcts --rollouts 1000 --n_images 200 --note bench_mcts_1000",
             "Benchmark - MCTS 1000r"),
            # Mixed
            (f"{base_cmd} --mixed_agents --honest_agent mcts --rollouts 500 --n_images 300 --note bench_mixed_mcts_h",
             "Benchmark - MCTS honest vs Greedy liar"),
            (f"{base_cmd} --mixed_agents --honest_agent greedy --rollouts 500 --n_images 300 --note bench_mixed_greedy_h",
             "Benchmark - Greedy honest vs MCTS liar"),
        ]
        
        self.experiments.extend(experiments)

    def _create_ablation_template(self):
        """Template para estudio de ablación."""
        print("\n🎯 Configurando Ablation Study Template...")
        
        # Ablación de k
        for k in [3, 4, 5, 6, 8, 10]:
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {k} "
                   f"--thr {self.config['thr']} --agent_type greedy --n_images 500 "
                   f"--note ablation_k{k}")
            self.experiments.append((cmd, f"Ablation - k={k}"))
        
        # Ablación de rollouts
        for r in [50, 100, 200, 500, 1000, 2000]:
            cmd = (f"python run_debate.py --judge_name {self.config['judge_name']} "
                   f"--resolution {self.config['resolution']} --k {self.config['k']} "
                   f"--thr {self.config['thr']} --agent_type mcts --rollouts {r} "
                   f"--n_images 200 --note ablation_r{r}")
            self.experiments.append((cmd, f"Ablation - {r} rollouts"))

def main():
    """Punto de entrada principal."""
    manager = ExperimentManager()
    manager.print_banner()
    manager.scan_available_judges()
    manager.show_main_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)