import subprocess
import sys
import os
import json
from datetime import datetime
from collections import defaultdict
from utils.paths import get_config_path

class ExperimentManager:
    def __init__(self):
        self.experiments = []
        self.available_judges = []
        # Global configuration - set once and reused
        self.global_config = {
            'judge_name': "28",  # default judge
            'resolution': 28,  # default resolution
            'k': 6,  # default pixels
            'thr': 0.0,
            'default_images_greedy': 1000,
            'default_images_mcts': 100,  # updated value
            'default_rollouts': 512,  # updated value
            'seed': 42,
            'allow_all_pixels': False,
            'track_confidence': False
        }
        # Experiment-specific configuration
        self.config = {
            'viz_enabled': False,
            'viz_colored': False,
            'viz_metadata': False,
            'save_images': False,
            'save_masks': False,
            'save_play': False,
            'save_confidence': False,
            'save_json': False,  # New comprehensive JSON format
            'track_logits': False,
            'track_logits_progressive': False,
            'save_raw_logits': False
        }
        
    def print_banner(self):
        print("=" * 80)
        print()
        print("🧪 AI SAFETY DEBATE - EXPERIMENT AUTOMATION v3.5 🧪")


    def get_input(self, prompt, default=None, options=None, input_type=str):
        """Gets user input with validation."""
        while True:
            if default is not None:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if options and user_input.lower() not in [opt.lower() for opt in options]:
                print(f"❌ Invalid option. Options: {', '.join(options)}")
                continue
            
            try:
                if input_type == int:
                    return int(user_input)
                elif input_type == float:
                    return float(user_input)
                return user_input
            except ValueError:
                print(f"❌ Please enter a valid {input_type.__name__}")

    def get_yes_no(self, prompt, default="n"):
        """Gets yes/no response from user."""
        response = self.get_input(f"{prompt} (y/n)", default, ["y", "n", "yes", "no"])
        return response.lower() in ["y", "yes"]

    def check_judge_exists(self, judge_name):
        """Checks if a judge model exists."""
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

    def ask_override_global(self, param_name, current_value, description):
        """Pregunta si el usuario quiere cambiar un parámetro global."""
        if current_value is not None:
            return self.get_yes_no(f"¿Cambiar {description} (actual: {current_value})?")
        return True

    def show_main_menu(self):
        """Muestra el menú principal."""
        while True:
            print("\n" + "="*80)
            print("📋 MENÚ PRINCIPAL")
            print("="*80)
            print("1. 🎓 Entrenar nuevos modelos juez")
            print("2. 🔬 Configurar y ejecutar experimentos")
            print("3. 🎯 Evaluar capacidades del juez")
            print("4. 📊 Análisis de tesis y visualizaciones")
            print("5. 📈 Ver resultados anteriores")
            print("6. 💾 Cargar configuración guardada")
            print("7. 📝 Crear template de experimentos")
            print("8. ❌ Salir")
            
            choice = self.get_input("Selecciona una opción", "2", ["1", "2", "3", "4", "5", "6", "7", "8"])
            
            if choice == "1":
                self.train_judges()
            elif choice == "2":
                self.configure_experiments()
            elif choice == "3":
                self.evaluate_judge()
            elif choice == "4":
                self.thesis_analysis()
            elif choice == "5":
                self.show_previous_results()
            elif choice == "6":
                self.load_configuration()
            elif choice == "7":
                self.create_template()
            elif choice == "8":
                print("👋 ¡Hasta luego!")
                sys.exit(0)

    def thesis_analysis(self):
        """Sistema completo de análisis de tesis."""
        print("\n" + "="*80)
        print("📊 ANÁLISIS DE TESIS Y VISUALIZACIONES")
        print("="*80)
        print("Complete system for generating debate experiment data.")
        print("\nCaracterísticas:")
        print("• CSV data export for statistical analysis")
        print("• Análisis estadístico con intervalos de confianza")
        print("• Tablas LaTeX para la tesis")
        print("• Notebook interactivo Jupyter")
        print("• Validación de consistencia de datos")
        
        # Check for available data files
        required_files = ["outputs/debates.csv", "outputs/evaluations.csv"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print("⚠️ Archivos de datos faltantes:")
            for f in missing_files:
                print(f"  • {f}")
            print("\nEjecuta experimentos primero para generar datos.")
            if not self.get_yes_no("¿Continuar de todas formas?"):
                return
        
        while True:
            print("\n" + "="*60)
            print("📋 OPCIONES DE ANÁLISIS")
            print("="*60)
            print("1. 📈 Generate analysis data (CSV export)")
            print("2. 🔍 Análisis de datos interactivo")
            print("3. 📊 Resumen estadístico completo")
            print("4. 📋 Validación de consistencia de datos")
            print("5. 📈 Custom data analysis")
            print("6. 📝 Generar tablas LaTeX")
            print("7. 📓 Abrir Jupyter notebook interactivo")
            print("8. 🚀 Complete data analysis workflow")
            print("0. ↩️ Volver al menú principal")
            
            choice = self.get_input("Selecciona opción", "1", ["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            
            if choice == "0":
                break
            elif choice == "1":
                self._generate_all_thesis_figures()
            elif choice == "8":
                self._complete_thesis_analysis()

    def _generate_all_thesis_figures(self):
        """Analysis and visualization features have been moved to private development module."""
        print("\n📊 Analysis and Visualization Module")
        print("=" * 50)
        print("The analysis and visualization capabilities have been moved to a private")
        print("development module for specialized thesis work.")
        print()
        print("Available analysis tools:")
        print("• scripts/analyze_results.py - Basic statistical analysis")
        print("• Manual data analysis using CSV outputs")
        print("• Custom visualization development as needed")
        print()
        print("💡 Tip: Use the CSV outputs (debates.csv, evaluations.csv) for custom analysis")

    def _complete_thesis_analysis(self):
        """Análisis completo de tesis."""
        print("\n🚀 ANÁLISIS COMPLETO DE TESIS")
        print("=" * 50)
        print("Executing complete data analysis workflow")
        
        self._generate_all_thesis_figures()
        print("\n🎉 ANÁLISIS COMPLETO FINALIZADO")
        print("📋 Archivos generados:")
        print("• analysis/output/ - Figuras PNG de alta calidad")
        print("• outputs/ - CSVs con datos de experimentos")
        print("• Salida de consola - Estadísticas y tablas LaTeX")

    def train_judges(self):
        """Maneja el entrenamiento de modelos juez."""
        print("\n" + "="*80)
        print("🎓 ENTRENAMIENTO DE MODELOS JUEZ")
        print("="*80)
        print("Entrena modelos SparseCNN para diferentes configuraciones de resolución y k.")
        
        # Configuraciones estándar expandidas
        standard_configs = [
            ("28", 28, 6, "Resolución 28x28, 6 píxeles (PRINCIPAL)"),
            ("16", 16, 6, "Resolución 16x16, 6 píxeles"),
            ("28_4px", 28, 4, "Resolución 28x28, 4 píxeles"),
            ("16_4px", 16, 4, "Resolución 16x16, 4 píxeles"),
            ("28_3px", 28, 3, "Resolución 28x28, 3 píxeles"),
            ("28_5px", 28, 5, "Resolución 28x28, 5 píxeles"),
            ("28_8px", 28, 8, "Resolución 28x28, 8 píxeles"),
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

    def evaluate_judge(self):
        """Evalúa capacidades del juez con diferentes estrategias."""
        print("\n" + "="*80)
        print("🎯 EVALUACIÓN DE CAPACIDADES DEL JUEZ")
        print("="*80)
        print("Evalúa la precisión del juez con 8 estrategias diferentes de selección de píxeles:")
        print("• Estrategias estáticas: random, optimal, adversarial")
        print("• Agentes secuenciales: greedy, MCTS, adversariales")
        print("• Análisis comparativo y de escalabilidad")
        
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
        judge_name = self.available_judges[idx]
        resolution, default_k = self.detect_judge_params(judge_name)
        
        while True:
            print("\n" + "="*60)
            print("📋 ESTRATEGIAS DE EVALUACIÓN")
            print("="*60)
            print("1. 🎲 Random Pixels - Selección aleatoria (baseline)")
            print("2. ⭐ Optimal Pixels - Píxeles que maximizan confianza")
            print("3. 💀 Adversarial Pixels - Píxeles que minimizan confianza")
            print("4. 🚫 Adversarial Non-Zero - Adversariales SIN píxeles negros")
            print("5. 🤖 Greedy Agent - Selección secuencial con agente Greedy")
            print("6. 🧠 MCTS Agent - Selección secuencial con agente MCTS")
            print("7. 💀 Greedy Adversarial Agent - Minimiza logits de clase verdadera")
            print("8. 🧠 MCTS Adversarial Agent - MCTS que maximiza predicciones incorrectas")
            print("9. 📊 Comparison Suite - Comparar todas las estrategias")
            print("10. 🔬 K-Range Analysis - Analizar diferentes valores de k")
            print("0. ↩️  Volver al menú principal")
            
            choice = self.get_input("Selecciona estrategia", "9", 
                                  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
            
            if choice == "0":
                break
            elif choice == "1":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "random")
            elif choice == "2":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "optimal")
            elif choice == "3":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "adversarial")
            elif choice == "4":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "adversarial_nonzero")
            elif choice == "5":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "greedy_agent")
            elif choice == "6":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "mcts_agent")
            elif choice == "7":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "greedy_adversarial_agent")
            elif choice == "8":
                self._evaluate_single_strategy(judge_name, resolution, default_k, "mcts_adversarial_agent")
            elif choice == "9":
                self._evaluate_comparison_suite(judge_name, resolution, default_k)
            elif choice == "10":
                self._evaluate_k_range_analysis(judge_name, resolution, default_k)

    def _evaluate_comparison_suite(self, judge_name, resolution, default_k):
        """Ejecuta comparación completa de estrategias."""
        print("\n📊 SUITE DE COMPARACIÓN COMPLETA")
        print("-" * 50)
        print("Compara todas las estrategias de evaluación del juez:")
        print("• Estáticas: random, optimal, adversarial, adversarial_nonzero")
        print("• Agentes: greedy, MCTS, greedy adversarial, MCTS adversarial")
        
        k = self.get_input("Número de píxeles (k)", str(default_k), input_type=int)
        n_images = self.get_input("Imágenes por estrategia", "500", input_type=int)
        thr = self.get_input("Threshold", "0.0", input_type=float)
        
        include_agents = self.get_yes_no("¿Incluir estrategias de agentes (más lento)?", "y")
        allow_all_pixels = self.get_yes_no("¿Permitir píxeles negros para agentes?")
        save_comparison_outputs = self.get_yes_no("¿Guardar visualizaciones para comparación (una muestra por estrategia)?")
        
        strategies = ["random", "optimal", "adversarial", "adversarial_nonzero"]
        
        if include_agents:
            strategies.extend(["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"])
            rollouts = self.get_input("Rollouts para MCTS agents", "300", input_type=int)
        
        print(f"\n🚀 Ejecutando comparación con k={k}, {n_images} imágenes cada una...")
        print(f"Estrategias: {', '.join(strategies)}")
        
        for strategy in strategies:
            cmd = (f"python eval_judge.py --judge_name {judge_name} --resolution {resolution} "
                   f"--strategy {strategy} --k {k} --n_images {n_images} --thr {thr}")
            
            if strategy in ["mcts_agent", "mcts_adversarial_agent"]:
                cmd += f" --rollouts {rollouts}"
                
            if strategy in ["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"] and allow_all_pixels:
                cmd += " --allow_all_pixels"
            
            # Add saving options for comparison visualizations
            if save_comparison_outputs:
                # Save only a few samples for comparison purposes
                cmd += " --save_visualizations"
                # Limit to fewer images when saving outputs to avoid clutter
                if n_images > 50:
                    cmd = cmd.replace(f"--n_images {n_images}", "--n_images 10")
                    print(f"ℹ️  Limitando a 10 imágenes para {strategy} (visualización habilitada)")
            
            if self.run_command(cmd, f"Comparación - {strategy.replace('_', ' ').title()}"):
                print(f"✅ {strategy.replace('_', ' ').title()} completado")
            else:
                print(f"❌ Error en {strategy}")
                if not self.get_yes_no("¿Continuar con las demás estrategias?"):
                    break
        
        print("\n📊 Suite de comparación completada.")
        print("📁 Resultados guardados en outputs/evaluations.csv")
        if save_comparison_outputs:
            print("📁 Visualizaciones guardadas en outputs/visualizations/evaluations/")
        print("💡 Use option 4 (Data Analysis) to export CSV data for analysis")

    def _evaluate_single_strategy(self, judge_name, resolution, default_k, strategy):
        """Ejecuta evaluación de una estrategia específica."""
        print(f"\n🎯 EVALUACIÓN - {strategy.replace('_', ' ').title()}")
        print("-" * 50)
        
        k = self.get_input("Número de píxeles (k)", str(default_k), input_type=int)
        n_images = self.get_input("Número de imágenes", "500", input_type=int)
        thr = self.get_input("Threshold", "0.0", input_type=float)
        
        cmd = (f"python eval_judge.py --judge_name {judge_name} --resolution {resolution} "
               f"--strategy {strategy} --k {k} --n_images {n_images} --thr {thr} "
               f"--seed {self.global_config['seed']}")
        
        # Opciones específicas para agentes
        if strategy in ["mcts_agent", "mcts_adversarial_agent"]:
            rollouts = self.get_input("Rollouts para MCTS", "300", input_type=int)
            cmd += f" --rollouts {rollouts}"
        
        if strategy in ["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"]:
            if self.get_yes_no("¿Permitir selección de píxeles negros?"):
                cmd += " --allow_all_pixels"
        
        # Ask about saving evaluation outputs
        print("\n📁 Opciones de guardado para evaluación:")
        save_outputs = self.get_yes_no("¿Guardar imágenes y visualizaciones de la evaluación?")
        if save_outputs:
            cmd += " --save_metadata"  # This includes images, masks, and visualizations
        
        cmd += f' --note "single_eval_{strategy}"'
        
        if self.run_command(cmd, f"Evaluación {strategy.replace('_', ' ').title()}"):
            print(f"✅ Evaluación {strategy} completada")
            if save_outputs:
                print(f"📁 Archivos guardados en outputs/visualizations/evaluations/")
            print("📁 Resultados guardados en outputs/evaluations.csv")
        else:
            print(f"❌ Error en evaluación {strategy}")

    def _evaluate_k_range_analysis(self, judge_name, resolution, default_k):
        """Analiza diferentes valores de k para una estrategia."""
        print("\n🔬 ANÁLISIS DE RANGO K")
        print("-" * 50)
        print("Evalúa cómo cambia la precisión del juez con diferentes valores de k")
        
        strategy = self.get_input("Estrategia a analizar", "random", 
                                ["random", "optimal", "adversarial", "adversarial_nonzero", 
                                 "greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"])
        
        print("Selecciona valores de k a probar:")
        k_values = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
            if self.get_yes_no(f"  • k = {k}"):
                k_values.append(k)
        
        if not k_values:
            return
            
        n_images = self.get_input("Imágenes por valor de k", "300", input_type=int)
        thr = self.get_input("Threshold", "0.0", input_type=float)
        
        if strategy in ["mcts_agent", "mcts_adversarial_agent"]:
            rollouts = self.get_input("Rollouts para MCTS", "200", input_type=int)
        
        allow_all_pixels = False
        if strategy in ["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"]:
            allow_all_pixels = self.get_yes_no("¿Permitir píxeles negros?")
        
        save_k_analysis_outputs = self.get_yes_no("¿Guardar visualizaciones para análisis k (pocas muestras)?")
        
        print(f"\n🚀 Ejecutando análisis de k para {strategy}...")
        print(f"Valores de k: {k_values}")
        
        for k in k_values:
            cmd = (f"python eval_judge.py --judge_name {judge_name} --resolution {resolution} "
                   f"--strategy {strategy} --k {k} --n_images {n_images} --thr {thr} "
                   f"--seed {self.global_config['seed']}")
            
            if strategy in ["mcts_agent", "mcts_adversarial_agent"]:
                cmd += f" --rollouts {rollouts}"
            
            if allow_all_pixels:
                cmd += " --allow_all_pixels"
            
            # Add saving options for k analysis
            if save_k_analysis_outputs:
                cmd += " --save_visualizations"
                # Limit to very few images for k analysis
                if n_images > 20:
                    cmd = cmd.replace(f"--n_images {n_images}", "--n_images 5")
                    print(f"ℹ️  Limitando a 5 imágenes para k={k} (visualización habilitada)")
            
            cmd += f' --note "k_analysis_{strategy}_k{k}"'
            
            if self.run_command(cmd, f"K-Analysis {strategy} k={k}"):
                print(f"✅ k={k} completado")
            else:
                print(f"❌ Error con k={k}")
                if not self.get_yes_no("¿Continuar con los demás valores?"):
                    break
        
        print("\n📊 Análisis de rango K completado")
        print("📁 Resultados guardados en outputs/evaluations.csv")

    def configure_experiments(self):
        """Configuración principal de experimentos."""
        print("\n" + "="*80)
        print("🔬 CONFIGURACIÓN DE EXPERIMENTOS")
        print("="*80)
        print("Sistema inteligente de configuración con:")
        print("• Configuración global reutilizable")
        print("• 9 tipos de experimentos automatizados")
        print("• Opciones granulares de guardado")
        print("• Gestión de cola de experimentos")
        
        # Escanear jueces disponibles
        self.scan_available_judges()
        
        if not self.available_judges:
            print("❌ No hay modelos juez disponibles. Entrena uno primero.")
            return
        
        # Configurar parámetros globales
        self._configure_global_parameters()
        
        # Configurar visualizaciones
        self._configure_visualizations()
        
        # Configurar tracking de logits
        self._configure_logits_tracking()
        
        # Menú de experimentos
        self._experiment_selection_menu()

    def _configure_global_parameters(self):
        """Configura parámetros globales que se reutilizan."""
        print("\n⚙️ CONFIGURACIÓN GLOBAL")
        print("-" * 60)
        print("Estos parámetros se aplicarán a todos los experimentos")
        
        # Selección de juez
        if self.global_config['judge_name'] is None or self.ask_override_global('judge_name', self.global_config['judge_name'], 'juez'):
            print("\n📊 Modelos juez disponibles:")
            for i, judge in enumerate(self.available_judges, 1):
                res, k = self.detect_judge_params(judge)
                print(f"{i:2d}. {judge} (res: {res}x{res}, k: {k})")
            
            idx = self.get_input("Selecciona juez (número)", "1", input_type=int) - 1
            self.global_config['judge_name'] = self.available_judges[idx]
            detected_resolution, detected_k = self.detect_judge_params(self.global_config['judge_name'])
            self.global_config['resolution'] = detected_resolution
            
            # Preguntar sobre k
            if self.get_yes_no(f"¿Usar k={detected_k} detectado del juez?", "y"):
                self.global_config['k'] = detected_k
            else:
                self.global_config['k'] = self.get_input("Número de píxeles (k) global", "6", input_type=int)
        
        # Otros parámetros globales
        if self.ask_override_global('defaults', None, 'valores por defecto'):
            self.global_config['default_images_greedy'] = self.get_input(
                "Imágenes por defecto para Greedy", str(self.global_config['default_images_greedy']), input_type=int)
            self.global_config['default_images_mcts'] = self.get_input(
                "Imágenes por defecto para MCTS", str(self.global_config['default_images_mcts']), input_type=int)
            self.global_config['default_rollouts'] = self.get_input(
                "Rollouts por defecto para MCTS", str(self.global_config['default_rollouts']), input_type=int)
        
        if self.ask_override_global('advanced', None, 'opciones avanzadas'):
            self.global_config['allow_all_pixels'] = self.get_yes_no(
                "¿Permitir selección irrestricta de píxeles?")
            self.global_config['track_confidence'] = self.get_yes_no(
                "¿Trackear probabilidades del juez (para análisis estadístico)?")
        
        print("\n✅ Configuración global establecida:")
        print(f"  • Juez: {self.global_config['judge_name']} (res: {self.global_config['resolution']}, k: {self.global_config['k']})")
        print(f"  • Imágenes defecto: Greedy={self.global_config['default_images_greedy']}, MCTS={self.global_config['default_images_mcts']}")
        print(f"  • Rollouts defecto: {self.global_config['default_rollouts']}")

    def _configure_visualizations(self):
        """Configura opciones granulares de guardado."""
        print("\n🎨 CONFIGURACIÓN DE GUARDADO")
        print("-" * 50)
        print("Controla qué se guarda durante los experimentos")
        
        self.config['viz_enabled'] = self.get_yes_no("¿Habilitar visualizaciones/guardado?")
        if self.config['viz_enabled']:
            print("\n📦 Método de configuración:")
            print("1. Presets rápidos (recomendado)")
            print("2. Selección individual (control total)")
            method = self.get_input("Elige método", "1", options=["1", "2"])
            
            if method == "1":
                preset = self.get_input(
                    "Elige preset", "minimal", 
                    options=["minimal", "standard", "complete", "analysis", "json"])
                
                if preset == "minimal":
                    self.config.update({
                        'save_images': False, 'save_masks': False, 
                        'save_play': False, 'viz_colored': False, 
                        'save_confidence': False, 'save_json': False
                    })
                    print("✅ Minimal: Solo resultados CSV (máxima velocidad)")
                elif preset == "standard":
                    self.config.update({
                        'save_images': False, 'save_masks': False, 
                        'save_play': True, 'viz_colored': True, 
                        'save_confidence': False, 'save_json': False
                    })
                    print("✅ Standard: Debates coloreados + secuencias")
                elif preset == "complete":
                    self.config.update({
                        'save_images': True, 'save_masks': True, 
                        'save_play': True, 'viz_colored': True, 
                        'save_confidence': True, 'save_json': False
                    })
                    print("✅ Complete: Todas las visualizaciones y metadata")
                elif preset == "analysis":
                    self.config.update({
                        'save_images': False, 'save_masks': False, 
                        'save_play': True, 'viz_colored': False, 
                        'save_confidence': True, 'save_json': False
                    })
                    print("✅ Analysis: Optimizado para análisis estadístico")
                elif preset == "json":
                    self.config.update({
                        'save_images': False, 'save_masks': False, 
                        'save_play': False, 'viz_colored': False, 
                        'save_confidence': False, 'save_json': True
                    })
                    print("✅ JSON: Formato comprehensivo completo (reemplaza save_play)")
            else:
                # Selección individual
                print("\n📁 Opciones individuales:")
                self.config['save_images'] = self.get_yes_no("  🖼️  ¿Guardar imágenes originales?")
                self.config['save_masks'] = self.get_yes_no("  🎭 ¿Guardar imágenes enmascaradas?")
                self.config['save_play'] = self.get_yes_no("  📜 ¿Guardar secuencias de juego?")
                self.config['viz_colored'] = self.get_yes_no("  🎨 ¿Guardar debates coloreados?")
                self.config['save_confidence'] = self.get_yes_no("  📊 ¿Guardar análisis de probabilidades (track_confidence)?")
                self.config['save_json'] = self.get_yes_no("  📄 ¿Guardar JSON comprehensivo?")
                
                # Mostrar advertencia si se seleccionan opciones redundantes
                if self.config['save_json'] and self.config['save_play']:
                    print("ℹ️  NOTA: save_json incluye toda la funcionalidad de save_play")
                    if self.get_yes_no("  ¿Desactivar save_play para evitar redundancia?"):
                        self.config['save_play'] = False
            
            # Metadata legacy para compatibilidad
            any_metadata = any([
                self.config['save_images'], self.config['save_masks'], 
                self.config['save_play'], self.config['save_confidence'],
                self.config['save_json']
            ])
            self.config['viz_metadata'] = any_metadata

    def _configure_logits_tracking(self):
        """Configura opciones de tracking de logits del juez."""
        print("\n🧠 CONFIGURACIÓN DE TRACKING DE LOGITS")
        print("-" * 50)
        print("Trackea la evolución de las predicciones del juez durante el debate")
        
        track_enabled = self.get_yes_no("¿Habilitar tracking de logits?", default="n")
        
        if track_enabled:
            print("\n📊 Opciones de tracking:")
            print("1. Solo estado final (básico)")
            print("2. Progresivo - cada píxel (recomendado)")
            print("3. Completo - progresivo + logits crudos")
            
            option = self.get_input("Elige opción", "2", options=["1", "2", "3"])
            
            if option == "1":
                self.config['track_logits'] = True
                self.config['track_logits_progressive'] = False
                self.config['save_raw_logits'] = False
                print("✅ Tracking básico: Solo estado final del juez")
            elif option == "2":
                self.config['track_logits'] = False
                self.config['track_logits_progressive'] = True
                self.config['save_raw_logits'] = False
                print("✅ Tracking progresivo: Monitoreo después de cada píxel")
            elif option == "3":
                self.config['track_logits'] = False
                self.config['track_logits_progressive'] = True
                self.config['save_raw_logits'] = True
                print("✅ Tracking completo: Progresivo + logits crudos para análisis")
                
            # Automatically enable JSON when logits tracking is used
            if not self.config.get('save_json', False):
                auto_json = self.get_yes_no("¿Habilitar JSON automáticamente para guardar logits?", default=True)
                if auto_json:
                    self.config['save_json'] = True
                    print("📝 JSON habilitado automáticamente")
        else:
            self.config['track_logits'] = False
            self.config['track_logits_progressive'] = False
            self.config['save_raw_logits'] = False
            print("❌ Tracking de logits deshabilitado")

    def _get_advanced_flags(self):
        """Genera flags avanzados para comandos de debate."""
        flags = ""
        
        if self.global_config['allow_all_pixels']:
            flags += " --allow_all_pixels"
        
        if self.global_config['track_confidence'] or self.config['save_confidence']:
            flags += " --track_confidence"
        
        # Logits tracking options
        if self.config.get('track_logits'):
            flags += " --track_logits"
        
        if self.config.get('track_logits_progressive'):
            flags += " --track_logits_progressive"
        
        if self.config.get('save_raw_logits'):
            flags += " --save_raw_logits"
        
        # Comprehensive JSON format (replaces save_play, includes confidence_progression)
        if self.config['save_json']:
            flags += " --save_json"
        
        # Opciones granulares de guardado
        if self.config['viz_enabled']:
            if self.config['save_images']:
                flags += " --save_images"
            if self.config['save_masks']:
                flags += " --save_mask"
            if self.config['save_play']:
                flags += " --save_play"
            if self.config['viz_colored']:
                flags += " --save_colored_debate"
            
            # Flag metadata legacy
            if self.config['viz_metadata']:
                flags += " --save_metadata"
        
        return flags

    def _experiment_selection_menu(self):
        """Menú para selección de experimentos."""
        while True:
            print("\n" + "="*80)
            print("📋 SELECCIÓN DE EXPERIMENTOS")
            print("="*80)
            print(f"Juez: {self.global_config['judge_name']} | k: {self.global_config['k']} | "
                  f"Experimentos en cola: {len(self.experiments)}")
            print("-" * 80)
            
            print("1. 🔄 Experimentos Simétricos (mismo tipo de agente)")
            print("2. ⚔️  Experimentos Asimétricos (MCTS vs Greedy)")
            print("3. 🎯 Experimentos de Ablación (variar k)")
            print("4. 📈 Curvas de Escalabilidad (rollouts)")
            print("5. 🎲 Análisis de Robustez (múltiples semillas)")
            print("6. 🧠 Experimentos de Confianza")
            print("7. 🚫 Experimentos con Píxeles Irrestrictos")
            print("8. 🤖 Experimentos de Evaluación de Juez")
            print("9. 🔬 Experimentos Personalizados")
            print("10. 📊 Ver cola de experimentos")
            print("11. 🚀 Ejecutar experimentos")
            print("12. 💾 Guardar configuración")
            print("0. ↩️  Volver al menú principal")
            
            choice = self.get_input("Selecciona opción", "11")
            
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
                self._add_confidence_experiments()
            elif choice == "7":
                self._add_unrestricted_pixel_experiments()
            elif choice == "8":
                self._add_judge_evaluation_experiments()
            elif choice == "9":
                self._add_custom_experiments()
            elif choice == "10":
                self._show_experiment_queue()
            elif choice == "11":
                self._execute_experiments()
                break
            elif choice == "12":
                self._save_configuration()
            elif choice == "0":
                break

    def _add_symmetric_experiments(self):
        """Añade experimentos simétricos a la cola."""
        print("\n🔄 EXPERIMENTOS SIMÉTRICOS")
        print("-" * 40)
        print("Debates entre agentes del mismo tipo (Greedy vs Greedy, MCTS vs MCTS)")
        
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
                                str(self.global_config[f'default_images_{agent_type}']), input_type=int)
        
        rollouts = ""
        if agent_type == "mcts":
            r = self.get_input("Rollouts", str(self.global_config['default_rollouts']), input_type=int)
            rollouts = f" --rollouts {r}"
            
        for variant_name, flags, note in variants:
            advanced_flags = self._get_advanced_flags()
            cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                   f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                   f"--thr {self.global_config['thr']} --agent_type {agent_type} "
                   f"--n_images {n_images} --seed {self.global_config['seed']}{rollouts}{flags}{advanced_flags} "
                   f'--note "{agent_type}_{note}"')
            
            desc = f"{agent_type.upper()} vs {agent_type.upper()} - {variant_name}"
            self.experiments.append((cmd, desc))
            
        print(f"✅ Añadidos {len(variants)} experimentos simétricos")

    def _add_asymmetric_experiments(self):
        """Añade experimentos asimétricos a la cola."""
        print("\n⚔️ EXPERIMENTOS ASIMÉTRICOS")
        print("-" * 40)
        print("Debates entre agentes de diferente tipo (MCTS vs Greedy)")
        
        honest_agent = self.get_input("Agente honesto (greedy/mcts)", "mcts", ["greedy", "mcts"])
        liar_agent = "greedy" if honest_agent == "mcts" else "mcts"
        
        variants = []
        if self.get_yes_no("• Baseline"):
            variants.append(("baseline", "", ""))
        if self.get_yes_no("• Con precommit"):
            variants.append(("precommit", " --precommit", "precommit"))
        
        if not variants:
            return
            
        n_images = self.get_input("Número de imágenes", 
                                str(self.global_config['default_images_mcts']), input_type=int)
        rollouts = self.get_input("Rollouts para MCTS", 
                                str(self.global_config['default_rollouts']), input_type=int)
        
        for variant_name, flags, note in variants:
            advanced_flags = self._get_advanced_flags()
            cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                   f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                   f"--thr {self.global_config['thr']} --mixed_agents --honest_agent {honest_agent} "
                   f"--rollouts {rollouts} --n_images {n_images} --seed {self.global_config['seed']}{flags}{advanced_flags} "
                   f'--note "{honest_agent}_honest_vs_{liar_agent}_liar_{note}"')
            
            desc = f"{honest_agent.upper()} Honest vs {liar_agent.upper()} Liar - {variant_name}"
            self.experiments.append((cmd, desc))
            
        print(f"✅ Añadidos {len(variants)} experimentos asimétricos")

    def _add_ablation_experiments(self):
        """Añade experimentos de ablación variando k."""
        print("\n🎯 EXPERIMENTOS DE ABLACIÓN (VARIAR K)")
        print("-" * 40)
        print("Estudia el efecto del número de píxeles revelados (k) en el rendimiento")
        
        k_values = []
        print("Selecciona valores de k a probar:")
        for k in [3, 4, 5, 6, 7, 8, 10, 12]:
            if self.get_yes_no(f"  • k = {k}"):
                k_values.append(k)
        
        if not k_values:
            return
            
        agent_type = self.get_input("Tipo de agente (greedy/mcts)", "greedy", ["greedy", "mcts"])
        n_images = self.get_input("Imágenes por valor de k", "500", input_type=int)
        
        if agent_type == "mcts":
            rollouts = self.get_input("Rollouts", "200", input_type=int)
            
        # Seleccionar variantes experimentales
        print("\nSelecciona variantes a incluir:")
        variants = []
        if self.get_yes_no("• Baseline (liar primero, sin precommit)"):
            variants.append(("baseline", "", ""))
        if self.get_yes_no("• Con precommit"):
            variants.append(("precommit", " --precommit", "precommit"))
        if self.get_yes_no("• Honest primero"):
            variants.append(("honest_first", " --starts honest", "honest_first"))
        if self.get_yes_no("• Precommit + Honest primero"):
            variants.append(("precommit_honest", " --precommit --starts honest", "precommit_honest_first"))
            
        # Experimentos asimétricos
        asymmetric = self.get_yes_no("• Incluir experimentos asimétricos")
        if asymmetric:
            print("  Configuración asimétrica: un agente greedy vs uno MCTS")
            asymmetric_rollouts = 100
            if agent_type == "mcts":
                asymmetric_rollouts = self.get_input("  Rollouts para MCTS en asimétrico", "100", input_type=int)
        
        if not variants:
            variants = [("baseline", "", "")]  # Default to baseline if none selected
        
        for k in k_values:
            advanced_flags = self._get_advanced_flags()
            
            # Experimentos simétricos
            for variant_name, variant_flags, variant_suffix in variants:
                cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {k} "
                       f"--thr {self.global_config['thr']} --n_images {n_images} "
                       f"--seed {self.global_config['seed']}")
                
                if agent_type == "mcts":
                    cmd += f" --agent_type mcts --rollouts {rollouts}"
                    desc = f"Ablación k={k} - MCTS"
                else:
                    cmd += f" --agent_type greedy"
                    desc = f"Ablación k={k} - Greedy"
                
                cmd += variant_flags
                variant_note = f"_{variant_suffix}" if variant_suffix else ""
                cmd += f'{advanced_flags} --note "ablation_k{k}_{agent_type}{variant_note}"'
                
                if variant_suffix:
                    desc += f" - {variant_name}"
                    
                self.experiments.append((cmd, desc))
            
            # Experimentos asimétricos
            if asymmetric:
                for variant_name, variant_flags, variant_suffix in variants:
                    # Greedy vs MCTS
                    cmd_asym1 = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                                f"--resolution {self.global_config['resolution']} --k {k} "
                                f"--thr {self.global_config['thr']} --n_images {n_images} "
                                f"--seed {self.global_config['seed']} --mixed_agents "
                                f"--honest_agent greedy --rollouts {asymmetric_rollouts}")
                    cmd_asym1 += variant_flags
                    variant_note = f"_{variant_suffix}" if variant_suffix else ""
                    cmd_asym1 += f'{advanced_flags} --note "ablation_k{k}_asym_greedy_vs_mcts{variant_note}"'
                    desc_asym1 = f"Ablación k={k} - Asimétrico (Greedy vs MCTS)"
                    if variant_suffix:
                        desc_asym1 += f" - {variant_name}"
                    self.experiments.append((cmd_asym1, desc_asym1))
                    
                    # MCTS vs Greedy
                    cmd_asym2 = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                                f"--resolution {self.global_config['resolution']} --k {k} "
                                f"--thr {self.global_config['thr']} --n_images {n_images} "
                                f"--seed {self.global_config['seed']} --mixed_agents "
                                f"--honest_agent mcts --rollouts {asymmetric_rollouts}")
                    cmd_asym2 += variant_flags
                    cmd_asym2 += f'{advanced_flags} --note "ablation_k{k}_asym_mcts_vs_greedy{variant_note}"'
                    desc_asym2 = f"Ablación k={k} - Asimétrico (MCTS vs Greedy)"
                    if variant_suffix:
                        desc_asym2 += f" - {variant_name}"
                    self.experiments.append((cmd_asym2, desc_asym2))
            
        total_experiments = len(k_values) * len(variants) * (3 if asymmetric else 1)
        print(f"✅ Añadidos {total_experiments} experimentos de ablación")

    def _add_scalability_experiments(self):
        """Añade experimentos de escalabilidad de rollouts."""
        print("\n📈 CURVAS DE ESCALABILIDAD (ROLLOUTS)")
        print("-" * 40)
        print("Analiza cómo el rendimiento de MCTS escala con el número de rollouts")
        
        print("Configurar serie de rollouts:")
        print("1. Serie logarítmica (50, 100, 200, 500, 1000, 2000)")
        print("2. Serie lineal (100, 200, 300, ..., 1000)")
        print("3. Serie personalizada")
        
        choice = self.get_input("Selecciona", "1", ["1", "2", "3"])
        
        if choice == "1":
            rollout_values = [50, 100, 200, 500, 1000, 2000]
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
        
        n_images = self.get_input("Imágenes por punto", str(self.global_config['default_images_mcts']), input_type=int)
        fixed_seed = self.get_yes_no("¿Usar semilla fija para comparación?", "y")
        seed_to_use = self.global_config['seed'] if fixed_seed else None
        
        use_mixed = self.get_yes_no("¿Incluir agentes mixtos además de MCTS puro?")
        
        for rollouts in rollout_values:
            # MCTS puro
            advanced_flags = self._get_advanced_flags()
            cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                   f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                   f"--thr {self.global_config['thr']} --agent_type mcts --rollouts {rollouts} "
                   f"--n_images {n_images}")
            
            if seed_to_use:
                cmd += f" --seed {seed_to_use}"
            else:
                cmd += f" --seed {self.global_config['seed']}"
                
            cmd += f'{advanced_flags} --note "scalability_mcts_r{rollouts}"'
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
        print("Evalúa la consistencia de resultados con diferentes semillas aleatorias")
        
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
        n_images = self.get_input("Imágenes por semilla", str(self.global_config['default_images_mcts']), input_type=int)
        
        base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                    f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                    f"--thr {self.global_config['thr']} --n_images {n_images}")
        
        if agent_type == "greedy":
            base_cmd += " --agent_type greedy"
        elif agent_type == "mcts":
            rollouts = self.get_input("Rollouts", str(self.global_config['default_rollouts']), input_type=int)
            base_cmd += f" --agent_type mcts --rollouts {rollouts}"
        else:  # mixed
            honest = self.get_input("Agente honesto", "mcts", ["greedy", "mcts"])
            rollouts = self.get_input("Rollouts", str(self.global_config['default_rollouts']), input_type=int)
            base_cmd += f" --mixed_agents --honest_agent {honest} --rollouts {rollouts}"
        
        for seed in seeds:
            advanced_flags = self._get_advanced_flags()
            cmd = base_cmd + f' --seed {seed}{advanced_flags} --note "robustness_{agent_type}_s{seed}"'
            self.experiments.append((cmd, f"Robustez {agent_type} - Semilla {seed}"))
            
        print(f"✅ Añadidos {len(seeds)} experimentos de robustez")

    def _add_confidence_experiments(self):
        """Añade experimentos específicos para análisis de confianza."""
        print("\n🧠 EXPERIMENTOS DE CONFIANZA")
        print("-" * 40)
        print("Analiza la evolución de probabilidades del juez durante los debates")
        print("Estos experimentos fuerzan el tracking de probabilidades para análisis estadístico detallado.")
        
        # Temporalmente forzar track_confidence
        original_track = self.global_config['track_confidence']
        self.global_config['track_confidence'] = True
        
        agent_types = []
        if self.get_yes_no("• Incluir Greedy vs Greedy"):
            agent_types.append("greedy")
        if self.get_yes_no("• Incluir MCTS vs MCTS"):
            agent_types.append("mcts")
        if self.get_yes_no("• Incluir Mixed (MCTS vs Greedy)"):
            agent_types.append("mixed")
        
        if not agent_types:
            self.global_config['track_confidence'] = original_track
            return
        
        n_images = self.get_input("Imágenes por experimento", str(self.global_config['default_images_mcts']), input_type=int)
        k_values = []
        print("Valores de k para analizar confianza:")
        for k in [3, 4, 5, 6, 8]:
            if self.get_yes_no(f"  • k = {k}"):
                k_values.append(k)
        
        if not k_values:
            k_values = [self.global_config['k']]
        
        for agent_type in agent_types:
            for k in k_values:
                advanced_flags = self._get_advanced_flags()
                base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                           f"--resolution {self.global_config['resolution']} --k {k} "
                           f"--thr {self.global_config['thr']} --n_images {n_images} --seed {self.global_config['seed']}{advanced_flags}")
                
                if agent_type == "greedy":
                    cmd = base_cmd + f' --agent_type greedy --note "confidence_greedy_k{k}"'
                    desc = f"Confidence Analysis - Greedy k={k}"
                elif agent_type == "mcts":
                    rollouts = self.get_input(f"Rollouts para MCTS k={k}", str(self.global_config['default_rollouts']), input_type=int)
                    cmd = base_cmd + f' --agent_type mcts --rollouts {rollouts} --note "confidence_mcts_k{k}_r{rollouts}"'
                    desc = f"Confidence Analysis - MCTS k={k} ({rollouts}r)"
                else:  # mixed
                    honest = self.get_input("Agente honesto", "mcts", ["greedy", "mcts"])
                    rollouts = self.get_input(f"Rollouts para mixed k={k}", str(self.global_config['default_rollouts']), input_type=int)
                    cmd = base_cmd + f' --mixed_agents --honest_agent {honest} --rollouts {rollouts} --note "confidence_mixed_{honest}_k{k}"'
                    desc = f"Confidence Analysis - Mixed ({honest} honest) k={k}"
                
                self.experiments.append((cmd, desc))
        
        # Restaurar configuración original
        self.global_config['track_confidence'] = original_track
        
        total_experiments = len(agent_types) * len(k_values)
        print(f"✅ Añadidos {total_experiments} experimentos de análisis de confianza")

    def _add_unrestricted_pixel_experiments(self):
        """Añade experimentos con selección irrestricta de píxeles."""
        print("\n🚫 EXPERIMENTOS CON PÍXELES IRRESTRICTOS")
        print("-" * 40)
        print("Permite a los agentes seleccionar CUALQUIER píxel, incluyendo píxeles negros,")
        print("para analizar estrategias emergentes y robustez del sistema.")
        
        # Temporalmente forzar allow_all_pixels
        original_allow = self.global_config['allow_all_pixels']
        self.global_config['allow_all_pixels'] = True
        
        experiment_types = []
        if self.get_yes_no("• Comparación: restringido vs irrestricto (mismo setup)"):
            experiment_types.append("comparison")
        if self.get_yes_no("• Exploración de estrategias adversariales"):
            experiment_types.append("adversarial")
        if self.get_yes_no("• Análisis de robustez con píxeles negros"):
            experiment_types.append("robustness")
        
        if not experiment_types:
            self.global_config['allow_all_pixels'] = original_allow
            return
        
        agent_type = self.get_input("Tipo de agente principal", "greedy", ["greedy", "mcts", "mixed"])
        n_images = self.get_input("Imágenes por experimento", str(self.global_config['default_images_mcts']), input_type=int)
        
        if "comparison" in experiment_types:
            # Experimentos de comparación directa
            advanced_flags = self._get_advanced_flags()
            base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                       f"--thr {self.global_config['thr']} --n_images {n_images} --seed {self.global_config['seed']}")
            
            # Con restricción (temporalmente: quitar allow_all_pixels)
            self.global_config['allow_all_pixels'] = False
            restricted_flags = self._get_advanced_flags()
            self.global_config['allow_all_pixels'] = True
            
            if agent_type == "greedy":
                # Restringido
                cmd_restricted = base_cmd + restricted_flags + ' --agent_type greedy --note "comparison_restricted_greedy"'
                self.experiments.append((cmd_restricted, "Comparison - Greedy RESTRICTED pixels"))
                # Irrestricto
                cmd_unrestricted = base_cmd + advanced_flags + ' --agent_type greedy --note "comparison_unrestricted_greedy"'
                self.experiments.append((cmd_unrestricted, "Comparison - Greedy UNRESTRICTED pixels"))
        
        if "adversarial" in experiment_types:
            # Experimentos adversariales
            advanced_flags = self._get_advanced_flags()
            for k in [4, 6, 8]:
                cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {k} "
                       f"--thr 0.0 --n_images {n_images} --agent_type {agent_type} --seed {self.global_config['seed']}{advanced_flags} "
                       f'--note "adversarial_unrestricted_{agent_type}_k{k}"')
                self.experiments.append((cmd, f"Adversarial - {agent_type.upper()} k={k} (unrestricted)"))
        
        if "robustness" in experiment_types:
            # Experimentos de robustez
            advanced_flags = self._get_advanced_flags()
            for thr in [0.0, 0.1, 0.3]:
                cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                       f"--thr {thr} --n_images {n_images} --agent_type {agent_type} --seed {self.global_config['seed']}{advanced_flags} "
                       f'--note "robustness_unrestricted_{agent_type}_thr{thr}"')
                self.experiments.append((cmd, f"Robustness - {agent_type.upper()} thr={thr} (unrestricted)"))
        
        # Restaurar configuración original
        self.global_config['allow_all_pixels'] = original_allow
        
        print(f"✅ Añadidos experimentos de píxeles irrestrictos")
        print(f"    Los agentes pueden ahora seleccionar píxeles negros y explorar nuevas estrategias")

    def _add_judge_evaluation_experiments(self):
        """Añade experimentos específicos de evaluación del juez con agentes."""
        print("\n🤖 EXPERIMENTOS DE EVALUACIÓN DE JUEZ")
        print("-" * 40)
        print("Evalúa la capacidad del juez con diferentes estrategias de selección de píxeles")
        print("incluyendo agentes que seleccionan píxeles secuencialmente.")
        
        experiment_types = []
        if self.get_yes_no("• Comparación de todas las estrategias (8 estrategias)"):
            experiment_types.append("full_comparison")
        if self.get_yes_no("• Análisis de agentes vs estrategias estáticas"):
            experiment_types.append("agent_vs_static")
        if self.get_yes_no("• Análisis de escalabilidad de agentes (diferentes k)"):
            experiment_types.append("agent_scalability")
        if self.get_yes_no("• Análisis de rollouts para MCTS"):
            experiment_types.append("mcts_rollouts")
        
        if not experiment_types:
            return
        
        n_images = self.get_input("Imágenes por experimento", "300", input_type=int)
        
        if "full_comparison" in experiment_types:
            # Comparación completa de todas las estrategias
            strategies = ["random", "optimal", "adversarial", "adversarial_nonzero", 
                         "greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"]
            
            for strategy in strategies:
                cmd = (f"python eval_judge.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                       f"--thr {self.global_config['thr']} --strategy {strategy} --n_images {n_images} "
                       f"--seed {self.global_config['seed']}")
                
                if strategy in ["mcts_agent", "mcts_adversarial_agent"]:
                    cmd += " --rollouts 300"
                
                cmd += f' --note "judge_eval_comparison_{strategy}"'
                
                desc = f"Judge Eval - {strategy.replace('_', ' ').title()}"
                self.experiments.append((cmd, desc))
        
        if "agent_vs_static" in experiment_types:
            # Comparación directa agentes vs estáticas con mismas condiciones
            for k in [4, 6, 8]:
                base_cmd = (f"python eval_judge.py --judge_name {self.global_config['judge_name']} "
                           f"--resolution {self.global_config['resolution']} --k {k} "
                           f"--thr {self.global_config['thr']} --n_images {n_images} --seed {self.global_config['seed']}")
                
                # Estáticas
                for strategy in ["random", "optimal"]:
                    cmd = base_cmd + f' --strategy {strategy} --note "judge_eval_static_{strategy}_k{k}"'
                    self.experiments.append((cmd, f"Judge Eval - {strategy.title()} k={k}"))
                
                # Agentes
                for agent in ["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"]:
                    cmd = base_cmd + f' --strategy {agent}'
                    if agent in ["mcts_agent", "mcts_adversarial_agent"]:
                        cmd += " --rollouts 200"
                    cmd += f' --note "judge_eval_agent_{agent}_k{k}"'
                    self.experiments.append((cmd, f"Judge Eval - {agent.replace('_', ' ').title()} k={k}"))
        
        if "agent_scalability" in experiment_types:
            # Análisis de escalabilidad de agentes
            for agent in ["greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"]:
                for k in [3, 4, 5, 6, 7, 8, 10]:
                    cmd = (f"python eval_judge.py --judge_name {self.global_config['judge_name']} "
                           f"--resolution {self.global_config['resolution']} --k {k} "
                           f"--thr {self.global_config['thr']} --strategy {agent} --n_images {n_images} "
                           f"--seed {self.global_config['seed']}")
                    
                    if agent in ["mcts_agent", "mcts_adversarial_agent"]:
                        cmd += " --rollouts 200"
                    
                    cmd += f' --note "judge_eval_scalability_{agent}_k{k}"'
                    self.experiments.append((cmd, f"Judge Eval Scalability - {agent.replace('_', ' ').title()} k={k}"))
        
        if "mcts_rollouts" in experiment_types:
            # Análisis de rollouts para MCTS
            for rollouts in [50, 100, 200, 500, 1000]:
                cmd = (f"python eval_judge.py --judge_name {self.global_config['judge_name']} "
                       f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                       f"--thr {self.global_config['thr']} --strategy mcts_agent --rollouts {rollouts} "
                       f"--n_images {n_images} --seed {self.global_config['seed']} --note judge_eval_mcts_r{rollouts}")
                
                self.experiments.append((cmd, f"Judge Eval - MCTS {rollouts} rollouts"))
        
        total_added = len([exp for exp in self.experiments 
                          if "judge_eval" in exp[0]])
        print(f"✅ Añadidos experimentos de evaluación de juez")
        print(f"    Total: {total_added} experimentos que evalúan capacidades del juez")

    def _add_custom_experiments(self):
        """Añade experimentos completamente personalizados."""
        print("\n🔬 EXPERIMENTO PERSONALIZADO")
        print("-" * 40)
        print("Configura un experimento con control total sobre todos los parámetros")
        
        # Construir comando paso a paso
        cmd = f"python run_debate.py --judge_name {self.global_config['judge_name']}"
        cmd += f" --resolution {self.global_config['resolution']}"
        
        # k personalizado
        k = self.get_input("Píxeles (k)", str(self.global_config['k']), input_type=int)
        cmd += f" --k {k}"
        
        # threshold
        thr = self.get_input("Threshold", str(self.global_config['thr']), input_type=float)
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
        
        # Opciones avanzadas
        if self.get_yes_no("¿Permitir selección irrestricta de píxeles?"):
            cmd += " --allow_all_pixels"
        
        if self.get_yes_no("¿Trackear confianza del juez?"):
            cmd += " --track_confidence"
        
        if self.get_yes_no("¿Guardar datos completos en formato JSON?"):
            cmd += " --save_json"
        
        cmd += f" --seed {self.global_config['seed']}"
            
        note = self.get_input("Nota descriptiva", f"custom_{agent_desc}")
        cmd += f' --note "{note}"'
        
        # Visualizaciones
        advanced_flags = self._get_advanced_flags()
        cmd += advanced_flags
        
        desc = self.get_input("Descripción del experimento", f"Custom - {agent_desc}")
        
        self.experiments.append((cmd, desc))
        print("✅ Experimento personalizado añadido")

    def _show_experiment_queue(self):
        """Muestra y gestiona la cola de experimentos."""
        if not self.experiments:
            print("\n❌ No hay experimentos en la cola")
            return
            
        print("\n" + "="*80)
        print(f"📊 COLA DE EXPERIMENTOS ({len(self.experiments)} total)")
        print("="*80)
        
        for i, (cmd, desc) in enumerate(self.experiments, 1):
            print(f"{i:3d}. {desc}")
        
        print("\nOpciones de gestión:")
        print("1. 📝 Ver comando completo de un experimento")
        print("2. ❌ Eliminar experimento específico")
        print("3. 🗑️ Limpiar toda la cola")
        print("4. ⏭️ Reordenar experimentos")
        print("5. 📋 Exportar lista de comandos")
        print("0. ↩️ Volver")
        
        choice = self.get_input("Selecciona", "0")
        
        if choice == "1":
            idx = self.get_input("Número de experimento a ver", input_type=int) - 1
            if 0 <= idx < len(self.experiments):
                cmd, desc = self.experiments[idx]
                print(f"\n📋 Experimento {idx+1}: {desc}")
                print(f"🔧 Comando: {cmd}")
        elif choice == "2":
            idx = self.get_input("Número a eliminar", input_type=int) - 1
            if 0 <= idx < len(self.experiments):
                removed = self.experiments.pop(idx)
                print(f"✅ Eliminado: {removed[1]}")
        elif choice == "3":
            if self.get_yes_no("¿Seguro que quieres limpiar toda la cola?"):
                self.experiments = []
                print("✅ Cola limpiada")
        elif choice == "4":
            self._reorder_experiments()
        elif choice == "5":
            self._export_commands()

    def _reorder_experiments(self):
        """Reordena experimentos en la cola."""
        if len(self.experiments) < 2:
            print("❌ Necesitas al menos 2 experimentos para reordenar")
            return
        
        print("Opciones de reordenamiento:")
        print("1. Mover experimento a nueva posición")
        print("2. Intercambiar dos experimentos")
        print("3. Ordenar por tipo (simétricos primero)")
        
        choice = self.get_input("Selecciona", "1", ["1", "2", "3"])
        
        if choice == "1":
            idx1 = self.get_input("Mover experimento #", input_type=int) - 1
            idx2 = self.get_input("A posición #", input_type=int) - 1
            if 0 <= idx1 < len(self.experiments) and 0 <= idx2 < len(self.experiments):
                self.experiments.insert(idx2, self.experiments.pop(idx1))
                print("✅ Experimento movido")
        elif choice == "2":
            idx1 = self.get_input("Primer experimento #", input_type=int) - 1
            idx2 = self.get_input("Segundo experimento #", input_type=int) - 1
            if 0 <= idx1 < len(self.experiments) and 0 <= idx2 < len(self.experiments):
                self.experiments[idx1], self.experiments[idx2] = self.experiments[idx2], self.experiments[idx1]
                print("✅ Experimentos intercambiados")
        elif choice == "3":
            # Ordenar por tipo
            symmetric = [exp for exp in self.experiments if "vs" in exp[1] and exp[1].split("vs")[0].strip() == exp[1].split("vs")[1].strip().split("-")[0].strip()]
            asymmetric = [exp for exp in self.experiments if exp not in symmetric]
            self.experiments = symmetric + asymmetric
            print(f"✅ Reordenados: {len(symmetric)} simétricos, {len(asymmetric)} otros")

    def _export_commands(self):
        """Exporta la lista de comandos a un archivo."""
        if not self.experiments:
            print("❌ No hay experimentos para exportar")
            return
        
        filename = self.get_input("Nombre del archivo", "experiment_commands.txt")
        
        try:
            with open(filename, 'w') as f:
                f.write("# Lista de comandos de experimentos\n")
                f.write(f"# Generado: {datetime.now().isoformat()}\n")
                f.write(f"# Total de experimentos: {len(self.experiments)}\n\n")
                
                for i, (cmd, desc) in enumerate(self.experiments, 1):
                    f.write(f"# {i}. {desc}\n")
                    f.write(f"{cmd}\n\n")
            
            print(f"✅ Comandos exportados a {filename}")
        except Exception as e:
            print(f"❌ Error exportando: {e}")

    def _execute_experiments(self):
        """Ejecuta todos los experimentos en la cola."""
        if not self.experiments:
            print("\n❌ No hay experimentos para ejecutar")
            return
            
        print("\n" + "="*80)
        print("🚀 RESUMEN DE EJECUCIÓN")
        print("="*80)
        print(f"Total de experimentos: {len(self.experiments)}")
        print(f"Configuración global:")
        print(f"  • Juez: {self.global_config['judge_name']}")
        print(f"  • Resolución: {self.global_config['resolution']}")
        print(f"  • K: {self.global_config['k']}")
        print(f"  • Opciones de guardado: {'Habilitadas' if self.config['viz_enabled'] else 'Solo CSV'}")
        
        # Advertir sobre experimentos costosos
        costly_experiments = []
        for cmd, desc in self.experiments:
            if "--rollouts" in cmd:
                try:
                    rollouts = int(cmd.split("--rollouts")[1].split()[0])
                    if rollouts > 1000:
                        costly_experiments.append((desc, rollouts))
                except:
                    pass
        
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
        start_time = datetime.now()
        
        for i, (cmd, desc) in enumerate(self.experiments, 1):
            print(f"\n📍 Experimento {i}/{len(self.experiments)} ({i/len(self.experiments)*100:.0f}%)")
            
            if self.run_command(cmd, desc):
                successful += 1
            else:
                failed += 1
                if not self.get_yes_no("❌ Error. ¿Continuar con los siguientes?"):
                    break
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "="*80)
        print("🏁 EJECUCIÓN COMPLETADA")
        print("="*80)
        print(f"✅ Exitosos: {successful}")
        print(f"❌ Fallidos: {failed}")
        print(f"⏱️ Tiempo total: {duration}")
        print(f"\n📁 Resultados guardados en:")
        print(f"   • outputs/debates.csv - Debates simétricos")
        print(f"   • outputs/debates_asimetricos.csv - Debates asimétricos")
        print(f"   • outputs/evaluations.csv - Evaluaciones de juez")
        if self.config['viz_enabled']:
            print(f"   • outputs/visualizations/ - Imágenes y metadata")
        print(f"\n💡 Use option 4 (Data Analysis) to export experiment data")

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
        print("Resumen de experimentos ejecutados y datos generados")
        
        # Verificar archivos CSV
        files_to_check = [
            ("outputs/debates.csv", "Debates simétricos"),
            ("outputs/debates_asimetricos.csv", "Debates asimétricos"),
            ("outputs/evaluations.csv", "Evaluaciones de juez"),
            ("outputs/judges.csv", "Jueces entrenados")
        ]
        
        total_records = 0
        for filepath, desc in files_to_check:
            if os.path.exists(filepath):
                print(f"\n📄 {desc} ({filepath}):")
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            records = len(lines)-1
                            total_records += records
                            print(f"   📊 Total de registros: {records}")
                            print("   📈 Últimos resultados:")
                            for line in lines[-3:]:
                                if line.strip() and not line.startswith("timestamp"):
                                    parts = line.strip().split(',')
                                    if len(parts) > 2:
                                        if "judges.csv" in filepath:
                                            if len(parts) > 10:
                                                print(f"   - {parts[1]}: acc={parts[10]}")
                                        else:
                                            if len(parts) > 2:
                                                print(f"   - {parts[1]}: acc={parts[-2]}")
                except Exception as e:
                    print(f"   ❌ Error leyendo archivo: {e}")
            else:
                print(f"\n❌ No encontrado: {filepath}")
        
        print(f"\n📈 RESUMEN TOTAL:")
        print(f"   • Total de registros en todos los archivos: {total_records}")
        
        # Check data exports
        if os.path.exists("analysis/output"):
            csv_files = [f for f in os.listdir("outputs") if f.endswith('.csv')]
            print(f"   • CSV data files available: {len(csv_files)}")
        
        print(f"\n💡 Consejos:")
        print(f"   • Use option 4 to access data analysis tools")
        print(f"   • Los archivos CSV pueden abrirse con Excel o Python/pandas")
        print(f"   • CSV files ready for custom analysis and visualization")

    def _save_configuration(self):
        """Guarda la configuración actual."""
        print("\n💾 GUARDAR CONFIGURACIÓN")
        print("Guarda la configuración completa para reutilización futura")
        
        filename = self.get_input("Nombre del archivo", "config_experimentos")
        if not filename.endswith('.json'):
            filename += '.json'
            
        config_data = {
            "global_config": self.global_config,
            "config": self.config,
            "experiments": self.experiments,
            "timestamp": datetime.now().isoformat(),
            "version": "3.3",
            "description": f"Configuración con {len(self.experiments)} experimentos"
        }
        
        filepath = get_config_path(filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"✅ Configuración guardada en {filepath}")
            print(f"📋 Incluye:")
            print(f"   • Configuración global: {self.global_config['judge_name']}, k={self.global_config['k']}")
            print(f"   • Configuración de guardado: {'Habilitada' if self.config['viz_enabled'] else 'Solo CSV'}")
            print(f"   • Cola de experimentos: {len(self.experiments)} experimentos")
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")

    def load_configuration(self):
        """Carga una configuración guardada."""
        print("\n📂 CARGAR CONFIGURACIÓN")
        print("Carga una configuración previamente guardada")
        
        from utils.paths import CONFIGS_DIR
        if not CONFIGS_DIR.exists() or not any(CONFIGS_DIR.glob('*.json')):
            print("❌ No hay configuraciones guardadas")
            print("💡 Usa opción 12 en el menú de experimentos para guardar configuraciones")
            return
            
        configs = [f.name for f in CONFIGS_DIR.glob('*.json')]
        if not configs:
            print("❌ No hay configuraciones guardadas")
            return
            
        print("📁 Configuraciones disponibles:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config}")
            
        idx = self.get_input("Selecciona configuración", "1", input_type=int) - 1
        if 0 <= idx < len(configs):
            filepath = get_config_path(configs[idx])
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                if "global_config" in data:
                    self.global_config = data["global_config"]
                self.config = data["config"]
                self.experiments = data["experiments"]
                
                print(f"✅ Configuración cargada de {data.get('timestamp', 'fecha desconocida')}")
                print(f"📋 Configuración cargada:")
                print(f"   • Juez: {self.global_config['judge_name']}")
                print(f"   • Resolución: {self.global_config['resolution']}")
                print(f"   • K: {self.global_config['k']}")
                print(f"   • Experimentos: {len(self.experiments)}")
                print(f"   • Guardado: {'Habilitado' if self.config['viz_enabled'] else 'Solo CSV'}")
                
            except Exception as e:
                print(f"❌ Error cargando configuración: {e}")

    def create_template(self):
        """Crea templates predefinidos."""
        print("\n📝 CREAR TEMPLATE DE EXPERIMENTOS")
        print("-" * 40)
        print("Templates optimizados para diferentes propósitos de investigación")
        
        print("\n📚 Templates disponibles:")
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
        print("📋 Experimentos añadidos a la cola:")
        for i, (_, desc) in enumerate(self.experiments, 1):
            print(f"   {i}. {desc}")
            
        if self.get_yes_no("¿Guardar este template?"):
            self._save_configuration()

    def _create_paper_template(self):
        """Template basado en el paper original."""
        print("\n📚 Configurando Paper Replication Template...")
        print("Replica los experimentos principales del paper AI Safety via Debate")
        
        # Configurar juez por defecto
        if "28" in self.available_judges:
            self.global_config['judge_name'] = "28"
            self.global_config['resolution'] = 28
            self.global_config['k'] = 6
        
        base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                    f"--resolution {self.global_config['resolution']} --k 6 --thr 0.0 "
                    f"--seed {self.global_config['seed']}")
        
        # Experimentos del paper
        experiments = [
            (f'{base_cmd} --agent_type greedy --n_images 1000 --note "paper_greedy_baseline"',
             "Paper - Greedy baseline"),
            (f'{base_cmd} --agent_type greedy --n_images 1000 --precommit --note "paper_greedy_precommit"',
             "Paper - Greedy precommit"),
            (f'{base_cmd} --agent_type mcts --rollouts 100 --n_images 500 --note "paper_mcts_100"',
             "Paper - MCTS 100 rollouts"),
            (f'{base_cmd} --agent_type mcts --rollouts 500 --n_images 500 --note "paper_mcts_500"',
             "Paper - MCTS 500 rollouts"),
            (f'{base_cmd} --agent_type mcts --rollouts 1000 --n_images 300 --note "paper_mcts_1000"',
             "Paper - MCTS 1000 rollouts"),
        ]
        
        self.experiments.extend(experiments)

    def _create_quick_template(self):
        """Template para pruebas rápidas."""
        print("\n🚀 Configurando Quick Test Template...")
        print("Pruebas rápidas para verificar funcionalidad del sistema")
        
        base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                    f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                    f"--thr {self.global_config['thr']} --seed {self.global_config['seed']}")
        
        experiments = [
            (f"{base_cmd} --agent_type greedy --n_images 100 --note quick_greedy",
             "Quick - Greedy test"),
            (f"{base_cmd} --agent_type mcts --rollouts 50 --n_images 50 --note quick_mcts",
             "Quick - MCTS test"),
            (f"{base_cmd} --mixed_agents --honest_agent mcts --rollouts 50 --n_images 30 --note quick_mixed",
             "Quick - Mixed test"),
        ]
        
        self.experiments.extend(experiments)

    def _create_full_template(self):
        """Template para análisis exhaustivo."""
        print("\n🔬 Configurando Full Analysis Template...")
        print("Análisis exhaustivo con todas las combinaciones posibles")
        
        base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                    f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                    f"--thr {self.global_config['thr']} --seed {self.global_config['seed']}")
        
        # Combinaciones completas
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
                           f"--starts {starts}{precommit} --note \"{note}\"")
                    
                    desc = f"Full - {agent.upper()}"
                    if precommit:
                        desc += " + precommit"
                    desc += f" ({starts} first)"
                    
                    self.experiments.append((cmd, desc))

    def _create_benchmark_template(self):
        """Template para benchmarking."""
        print("\n📊 Configurando Benchmarking Template...")
        print("Comparación directa de agentes con condiciones controladas")
        
        base_cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                    f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                    f"--thr {self.global_config['thr']} --seed {self.global_config['seed']}")
        
        experiments = [
            (f"{base_cmd} --agent_type greedy --n_images 500 --note bench_greedy",
             "Benchmark - Greedy vs Greedy"),
            (f"{base_cmd} --agent_type mcts --rollouts 100 --n_images 300 --note bench_mcts_100",
             "Benchmark - MCTS 100r"),
            (f"{base_cmd} --agent_type mcts --rollouts 500 --n_images 300 --note bench_mcts_500",
             "Benchmark - MCTS 500r"),
            (f"{base_cmd} --agent_type mcts --rollouts 1000 --n_images 200 --note bench_mcts_1000",
             "Benchmark - MCTS 1000r"),
            (f"{base_cmd} --mixed_agents --honest_agent mcts --rollouts 500 --n_images 300 --note bench_mixed_mcts_h",
             "Benchmark - MCTS honest vs Greedy liar"),
            (f"{base_cmd} --mixed_agents --honest_agent greedy --rollouts 500 --n_images 300 --note bench_mixed_greedy_h",
             "Benchmark - Greedy honest vs MCTS liar"),
        ]
        
        self.experiments.extend(experiments)

    def _create_ablation_template(self):
        """Template para estudio de ablación."""
        print("\n🎯 Configurando Ablation Study Template...")
        print("Estudio sistemático de parámetros k y rollouts")
        
        # Ablación de k
        for k in [3, 4, 5, 6, 8, 10]:
            cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                   f"--resolution {self.global_config['resolution']} --k {k} "
                   f"--thr {self.global_config['thr']} --agent_type greedy --n_images 500 "
                   f"--seed {self.global_config['seed']} --note ablation_k{k}")
            self.experiments.append((cmd, f"Ablation - k={k}"))
        
        # Ablación de rollouts
        for r in [50, 100, 200, 500, 1000, 2000]:
            cmd = (f"python run_debate.py --judge_name {self.global_config['judge_name']} "
                   f"--resolution {self.global_config['resolution']} --k {self.global_config['k']} "
                   f"--thr {self.global_config['thr']} --agent_type mcts --rollouts {r} "
                   f"--n_images 200 --seed {self.global_config['seed']} --note ablation_r{r}")
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