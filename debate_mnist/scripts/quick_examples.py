#!/usr/bin/env python3
"""
Scripts de ejemplo para casos de uso comunes del sistema de debates.
Ejecuta directamente sin interacción del usuario.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n🚀 Ejecutando: {description}")
    print(f"Comando: {cmd}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        return False

def ejemplo_entrenamiento_basico():
    """Entrena un juez básico para empezar."""
    print("🎓 EJEMPLO: Entrenamiento de Juez Básico")
    print("=" * 50)
    
    # Entrenar un juez simple 28x28 con pocas épocas para testing
    cmd = "python train_judge.py --judge_name test_judge --resolution 28 --k 6 --epochs 5 --note 'ejemplo_rapido'"
    return run_command(cmd, "Entrenamiento de juez de prueba")

def ejemplo_debate_rapido():
    """Ejecuta un debate rápido para probar el sistema."""
    print("\n⚡ EJEMPLO: Debate Rápido de Prueba")
    print("=" * 50)
    
    # Verificar si existe un juez
    if not os.path.exists("models/28.pth") and not os.path.exists("models/test_judge.pth"):
        print("❌ No hay modelos juez disponibles. Ejecuta primero ejemplo_entrenamiento_basico()")
        return False
    
    judge_name = "28" if os.path.exists("models/28.pth") else "test_judge"
    
    # Debate greedy rápido con visualización
    cmd = f"python run_debate.py --judge_name {judge_name} --resolution 28 --agent_type greedy --n_images 10 --save_colored_debate --save_metadata --note 'test_rapido'"
    return run_command(cmd, "Debate Greedy con visualización")

def ejemplo_alta_precision():
    """Ejecuta experimento de alta precisión: 1 imagen, muchos rollouts."""
    print("\n🎯 EJEMPLO: Alta Precisión - 1 Imagen, Muchos Rollouts")
    print("=" * 50)
    
    if not os.path.exists("models/28.pth"):
        print("❌ Necesitas el modelo '28.pth'. Ejecuta ejemplo_entrenamiento_basico() primero")
        return False
    
    # Experimento de alta precisión
    cmd = "python run_debate.py --judge_name 28 --resolution 28 --agent_type mcts --rollouts 2000 --n_images 1 --seed 42 --save_colored_debate --save_metadata --note 'alta_precision_2k_rollouts'"
    return run_command(cmd, "Experimento de alta precisión MCTS")

def ejemplo_comparacion_agentes():
    """Compara diferentes tipos de agentes con misma semilla."""
    print("\n🔄 EJEMPLO: Comparación de Agentes (Misma Semilla)")
    print("=" * 50)
    
    if not os.path.exists("models/28.pth"):
        print("❌ Necesitas el modelo '28.pth'")
        return False
    
    experiments = [
        ("Greedy vs Greedy", "python run_debate.py --judge_name 28 --resolution 28 --agent_type greedy --n_images 20 --seed 123 --save_colored_debate --note 'comparacion_greedy'"),
        ("MCTS vs MCTS", "python run_debate.py --judge_name 28 --resolution 28 --agent_type mcts --rollouts 500 --n_images 20 --seed 123 --save_colored_debate --note 'comparacion_mcts'"),
        ("MCTS Honest vs Greedy Liar", "python run_debate.py --judge_name 28 --resolution 28 --mixed_agents --honest_agent mcts --rollouts 500 --n_images 20 --seed 123 --save_colored_debate --note 'comparacion_mixed_mcts_honest'"),
        ("Greedy Honest vs MCTS Liar", "python run_debate.py --judge_name 28 --resolution 28 --mixed_agents --honest_agent greedy --rollouts 500 --n_images 20 --seed 123 --save_colored_debate --note 'comparacion_mixed_greedy_honest'")
    ]
    
    success = 0
    for desc, cmd in experiments:
        if run_command(cmd, desc):
            success += 1
    
    print(f"\n📊 Completados {success}/{len(experiments)} experimentos de comparación")
    return success == len(experiments)

def ejemplo_escalabilidad_rollouts():
    """Prueba diferentes números de rollouts para ver el efecto."""
    print("\n📈 EJEMPLO: Escalabilidad de Rollouts")
    print("=" * 50)
    
    if not os.path.exists("models/28.pth"):
        print("❌ Necesitas el modelo '28.pth'")
        return False
    
    rollout_values = [50, 100, 200, 500, 1000]
    
    for rollouts in rollout_values:
        cmd = f"python run_debate.py --judge_name 28 --resolution 28 --agent_type mcts --rollouts {rollouts} --n_images 5 --seed 456 --save_colored_debate --note 'escalabilidad_{rollouts}_rollouts'"
        run_command(cmd, f"MCTS con {rollouts} rollouts")
    
    print(f"\n📊 Experimentos de escalabilidad completados")
    return True

def ejemplo_precommit_vs_no_precommit():
    """Compara experimentos con y sin precommit."""
    print("\n🔒 EJEMPLO: Precommit vs No Precommit")
    print("=" * 50)
    
    if not os.path.exists("models/28.pth"):
        print("❌ Necesitas el modelo '28.pth'")
        return False
    
    experiments = [
        ("MCTS sin precommit", "python run_debate.py --judge_name 28 --resolution 28 --agent_type mcts --rollouts 300 --n_images 30 --seed 789 --save_colored_debate --note 'mcts_no_precommit'"),
        ("MCTS con precommit", "python run_debate.py --judge_name 28 --resolution 28 --agent_type mcts --rollouts 300 --n_images 30 --seed 789 --precommit --save_colored_debate --note 'mcts_precommit'"),
        ("Mixed sin precommit", "python run_debate.py --judge_name 28 --resolution 28 --mixed_agents --honest_agent mcts --rollouts 300 --n_images 30 --seed 789 --save_colored_debate --note 'mixed_no_precommit'"),
        ("Mixed con precommit", "python run_debate.py --judge_name 28 --resolution 28 --mixed_agents --honest_agent mcts --rollouts 300 --n_images 30 --seed 789 --precommit --save_colored_debate --note 'mixed_precommit'")
    ]
    
    success = 0
    for desc, cmd in experiments:
        if run_command(cmd, desc):
            success += 1
    
    print(f"\n📊 Completados {success}/{len(experiments)} experimentos de precommit")
    return success == len(experiments)

def mostrar_menu():
    """Muestra el menú de ejemplos disponibles."""
    print("🧪 EJEMPLOS RÁPIDOS - DEBATE EXPERIMENTS")
    print("=" * 50)
    print("1. Entrenamiento básico de juez")
    print("2. Debate rápido de prueba")
    print("3. Alta precisión (1 imagen, muchos rollouts)")
    print("4. Comparación de agentes")
    print("5. Escalabilidad de rollouts")
    print("6. Precommit vs No Precommit")
    print("7. Ejecutar todos los ejemplos")
    print("0. Salir")
    print()

def main():
    """Función principal del script de ejemplos."""
    while True:
        mostrar_menu()
        try:
            opcion = input("Selecciona un ejemplo (0-7): ").strip()
            
            if opcion == "0":
                print("👋 ¡Hasta luego!")
                break
            elif opcion == "1":
                ejemplo_entrenamiento_basico()
            elif opcion == "2":
                ejemplo_debate_rapido()
            elif opcion == "3":
                ejemplo_alta_precision()
            elif opcion == "4":
                ejemplo_comparacion_agentes()
            elif opcion == "5":
                ejemplo_escalabilidad_rollouts()
            elif opcion == "6":
                ejemplo_precommit_vs_no_precommit()
            elif opcion == "7":
                print("🚀 Ejecutando todos los ejemplos...")
                ejemplo_entrenamiento_basico()
                ejemplo_debate_rapido()
                ejemplo_alta_precision()
                ejemplo_comparacion_agentes()
                ejemplo_escalabilidad_rollouts()
                ejemplo_precommit_vs_no_precommit()
                print("\n🎉 ¡Todos los ejemplos completados!")
            else:
                print("❌ Opción inválida. Intenta de nuevo.")
            
            input("\nPresiona Enter para continuar...")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except EOFError:
            print("\n👋 ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()