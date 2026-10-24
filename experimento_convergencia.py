"""Experimento de convergencia para la simulación de Yahtzee.

Usa el mismo generador de dado justo del código original: random.randint(1, 6).
Ejecuta pruebas de 10, 1.000 y 100.000 lanzamientos y compara la frecuencia
observada de cada cara con su probabilidad teórica (1/6).

Ejecutar: python experimento_convergencia.py
No requiere instalar paquetes externos.
"""
import random
from collections import Counter

SEED = 42  # Usa None para resultados diferentes en cada ejecución.
ITERACIONES = [10, 1_000, 100_000]
CARAS = range(1, 7)
PROBABILIDAD_TEORICA = 1 / 6

if SEED is not None:
    random.seed(SEED)


def roll_single_die() -> int:
    """Misma función de lanzamiento del código original de Yahtzee."""
    return random.randint(1, 6)


def ejecutar_experimento(iteraciones: int) -> None:
    """Lanza un dado N veces y calcula frecuencias y diferencias."""
    resultados = [roll_single_die() for _ in range(iteraciones)]
    conteos = Counter(resultados)

    print(f"\n{'=' * 58}")
    print(f"EXPERIMENTO: {iteraciones:,} lanzamientos")
    print(f"Probabilidad teórica por cara: {PROBABILIDAD_TEORICA:.4%}")
    print("Cara | Frecuencia | Prob. observada | Diferencia vs. teórica")
    print("-" * 58)

    for cara in CARAS:
        frecuencia = conteos.get(cara, 0)
        prob_observada = frecuencia / iteraciones
        diferencia = prob_observada - PROBABILIDAD_TEORICA
        print(
            f"{cara:>4} | {frecuencia:>10,} | {prob_observada:>14.4%} | "
            f"{diferencia:+21.4%}"
        )

    # La desviación absoluta media resume qué tan lejos están las frecuencias
    # observadas del valor teórico de 1/6.
    error_medio = sum(
        abs(conteos.get(cara, 0) / iteraciones - PROBABILIDAD_TEORICA)
        for cara in CARAS
    ) / 6
    print(f"Desviación absoluta media: {error_medio:.4%}")


def main() -> None:
    print("EXPERIMENTO DE CONVERGENCIA — LEY DE LOS GRANDES NÚMEROS")
    print("Un dado justo tiene probabilidad teórica de 1/6 (16,6667%) por cara.")
    for n in ITERACIONES:
        ejecutar_experimento(n)
    print("\nConclusión: al aumentar los lanzamientos, las frecuencias relativas")
    print("suelen acercarse a 1/6, aunque no tienen que ser exactamente iguales.")


if __name__ == "__main__":
    main()
