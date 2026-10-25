"""
Simulación automática de Yahtzee con estrategia Montecarlo.
Ejecutar en Visual Studio Code con: python yahtzee.py

No requiere instalar paquetes externos; usa únicamente la biblioteca estándar.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ===============================
# Parámetros Montecarlo
# ===============================
SIMS_PER_MASK = 300  # nº de simulaciones por máscara de bloqueo (32 máx). Ajusta si quieres más/menos precisión.
SEED = None          # fija semilla para reproducibilidad; pon None para aleatorio

if SEED is not None:
    random.seed(SEED)

# ===============================
# Utilidades de dados
# ===============================
def roll_single_die() -> int:
    """Devuelve un entero uniforme en [1,6]."""
    return random.randint(1, 6)


def roll_dice(values: List[int], locked: List[bool]) -> None:
    """Lanza dados no bloqueados, actualizando 'values' in place."""
    for i, is_locked in enumerate(locked):
        if not is_locked:
            values[i] = roll_single_die()

def has_sequence(values: List[int], needed_len: int) -> bool:
    """¿Existe una secuencia consecutiva de longitud 'needed_len'? (ignora duplicados)."""
    uniq = sorted(set(values))
    if not uniq:
        return False
    longest = curr = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1] + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    return longest >= needed_len

# ===============================
# Puntuación
# ===============================
CATEGORIES_ORDER = [
    "ones","twos","threes","fours","fives","sixes",
    "threeOfKind","fourOfKind","fullHouse","smallStraight",
    "largeStraight","yahtzee","chance"
]

def score_hand(category: str, dice: List[int]) -> int:
    counts = Counter(dice)
    freqs = list(counts.values())
    total = sum(dice)
    if category in ["ones","twos","threes","fours","fives","sixes"]:
        num = {"ones":1, "twos":2, "threes":3, "fours":4, "fives":5, "sixes":6}[category]
        return counts.get(num, 0) * num
    if category == "threeOfKind":
        return total if any(f >= 3 for f in freqs) else 0
    if category == "fourOfKind":
        return total if any(f >= 4 for f in freqs) else 0
    if category == "fullHouse":
        return 25 if (2 in freqs and 3 in freqs) else 0
    if category == "smallStraight":
        return 30 if has_sequence(dice, 4) else 0
    if category == "largeStraight":
        return 40 if has_sequence(dice, 5) else 0
    if category == "yahtzee":
        return 50 if 5 in freqs else 0
    if category == "chance":
        return total
    return 0

def best_category_score(dice: List[int], available: List[str]) -> Tuple[str, int]:
    """Devuelve (mejor_categoria, puntuación) para los dados dados."""
    best_cat, best = None, -1
    for cat in available:
        sc = score_hand(cat, dice)
        if sc > best:
            best = sc
            best_cat = cat
    return best_cat, best

# ===============================
# Estructuras de juego
# ===============================
@dataclass
class PlayerState:
    name: str
    scores: Dict[str, Optional[int]] = field(default_factory=lambda: {c: None for c in CATEGORIES_ORDER})
    total: int = 0

    def available_categories(self) -> List[str]:
        return [c for c,v in self.scores.items() if v is None]

    def set_score(self, category: str, points: int) -> None:
        self.scores[category] = points
        self.total = sum(v for v in self.scores.values() if v is not None)

@dataclass
class GameStats:
    total_rolls: int = 0
    face_hist: Counter = field(default_factory=Counter)

    def update_from_roll(self, values: List[int]) -> None:
        self.total_rolls += sum(1 for _ in values)  # 5 por tirada
        self.face_hist.update(values)

@dataclass
class GameState:
    players: List[PlayerState]
    current_player_idx: int = 0
    turn_in_round: int = 0  # 0..12 (13 turnos)
    dice_values: List[int] = field(default_factory=lambda: [0]*5)
    locked: List[bool] = field(default_factory=lambda: [False]*5)
    rolls_left: int = 3
    stats: GameStats = field(default_factory=GameStats)

    def reset_turn(self):
        self.dice_values = [0]*5
        self.locked = [False]*5
        self.rolls_left = 3

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    def advance_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        if self.current_player_idx == 0:
            self.turn_in_round += 1

# ===============================
# Montecarlo para decidir bloqueos
# ===============================
def all_keep_masks() -> List[List[bool]]:
    """Genera las 32 máscaras posibles de bloqueo para 5 dados."""
    masks = []
    for m in range(32):  # 0..31
        mask = [(m >> i) & 1 == 1 for i in range(5)]
        masks.append(mask)
    return masks

def simulate_expected_value_after_keeps(
    current_values: List[int],
    keep_mask: List[bool],
    rolls_remaining: int,
    available_categories: List[str],
    sims: int
) -> float:
    """
    Estima el valor esperado de la mejor categoría si:
    - conservas (keep_mask) algunos dados ahora,
    - y después solo rerolléas los no conservados el número de veces restante,
    - sin volver a cambiar la máscara (aprox. Montecarlo de 1 paso).
    """
    if rolls_remaining <= 0:
        # Sin tiradas: solo puntuar ahora
        _, sc = best_category_score(current_values, available_categories)
        return float(sc)

    kept = [v if keep_mask[i] else None for i, v in enumerate(current_values)]
    unlocked_idx = [i for i,b in enumerate(keep_mask) if not b]

    total_score = 0.0
    for _ in range(sims):
        # Copia de valores
        vals = kept[:]
        # Rerolls: tirar los no conservados 'rolls_remaining' veces, quedándose con la última
        temp = current_values[:]  # por si keep None
        for _r in range(rolls_remaining):
            for i in unlocked_idx:
                temp[i] = roll_single_die()
        # Construir resultado final
        for i in range(5):
            vals[i] = vals[i] if vals[i] is not None else temp[i]

        # Mejor categoría al final
        _, sc = best_category_score(vals, available_categories)
        total_score += sc

    return total_score / sims if sims > 0 else 0.0

def choose_keep_mask_montecarlo(
    current_values: List[int],
    rolls_remaining: int,
    available_categories: List[str],
    sims_per_mask: int = SIMS_PER_MASK
) -> List[bool]:
    """
    Elige la máscara de bloqueo que maximiza el valor esperado estimado al final del turno.
    Nota: aproximación de 1 paso (no recalcula bloqueos tras la siguiente tirada durante la simulación).
    """
    best_mask = None
    best_ev = -1.0
    for mask in all_keep_masks():
        # Si el jugador decide "plantarse" ahora, la máscara sería keep todo (no cambia nada).
        ev = simulate_expected_value_after_keeps(
            current_values, mask, rolls_remaining, available_categories, sims_per_mask
        )
        if ev > best_ev:
            best_ev = ev
            best_mask = mask
    return best_mask

# ===============================
# Flujo del juego
# ===============================
def play_one_turn(state: GameState, verbose: bool = True):
    """Un turno completo para el jugador actual: hasta 3 tiradas + elección de categoría."""
    player = state.current_player
    state.reset_turn()

    # Tirada inicial
    roll_dice(state.dice_values, state.locked)
    state.stats.update_from_roll(state.dice_values)
    state.rolls_left -= 1
    if verbose:
        print(f"\n{player.name} — Tirada 1:", state.dice_values)

    # Hasta 2 rerolls con estrategia MC
    while state.rolls_left > 0:
        keep_mask = choose_keep_mask_montecarlo(
            state.dice_values, state.rolls_left, player.available_categories()
        )

        # Si la máscara "keep_mask" implica mantener todo tal cual (keep 5), podemos decidir plantarnos:
        if all(keep_mask):
            if verbose:
                print(f"{player.name} decide plantarse con:", state.dice_values)
            break

        # Bloquear según la máscara
        state.locked = keep_mask[:]
        # Reroll
        roll_dice(state.dice_values, state.locked)
        state.stats.update_from_roll(state.dice_values)
        state.rolls_left -= 1
        if verbose:
            tirada_num = 3 - state.rolls_left
            print(f"{player.name} — Tirada {tirada_num}:", state.dice_values, "| keep:", keep_mask)

    # Elegir mejor categoría disponible para la mano final
    cat, sc = best_category_score(state.dice_values, player.available_categories())
    player.set_score(cat, sc)
    if verbose:
        print(f"→ {player.name} elige categoría '{cat}' y anota {sc} puntos. Total = {player.total}")

def play_game(verbose_each_turn: bool = False) -> GameState:
    players = [PlayerState("Jugador 1"), PlayerState("Jugador 2")]
    state = GameState(players=players)
    total_rounds = 13

    print("🎲 Iniciando simulación de Montecarlo (Motor de IA tomando decisiones)...")
    for r in range(total_rounds):
        for _p in range(len(players)):
            play_one_turn(state, verbose=verbose_each_turn)
            state.advance_player()  # <--- ¡Esta es la línea que faltaba!
    print("✅ Simulación completada.\n")
    return state

# ===============================
# Ejecutar simulación completa
# ===============================
state = play_game(verbose_each_turn=True)

# ===============================
# Mostrar resultados
# ===============================
def print_scoreboard(state: GameState):
    print("\n\n===== MARCADOR FINAL =====")
    for p in state.players:
        print(f"\n{p.name}: {p.total} puntos")
        for cat in CATEGORIES_ORDER:
            v = p.scores[cat]
            print(f"  - {cat:13s}: {v if v is not None else 0}")
    # Ganador
    if state.players[0].total > state.players[1].total:
        winner = state.players[0].name
    elif state.players[1].total > state.players[0].total:
        winner = state.players[1].name
    else:
        winner = "Empate"
    print(f"\n🏆 Ganador: {winner}")

def print_stats(state: GameState):
    print("\n===== ESTADÍSTICAS =====")
    print(f"Total de lanzamientos (dados individuales): {state.stats.total_rolls}")
    total_faces = sum(state.stats.face_hist.values())
    for face in range(1,7):
        c = state.stats.face_hist.get(face, 0)
        p = (c/total_faces)*100 if total_faces else 0.0
        print(f" Cara {face}: {c} veces ({p:.2f}%)")

print_scoreboard(state)
print_stats(state)


def analizar_riesgo_tres_iguales(valor_base: int, d4: int, d5: int, categorias_disponibles: list, sims: int = 10000):
    """
    Simula el escenario de decisión: conservar 3 dados iguales + 2 distintos vs relanzar los 2 distintos.
    """
    print(f"\n--- ANÁLISIS DE RIESGO: 3 DADOS IGUALES (Valor: {valor_base}) ---")
    dados_actuales = [valor_base, valor_base, valor_base, d4, d5]
    
    # 1. Decisión A: Detenerse (Conservar puntaje actual)
    mejor_cat_actual, puntaje_actual = best_category_score(dados_actuales, categorias_disponibles)
    
    # 2. Decisión B: Arriesgarse (Lanzar d4 y d5)
    puntaje_total_simulado = 0
    frecuencias_categorias = Counter()
    
    for _ in range(sims):
        nuevo_d4 = roll_single_die()
        nuevo_d5 = roll_single_die()
        dados_nuevos = [valor_base, valor_base, valor_base, nuevo_d4, nuevo_d5]
        
        cat_simulada, puntaje_simulado = best_category_score(dados_nuevos, categorias_disponibles)
        puntaje_total_simulado += puntaje_simulado
        frecuencias_categorias[cat_simulada] += 1
        
    valor_esperado_arriesgar = puntaje_total_simulado / sims
    
    # 3. Mostrar Resultados
    print(f"Dados iniciales: {dados_actuales}")
    print(f"Categorías disponibles: {categorias_disponibles}")
    print(f"\n[OPCIÓN 1] DETENERSE:")
    print(f"Mejor categoría: '{mejor_cat_actual}' -> Puntaje asegurado: {puntaje_actual}")
    
    print(f"\n[OPCIÓN 2] ARRIESGARSE (Relanzar {d4} y {d5} - {sims} simulaciones):")
    print(f"Valor Esperado (EV): {valor_esperado_arriesgar:.2f} puntos")
    
    print("\nDistribución de mejores categorías obtenidas al arriesgar:")
    for cat, freq in frecuencias_categorias.most_common():
        probabilidad = (freq / sims) * 100
        print(f"  - {cat}: {probabilidad:.2f}%")
        
    # 4. Conclusión de la Teoría de Decisiones
    print("\nCONCLUSIÓN:")
    if valor_esperado_arriesgar > puntaje_actual:
        print(f"Matemáticamente, la mejor decisión es ARRIESGARSE (Ventaja de {(valor_esperado_arriesgar - puntaje_actual):.2f} puntos a largo plazo).")
    elif valor_esperado_arriesgar < puntaje_actual:
        print(f"Matemáticamente, la mejor decisión es DETENERSE (Pérdida esperada de {(puntaje_actual - valor_esperado_arriesgar):.2f} puntos al arriesgar).")
    else:
        print("Ambas opciones tienen el mismo valor esperado. Es una decisión neutral.")

# ===============================
# Ejecución del Caso de Prueba (Fase 3)
# ===============================
if __name__ == "__main__":

    categorias_prueba = ["threeOfKind", "fourOfKind", "fullHouse", "yahtzee", "chance"]
    analizar_riesgo_tres_iguales(valor_base=4, d4=2, d5=1, categorias_disponibles=categorias_prueba, sims=50000)
    