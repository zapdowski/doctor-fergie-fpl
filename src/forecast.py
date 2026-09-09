"""Team-level expected-goals model, fit from this season's actual match
results — the real-forecasting counterpart to the hand-picked FDR
multiplier table in fixtures.py.

Each team gets a fitted attack strength (expected goals scored against a
league-average defence at a neutral venue) and defence strength (expected
goals conceded against a league-average attack), plus a single league-wide
home-advantage factor. A fixture's expected goals for each side are then:

    expected_goals(home=i, away=j) = attack[i] * defence[j] * home_advantage
    expected_goals(away=j, home=i) = attack[j] * defence[i]

Fit via iterative proportional fitting — coordinate-ascent maximum
likelihood for this independent-Poisson model, the same well-established
approach behind most public football-goals models (minus Dixon-Coles' own
low-score correlation correction, which isn't worth the added complexity
here). The equations themselves are self-normalizing: because attack/
defence are anchored to real goal totals (not just relative comparisons),
there's no separate scale-identifiability step needed.

From expected goals, two real (if simplified) forecasts fall out:
  - attacking_multiplier: how much a specific fixture's expected goals for
    a team deviate from that team's own season-average attack — replaces
    the FDR lookup table for MID/FWD scoring.
  - defensive_multiplier: how much a specific fixture's clean-sheet
    probability (via the Poisson P(opponent scores 0) formula) deviates
    from that team's own season-average clean-sheet probability — replaces
    the FDR lookup table for GKP/DEF scoring.

Every caller must be ready for fit_team_strengths to return (None, None,
None): with too few finished matches (very early preseason), there's
nothing meaningful to fit, and callers should fall back to the FDR-based
proxy instead.
"""

import math
from collections import defaultdict

DEFENSIVE_POSITIONS = {"GKP", "DEF"}  # score via clean-sheet probability, not expected goals for

# Below this many finished matches league-wide, there's not enough signal
# to fit 20 teams' worth of attack/defence/home-advantage parameters —
# callers should fall back to the FDR-based proxy instead.
MIN_FINISHED_MATCHES = 10
# Confidence in a team's own fitted strength ramps up over this many
# matches played, same "small sample -> shrink toward neutral" idea used
# for player underlying-form confidence in optimizer.py — early on, every
# team is in that position together, so this mostly just damps the whole
# league's spread until more games are in the book.
SHRINKAGE_MATCHES = 8
MAX_ITER = 200
CONVERGENCE_TOL = 1e-6
# Typical historical top-flight home advantage; used as the starting value
# for fitting, and as the fallback if there's ever nothing to fit against.
DEFAULT_HOME_ADVANTAGE = 1.35


def _finished_matches(fixtures_data):
    return [
        f
        for f in fixtures_data
        if f.get("finished") and f.get("team_h_score") is not None and f.get("team_a_score") is not None
    ]


def fit_team_strengths(fixtures_data, team_ids):
    """Fit attack/defence strength for each team in team_ids from this
    season's finished matches.

    Returns (strengths, home_advantage, league_avg_attack), where
    strengths is {team_id: {"attack": float, "defence": float,
    "matches_played": int}}. Returns (None, None, None) if there aren't
    enough finished matches league-wide to fit anything meaningful (see
    MIN_FINISHED_MATCHES).

    Each team's raw fitted strength is shrunk toward the league average in
    proportion to how few matches they've played (see SHRINKAGE_MATCHES) —
    with only a handful of games played, a fitted strength is barely more
    reliable than the average.
    """
    matches = _finished_matches(fixtures_data)
    if len(matches) < MIN_FINISHED_MATCHES:
        return None, None, None

    attack = {tid: 1.0 for tid in team_ids}
    defence = {tid: 1.0 for tid in team_ids}
    home_advantage = DEFAULT_HOME_ADVANTAGE

    # (opponent_id, goals_for, goals_against) per team, split by venue —
    # doesn't change across iterations, so precompute once.
    home_matches = {tid: [] for tid in team_ids}
    away_matches = {tid: [] for tid in team_ids}
    for m in matches:
        h, a = m["team_h"], m["team_a"]
        hs, a_s = m["team_h_score"], m["team_a_score"]
        if h not in attack or a not in attack:
            continue  # defensive: a team not in the current bootstrap (shouldn't happen)
        home_matches[h].append((a, hs, a_s))
        away_matches[a].append((h, a_s, hs))

    matches_played = {tid: len(home_matches[tid]) + len(away_matches[tid]) for tid in team_ids}

    for _ in range(MAX_ITER):
        new_attack = {}
        for tid in team_ids:
            goals_for = sum(gf for _, gf, _ in home_matches[tid]) + sum(gf for _, gf, _ in away_matches[tid])
            denom = home_advantage * sum(defence[opp] for opp, _, _ in home_matches[tid]) + sum(
                defence[opp] for opp, _, _ in away_matches[tid]
            )
            new_attack[tid] = goals_for / denom if denom > 0 else attack[tid]

        new_defence = {}
        for tid in team_ids:
            goals_against = sum(ga for _, _, ga in home_matches[tid]) + sum(ga for _, _, ga in away_matches[tid])
            denom = sum(new_attack[opp] for opp, _, _ in home_matches[tid]) + home_advantage * sum(
                new_attack[opp] for opp, _, _ in away_matches[tid]
            )
            new_defence[tid] = goals_against / denom if denom > 0 else defence[tid]

        total_home_goals = sum(m["team_h_score"] for m in matches)
        expected_home_goals_at_1x = sum(new_attack[m["team_h"]] * new_defence[m["team_a"]] for m in matches)
        new_home_advantage = (
            total_home_goals / expected_home_goals_at_1x if expected_home_goals_at_1x > 0 else home_advantage
        )

        max_delta = max(
            (abs(new_attack[tid] - attack[tid]) for tid in team_ids),
            default=0.0,
        )
        max_delta = max(max_delta, max((abs(new_defence[tid] - defence[tid]) for tid in team_ids), default=0.0))
        max_delta = max(max_delta, abs(new_home_advantage - home_advantage))

        attack, defence, home_advantage = new_attack, new_defence, new_home_advantage
        if max_delta < CONVERGENCE_TOL:
            break

    league_avg_attack = sum(attack.values()) / len(attack)
    strengths = {}
    for tid in team_ids:
        n = matches_played[tid]
        confidence = n / (n + SHRINKAGE_MATCHES)
        strengths[tid] = {
            "attack": confidence * attack[tid] + (1 - confidence) * league_avg_attack,
            "defence": confidence * defence[tid] + (1 - confidence) * 1.0,
            "matches_played": n,
        }
    return strengths, home_advantage, league_avg_attack


def expected_goals(strengths, home_advantage, home_team_id, away_team_id):
    """(expected_goals_home, expected_goals_away) for a fixture, from
    fitted strengths. Unknown teams (shouldn't happen) fall back neutral.
    """
    home = strengths.get(home_team_id, {"attack": 1.0, "defence": 1.0})
    away = strengths.get(away_team_id, {"attack": 1.0, "defence": 1.0})
    exp_home = home["attack"] * away["defence"] * home_advantage
    exp_away = away["attack"] * home["defence"]
    return exp_home, exp_away


def expected_goals_for_team(strengths, home_advantage, team_id, opponent_id, is_home):
    """(expected_goals_for, expected_goals_against) for `team_id` in a
    fixture against `opponent_id`, from team_id's own perspective —
    handles the home/away swap so callers never have to.
    """
    if is_home:
        return expected_goals(strengths, home_advantage, team_id, opponent_id)
    exp_opponent, exp_team = expected_goals(strengths, home_advantage, opponent_id, team_id)
    return exp_team, exp_opponent


def clean_sheet_probability(expected_goals_against):
    """Poisson P(opponent scores 0) — a properly calibrated probability,
    not the ad hoc "1 - expected goals conceded per 90" proxy it replaces.
    """
    return math.exp(-expected_goals_against)


def attacking_multiplier(strengths, team_id, expected_goals_for):
    """How much this fixture's expected goals for `team_id` deviate from
    the team's own fitted attack strength (their average output against a
    league-average defence) — 1.0 is an average fixture, >1 easier, <1
    harder. Scales a player's own season-long attacking rate for this
    specific fixture, in place of the FDR lookup table.
    """
    team_attack = strengths.get(team_id, {}).get("attack")
    if not team_attack:
        return 1.0
    return expected_goals_for / team_attack


def defensive_multiplier(strengths, league_avg_attack, team_id, expected_goals_against):
    """How much this fixture's clean-sheet probability for `team_id`
    deviates from their own season-average clean-sheet probability (against
    a league-average attack) — 1.0 is an average fixture, >1 a better
    clean-sheet chance, <1 worse. Scales a GKP/DEF's clean-sheet-derived
    score for this specific fixture, in place of the FDR lookup table.
    """
    team_defence = strengths.get(team_id, {}).get("defence")
    if not team_defence or league_avg_attack <= 0:
        return 1.0
    baseline_goals_against = team_defence * league_avg_attack
    baseline_prob = clean_sheet_probability(baseline_goals_against)
    if baseline_prob <= 0:
        return 1.0
    return clean_sheet_probability(expected_goals_against) / baseline_prob


def team_window_multipliers(fixtures, start_gw, num_gws, strengths, home_advantage, league_avg_attack):
    """{team_id: {"attack": avg_multiplier, "defence": avg_multiplier}} over
    [start_gw, start_gw + num_gws) — same blank/double-gameweek handling as
    fixtures.team_fixture_multipliers: a double sums both fixtures'
    multipliers before averaging across weeks (rewarding the extra
    fixture); a blank contributes 0 for that week.
    """
    end_gw = start_gw + num_gws - 1
    attack_sums = defaultdict(float)
    defence_sums = defaultdict(float)
    for f in fixtures:
        gw = f.get("event")
        if gw is None or not (start_gw <= gw <= end_gw):
            continue
        h, a = f["team_h"], f["team_a"]
        exp_h, exp_a = expected_goals(strengths, home_advantage, h, a)
        attack_sums[h] += attacking_multiplier(strengths, h, exp_h)
        attack_sums[a] += attacking_multiplier(strengths, a, exp_a)
        defence_sums[h] += defensive_multiplier(strengths, league_avg_attack, h, exp_a)
        defence_sums[a] += defensive_multiplier(strengths, league_avg_attack, a, exp_h)

    all_teams = set(attack_sums) | set(defence_sums)
    return {
        tid: {
            "attack": attack_sums.get(tid, 0.0) / num_gws,
            "defence": defence_sums.get(tid, 0.0) / num_gws,
        }
        for tid in all_teams
    }
