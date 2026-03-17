"""equilibrist_parser.py"""
import re
import math
from collections import Counter
import numpy as np

__all__ = ['parse_value_with_units', 'convert_concentration_to_mM', 'convert_volume_to_mL',
           'convert_temperature_to_K', 'parse_equilibrium_constant', 'parse_species_with_stoich',
           'parse_script', 'check_script_syntax']


def parse_value_with_units(value_str):
    """
    Parse a value string that may contain units.
    
    Examples:
        "1.50"     → (1.50, None)
        "1.50 mM"  → (1.50, "mM") 
        "500 uL"   → (500, "uL")
        "1e-6 M"   → (1e-6, "M")
    
    Returns (value, unit) where unit is None if not specified
    """
    import re
    value_str = value_str.strip()
    
    # Match number (including scientific notation) optionally followed by units
    match = re.match(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(\w+)?$', value_str)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else None
        return value, unit
    else:
        # Fallback: try to parse as pure number
        try:
            return float(value_str), None
        except ValueError:
            raise ValueError(f"Could not parse value: '{value_str}'")

def convert_concentration_to_mM(value, unit):
    """Convert concentration to mM (internal standard)."""
    if unit is None or unit == "mM":
        return value  # Already in mM (default)
    elif unit == "M":
        return value * 1000  # M to mM
    elif unit == "uM":
        return value / 1000  # uM to mM
    else:
        raise ValueError(f"Unsupported concentration unit: {unit}")

def convert_volume_to_mL(value, unit):
    """Convert volume to mL (internal standard)."""
    if unit is None or unit == "mL":
        return value  # Already in mL (default)
    elif unit == "L":
        return value * 1000  # L to mL
    elif unit == "uL":
        return value / 1000  # uL to mL
    else:
        raise ValueError(f"Unsupported volume unit: {unit}")

def convert_temperature_to_K(value, unit):
    """Convert temperature to Kelvin (internal standard).

    Accepts:
        unit = None | "K"  → returned as-is
        unit = "C"         → Celsius to Kelvin
    """
    if unit is None or unit.upper() == "K":
        return float(value)
    elif unit.upper() == "C":
        return float(value) + 273.15
    else:
        raise ValueError(f"Unsupported temperature unit: '{unit}'. Use K or C.")


# Physical constants (CODATA 2018)
_KB   = 1.380649e-23    # J K⁻¹
_H    = 6.62607015e-34  # J s
_R_KCAL = 1.9872036e-3  # kcal mol⁻¹ K⁻¹


def _eyring_log10k(dg_act_kcal, T_K):
    """Return log10(k) for a rate constant from the Eyring equation.

    k = (kB·T / h) · exp(−ΔG‡ / RT)

    Args:
        dg_act_kcal : activation free energy ΔG‡ in kcal/mol (must be > 0)
        T_K         : temperature in Kelvin

    Returns:
        log10(k) in s⁻¹  (pre-exponential factor only; no concentration units —
        those are handled by the mM conversion layer in the ODE integrator).
    """
    if T_K <= 0:
        raise ValueError(f"Temperature must be positive, got {T_K} K.")
    log10_prefactor = math.log10(_KB * T_K / _H)          # log10(kB·T/h)
    exponent_log10  = -dg_act_kcal / (_R_KCAL * T_K * math.log(10))
    return log10_prefactor + exponent_log10


def parse_equilibrium_constant(k_str):
    """
    Parse equilibrium constant that may be log K or K.
    
    Examples:
        "log K = 1.50"  → (1.50, True)   # Already log K
        "Log K = 1.50"  → (1.50, True)   # Case insensitive
        "K = 1e5"       → (5.0, False)   # Converted to log K
    
    Returns (log_k_value, was_already_log)
    """
    import re
    import math
    k_str = k_str.strip()
    
    # Check for "log K" pattern (case insensitive)
    log_match = re.match(r'^log\s+k\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$', k_str, re.IGNORECASE)
    if log_match:
        log_k = float(log_match.group(1))
        return log_k, True
    
    # Check for plain "K" pattern
    k_match = re.match(r'^k\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$', k_str, re.IGNORECASE)
    if k_match:
        k_value = float(k_match.group(1))
        if k_value <= 0:
            raise ValueError(f"Equilibrium constant must be positive: K = {k_value}")
        log_k = math.log10(k_value)
        return log_k, False
    
    raise ValueError(f"Could not parse equilibrium constant: '{k_str}'")

# ─────────────────────────────────────────────
# EXCEL EXPORT FUNCTIONALITY 
# ─────────────────────────────────────────────


def parse_species_with_stoich(species_str):
    """Parse species string with stoichiometric coefficient.
    
    REQUIRES SPACE between coefficient and species name to distinguish from
    numbers that are part of the species name.
    
    Examples:
        "A" → (1, "A")
        "GH2M" → (1, "GH2M")     ← Numbers in name, no coefficient  
        "2 A" → (2, "A")          ← Coefficient 2, species A
        "2 GH2M" → (2, "GH2M")    ← Coefficient 2, species GH2M
        "10 H2O" → (10, "H2O")    ← Coefficient 10, species H2O
        "CO2" → (1, "CO2")        ← Just a species name
    """
    import re
    species_str = species_str.strip()
    
    # Look for pattern: digits + SPACE + species name
    match = re.match(r'(\d+)\s+([A-Za-z]\w*)', species_str)
    if match:
        coeff_str, species = match.groups()
        coefficient = int(coeff_str)
        return coefficient, species
    else:
        # No space-separated coefficient found, entire string is species name
        return 1, species_str


def _tokenize_expr(s, param_names):
    """
    Tokenize a constraint expression, matching parameter names first using
    longest-match so that hyphenated names like 'k-1' are found before
    the '-' is mistaken for a subtraction operator.

    Token types: LOG, NAME, NUMBER, LPAREN, RPAREN, OP (one of + - * /), WS
    Returns list of (type, value) pairs.
    """
    # Sort param names longest-first so 'k-1' beats 'k' at the same position
    sorted_params = sorted(param_names, key=len, reverse=True)

    tokens = []
    i = 0
    while i < len(s):
        # Skip whitespace
        if s[i].isspace():
            i += 1
            continue

        # 'log' keyword (only if NOT the start of a param name)
        if s[i:i+3].lower() == 'log' and not any(
                s[i:].startswith(p) and len(p) > 3 for p in sorted_params):
            # Make sure it's not part of a longer param name
            rest = s[i+3:]
            if not rest or not (rest[0].isalnum() or rest[0] == '_' or rest[0] == '-'):
                tokens.append(('LOG', 'log'))
                i += 3
                continue

        # Parameter names (longest-match)
        matched_param = None
        for p in sorted_params:
            if s[i:i+len(p)] == p:
                # Make sure it's not a prefix of a longer identifier
                end = i + len(p)
                after = s[end] if end < len(s) else ''
                if not (after.isalnum() or after == '_'):
                    matched_param = p
                    break
        if matched_param is not None:
            tokens.append(('NAME', matched_param))
            i += len(matched_param)
            continue

        # Number (no leading sign — sign handled as OP)
        m = re.match(r'(\d+\.?\d*|\.\d+)([eE][+\-]?\d+)?', s[i:])
        if m:
            tokens.append(('NUMBER', float(m.group(0))))
            i += len(m.group(0))
            continue

        # Operators and parentheses
        if s[i] == '(':
            tokens.append(('LPAREN', '('))
        elif s[i] == ')':
            tokens.append(('RPAREN', ')'))
        elif s[i] in '+-*/':
            tokens.append(('OP', s[i]))
        else:
            raise ValueError(f"Unexpected character '{s[i]}' in constraint expression '{s}'")
        i += 1

    return tokens


def _parse_constraint_expr(s, param_names, line_has_log):
    """
    Parse one side of a constraint as a linear expression in log K space.

    Grammar (all operators have log-space semantics):
        expr   := ['+' | '-'] term (( '+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := '(' expr ')' | 'log' NAME | NAME | NUMBER

    Log-space semantics:
      * and / between factors → add / subtract log-coefficients
      + and - between terms   → add / subtract log-coefficients (already in log units)

    line_has_log = True  → numbers are already in log units (const += n)
    line_has_log = False → numbers are K-values, convert via log10 (const += log10(n))

    Bare NAME (no 'log' prefix):
      line_has_log = False → treated as log K (coefficient 1 on NAME)
      line_has_log = True  → also treated as log K for backward-compat

    Returns {"coeffs": {name: float}, "const": float} or raises ValueError.
    """
    import math as _math

    param_set = set(param_names)
    tokens = _tokenize_expr(s.strip(), param_names)

    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else ('EOF', '')

    def consume():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def add_terms(a, b, sign=1.0):
        """Combine two {coeffs, const} dicts: return a + sign*b."""
        c = dict(a['coeffs'])
        for k, v in b['coeffs'].items():
            c[k] = c.get(k, 0.0) + sign * v
        return {
            'coeffs': {k: v for k, v in c.items() if abs(v) > 1e-15},
            'const':  a['const'] + sign * b['const'],
        }

    def parse_expr():
        # Optional leading sign
        sign = 1.0
        if peek() == ('OP', '-'):
            consume(); sign = -1.0
        elif peek() == ('OP', '+'):
            consume()

        result = parse_term()
        if sign == -1.0:
            result = {'coeffs': {k: -v for k, v in result['coeffs'].items()},
                      'const':  -result['const']}

        while peek()[0] == 'OP' and peek()[1] in ('+', '-'):
            op = consume()[1]
            rhs = parse_term()
            s_op = 1.0 if op == '+' else -1.0
            result = add_terms(result, rhs, s_op)

        return result

    def parse_term():
        result = parse_factor()
        while peek()[0] == 'OP' and peek()[1] in ('*', '/'):
            op = consume()[1]
            rhs = parse_factor()
            if op == '*':
                result = add_terms(result, rhs, 1.0)   # log(A*B) = logA + logB
            else:
                result = add_terms(result, rhs, -1.0)  # log(A/B) = logA - logB
        return result

    def parse_factor():
        t = peek()

        if t[0] == 'LPAREN':
            consume()
            result = parse_expr()
            if peek()[0] != 'RPAREN':
                raise ValueError("Missing closing parenthesis in constraint expression")
            consume()
            return result

        if t[0] == 'LOG':
            consume()
            name_tok = peek()
            if name_tok[0] != 'NAME':
                raise ValueError(f"Expected parameter name after 'log', got '{name_tok[1]}'")
            consume()
            name = name_tok[1]
            if name not in param_set:
                raise ValueError(f"Unknown parameter '{name}' in constraint")
            return {'coeffs': {name: 1.0}, 'const': 0.0}

        if t[0] == 'NAME':
            consume()
            name = t[1]
            if name not in param_set:
                raise ValueError(f"Unknown parameter '{name}' in constraint")
            return {'coeffs': {name: 1.0}, 'const': 0.0}

        if t[0] == 'NUMBER':
            consume()
            n = t[1]
            if line_has_log:
                return {'coeffs': {}, 'const': n}
            if n <= 0:
                raise ValueError(f"Non-positive number {n} cannot be converted to log scale")
            return {'coeffs': {}, 'const': _math.log10(n)}

        raise ValueError(f"Unexpected token '{t[1]}' in constraint expression '{s}'")

    result = parse_expr()
    if peek()[0] != 'EOF':
        raise ValueError(
            f"Unexpected '{peek()[1]}' in constraint expression '{s}' — "
            "use K-space syntax for direct arithmetic (e.g. K1 + K2 = 1e7)"
        )
    return result


def _needs_kspace_eval(expr_str, param_names):
    """
    Returns True if expr_str should be evaluated in K-space (actual K values)
    rather than log-space (linear in log K).

    Rule: if any bare (non-log) parameter name appears in an expression that
    also has a binary + or - operator, the user is doing K-space arithmetic.

    Critical: parameter names that contain hyphens (e.g. 'k-1') are masked
    out FIRST so their internal hyphen is never mistaken for a minus operator.

    Examples that return True:
        "K1 + K2"         -> K1 and K2 added directly
        "2 * K1 + K2"     -> linear combination with addition

    Examples that return False:
        "K1"              -> single bare param -> log-space
        "K1 * K2"         -> multiplicative -> log-space
        "K1 / K2"         -> ratio -> log-space
        "k1 / k-1"        -> ratio, 'k-1' is a param name, not k minus 1
        "log K1 + log K2" -> explicit log -> log-space
        "1e5"             -> pure number
    """
    if not param_names:
        return False
    # Mask out param names (longest first) to avoid confusing hyphens/symbols in names
    cleaned = expr_str
    for p in sorted(param_names, key=len, reverse=True):
        cleaned = cleaned.replace(p, '__P__')
    # Also mask out 'log NAME' tokens that are explicitly log-space
    cleaned = re.sub(r'\blog\s+\w+', '__LOG__', cleaned, flags=re.I)
    cleaned = re.sub(r'\blog\s+__P__', '__LOG__', cleaned, flags=re.I)
    # Does the cleaned expression still contain any bare parameter placeholder?
    if '__P__' not in cleaned:
        return False
    # Is there a binary + or - (i.e. not part of an exponent like 1e-5)?
    return bool(re.search(r'(?<![eE])\s*[+-]\s*(?=[^\s])', cleaned[1:]))  # skip leading sign


def _parse_constraint_line(line, param_names):
    """
    Parse one $constraints line into a list of constraint dicts.

    Auto-detects whether each expression is log-space or K-space:
    - Log-space (linear in log K): K1 > K2, K1 = 2*K2, K1 > 1e5, log K1 + log K2 < 15
    - K-space (arithmetic in K):   K1 + K2 = 1e7, 2*K1 + K2 < 1e8

    Supports '; hard' modifier for strict enforcement.
    Supports chained comparisons: 3 < log K1 < 8

    Log-space dict keys: "mode"="log", "coeffs", "const", "op", "hard"
    K-space dict keys:   "mode"="K",   "lhs_expr", "rhs_expr", "op", "hard", "param_names"
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return []

    # ── Strip '; hard' modifier ───────────────────────────────────────────
    hard = False
    if ';' in line:
        main_part, flag_part = line.split(';', 1)
        if flag_part.strip().lower() == 'hard':
            hard = True
        line = main_part.strip()

    # ── Split on comparison operators ─────────────────────────────────────
    OP_RE = re.compile(r'(<=|>=|<|>|=)')
    parts = OP_RE.split(line)
    if len(parts) < 3:
        return []

    exprs_raw = [parts[i].strip() for i in range(0, len(parts), 2)]
    ops       = [parts[i].strip() for i in range(1, len(parts), 2)]

    valid_ops = {'<', '>', '=', '<=', '>='}
    if not all(o in valid_ops for o in ops):
        return []
    ops = ['<' if o == '<=' else '>' if o == '>=' else o for o in ops]

    # ── Detect mode: K-space if any segment needs it ───────────────────────
    if any(_needs_kspace_eval(e, param_names) for e in exprs_raw):
        # K-space: store raw expressions for numeric evaluation at fit time
        result = []
        for i, op in enumerate(ops):
            result.append({
                "mode":        "K",
                "lhs_expr":    exprs_raw[i],
                "rhs_expr":    exprs_raw[i + 1],
                "op":          op,
                "hard":        hard,
                "param_names": list(param_names),
            })
        return result

    # ── Log-space: parse to linear coefficients ───────────────────────────
    line_has_log = bool(re.search(r'\blog\s+\w', line, re.I))
    parsed_exprs = []
    for e in exprs_raw:
        try:
            parsed_exprs.append(_parse_constraint_expr(e, param_names, line_has_log))
        except ValueError:
            return []

    result = []
    for i, op in enumerate(ops):
        lhs = parsed_exprs[i]
        rhs = parsed_exprs[i + 1]
        combined_coeffs = dict(lhs["coeffs"])
        for name, coeff in rhs["coeffs"].items():
            combined_coeffs[name] = combined_coeffs.get(name, 0.0) - coeff
        combined_coeffs = {k: v for k, v in combined_coeffs.items() if abs(v) > 1e-15}
        combined_const  = lhs["const"] - rhs["const"]
        result.append({
            "mode":   "log",
            "coeffs": combined_coeffs,
            "const":  combined_const,
            "op":     op,
            "hard":   hard,
        })
    return result


def _eval_kspace_expr(expr_str, logK_dict, param_names):
    """
    Safely evaluate a K-space arithmetic expression.
    Parameter names are mapped to their actual K values (10^logK).
    Only arithmetic operations and standard math functions are permitted.

    Handles param names containing hyphens (e.g. 'k-1') by replacing them
    with safe Python identifiers before calling eval.

    Returns float, or raises ValueError on failure.
    """
    import math as _math

    # Build a safe-identifier mapping for any param name that isn't a valid
    # Python identifier (e.g. 'k-1' → '_p0_', 'k_1' left as-is)
    safe_map = {}   # original_name -> safe_identifier
    for name in sorted(param_names, key=len, reverse=True):
        if not name.isidentifier():
            safe_id = f"_p{len(safe_map)}_"
            safe_map[name] = safe_id

    # Substitute in the expression string (longest names first)
    expr_safe = expr_str
    for orig, safe in safe_map.items():
        expr_safe = expr_safe.replace(orig, safe)

    # Build namespace: (possibly renamed) param -> 10^logK
    ns = {}
    for name in param_names:
        val = 10 ** logK_dict.get(name, 0.0)
        ns[safe_map.get(name, name)] = val
    ns.update({
        "log": _math.log10,
        "ln":  _math.log,
        "exp": _math.exp,
        "sqrt": _math.sqrt,
        "abs": abs,
        "pi":  _math.pi,
        "e":   _math.e,
    })

    try:
        result = eval(compile(expr_safe, "<constraint>", "eval"),  # noqa: S307
                      {"__builtins__": {}}, ns)
        return float(result)
    except Exception as exc:
        raise ValueError(f"Cannot evaluate K-space expression '{expr_str}': {exc}") from exc


def constraints_penalty(constraints, logK_dict, ssr_scale: float = 1.0):
    """
    Compute total penalty for all violated constraints.

    Weights scale with ssr_scale so the penalty is always commensurate with
    the data objective regardless of units (mM², AU², ppm², …).

    Target: penalty ≈ 1000 × ssr_scale when constraint violated by 0.1 log units.
    → W = 1000 × ssr_scale / (0.1)² = 100 000 × ssr_scale

    Hard constraints use W × 1000.

    Log-space constraint f = Σ(aᵢ·logKᵢ) + c:
        '<' penalty: W * max(0,  f)²
        '>' penalty: W * max(0, -f)²
        '=' penalty: W * f²

    K-space constraint: evaluate lhs and rhs in linear space,
        f = (lhs - rhs) / max(|rhs|, 1e-30)   (relative deviation)
    then same penalty formula.
    """
    if not constraints:
        return 0.0

    _scale = max(abs(ssr_scale), 1e-30)
    W_SOFT = _scale * 1e5    # ×100 000 so 0.1-unit violation → 1000× SSR
    W_HARD = _scale * 1e8

    penalty = 0.0
    for c in constraints:
        hard = c.get("hard", False)
        W = W_HARD if hard else W_SOFT

        if c.get("mode", "log") == "K":
            try:
                lhs_val = _eval_kspace_expr(c["lhs_expr"], logK_dict, c["param_names"])
                rhs_val = _eval_kspace_expr(c["rhs_expr"], logK_dict, c["param_names"])
                # Normalise to relative deviation so K-space and log-space are comparable
                val = (lhs_val - rhs_val) / max(abs(rhs_val), 1e-30)
            except Exception:
                continue
        else:
            val = (sum(coeff * logK_dict.get(name, 0.0)
                       for name, coeff in c["coeffs"].items())
                   + c["const"])

        op = c["op"]
        if op == '<':
            if val > 0:
                penalty += W * val * val
        elif op == '>':
            if val < 0:
                penalty += W * val * val
        else:
            penalty += W * val * val

    return penalty


def parse_script(text: str) -> dict:
    result = {
        "concentrations":   {},
        "conc_bounds":      {},   # name -> (lo_mM, hi_mM) or (None, None)
        "volumes":          {},
        "titrant":          {},
        "titrant_bounds":   {},   # tkey -> (lo_mM, hi_mM) or (None, None)
        "titrant_is_solid": False,
        "titrant_ratios":   {},
        "equilibria":       [],      # pre-equilibrium reactions (= type)
        "kinetics":         [],      # kinetic reactions (> and <> types)
        "is_kinetics":      False,   # True if any > or <> found
        "variables":        {},
        "plot_xmax":        3.0,
        "plot_x_expr":      None,
        "plot_y":           [],
        "nmr":              None,   # None | {"mode": "shift"|"integration", "targets": [str]}
        "spectra":          None,   # None | {"transparent": [str]}
        "constraints":      [],     # parsed constraint objects (populated after knames known)
        "temperature_K":    298.15, # default 298.15 K; overridden by $temperature section
        "warnings":         [],     # non-fatal issues collected during parsing
    }
    _raw_constraint_lines = []   # collected during section pass, parsed after all knames known

    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"(?<!\w)#.*$", "", line).strip()  # strip comments but preserve DG#
        if not line:
            continue

        if line.startswith("$"):
            section_raw = line[1:].lower().strip()
            if section_raw.startswith("titrant"):
                section = "titrant"
                result["titrant_is_solid"] = "solid" in section_raw
            elif section_raw == "reactions":
                section = "equilibria"   # map new name to internal key
            elif section_raw == "temperature":
                section = "temperature"
            elif section_raw == "spectra":
                section = "spectra"
                if result["spectra"] is None:
                    result["spectra"] = {"transparent": []}  # init on section entry
            else:
                section = section_raw
            continue

        num_re = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"

        if section == "temperature":
            # Accepts: "T = 300 K"  or  "T = 25 C"  (unit optional, default K)
            m = re.match(r"T\s*=\s*(.+)", line, re.IGNORECASE)
            if m:
                val_str = m.group(1).strip()
                parts   = val_str.split()
                try:
                    val  = float(parts[0])
                    unit = parts[1] if len(parts) > 1 else "K"
                    result["temperature_K"] = convert_temperature_to_K(val, unit)
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Error parsing temperature '{line}': {e}")
            else:
                raise ValueError(
                    f"Expected format 'T = value [K|C]' in $temperature, got: '{line}'")

        elif section == "concentrations":
            m = re.match(r"(\w+)\s*=\s*(.+)", line)
            if m:
                name = m.group(1)
                rest = m.group(2).strip()
                # Split on ";" to separate value from optional bounds
                parts = [p.strip() for p in rest.split(";")]
                value_str = parts[0]
                try:
                    value, unit = parse_value_with_units(value_str)
                    mM_value = convert_concentration_to_mM(value, unit)
                    result["concentrations"][name] = mM_value
                except ValueError as e:
                    raise ValueError(f"Error parsing concentration '{line}': {e}")
                # Parse optional "from X to Y" bounds (in same units as value)
                _num_pat = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                _lo_mM, _hi_mM = None, None
                for bp in parts[1:]:
                    _m_ft = re.match(rf"from\s+({_num_pat})\s+to\s+({_num_pat})", bp, re.IGNORECASE)
                    if _m_ft:
                        try:
                            _lo_mM = convert_concentration_to_mM(float(_m_ft.group(1)), unit)
                            _hi_mM = convert_concentration_to_mM(float(_m_ft.group(2)), unit)
                        except Exception:
                            pass
                # Hard-clip lower bound to 0
                if _lo_mM is not None and _lo_mM < 0:
                    _lo_mM = 0.0
                result["conc_bounds"][name] = (_lo_mM, _hi_mM)

        elif section == "volumes":
            m = re.match(r"(\w+)\s*=\s*(.+)", line)
            if m:
                name = m.group(1)
                value_str = m.group(2).strip()
                try:
                    value, unit = parse_value_with_units(value_str)
                    # Convert to mL (internal standard)
                    mL_value = convert_volume_to_mL(value, unit)
                    result["volumes"][name] = mL_value
                except ValueError as e:
                    raise ValueError(f"Error parsing volume '{line}': {e}")

        elif section == "titrant":
            if result["titrant_is_solid"]:
                # Solid syntax: "Mt" or "Mt; 1.3"
                parts = line.split(";")
                tkey  = parts[0].strip()
                ratio = float(parts[1].strip()) if len(parts) > 1 else 1.0
                # Store sentinel 0.0 as stock conc (not used), record ratio separately
                result["titrant"][tkey]        = 0.0
                result["titrant_ratios"][tkey] = ratio
            else:
                # Liquid syntax: "Mt = 10.00 mM; from 8.0 to 12.0"
                m = re.match(r"(\w+)\s*=\s*(.+)", line)
                if m:
                    name = m.group(1)
                    rest = m.group(2).strip()
                    parts = [p.strip() for p in rest.split(";")]
                    value_str = parts[0]
                    try:
                        value, unit = parse_value_with_units(value_str)
                        mM_value = convert_concentration_to_mM(value, unit)
                        result["titrant"][name] = mM_value
                    except ValueError as e:
                        raise ValueError(f"Error parsing titrant '{line}': {e}")
                    # Parse optional "from X to Y" bounds
                    _num_pat = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                    _lo_mM, _hi_mM = None, None
                    for bp in parts[1:]:
                        _m_ft = re.match(rf"from\s+({_num_pat})\s+to\s+({_num_pat})", bp, re.IGNORECASE)
                        if _m_ft:
                            try:
                                _lo_mM = convert_concentration_to_mM(float(_m_ft.group(1)), unit)
                                _hi_mM = convert_concentration_to_mM(float(_m_ft.group(2)), unit)
                            except Exception:
                                pass
                    if _lo_mM is not None and _lo_mM < 0:
                        _lo_mM = 0.0
                    result["titrant_bounds"][name] = (_lo_mM, _hi_mM)

        elif section == "equilibria":
            if ";" not in line:
                raise ValueError(f"Missing ';' in equilibrium line: '{line}'")

            parts    = line.split(";")
            rxn_part = parts[0].strip()
            k_parts  = [p.strip() for p in parts[1:] if p.strip()]

            # ── Detect reaction type ─────────────────────────────────
            if "<>" in rxn_part:
                lhs, rhs_rxn = rxn_part.split("<>", 1)
                rxn_type = "reversible_kinetic"
                result["is_kinetics"] = True
            elif ">" in rxn_part:
                lhs, rhs_rxn = rxn_part.split(">", 1)
                rxn_type = "irreversible"
                result["is_kinetics"] = True
            elif "=" in rxn_part:
                lhs, rhs_rxn = rxn_part.split("=", 1)
                rxn_type = "equilibrium"
            else:
                raise ValueError(f"No reaction arrow (=, >, <>) in: '{line}'")

            # ── Parse species lists ──────────────────────────────────
            def _parse_side(side_str):
                side_str = side_str.strip()
                if "+" in side_str:
                    return [p.strip() for p in side_str.split(" + ") if p.strip()]
                return [side_str] if side_str else []

            reactants_str = _parse_side(lhs)
            products_str  = _parse_side(rhs_rxn)

            reactants = [parse_species_with_stoich(r) for r in reactants_str]
            products  = [parse_species_with_stoich(p) for p in products_str]

            # ── Classify input mode: DG-based or k/K-based ───────────
            # DG# prefix  → activation free energy ΔG‡ (kcal/mol)
            # DG prefix   → reaction free energy ΔG   (kcal/mol)
            # Patterns are mutually exclusive because '#' is not \w.
            _DG_ACT_RE  = re.compile(r"^DG#(\w*)\s*=\s*(.+)$",  re.IGNORECASE)
            _DG_RXN_RE  = re.compile(r"^DG(\w*)\s*=\s*(.+)$",   re.IGNORECASE)
            _KRATE_RE   = re.compile(r"^(log\s+)?(k[A-Za-z0-9_\-]*)\s*=\s*(.+)$", re.IGNORECASE)
            _KEQ_RE     = re.compile(r"^(log\s+)?([A-Za-z0-9_,]+)\s*=\s*(.+)$",   re.IGNORECASE)

            _dg_act_parts = [kp for kp in k_parts if _DG_ACT_RE.match(kp)]
            _dg_rxn_parts = [kp for kp in k_parts if _DG_RXN_RE.match(kp)
                                                    and not _DG_ACT_RE.match(kp)]
            _k_parts_mode = [kp for kp in k_parts
                             if not _DG_ACT_RE.match(kp) and not _DG_RXN_RE.match(kp)]

            _is_dg_mode = bool(_dg_act_parts or _dg_rxn_parts)

            # Mixing DG and k/K inputs is not allowed
            if _is_dg_mode and _k_parts_mode:
                raise ValueError(
                    f"Cannot mix DG/DG# and k/K inputs in the same reaction: '{line}'. "
                    f"Use either free-energy (DG#/DG) or rate/equilibrium constants (k/K) — not both.")

            # ── Parse constants ──────────────────────────────────────
            import math

            if _is_dg_mode:
                # ── Free-energy (Eyring / van 't Hoff) input mode ────
                T_K = result["temperature_K"]

                if rxn_type == "equilibrium":
                    # Only DG is meaningful; DG# (activation barrier) makes no sense
                    # for a pseudo-equilibrium where we never track individual k values.
                    if _dg_act_parts:
                        raise ValueError(
                            f"DG# (activation energy) is not valid for equilibrium (=) "
                            f"reactions — use DG alone, or change '=' to '<>': '{line}'")
                    if not _dg_rxn_parts:
                        raise ValueError(f"Missing DG in DG-mode equilibrium reaction: '{line}'")
                    _m = _DG_RXN_RE.match(_dg_rxn_parts[0])
                    _label = _m.group(1) or "rxn"
                    _dg    = float(_m.group(2))
                    _logK  = -_dg / (2.303 * _R_KCAL * T_K)
                    _kname = f"K{_label}"
                    result["equilibria"].append({
                        "type":       "equilibrium",
                        "reactants":  reactants,
                        "products":   products,
                        "kname":      _kname,
                        "logK":       _logK,
                        "logK_lo":    None,
                        "logK_hi":    None,
                        "input_mode": "dg",
                        "dg_kcal":    _dg,
                    })

                elif rxn_type == "reversible_kinetic":
                    # Requires both DG# (forward barrier) and DG (reaction energy)
                    if not _dg_act_parts:
                        raise ValueError(
                            f"Reversible kinetic (<>) reaction in DG mode requires DG# "
                            f"(activation energy): '{line}'")
                    if not _dg_rxn_parts:
                        raise ValueError(
                            f"Reversible kinetic (<>) reaction in DG mode requires DG "
                            f"(reaction energy): '{line}'")
                    _m_act  = _DG_ACT_RE.match(_dg_act_parts[0])
                    _m_rxn  = _DG_RXN_RE.match(_dg_rxn_parts[0])
                    _label  = _m_act.group(1) or "1"
                    _dg_act = float(_m_act.group(2))
                    _dg_rxn = float(_m_rxn.group(2))
                    # Reverse barrier: ΔG‡_rev = ΔG‡_fwd − ΔG_rxn  (must be > 0)
                    _dg_rev = _dg_act - _dg_rxn
                    if _dg_rev <= 0:
                        raise ValueError(
                            f"Unphysical energy profile: ΔG‡_rev = DG# − DG = "
                            f"{_dg_act} − {_dg_rxn} = {_dg_rev:.3f} kcal/mol ≤ 0. "
                            f"The transition state cannot be lower than the products: '{line}'")
                    _log_kf = _eyring_log10k(_dg_act, T_K)
                    _log_kr = _eyring_log10k(_dg_rev, T_K)
                    _kname_fwd = f"kf{_label}"
                    _kname_rev = f"kr{_label}"
                    result["kinetics"].append({
                        "type":       "reversible_kinetic",
                        "reactants":  reactants,
                        "products":   products,
                        "kname":      _kname_fwd,
                        "log_k":      _log_kf,
                        "krname":     _kname_rev,
                        "log_kr":     _log_kr,
                        "input_mode": "dg",
                        "dg_act_kcal": _dg_act,
                        "dg_rxn_kcal": _dg_rxn,
                    })

                else:  # irreversible
                    if not _dg_act_parts:
                        raise ValueError(
                            f"Irreversible kinetic (>) reaction in DG mode requires DG# "
                            f"(activation energy): '{line}'")
                    if _dg_rxn_parts:
                        result["warnings"].append(
                            f"DG (reaction energy) is ignored for irreversible (>) reaction "
                            f"'{rxn_part.strip()}' — only DG# is used.")
                    _m_act  = _DG_ACT_RE.match(_dg_act_parts[0])
                    _label  = _m_act.group(1) or "1"
                    _dg_act = float(_m_act.group(2))
                    _log_kf = _eyring_log10k(_dg_act, T_K)
                    _kname_fwd = f"kf{_label}"
                    result["kinetics"].append({
                        "type":        "irreversible",
                        "reactants":   reactants,
                        "products":    products,
                        "kname":       _kname_fwd,
                        "log_k":       _log_kf,
                        "input_mode":  "dg",
                        "dg_act_kcal": _dg_act,
                    })

            elif rxn_type == "equilibrium":
                # ── Classic K / log K input ───────────────────────────
                if not k_parts:
                    raise ValueError(f"Missing K in: '{line}'")
                kp = k_parts[0]
                k_match = re.match(r"^(log\s+)?([A-Za-z0-9_,]+)\s*=\s*(.+)$", kp, re.IGNORECASE)
                if not k_match:
                    raise ValueError(f"Cannot parse K in: '{line}'")
                is_log   = k_match.group(1) is not None
                kname    = k_match.group(2)
                kval_str = k_match.group(3)
                kval     = float(kval_str)
                logK     = kval if is_log else math.log10(kval)

                # ── Optional soft bounds: "from X to Y", "from X", "to Y" ──
                _num_pat  = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                _bound_lo = None
                _bound_hi = None
                for _bp in k_parts[1:]:
                    _bp = _bp.strip()
                    _m_from_to = re.match(
                        rf"from\s+({_num_pat})\s+to\s+({_num_pat})", _bp, re.IGNORECASE)
                    _m_from    = re.match(rf"from\s+({_num_pat})$", _bp, re.IGNORECASE)
                    _m_to      = re.match(rf"to\s+({_num_pat})$",   _bp, re.IGNORECASE)
                    if _m_from_to:
                        _lo = float(_m_from_to.group(1))
                        _hi = float(_m_from_to.group(2))
                        _bound_lo = _lo if is_log else math.log10(_lo)
                        _bound_hi = _hi if is_log else math.log10(_hi)
                    elif _m_from:
                        _lo = float(_m_from.group(1))
                        _bound_lo = _lo if is_log else math.log10(_lo)
                    elif _m_to:
                        _hi = float(_m_to.group(1))
                        _bound_hi = _hi if is_log else math.log10(_hi)

                result["equilibria"].append({
                    "type":      "equilibrium",
                    "reactants": reactants,
                    "products":  products,
                    "kname":     kname,
                    "logK":      logK,
                    "logK_lo":   _bound_lo,
                    "logK_hi":   _bound_hi,
                })

            else:
                # ── Classic k / log k kinetic input ──────────────────
                krate_pat = re.compile(
                    r"^(log\s+)?(k[A-Za-z0-9_\-]*)\s*=\s*(.+)$", re.IGNORECASE)
                parsed_ks = {}
                for kp in k_parts:
                    m = krate_pat.match(kp)
                    if not m:
                        raise ValueError(f"Cannot parse rate constant '{kp}' in: '{line}'")
                    is_log = m.group(1) is not None
                    kname  = m.group(2)
                    kval   = float(m.group(3))
                    log_k  = kval if is_log else math.log10(kval)
                    parsed_ks[kname] = log_k

                if not parsed_ks:
                    raise ValueError(f"No rate constants found in: '{line}'")

                knames_list = list(parsed_ks.keys())
                kname_fwd   = knames_list[0]
                entry = {
                    "type":      rxn_type,
                    "reactants": reactants,
                    "products":  products,
                    "kname":     kname_fwd,
                    "log_k":     parsed_ks[kname_fwd],
                }
                if rxn_type == "reversible_kinetic":
                    if len(knames_list) < 2:
                        raise ValueError(
                            f"Reversible reaction '<>' needs k_fwd and k_rev: '{line}'")
                    kname_rev = knames_list[1]
                    entry["krname"] = kname_rev
                    entry["log_kr"] = parsed_ks[kname_rev]
                result["kinetics"].append(entry)

        elif section == "variables":
            m = re.match(r"(%?\w+)\s*=\s*(.+)", line)
            if m:
                varname = m.group(1).strip()
                expression = m.group(2).strip()
                result["variables"][varname] = expression

        elif section == "plot":
            m_xmax = re.match(rf"xmax\s*=\s*({num_re})", line)
            m_x    = re.match(r"x\s*=\s*(.+)",           line)
            m_y    = re.match(r"y\s*=\s*(.+)",           line)
            if m_xmax:
                result["plot_xmax"] = float(m_xmax.group(1))
            if m_x and not m_xmax:   # "x = expr" but not "xmax = ..."
                result["plot_x_expr"] = m_x.group(1).strip()
            if m_y:
                result["plot_y"] = [s.strip() for s in m_y.group(1).split(",") if s.strip()]

        elif section == "nmr":
            # "shift: Gtot"           -> fast-exchange chemical shift mode
            # "integration: 1, 3, 1"  -> slow-exchange integration mode
            # Both lines together     -> mixed slow+fast exchange mode
            m_shift = re.match(r"shift\s*:\s*(.+)", line, re.IGNORECASE)
            m_integ = re.match(r"integration\s*:\s*(.+)", line, re.IGNORECASE)
            if m_shift:
                targets = [t.strip() for t in m_shift.group(1).split(",") if t.strip()]
                if result["nmr"] is None:
                    result["nmr"] = {"mode": "shift", "targets": targets,
                                     "n_H_list": [], "n_integ": 0}
                else:
                    result["nmr"]["mode"]    = "mixed"
                    result["nmr"]["targets"] = targets
            elif m_integ:
                parts    = [t.strip() for t in m_integ.group(1).split(",") if t.strip()]
                n_H_list = []
                for p in parts:
                    try:    n_H_list.append(float(p))
                    except ValueError: pass
                if result["nmr"] is None:
                    result["nmr"] = {"mode": "integration", "n_H_list": n_H_list,
                                     "n_integ": len(n_H_list), "targets": []}
                else:
                    result["nmr"]["mode"]     = "mixed"
                    result["nmr"]["n_H_list"] = n_H_list
                    result["nmr"]["n_integ"]  = len(n_H_list)

        elif section == "spectra":
            # "transparent: H, X"  → species with zero absorbance everywhere
            # Also accept "transparent:" with no species (all species absorb)
            m_trans = re.match(r"transparent\s*:(.*)", line, re.IGNORECASE)
            if m_trans:
                raw = m_trans.group(1).strip()
                transparent = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
                if result["spectra"] is None:
                    result["spectra"] = {"transparent": transparent}
                else:
                    result["spectra"]["transparent"] = transparent

        elif section == "constraints":
            # Collect raw lines; parsed after all parameter names are known
            _raw_constraint_lines.append(line)

    # ── Deduplicate knames across both equilibria and kinetics ───────────────
    from collections import Counter
    all_knames = ([eq["kname"] for eq in result["equilibria"]] +
                  [r["kname"]  for r in result["kinetics"]] +
                  [r["krname"] for r in result["kinetics"] if "krname" in r])
    kname_counts = Counter(all_knames)
    kname_seen   = Counter()
    for eq in result["equilibria"]:
        kn = eq["kname"]
        if kname_counts[kn] > 1:
            kname_seen[kn] += 1
            eq["kname"] = f"{kn}{kname_seen[kn]}"
    for rxn in result["kinetics"]:
        for attr in ("kname", "krname"):
            if attr in rxn:
                kn = rxn[attr]
                if kname_counts[kn] > 1:
                    kname_seen[kn] += 1
                    rxn[attr] = f"{kn}{kname_seen[kn]}"

    # ── Parse $constraints now that all parameter names are known ─────────────
    if _raw_constraint_lines:
        _all_param_names = (
            [eq["kname"] for eq in result["equilibria"]] +
            [r["kname"]  for r in result["kinetics"]] +
            [r["krname"] for r in result["kinetics"] if "krname" in r]
        )
        for _cline in _raw_constraint_lines:
            try:
                _cs = _parse_constraint_line(_cline, _all_param_names)
                if _cs:
                    result["constraints"].extend(_cs)
            except Exception:
                pass  # silently skip unparseable lines in fast parse

    return result



def check_script_syntax(text: str) -> list:
    """
    Line-by-line syntax checker. Returns a list of (line_number, original_line, error_message).
    Does NOT raise — collects all errors so every problem is reported at once.
    """
    import re as _re

    VALID_SECTIONS = {
        "concentrations", "volumes", "titrant", "reactions", "equilibria",
        "variables", "plot", "nmr", "spectra", "constraints", "temperature",
    }
    CONC_UNITS  = {"m", "mm", "um", "µm"}
    VOL_UNITS   = {"l", "ml", "ul", "µl"}
    NUM_RE      = _re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
    ARROW_RE    = _re.compile(r"(<>|>|=)")
    NAME_RE     = _re.compile(r"^[A-Za-z_][A-Za-z0-9_,]*$")

    errors  = []
    section = None

    # ── Cross-section tracking (populated during line pass) ──────────────
    defined_concs         = {}   # name -> (lineno, raw)
    defined_vols          = {}   # name -> (lineno, raw)
    defined_titrant       = {}   # name -> (lineno, raw)
    defined_vars          = {}   # name -> (lineno, raw)
    reaction_species      = set()
    reaction_species_refs = []   # (lineno, raw, name)
    plot_y_refs           = []   # (lineno, raw, name)
    plot_x_tokens         = []   # (lineno, raw, token)
    nmr_shift_refs        = []   # (lineno, raw, name)
    var_expr_refs         = []   # (lineno, raw, token)

    def err(lineno, raw, msg):
        errors.append((lineno, raw, msg))

    def is_number(s):
        return bool(NUM_RE.match(s.strip()))

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _re.sub(r"(?<!\w)#.*$", "", line).strip()  # strip comments but preserve DG#
        if not line:
            continue

        # ── Section headers ──────────────────────────────────────────
        if line.startswith("$"):
            sec_raw = line[1:].lower().strip()
            # $titrant or $titrant solid
            base = sec_raw.split()[0]
            if base == "titrant":
                section = "titrant"
            elif base in VALID_SECTIONS:
                section = base
                if "solid" in sec_raw and base != "titrant":
                    err(lineno, raw_line, f"'solid' modifier is only valid for $titrant, not ${base}.")
            else:
                err(lineno, raw_line, f"Unknown section '${sec_raw}'. Valid sections: {', '.join('$'+s for s in sorted(VALID_SECTIONS))}.")
                section = None
            continue

        if section is None:
            err(lineno, raw_line, "Line appears before any section header ($concentrations, $reactions, etc.).")
            continue

        # ── $concentrations ──────────────────────────────────────────
        if section == "concentrations":
            m = _re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: Name = value [unit]  (e.g. G0 = 1.00 mM).")
                continue
            defined_concs[m.group(1)] = (lineno, raw_line)
            if not m.group(1).endswith('0'):
                err(lineno, raw_line,
                    f"Concentration name '{m.group(1)}' should end in '0' to indicate an initial amount (e.g. '{m.group(1)}0').")
            val_str = m.group(2).strip()
            parts   = val_str.split()
            num_part = parts[0]
            unit_part = parts[1].lower() if len(parts) > 1 else "mm"
            if not is_number(num_part):
                err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
            elif unit_part not in CONC_UNITS:
                err(lineno, raw_line, f"Unknown concentration unit '{parts[1]}'. Use M, mM, or uM.")

        # ── $volumes ─────────────────────────────────────────────────
        elif section == "volumes":
            m = _re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: V0 = value [unit]  (e.g. V0 = 500 uL).")
                continue
            defined_vols[m.group(1)] = (lineno, raw_line)
            val_str   = m.group(2).strip()
            parts     = val_str.split()
            num_part  = parts[0]
            unit_part = parts[1].lower() if len(parts) > 1 else "ml"
            if not is_number(num_part):
                err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
            elif unit_part not in VOL_UNITS:
                err(lineno, raw_line, f"Unknown volume unit '{parts[1]}'. Use L, mL, or uL.")

        # ── $titrant ─────────────────────────────────────────────────
        elif section == "titrant":
            # Detect whether solid from the section header (stored externally);
            # here we just check the line format.
            parts = [p.strip() for p in line.split(";")]
            # Liquid: "Ht = 10.00 mM" or solid: "Ht" or "Ht; 1.3"
            if "=" in parts[0]:
                # liquid-mode line
                m = _re.match(r"^(\w+)\s*=\s*(.+)$", parts[0])
                if not m:
                    err(lineno, raw_line, "Expected format: Name = value [unit]  (e.g. Ht = 10.00 mM).")
                    continue
                defined_titrant[m.group(1)] = (lineno, raw_line)
                val_str   = m.group(2).strip()
                tparts    = val_str.split()
                num_part  = tparts[0]
                unit_part = tparts[1].lower() if len(tparts) > 1 else "mm"
                if not is_number(num_part):
                    err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
                elif unit_part not in CONC_UNITS:
                    err(lineno, raw_line, f"Unknown concentration unit '{tparts[1]}'. Use M, mM, or uM.")
            else:
                # solid-mode line: just a name, optionally followed by "; ratio"
                name = parts[0]
                if not _re.match(r"^\w+$", name):
                    err(lineno, raw_line, f"Invalid titrant name '{name}'.")
                else:
                    defined_titrant[name] = (lineno, raw_line)
                if len(parts) > 1:
                    if not is_number(parts[1]):
                        err(lineno, raw_line, f"Stoichiometric ratio '{parts[1]}' is not a valid number.")

        # ── $temperature ──────────────────────────────────────────────
        elif section == "temperature":
            m = _re.match(r"^T\s*=\s*(.+)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line,
                    "Expected format: T = value [K|C]  (e.g. T = 300 K  or  T = 25 C).")
            else:
                val_str   = m.group(1).strip()
                parts     = val_str.split()
                num_part  = parts[0]
                unit_part = parts[1].upper() if len(parts) > 1 else "K"
                if not is_number(num_part):
                    err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
                elif unit_part not in ("K", "C"):
                    err(lineno, raw_line,
                        f"Unknown temperature unit '{parts[1]}'. Use K (Kelvin) or C (Celsius).")

        # ── $reactions / $equilibria ──────────────────────────────────
        elif section in ("reactions", "equilibria"):
            # Must contain an arrow
            if not ARROW_RE.search(line):
                err(lineno, raw_line, "Reaction line must contain an arrow: = (equilibrium), > (irreversible), or <> (reversible).")
                continue
            # Split off the semicolon parameters
            main_part = line.split(";")[0].strip()
            arrow_m = ARROW_RE.search(main_part)
            if not arrow_m:
                err(lineno, raw_line, "Cannot find arrow in reaction left-hand/right-hand split.")
                continue
            arrow     = arrow_m.group(0)
            arrow_pos = arrow_m.start()
            lhs = main_part[:arrow_pos].strip()
            rhs = main_part[arrow_pos + len(arrow):].strip()
            if not lhs:
                err(lineno, raw_line, "Reaction has no reactants (left side is empty).")
            if not rhs:
                err(lineno, raw_line, "Reaction has no products (right side is empty).")
            # Determine reaction type from arrow
            if "<>" in main_part:
                _rxn_type = "reversible_kinetic"
            elif ">" in main_part:
                _rxn_type = "irreversible"
            else:
                _rxn_type = "equilibrium"
            # Collect species names for cross-section checks
            for _side_str in (lhs, rhs):
                for _tok in _side_str.split("+"):
                    _tok = _tok.strip()
                    _m_sp = _re.match(r"^\d*\.?\d*\s*([A-Za-z_]\w*)$", _tok)
                    if _m_sp:
                        _sp = _m_sp.group(1)
                        reaction_species.add(_sp)
                        reaction_species_refs.append((lineno, raw_line, _sp))
            # Check for K/k/DG parameter
            k_parts = [p.strip() for p in line.split(";")[1:] if p.strip()]
            if not k_parts:
                err(lineno, raw_line,
                    "Reaction is missing equilibrium/rate constant (e.g. ; log K1 = 4.0  or  ; DG#1 = 10.0).")
            else:
                _DG_ACT_RE_s = _re.compile(r"^DG#(\w*)\s*=\s*(.+)$", _re.IGNORECASE)
                _DG_RXN_RE_s = _re.compile(r"^DG(\w*)\s*=\s*(.+)$",   _re.IGNORECASE)
                _K_ANY_RE_s  = _re.compile(r"^(log\s+)?([A-Za-z0-9_,\-]+)\s*=\s*(.+)$", _re.IGNORECASE)

                _has_dg_act = any(_DG_ACT_RE_s.match(kp) for kp in k_parts)
                _has_dg_rxn = any(
                    _DG_RXN_RE_s.match(kp) and not _DG_ACT_RE_s.match(kp)
                    for kp in k_parts)
                _has_k      = any(
                    not _DG_ACT_RE_s.match(kp) and not _DG_RXN_RE_s.match(kp)
                    for kp in k_parts)
                _is_dg_mode = _has_dg_act or _has_dg_rxn

                # Mixing check
                if _is_dg_mode and _has_k:
                    err(lineno, raw_line,
                        "Cannot mix DG/DG# and k/K inputs in the same reaction. "
                        "Use either free-energy (DG#/DG) or rate/equilibrium constants (k/K).")

                elif _is_dg_mode:
                    # ── Validate DG-mode entries ──────────────────────
                    if _rxn_type == "equilibrium" and _has_dg_act:
                        err(lineno, raw_line,
                            "DG# (activation energy) is not valid for equilibrium (=) reactions. "
                            "Use DG alone, or change '=' to '<>'.")
                    if _rxn_type == "reversible_kinetic" and not _has_dg_act:
                        err(lineno, raw_line,
                            "Reversible kinetic (<>) reaction in DG mode requires DG# "
                            "(activation energy).")
                    if _rxn_type == "reversible_kinetic" and not _has_dg_rxn:
                        err(lineno, raw_line,
                            "Reversible kinetic (<>) reaction in DG mode requires DG "
                            "(reaction free energy).")
                    if _rxn_type == "irreversible" and not _has_dg_act:
                        err(lineno, raw_line,
                            "Irreversible kinetic (>) reaction in DG mode requires DG# "
                            "(activation energy).")
                    # Check all DG values are numbers
                    for kp in k_parts:
                        m_da = _DG_ACT_RE_s.match(kp)
                        m_dr = _DG_RXN_RE_s.match(kp)
                        if m_da:
                            if not is_number(m_da.group(2).strip()):
                                err(lineno, raw_line,
                                    f"DG# value '{m_da.group(2).strip()}' is not a valid number.")
                        elif m_dr:
                            if not is_number(m_dr.group(2).strip()):
                                err(lineno, raw_line,
                                    f"DG value '{m_dr.group(2).strip()}' is not a valid number.")

                else:
                    # ── Validate classic k/K entries ──────────────────
                    kp = k_parts[0]
                    km = _re.match(r"^(log\s+)?([A-Za-z0-9_,\-]+)\s*=\s*(.+)$", kp, _re.IGNORECASE)
                    if not km:
                        err(lineno, raw_line,
                            f"Cannot parse constant '{kp}'. Expected: log Name = value  or  Name = value.")
                    else:
                        val = km.group(3).strip()
                        if not is_number(val):
                            err(lineno, raw_line,
                                f"Constant value '{val}' is not a valid number.")
                    # Validate optional bounds or second k for reversible
                    for extra in k_parts[1:]:
                        bm = _re.match(
                            r"^(from\s+\S+\s+to\s+\S+|from\s+\S+|to\s+\S+)$",
                            extra.strip(), _re.IGNORECASE)
                        if not bm:
                            km2 = _re.match(
                                r"^(log\s+)?([A-Za-z0-9_,\-]+)\s*=\s*(.+)$",
                                extra, _re.IGNORECASE)
                            if not km2:
                                err(lineno, raw_line,
                                    f"Unrecognised parameter clause '{extra}'. "
                                    "Expected: log Name = value  or  from X to Y.")

        # ── $variables ────────────────────────────────────────────────
        elif section == "variables":
            m = _re.match(r"^(%?\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: Name = expression  (e.g. Gtot = G + GH + GH2).")
            else:
                expr = m.group(2).strip()
                if not expr:
                    err(lineno, raw_line, "Variable expression is empty.")
                else:
                    defined_vars[m.group(1)] = (lineno, raw_line)
                    for _tok in (_re.findall(r"[A-Za-z_]\w*", expr) +
                                 _re.findall(r"%\w+", expr)):
                        var_expr_refs.append((lineno, raw_line, _tok))

        # ── $plot ─────────────────────────────────────────────────────
        elif section == "plot":
            m = _re.match(r"^(xmax|x|y)\s*=\s*(.+)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Expected one of: xmax = value, x = expression, y = species1, species2, …")
            else:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "xmax" and not is_number(val):
                    err(lineno, raw_line, f"xmax value '{val}' is not a valid number.")
                elif key == "y":
                    for _name in [n.strip() for n in val.split(",") if n.strip()]:
                        plot_y_refs.append((lineno, raw_line, _name))
                        # Also register %name in defined_vars if not already there
                        # so the cross-section check doesn't flag it as undefined
                elif key == "x":
                    for _tok in _re.findall(r"[A-Za-z_]\w*", val):
                        plot_x_tokens.append((lineno, raw_line, _tok))

        # ── $nmr ─────────────────────────────────────────────────────
        elif section == "nmr":
            m = _re.match(r"^(shift|integration)\s*:\s*(.*)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Expected: shift: TargetName  or  integration: n1, n2, …")
            else:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "integration" and val:
                    for tok in [t.strip() for t in val.split(",") if t.strip()]:
                        if not is_number(tok):
                            err(lineno, raw_line, f"integration: expects numbers separated by commas; '{tok}' is not a valid number.")
                elif key == "shift" and val:
                    for _name in [t.strip() for t in val.split(",") if t.strip()]:
                        nmr_shift_refs.append((lineno, raw_line, _name))

        # ── $spectra ──────────────────────────────────────────────────
        elif section == "spectra":
            m = _re.match(r"^transparent\s*:\s*(.*)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Only 'transparent: Species1, Species2, …' is valid inside $spectra (or leave the section body empty).")

    # ── Cross-section validation ──────────────────────────────────────────
    # Each 'X0' in $concentrations implicitly defines 'X' as a free species
    implicit_species = {name[:-1] for name in defined_concs if name.endswith('0') and len(name) > 1}
    # X0 variables: each titrant key 'Xt' defines 'X0' (e.g. Ht → H0)
    # Each concentration 'G0' already in defined_concs; titrant 'Ht' → 'H0' is new
    titrant_x0_names = set()
    for _tkey in defined_titrant:
        _tfree = _tkey[:-1] if _tkey.endswith('t') else _tkey
        titrant_x0_names.add(_tfree + '0')
    valid_names = (reaction_species
                   | set(defined_concs)
                   | set(defined_titrant)
                   | set(defined_vars)
                   | implicit_species
                   | titrant_x0_names)
    vol_names   = set(defined_vols)

    # Concentrations defined but never used in any reaction, plot x, plot y, or variable expression
    # Also count usage of the implicit species (e.g. 'G' counts as usage of 'G0')
    _all_refs = (reaction_species
                 | {tok for _, _, tok in plot_x_tokens}
                 | {tok for _, _, tok in plot_y_refs}
                 | {tok for _, _, tok in var_expr_refs}
                 | {tok for _, _, tok in nmr_shift_refs})
    for _name in defined_concs:
        _implicit = _name[:-1] if _name.endswith('0') and len(_name) > 1 else None
        if _name not in _all_refs and (_implicit is None or _implicit not in _all_refs):
            _ln, _raw = defined_concs[_name]
            err(_ln, _raw,
                f"'{_name}' is defined in $concentrations but never appears in any reaction or expression. Possible typo?")

    # Volume names must not appear as reaction species
    for _ln, _raw, _sp in reaction_species_refs:
        if _sp in vol_names:
            err(_ln, _raw,
                f"'{_sp}' is defined as a volume in $volumes and cannot be used as a chemical species.")

    # All names in $plot y must be defined
    for _ln, _raw, _name in plot_y_refs:
        if _name not in valid_names:
            err(_ln, _raw,
                f"'{_name}' in '$plot y' is not defined as a species, concentration, titrant, or variable.")

    # All identifier tokens in $plot x expression must be defined
    for _ln, _raw, _tok in plot_x_tokens:
        if _tok in vol_names:
            err(_ln, _raw,
                f"'{_tok}' is a volume ($volumes) and cannot be used in a '$plot x' expression.")
        elif _tok not in valid_names:
            err(_ln, _raw,
                f"'{_tok}' in '$plot x' expression is not defined as a species, concentration, titrant, or variable.")

    # All targets in $nmr shift must be defined
    for _ln, _raw, _name in nmr_shift_refs:
        if _name not in valid_names:
            err(_ln, _raw,
                f"'{_name}' in '$nmr shift' is not defined as a species, concentration, or variable.")

    # All identifier tokens in $variables expressions must be defined
    for _ln, _raw, _tok in var_expr_refs:
        if _tok in vol_names:
            err(_ln, _raw,
                f"'{_tok}' is a volume ($volumes) and cannot be used in a '$variables' expression.")
        elif _tok not in valid_names:
            err(_ln, _raw,
                f"'{_tok}' in '$variables' expression is not defined as a species, concentration, titrant, or variable.")

    return errors
