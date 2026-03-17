"""equilibrist_network.py"""
import re
import re
import warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space as _null_space

__all__ = ['extract_identifiers_from_expression', 'resolve_variable_dependencies', '_sanitise_pct', '_sanitise_expr', 'evaluate_variable_expression', 'compute_variable_curve', 'find_variable_for_exp_col', 'transform_exp_via_variable', 'build_network', 'solve_equilibria_general', 'solve_equilibria_rigorous', 'detect_thermodynamic_cycles', '_LN_LO', '_LN_HI', 'solve_free_species']


# ─────────────────────────────────────────────
# VARIABLE EXPRESSION SYSTEM
# ─────────────────────────────────────────────

def extract_identifiers_from_expression(expr: str) -> set:
    """Extract variable/species names from a mathematical expression.
    Handles both plain names (Gtot) and %-prefixed names (%AB)."""
    import re
    identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))  # plain names
    identifiers |= set(re.findall(r'%\w+', expr))              # %name tokens
    return identifiers


def resolve_variable_dependencies(variables: dict) -> list:
    """
    Resolve variable dependencies using topological sort.
    
    Returns list of variable names in evaluation order.
    """
    # Build dependency graph
    dependencies = {}
    for var_name, expression in variables.items():
        deps = extract_identifiers_from_expression(expression)
        # Only include dependencies that are other variables (not species)
        var_deps = deps.intersection(set(variables.keys()))
        dependencies[var_name] = var_deps
    
    # Topological sort using Kahn's algorithm
    in_degree = {var: 0 for var in variables}
    for var in variables:
        for dep in dependencies[var]:
            in_degree[var] += 1
    
    queue = [var for var, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        var = queue.pop(0)
        result.append(var)
        
        # Reduce in-degree for variables that depend on this one
        for other_var, deps in dependencies.items():
            if var in deps:
                in_degree[other_var] -= 1
                if in_degree[other_var] == 0:
                    queue.append(other_var)
    
    if len(result) != len(variables):
        # Circular dependency detected
        remaining = set(variables.keys()) - set(result)
        raise ValueError(f"Circular dependency detected in variables: {remaining}")
    
    return result



def _sanitise_pct(name: str) -> str:
    """Convert %VarName to a valid Python identifier _pct_VarName for eval."""
    return name.replace("%", "_pct_") if name.startswith("%") else name

def _sanitise_expr(expr: str, pct_names: set) -> str:
    """Replace %name tokens in an expression with _pct_name so eval works."""
    import re as _re
    result = expr
    for name in sorted(pct_names, key=len, reverse=True):  # longest first
        if name.startswith("%"):
            safe = _sanitise_pct(name)
            # Replace %name that is not immediately followed by more word chars
            result = _re.sub(r"%" + _re.escape(name[1:]) + r"(?!\w)", safe, result)
    return result

def evaluate_variable_expression(expr: str, species_values: dict, variable_values: dict) -> float:
    """
    Evaluate a mathematical expression with species and variable substitution.
    
    Args:
        expr: Mathematical expression string
        species_values: {species_name: concentration_value}
        variable_values: {variable_name: computed_value}
        
    Returns:
        Evaluated result as float
    """
    # Sanitise %names → _pct_names for eval (% is not a valid identifier char)
    all_pct = {k for k in list(species_values) + list(variable_values) if k.startswith("%")}
    safe_species  = {_sanitise_pct(k): v for k, v in species_values.items()}
    safe_vars     = {_sanitise_pct(k): v for k, v in variable_values.items()}
    safe_expr     = _sanitise_expr(expr, all_pct)

    namespace = {}
    namespace.update(safe_species)
    namespace.update(safe_vars)

    try:
        result = eval(safe_expr, {"__builtins__": {}}, namespace)
        return float(result) if np.isfinite(result) else 0.0
    except Exception:
        return 0.0


def compute_variable_curve(var_name: str, variables: dict, curve: dict, network: dict, x_vals: np.ndarray) -> np.ndarray:
    """
    Compute values of an expression variable for all points in the titration curve.
    
    Args:
        var_name: Name of the variable to compute
        variables: Dict of variable definitions  
        curve: Curve data with species concentrations
        network: Network information
        x_vals: X-axis values
        
    Returns:
        Array of variable values for each titration point
    """
    n_points = len(x_vals)
    result = np.zeros(n_points)
    
    try:
        # Get variable evaluation order
        var_order = resolve_variable_dependencies(variables)
        
        # Compute variable for each titration point
        for i in range(n_points):
            # Get species concentrations at this point
            species_values = {}
            for sp in network["all_species"]:
                species_values[sp] = curve.get(sp, np.zeros(n_points))[i]
            
            # Compute variables in dependency order
            variable_values = {}
            for v_name in var_order:
                expr = variables[v_name]
                var_value = evaluate_variable_expression(expr, species_values, variable_values)
                variable_values[v_name] = var_value
            
            result[i] = variable_values.get(var_name,
                          variable_values.get(_sanitise_pct(var_name), 0.0))
            
    except Exception:
        # If computation fails, return zeros
        result[:] = 0.0
        
    return result


# ─────────────────────────────────────────────
# NETWORK BUILDER
# ─────────────────────────────────────────────


def find_variable_for_exp_col(col_name: str, variables: dict, plot_y: list,
                               all_species: list) -> str | None:
    """
    Given an experimental column name (e.g. "A"), find which variable in plot_y
    most directly depends on it.  Returns the variable name, or None if no match.

    Matching rule: the variable expression (after expanding intermediate variables)
    contains col_name as an identifier token, AND col_name is a species (not itself
    a variable).  If multiple variables match, prefer the one whose expression
    mentions col_name first / most directly.
    """
    import re as _re
    if col_name not in all_species:
        return None                    # only match raw species columns

    # Expand all variable expressions transitively so we can check deep deps
    def expand(expr, vars_dict, depth=0):
        if depth > 10:
            return expr
        for vname in sorted(vars_dict, key=len, reverse=True):
            safe_v = _sanitise_pct(vname)
            expr = _re.sub(r"(?<![\w%])" + _re.escape(vname) + r"(?!\w)", safe_v, expr)
        return expr

    for var_name in plot_y:
        if var_name not in variables:
            continue
        expanded = expand(variables[var_name], variables)
        tokens = (set(_re.findall(r"[A-Za-z_]\w*", expanded)) |
                  set(_re.findall(r"%\w+", expanded)))
        if col_name in tokens:
            return var_name
    return None


def transform_exp_via_variable(var_name: str, variables: dict,
                                exp_col: str, exp_x: np.ndarray, exp_y: np.ndarray,
                                curve: dict, x_vals: np.ndarray,
                                network: dict) -> np.ndarray:
    """
    Convert raw experimental species values into the variable's coordinate system.

    At each experimental point i:
      1. Interpolate all theoretical species from the curve at exp_x[i].
      2. Override the matched species (exp_col) with exp_y[i].
      3. Evaluate the full variable expression chain → transformed y value.

    Example:  exp_col="A", var_name="%A", expression "A/S" where S=A+B+C+D
      → result[i] = exp_y[i] / S_theory(exp_x[i])
    """
    n = len(exp_x)
    result = np.full(n, np.nan)
    if n == 0:
        return result

    var_order = resolve_variable_dependencies(variables)

    for i in range(n):
        # Interpolate theoretical species at this x-point
        species_values = {}
        for sp in network["all_species"]:
            sp_curve = curve.get(sp, np.zeros_like(x_vals))
            species_values[sp] = float(np.interp(exp_x[i], x_vals, sp_curve))

        # Override the measured species with the experimental value
        species_values[exp_col] = float(exp_y[i])

        # Evaluate variable chain in dependency order
        variable_values = {}
        for v_name in var_order:
            expr = variables[v_name]
            val = evaluate_variable_expression(expr, species_values, variable_values)
            variable_values[v_name] = val

        result[i] = variable_values.get(var_name,
                    variable_values.get(_sanitise_pct(var_name), np.nan))

    return result


def build_network(parsed: dict) -> dict:
    """
    Build the species graph from the equilibria list.
    """
    equilibria = parsed["equilibria"]
    
    # Extract product species (handle both old and new formats)
    products_set = set()
    for eq in equilibria:
        if "products" in eq:
            # New multiple products format
            for prod_coeff, prod_species in eq["products"]:
                products_set.add(prod_species)
        elif "product" in eq:
            # Old single product format (backwards compatibility)
            products_set.add(eq["product"][1])

    all_sp_set = set()
    for eq in equilibria:
        for coeff, species in eq["reactants"]:  # Now (coeff, species) tuples
            all_sp_set.add(species)
        
        # Handle both product formats
        if "products" in eq:
            for prod_coeff, prod_species in eq["products"]:
                all_sp_set.add(prod_species)
        elif "product" in eq:
            all_sp_set.add(eq["product"][1])

    # Make sure declared concentrations and titrant are in the species set
    for cname in parsed["concentrations"]:
        root = cname[:-1] if cname.endswith("0") else cname
        all_sp_set.add(root)

    # Support both old-style (M0) and new-style (Mt) titrant keys
    titrant_items = list(parsed["titrant"].items())
    titrant_free_names = []  # free species names, e.g. ['M', 'Q']
    titrant_keys = []        # script keys, e.g. ['Mt', 'Qt']
    for tkey, _ in titrant_items:
        if tkey.endswith("t"):
            tfree = tkey[:-1]
        elif tkey.endswith("0"):
            tfree = tkey[:-1]
        else:
            tfree = tkey
        titrant_free_names.append(tfree)
        titrant_keys.append(tkey)
        all_sp_set.add(tfree)

    # Primary titrant is the first one (drives volume addition)
    titrant_name = titrant_free_names[0] if titrant_free_names else "M"
    titrant_key  = titrant_keys[0]       if titrant_keys       else "Mt"

    all_species  = sorted(all_sp_set)
    free_species = sorted(all_sp_set - products_set)

    # Detect duplicate product names — two reactions producing the same species
    # would silently overwrite each other.  Raise a clear error instead.
    from collections import Counter
    product_counts = Counter()
    for eq in equilibria:
        if "products" in eq:
            for prod_coeff, prod_species in eq["products"]:
                product_counts[prod_species] += 1
        elif "product" in eq:
            product_counts[eq["product"][1]] += 1
    duplicates = [p for p, n in product_counts.items() if n > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate product name(s) in $reactions: {duplicates}. "
            "Each product must have a unique name. "
            "Check for typos (e.g. 'Q + M = QM' not 'Q + M = GM')."
        )

    # Create product-to-equilibrium mapping (simplified for multiple products)
    prod_to_eq = {}
    for eq in equilibria:
        if "products" in eq:
            # New format: multiple products - only map if single product
            if len(eq["products"]) == 1:
                prod_coeff, prod_species = eq["products"][0]
                prod_to_eq[prod_species] = eq
        elif "product" in eq:
            # Old format: backwards compatibility  
            prod_to_eq[eq["product"][1]] = eq

    # ── Recursive concentration evaluator ──
    def species_conc(name, free_concs, logK_vals, memo=None):
        # PRIORITY: Use rigorous solver results if available
        if "_rigorous_concentrations" in network:
            rigorous_concs = network["_rigorous_concentrations"]
            if name in rigorous_concs:
                return rigorous_concs[name]
        
        if memo is None:
            memo = {}
        if name in memo:
            return memo[name]
        if name in free_concs:
            memo[name] = free_concs[name]
            return free_concs[name]
        if name not in prod_to_eq:
            memo[name] = 0.0
            return 0.0
        eq  = prod_to_eq[name]
        
        # For stoichiometry: K = [product]^coeff_prod / (∏[reactant]^coeff_reactant)
        # Rearranging: [product] = K^(1/coeff_prod) * ∏[reactant]^(coeff_reactant/coeff_prod)
        
        # Handle both old and new product formats
        if "products" in eq:
            if len(eq["products"]) == 1:
                # Single product in new format
                prod_coeff, prod_species = eq["products"][0]
            else:
                # Multiple products - should use rigorous solver
                memo[name] = 1e-20  # Fallback
                return 1e-20
        else:
            # Old format  
            prod_coeff, prod_species = eq["product"]
        
        val = (10.0 ** logK_vals[eq["kname"]]) ** (1.0 / prod_coeff)
        for coeff, reactant_species in eq["reactants"]:
            reactant_conc = species_conc(reactant_species, free_concs, logK_vals, memo)
            val *= reactant_conc ** (coeff / prod_coeff)
        memo[name] = val
        return val

    # ── Stoichiometry table ──
    def count_free(name, memo=None):
        """How many of each free species goes into `name`?"""
        if memo is None:
            memo = {}
        if name in memo:
            return dict(memo[name])
        if name not in prod_to_eq:
            memo[name] = {name: 1}
            return {name: 1}
        eq = prod_to_eq[name]
        
        # Handle both old and new product formats
        if "products" in eq:
            if len(eq["products"]) == 1:
                prod_coeff, prod_species = eq["products"][0]
            else:
                # Multiple products - simplified handling
                prod_coeff = 1
        else:
            prod_coeff, prod_species = eq["product"]
        total = {}
        
        for coeff, reactant_species in eq["reactants"]:
            for k, v in count_free(reactant_species, memo).items():
                # Account for stoichiometric coefficient and product coefficient
                total[k] = total.get(k, 0) + v * coeff / prod_coeff
        memo[name] = total
        return dict(total)

    stoich = {}
    for sp in all_species:
        for fs, cnt in count_free(sp).items():
            stoich[(fs, sp)] = cnt

    return_dict = {
        "all_species":        all_species,
        "equilibria":         equilibria,        # ← THIS WAS MISSING!
        "free_species":       free_species,
        "prod_to_eq":         prod_to_eq,
        "stoich":             stoich,
        "species_conc_fn":    species_conc,
        "titrant_name":       titrant_name,       # primary free species name, e.g. 'M'
        "titrant_key":        titrant_key,         # primary script key, e.g. 'Mt'
        "titrant_free_names": titrant_free_names,  # all free names, e.g. ['M','Q']
        "titrant_keys":       titrant_keys,        # all script keys, e.g. ['Mt','Qt']
    }
    
    return return_dict


def solve_equilibria_general(totals_M, equilibria, all_species, logK_vals,
                              y0_warm=None):
    """
    Fully general equilibrium solver. Handles any combination of reactions
    with any stoichiometry: multiple reactants, multiple products, non-unit
    coefficients, coupled reactions — everything.

    Formulation (variables: y_j = ln [S_j] for every species j):
    ─────────────────────────────────────────────────────────────
    • n_rxn equilibrium equations:
          Σ_j ν_ij · y_j = ln(K_i)          ← LINEAR in y  ✓
      where ν_ij > 0 for products, < 0 for reactants.

    • n_cons mass-balance equations  (n_cons = n_sp − rank(ν)):
          Λ @ exp(y) = T = Λ @ c₀
      where Λ is the conservation matrix (null space of νᵀ) and
      c₀ is the vector of analytical (input) total concentrations.

    y0_warm: optional warm-start vector (log-concentrations of all species
             from a previous nearby solve).  When provided it is tried first
             before the heuristic cold-start guesses.

    Returns (concs_dict, success, best_y) where best_y is the raw solution
    vector in log-concentration space — suitable for warm-starting the next
    nearby point.
    """
    from scipy.linalg import null_space as _null_space
    from scipy.optimize import least_squares

    n_sp  = len(all_species)
    n_rxn = len(equilibria)
    sp_idx = {sp: i for i, sp in enumerate(all_species)}

    # ── Trivial: no reactions ────────────────────────────────────────────
    if n_rxn == 0:
        c0_trivial = np.array([max(totals_M.get(sp, 0.0), 1e-20) for sp in all_species])
        return ({sp: max(totals_M.get(sp, 0.0), 1e-20) for sp in all_species},
                True, np.log(c0_trivial))

    # ── Stoichiometric matrix ν  (n_rxn × n_sp) ─────────────────────────
    nu      = np.zeros((n_rxn, n_sp))
    lnK_vec = np.zeros(n_rxn)       # natural log of each K

    for i, eq in enumerate(equilibria):
        lnK_vec[i] = logK_vals[eq['kname']] * np.log(10.0)
        for coeff, sp in eq['reactants']:
            nu[i, sp_idx[sp]] -= float(coeff)
        for coeff, sp in eq['products']:
            nu[i, sp_idx[sp]] += float(coeff)

    # ── Conservation laws: null space of νᵀ  (n_cons × n_sp) ────────────
    Lambda  = _null_space(nu, rcond=1e-10).T      # (n_cons, n_sp): ν @ λ = 0
    c0      = np.array([max(totals_M.get(sp, 0.0), 0.0) for sp in all_species])
    T_cons  = Lambda @ c0           # conserved quantities (constants)

    # ── Residual function ────────────────────────────────────────────────
    def residuals(y):
        c      = np.exp(y)
        eq_res = lnK_vec - (nu @ y)       # equilibrium  (linear in y)
        mb_res = (Lambda @ c) - T_cons    # mass balance (nonlinear)
        return np.concatenate([eq_res, mb_res])

    # ── Cold-start initial guess ─────────────────────────────────────────
    c_guess = np.maximum(c0, 1e-20)

    # Seed product concentrations from equilibrium expressions at c₀
    for i, eq in enumerate(equilibria):
        log_react = sum(float(coeff) * np.log(c_guess[sp_idx[sp]])
                        for coeff, sp in eq['reactants'])
        n_prod = len(eq['products'])
        if n_prod == 0:
            continue
        log_prod_unit = (lnK_vec[i] + log_react) / n_prod
        for coeff, sp in eq['products']:
            j = sp_idx[sp]
            if c0[j] < 1e-15:    # pure product — not given in $concentrations
                log_c = log_prod_unit / float(coeff)
                c_guess[j] = max(c_guess[j], np.exp(np.clip(log_c, -46, 2)))

    y0 = np.log(np.maximum(c_guess, 1e-20))

    # ── Multi-start Levenberg-Marquardt ──────────────────────────────────
    # Try warm start first (if provided), then cold start, then random perturbations
    rng = np.random.default_rng(42)
    if y0_warm is not None:
        guesses = [y0_warm, y0] + [y0 + rng.normal(0, s, n_sp) for s in [1.0, 2.0]]
    else:
        guesses = [y0] + [y0 + rng.normal(0, s, n_sp) for s in [1.0, 2.0, 3.0, 5.0]]

    best_sol  = None
    best_norm = np.inf

    for y_init in guesses:
        try:
            sol  = least_squares(residuals, y_init, method='lm',
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                 max_nfev=1000)
            norm = float(np.max(np.abs(sol.fun)))
            if norm < best_norm:
                best_norm = norm
                best_sol  = sol
            if best_norm < 1e-8:
                break
        except Exception:
            continue

    if best_sol is not None and best_norm < 1e-4:
        c_final = np.maximum(np.exp(best_sol.x), 1e-20)
        return ({sp: float(c_final[sp_idx[sp]]) for sp in all_species},
                best_norm < 1e-6,
                best_sol.x)

    # Hard fallback: return analytical totals unchanged
    return ({sp: max(totals_M.get(sp, 0.0), 1e-20) for sp in all_species},
            False, y0)


# ─── kept for import-compatibility only; not used internally ─────────────────
def solve_equilibria_rigorous(totals_M, equilibria, all_species, logK_vals):
    concs, success, _ = solve_equilibria_general(totals_M, equilibria, all_species, logK_vals)
    return concs, success



def detect_thermodynamic_cycles(parsed: dict, logK_vals: dict, tol: float = 0.01) -> list:
    """
    Detect thermodynamic inconsistencies in the reaction network.

    For each non-root species, finds ALL possible reaction paths back to the
    ultimate free species (roots). In a consistent network each path must give
    the same total log10(K). If paths disagree the K values are contradictory.
    If a species appears in its own ancestry a circular dependency is flagged.

    Returns a list of warning strings (empty = no problems).
    """
    equilibria  = parsed["equilibria"]
    if not equilibria:
        return []

    # Create product-to-equilibrium mapping (handle new products format)
    prod_to_eq = {}
    for eq in equilibria:
        if "products" in eq:
            # New format: only consider single products for cycle detection
            if len(eq["products"]) == 1:
                prod_coeff, prod_species = eq["products"][0]
                prod_to_eq[prod_species] = eq
        elif "product" in eq:
            # Old format: backwards compatibility
            prod_to_eq[eq["product"][1]] = eq

    all_sp = set()
    for eq in equilibria:
        for coeff, species in eq["reactants"]:  # Now (coeff, species) tuples
            all_sp.add(species)
        
        # Handle both product formats
        if "products" in eq:
            for prod_coeff, prod_species in eq["products"]:
                all_sp.add(prod_species)
        elif "product" in eq:
            all_sp.add(eq["product"][1])

    roots = all_sp - set(prod_to_eq.keys())

    def all_logK_paths(species, visited=None):
        """
        Return list of total log10(K) values for every path from roots to species.
        Returns [] if a circular dependency is encountered.
        """
        if visited is None:
            visited = frozenset()
        if species in roots:
            return [0.0]
        if species not in prod_to_eq:
            return [0.0]
        if species in visited:
            return []           # circular dependency
        eq   = prod_to_eq[species]
        
        # Handle both old and new product formats
        if "products" in eq:
            if len(eq["products"]) == 1:
                prod_coeff, prod_species = eq["products"][0]
            else:
                # Multiple products - skip cycle detection for now
                return [0.0]
        else:
            prod_coeff, prod_species = eq["product"]
        
        # For stoichiometry, adjust K value by coefficient: log(K_eff) = log(K) / prod_coeff
        kval = logK_vals[eq["kname"]] / prod_coeff
        
        # Accumulate logK across all reactants
        result_paths = [kval]
        for coeff, reactant_species in eq["reactants"]:
            sub = all_logK_paths(reactant_species, visited | {species})
            if not sub:
                return []       # circular dependency propagated
            result_paths = [p + sp * coeff / prod_coeff for p in result_paths for sp in sub]
        return result_paths

    warnings    = []
    reported    = set()

    for sp in sorted(all_sp - roots):
        if sp in reported:
            continue
        paths = all_logK_paths(sp)

        if not paths:
            # Circular dependency: species is its own ancestor
            reported.add(sp)
            warnings.append(
                f"⚠️ **Circular dependency** involving **{sp}**: "
                f"this species appears in a closed reaction loop. "
                f"Check your $reactions for redundant or contradictory reactions."
            )

        elif len(set(round(p, 6) for p in paths)) > 1:
            # Multiple paths give different logK → thermodynamically inconsistent
            path_strs = ", ".join(f"{p:.4g}" for p in sorted(set(paths)))
            warnings.append(
                f"⚠️ **Thermodynamic inconsistency** for **{sp}**: "
                f"different reaction paths imply different log K values ({path_strs}). "
                f"The product of K values around any closed cycle must equal 1. "
                f"Check your $reactions for contradictory constants."
            )

    return warnings



_LN_LO = -30.0   # log(~1e-13 M) - more reasonable lower bound  
_LN_HI =   2.0   # log(~7 M) — generous upper bound to avoid out-of-bounds errors

def solve_free_species(totals: dict, network: dict, logK_vals: dict, x0: np.ndarray):
    """
    Solve equilibrium at one titration point.

    'totals' maps every species name that has an analytical total concentration
    (before equilibrium) to that value in Molar.  Species absent from 'totals'
    start at zero (pure products).

    Always delegates to solve_equilibria_general, which handles any network.
    The MockSolution wrapper preserves the interface expected by compute_curve.
    """
    equilibria  = network.get("equilibria", [])
    all_species = network["all_species"]

    final_concs, success, _ = solve_equilibria_general(
        totals, equilibria, all_species, logK_vals
    )

    # Store for downstream extraction in compute_curve
    network["_rigorous_concentrations"] = final_concs

    residual_indicator = 0.0 if success else 1e-4

    class MockSolution:
        success = True
        x   = np.zeros(max(len(network.get("free_species", [])), 1))
        fun = np.array([residual_indicator])

    free_concs = {sp: final_concs.get(sp, 1e-20)
                  for sp in network.get("free_species", [])}
    return MockSolution(), free_concs




# ─────────────────────────────────────────────
# KINETICS ENGINE
# ─────────────────────────────────────────────