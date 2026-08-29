"""User-defined SDE models: sympy parsing + a generic torch model satisfying the solver contract.

A user declares state variables and, per variable, types (1) the deterministic RHS of dx/dt (first-order,
nondimensional), (2) a white-noise strength D (parameters only -- additive noise; the solver diffusion is
g = sqrt(2*D), matching <xi(t)xi(t')> = 2 D delta(t-t')), and (3) optionally a time-driven forcing (built
separately in core/forcing.py). ``parse_user_model`` compiles the typed expressions into a
``CompiledUserModel``; ``UserModel`` is the ONE generic runtime class every user model shares -- it mirrors
the duck-typed contract of core/Models/hopf_model.py (f(x, t) with the integer step index, g() returning a
(batch, d) diagonal vector, a settable public ``force`` (batch, n_vars, T) attribute, device/dtype applied
to every param tensor).

Parsing is sandboxed: a token pre-filter rejects everything outside identifiers/numbers/operators, the
sympy namespace is locked to an explicit whitelist, and evaluation of the parsed tree is a hand-rolled
walk over a fixed set of node types -- no eval of arbitrary Python anywhere.
"""
import itertools
import linecache
import math
import re
from dataclasses import dataclass, field

import sympy
import torch
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr


class ModelParseError(ValueError):
    """Raised when a user-typed model definition cannot be parsed/compiled. Message is user-facing."""


# ── the expression whitelist ─────────────────────────────────────────────────
# name -> (sympy class for parsing, torch fn for tensor args, python fn for constant args)
_FUNCS = {
    "sin":  (sympy.sin,  torch.sin,  math.sin),
    "cos":  (sympy.cos,  torch.cos,  math.cos),
    "tan":  (sympy.tan,  torch.tan,  math.tan),
    "asin": (sympy.asin, torch.asin, math.asin),
    "acos": (sympy.acos, torch.acos, math.acos),
    "atan": (sympy.atan, torch.atan, math.atan),
    "sinh": (sympy.sinh, torch.sinh, math.sinh),
    "cosh": (sympy.cosh, torch.cosh, math.cosh),
    "tanh": (sympy.tanh, torch.tanh, math.tanh),
    "exp":  (sympy.exp,  torch.exp,  math.exp),
    "log":  (sympy.log,  torch.log,  math.log),
    "sqrt": (sympy.sqrt, torch.sqrt, math.sqrt),   # note: sympy.sqrt(x) becomes Pow(x, 1/2)
    "Abs":  (sympy.Abs,  torch.abs,  abs),
    "abs":  (sympy.Abs,  torch.abs,  abs),
    "sign": (sympy.sign, torch.sign, lambda v: float((v > 0) - (v < 0))),
}
# NOTE: no "E" constant -- a physics parameter named E (modulus, field, energy) would silently become
# Euler's number; exp() covers the use case, so E parses as an ordinary user parameter instead.
_CONSTANTS = {"pi": sympy.pi}

# The minimal constructors sympy's parser transformations reference at eval time. Nothing else.
# Their NAMES are also refused as user identifiers: a parameter named e.g. "Float" would shadow the
# constructor inside parse_expr's namespace and break every numeric literal with a baffling error.
_GLOBAL_DICT = {
    "Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
    "Rational": sympy.Rational, "Function": sympy.Function,
}
RESERVED_NAMES = frozenset(_FUNCS) | frozenset(_CONSTANTS) | frozenset(_GLOBAL_DICT) | {"t", "force", "I"}

# Legal characters for a user expression. Everything else (brackets, quotes, '=', '@', ';', ...) is
# rejected BEFORE sympy ever sees the string.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s]*$")
# A '.' adjacent to a letter/underscore is attribute access, not a decimal point -- reject it.
_ATTR_DOT_RE = re.compile(r"[A-Za-z_]\s*\.|\.\s*[A-Za-z_]")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# A numeric literal (incl. scientific notation and hex) NOT glued to an identifier. Stripped before
# identifier discovery: the tokenizer reads "1e-3"/"0x1F" as single NUMBER tokens, so scanning the
# raw text would otherwise shed phantom "parameters" like e/e3/x1F out of the mantissa tail.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:0[xX][0-9A-Fa-f]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parse_one(text: str, where: str, local_dict: dict) -> sympy.Expr:
    """Parse a single user expression inside the locked-down namespace, or raise ModelParseError."""
    text = (text or "").strip()
    if not text:
        raise ModelParseError(f"{where}: expression is empty.")
    if not _TOKEN_RE.match(text):
        raise ModelParseError(f"{where}: only names, numbers and + - * / ^ ( ) are allowed.")
    if _ATTR_DOT_RE.search(text):
        raise ModelParseError(f"{where}: '.' is only allowed inside a number (e.g. 0.5).")
    try:
        expr = parse_expr(text.replace("^", "**"), local_dict=local_dict, global_dict=_GLOBAL_DICT)
    except ModelParseError:
        raise
    except Exception as e:                                     # noqa: BLE001 -- any parse failure
        raise ModelParseError(f"{where}: could not parse '{text}' ({e}).") from e
    if not isinstance(expr, sympy.Expr):
        raise ModelParseError(f"{where}: '{text}' is not a numeric expression.")
    undef = expr.atoms(AppliedUndef)
    if undef:
        names = ", ".join(sorted(str(u.func) for u in undef))
        raise ModelParseError(
            f"{where}: unknown function(s) {names}. Allowed: {', '.join(sorted(set(_FUNCS) - {'Abs'}))}.")
    return expr


def _identifiers(text: str) -> list[str]:
    """Identifiers in TEXT order (first appearance), with numeric literals stripped first -- the
    user-visible parameter discovery order."""
    return [m.group(0) for m in _IDENT_RE.finditer(_NUMBER_RE.sub(" ", text or ""))]


def _compile_tree(expr: sympy.Expr, arg_index: dict, where: str):
    """Recursively compile a sympy tree into a closure ``fn(args) -> tensor | float``.

    ``args`` is the flat tuple (*state_columns, *param_tensors); ``arg_index`` maps Symbol -> position.
    Only the whitelisted node types below are compilable -- anything else is a parse-stage bug or an
    exotic sympy object the UI should not accept, so raise ModelParseError.
    """
    if expr is sympy.pi:
        return lambda args: math.pi
    if expr is sympy.E:
        return lambda args: math.e
    if isinstance(expr, sympy.Symbol):
        idx = arg_index[expr]
        return lambda args: args[idx]
    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        val = float(expr)
        return lambda args: val
    if isinstance(expr, sympy.Add):
        parts = [_compile_tree(a, arg_index, where) for a in expr.args]
        def _add(args, _parts=parts):
            acc = _parts[0](args)
            for p in _parts[1:]:
                acc = acc + p(args)
            return acc
        return _add
    if isinstance(expr, sympy.Mul):
        parts = [_compile_tree(a, arg_index, where) for a in expr.args]
        def _mul(args, _parts=parts):
            acc = _parts[0](args)
            for p in _parts[1:]:
                acc = acc * p(args)
            return acc
        return _mul
    if isinstance(expr, sympy.Pow):
        base = _compile_tree(expr.base, arg_index, where)
        exponent = _compile_tree(expr.exp, arg_index, where)
        return lambda args: base(args) ** exponent(args)
    if isinstance(expr, sympy.Function):
        name = expr.func.__name__
        if name in _FUNCS:
            _, torch_fn, py_fn = _FUNCS[name]
            arg = _compile_tree(expr.args[0], arg_index, where)
            def _apply(args, _arg=arg, _t=torch_fn, _p=py_fn):
                v = _arg(args)
                return _t(v) if torch.is_tensor(v) else _p(v)
            return _apply
    raise ModelParseError(f"{where}: unsupported construct '{expr}'.")


# ── torchscript codegen (the euler_compiled fast path) ───────────────────────
_STEP_ID = itertools.count()   # unique compile filenames so each model's linecache entry is distinct


def _print_torch(expr: sympy.Expr, name_src: dict) -> str:
    """Print a sympy expression as torch source, PARALLEL to _compile_tree (same node types, same
    left-fold order). ``name_src`` maps a symbol NAME to its source ("x{j}" for a state column,
    "p{k}" for a parameter arg). Constant subexpressions -- incl. pi, E, and f(const) like exp(2) --
    fold to a float literal (matching _compile_tree, which routes constant-arg functions to the python
    fn), which also avoids torch.<fn>(<float>) calls that TorchScript rejects."""
    if expr.is_number:
        return repr(float(expr))
    if isinstance(expr, sympy.Symbol):
        return name_src[expr.name]
    if isinstance(expr, sympy.Add):
        return "(" + " + ".join(_print_torch(a, name_src) for a in expr.args) + ")"
    if isinstance(expr, sympy.Mul):
        return "(" + " * ".join(_print_torch(a, name_src) for a in expr.args) + ")"
    if isinstance(expr, sympy.Pow):
        return "(" + _print_torch(expr.base, name_src) + " ** " + _print_torch(expr.exp, name_src) + ")"
    if isinstance(expr, sympy.Function):
        name = expr.func.__name__
        if name in _FUNCS:
            torch_fn = _FUNCS[name][1]
            return f"torch.{torch_fn.__name__}(" + _print_torch(expr.args[0], name_src) + ")"
    raise ModelParseError(f"cannot print '{expr}' to torch source")


def build_compiled_step(compiled: "CompiledUserModel"):
    """Best-effort: emit a @torch.jit.script one-step Euler-Maruyama function for the euler_compiled fast
    path (core/Solvers/sdeint.py euler_compiled -- auto-selected only on CUDA + hasattr(compiled_step)),
    or return None so the model falls back to the eager `euler` loop. Signature matches the built-ins:
    ``step(x, force_step, dW, *params, dt, sqrt_dt) -> next_x``, recomputing g INLINE (Nadrowski-style) so
    one codegen covers additive + state-dependent noise. The g math mirrors UserModel EXACTLY: additive
    -> sqrt(2 D) (unclamped -- a negative constant D must still NaN so the stability sweep screens it),
    state-dependent -> sqrt(clamp(2 D, 0)). The torch.zeros_like(x[:,0]) anchor keeps a purely-constant D
    (e.g. "0") a tensor. Any codegen/script failure returns None (pure-speedup contract: never a
    correctness risk). The exec runs OUR printer output over the already-sandboxed sympy tree, in a
    minimal {torch, math} namespace -- same trust level as the closure interpreter."""
    try:
        var_names, param_names = compiled.var_names, compiled.param_names
        n_vars = len(var_names)
        name_src = {vn: f"x{j}" for j, vn in enumerate(var_names)}
        name_src.update({pn: f"p{k}" for k, pn in enumerate(param_names)})

        drift_src = [_print_torch(e, name_src) for e in compiled.drift_exprs]
        diff_src = [_print_torch(e, name_src) for e in compiled.diff_exprs]

        params_sig = "".join(f", p{k}: torch.Tensor" for k in range(len(param_names)))
        lines = [f"def _user_step(x: torch.Tensor, force_step: torch.Tensor, dW: torch.Tensor"
                 f"{params_sig}, dt: float, sqrt_dt: float) -> torch.Tensor:"]
        for j in range(n_vars):
            lines.append(f"    x{j} = x[:, {j}]")
        for j in range(n_vars):
            lines.append(f"    dx{j} = ({drift_src[j]}) + force_step[:, {j}]")
        lines.append("    drift = torch.stack([" + ", ".join(f"dx{j}" for j in range(n_vars)) + "], dim=1)")
        for j in range(n_vars):
            two_d = f"2.0 * ({diff_src[j]}) + torch.zeros_like(x[:, 0])"
            lines.append(f"    g{j} = torch.sqrt(torch.clamp({two_d}, min=0.0))"
                         if compiled.state_dep_noise else f"    g{j} = torch.sqrt({two_d})")
        lines.append("    g = torch.stack([" + ", ".join(f"g{j}" for j in range(n_vars)) + "], dim=-1)")
        lines.append("    return x + drift * dt + g * dW * sqrt_dt")

        src = "\n".join(lines)
        # torch.jit.script needs (a) a qualified name -> __name__ in the exec globals gives __module__,
        # and (b) SOURCE ACCESS via inspect/linecache -- an exec'd function has no source file, so
        # register the generated source under a UNIQUE compile filename in linecache.
        filename = f"<user_model_step_{next(_STEP_ID)}>"
        linecache.cache[filename] = (len(src), None, [ln + "\n" for ln in src.split("\n")], filename)
        g: dict = {"torch": torch, "math": math, "__name__": "core.Models._user_model_compiled"}
        exec(compile(src, filename, "exec"), g, g)
        return torch.jit.script(g["_user_step"])
    except Exception:                                          # noqa: BLE001 -- any failure -> eager
        return None


@dataclass
class CompiledUserModel:
    """The runnable form of a user model definition.

    ``param_names`` order is LOAD-BEARING: it is the constructor positional order, the
    ``torch.unbind(params, dim=1)`` column order, and the Bounds-file ND section order -- all three must
    stay identical (the constructor is positional -- the model/solver contract).
    """
    var_names: list = field(default_factory=list)     # declared order; index 0 = the observable
    param_names: list = field(default_factory=list)   # first appearance in the typed expressions
    drift_fns: list = field(default_factory=list)     # fn(args) per variable; args = (*states, *params)
    diff_fns: list = field(default_factory=list)      # D_j as fn(args); same flat args (may use state)
    state_dep_noise: bool = False                     # True if any D references a state variable ->
                                                      # multiplicative noise; g(x) is evaluated per step
    drift_exprs: list = field(default_factory=list)   # sympy Expr per variable (for the torchscript codegen)
    diff_exprs: list = field(default_factory=list)    # sympy Expr per variable
    compiled_step_fn: object = None                   # @torch.jit.script euler_compiled step, or None
                                                      # (built lazily by build_compiled_step; eager fallback)


def parse_user_model(variables: list) -> CompiledUserModel:
    """Compile ``[{name, drift, D}, ...]`` (declared order) into a CompiledUserModel.

    Every identifier that is not a declared state variable (or a whitelisted function/constant) is a
    PARAMETER, ordered by first appearance across the drift/D texts. D MAY reference state variables:
    if any does, the model has multiplicative (state-dependent) noise and ``state_dep_noise`` is set,
    so the solver evaluates g(x) = sqrt(2 D(x)) each step; otherwise D is a constant amplitude cached once.
    """
    if not variables:
        raise ModelParseError("Declare at least one state variable.")
    var_names = [str(v.get("name", "")).strip() for v in variables]
    for name in var_names:
        if not _NAME_RE.match(name):
            raise ModelParseError(f"'{name}' is not a valid variable name (letters/digits/_, no leading _ or digit).")
        if name in RESERVED_NAMES:
            raise ModelParseError(f"'{name}' is reserved and cannot be a variable name.")
    if len(set(var_names)) != len(var_names):
        raise ModelParseError("Variable names must be unique.")

    # Parameter discovery in text order (numeric literals pre-stripped): drift then D, per variable.
    param_names: list[str] = []
    for v, name in zip(variables, var_names):
        for text in (v.get("drift", ""), v.get("D", "0")):
            for ident in _identifiers(str(text).replace("^", "**")):
                if ident in var_names or ident in param_names:
                    continue
                if ident in _GLOBAL_DICT:
                    raise ModelParseError(f"'{ident}' is a reserved name and cannot be a parameter.")
                if ident in RESERVED_NAMES:
                    continue                                   # functions/constants, not parameters
                if not _NAME_RE.match(ident):
                    raise ModelParseError(f"'{ident}' is not a valid parameter name.")
                param_names.append(ident)

    state_syms = [sympy.Symbol(n) for n in var_names]
    param_syms = [sympy.Symbol(n) for n in param_names]
    local_dict = {n: s for n, s in zip(var_names, state_syms)}
    local_dict.update({n: s for n, s in zip(param_names, param_syms)})
    local_dict.update({n: cls for n, (cls, _, _) in _FUNCS.items()})
    local_dict.update(_CONSTANTS)
    allowed = set(state_syms) | set(param_syms)

    # Pass 1 -- parse + validate every expression (so the used-symbol set is known before indices
    # are frozen: a discovered name the tokenizer never emits as a Symbol must not become a column).
    drift_exprs, diff_exprs = [], []
    used: set = set()
    for v, name in zip(variables, var_names):
        where_d = f"d{name}/dt"
        expr = _parse_one(str(v.get("drift", "")), where_d, local_dict)
        stray = expr.free_symbols - allowed
        if stray:
            raise ModelParseError(f"{where_d}: unknown symbol(s) {sorted(map(str, stray))}.")
        drift_exprs.append(expr)
        used |= expr.free_symbols

        where_n = f"D of {name}"
        d_expr = _parse_one(str(v.get("D", "0") or "0"), where_n, local_dict)
        stray = d_expr.free_symbols - allowed
        if stray:
            raise ModelParseError(f"{where_n}: unknown symbol(s) {sorted(map(str, stray))}.")
        diff_exprs.append(d_expr)
        used |= d_expr.free_symbols

    # Keep only parameters the parsed expressions actually reference: this drops any phantom the
    # literal-stripping missed AND unused typo-parameters, keeping the positional contract honest.
    param_syms = [s for s in param_syms if s in used]
    param_names = [str(s) for s in param_syms]

    # Multiplicative noise iff any diffusion expression references a state variable.
    state_dep_noise = any(bool(d.free_symbols & set(state_syms)) for d in diff_exprs)

    # Pass 2 -- compile against the final (state + used-param) argument layout.
    arg_index = {s: i for i, s in enumerate(state_syms + param_syms)}
    drift_fns = [_compile_tree(expr, arg_index, f"d{name}/dt")
                 for name, expr in zip(var_names, drift_exprs)]
    diff_fns = [_compile_tree(expr, arg_index, f"D of {name}")
                for name, expr in zip(var_names, diff_exprs)]

    return CompiledUserModel(var_names=var_names, param_names=param_names,
                             drift_fns=drift_fns, diff_fns=diff_fns, state_dep_noise=state_dep_noise,
                             drift_exprs=drift_exprs, diff_exprs=diff_exprs)


class UserModel:
    """Generic runtime model for user-defined SDEs; satisfies the eager-solver duck-typed contract."""

    def __init__(self, compiled: CompiledUserModel, params, force: torch.Tensor, batch_size: int,
                 device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._compiled = compiled

        params = tuple(params)
        if len(params) != len(compiled.param_names):
            raise ValueError(
                f"UserModel expects {len(compiled.param_names)} parameter columns "
                f"({compiled.param_names}), got {len(params)}.")
        self.params = tuple(p.to(dtype=self.dtype, device=self.device) for p in params)

        n_vars = len(compiled.var_names)
        if force.shape[1] != n_vars:
            raise ValueError(
                f"UserModel force tensor must have one channel per state variable "
                f"({n_vars}), got shape {tuple(force.shape)}.")
        self.force = force.to(dtype=self.dtype, device=self.device)

        self.state_dep_noise = compiled.state_dep_noise
        if self.state_dep_noise:
            # Multiplicative noise: g depends on state, so it is recomputed per step in g(x). Nothing
            # to cache; the solver takes the state_dep_drift=True branch and calls g(x_curr).
            self._g = None
        else:
            # Additive noise: D depends on params only, so g = sqrt(2 D) is evaluated ONCE and cached
            # (the solver's state_dep_drift=False path reads g() once before the loop). A NEGATIVE
            # constant D yields NaN here (sqrt of a negative): per-sample this screens that parameter
            # set out downstream -- the SBI stability sweep drops non-finite trajectories, and the
            # builder smoke integration flags it as diverged -- rather than raising and killing an
            # entire batch (the stability screen Sobol-samples a whole bounds box, negatives included).
            zeros = torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
            args = tuple(zeros for _ in range(n_vars)) + self.params
            cols = [torch.sqrt(2.0 * self._as_batch(fn(args))) for fn in compiled.diff_fns]
            self._g = torch.stack(cols, dim=-1)                  # (batch, d) diagonal amplitudes

        # Fast path: expose a @torch.jit.script step ONLY when one was built (registry.load_user_models).
        # Its presence is what Simulator.__sols keys on to pick euler_compiled on CUDA; absent -> eager.
        if compiled.compiled_step_fn is not None:
            self.compiled_step = compiled.compiled_step_fn

    def compiled_params(self) -> tuple:
        """The param tuple passed to compiled_step after (x, force_step, dW); g is recomputed inline."""
        return self.params

    def _as_batch(self, val) -> torch.Tensor:
        """Normalize a compiled-expression result (tensor or python number) to a (batch,) tensor."""
        if not torch.is_tensor(val):
            val = torch.tensor(float(val), dtype=self.dtype, device=self.device)
        if val.dim() == 0:
            val = val.expand(self.batch_size)
        return val

    def f(self, x: torch.Tensor, t) -> torch.Tensor:
        args = tuple(x[:, j] for j in range(len(self._compiled.var_names))) + self.params
        cols = [self._as_batch(fn(args)) + self.force[:, j, t]
                for j, fn in enumerate(self._compiled.drift_fns)]
        return torch.stack(cols, dim=1)

    def g(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Diagonal diffusion amplitudes (batch, d). Additive: the cached sqrt(2D) (``x`` ignored).
        Multiplicative: sqrt(2 D(x)) recomputed from the state (the solver passes x_curr each step)."""
        if not self.state_dep_noise:
            return self._g
        args = tuple(x[:, j] for j in range(len(self._compiled.var_names))) + self.params
        # clamp at 0: a transient negative-D excursion from a badly-typed expression must not NaN the
        # whole batch (a grossly-wrong D still surfaces via the builder smoke test / divergence guard).
        cols = [torch.sqrt(torch.clamp(2.0 * self._as_batch(fn(args)), min=0.0))
                for fn in self._compiled.diff_fns]
        return torch.stack(cols, dim=-1)
