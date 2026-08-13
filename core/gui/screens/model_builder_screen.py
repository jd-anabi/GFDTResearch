"""The user-defined model builder (reached from Settings -> "User-defined models").

Declare state variables, then per variable type the deterministic RHS of dx/dt, the white-noise
strength D (the solver diffusion is g = sqrt(2*D); parameters only -- additive noise), and optionally
one time-driven forcing. Equations are NONDIMENSIONAL (nothing auto-nondimensionalizes); x_scale /
t_scale only label + redimensionalize the Simulate axes. Variable 1 is the observable Simulate plots.

Everything here is light (sympy parse + a 100-step batch-1 smoke integration, milliseconds), so it all
runs on the GUI thread -- no BasePanel/dispatch. Saving is refused while a task runs anywhere in the
app (refreshing the model combos mid-run would fight the app-wide control lock), and persistence goes
through core/Helpers/model_store.py (the JSON + the Bounds/Cells/Units triple).
"""
import torch
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QScrollArea, QStackedWidget,
                               QVBoxLayout, QWidget)

from core import config, registry
from core.config import DT_EXP_S
from core.forcing import FORCING_PARAM_NAMES
from core.Helpers import model_store
from core.Models.user_model import ModelParseError, UserModel, parse_user_model
from core.Solvers import sdeint

from ..design import SPACE
from ..panels.base_panel import BasePanel
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField
from ..widgets.field_row import LabeledFieldRow
from ..widgets.forms import make_form
from ..widgets.adaptive_stack import AdaptiveStack

HELP = {
    "name": "The model's name as it appears in every model dropdown. Letters/digits/_ (max 24).",
    "variables": "Comma-separated state variable names, e.g. `x, y`. One first-order equation per "
                 "variable; the FIRST variable is the observable Simulate plots. A 2nd-order system "
                 "must be written as two first-order equations.",
    "drift": "The deterministic right-hand side of dx/dt (nondimensional; may be nonlinear). Use the "
             "declared variables, parameters (any new name), numbers, + - * / ^ ( ) and "
             "sin cos tan asin acos atan sinh cosh tanh exp log sqrt abs sign, pi.",
    "D": "White-noise strength D (<xi(t)xi(t')> = 2 D delta): the solver's noise amplitude is "
         "sqrt(2*D). Use parameters/numbers for additive noise, or reference state variables for "
         "multiplicative (state-dependent) noise; 0 = noiseless.",
    "init": "Initial condition for this variable (nondimensional).",
    "forcing": "Optional time-driven external force added to this variable's RHS. For a restoring "
               "(spring) force, put -k*(x - x0) directly in the RHS instead -- it depends on state.",
    "x_scale": "Length scale (nm per ND unit) used to redimensionalize the Simulate displacement axis.",
    "t_scale": "Time scale (seconds per ND time unit) used to map ND time to the seconds axis.",
}

# (kind key, display label); "" = no forcing.
_FORCE_KINDS = (("", "None"), ("sin", "Sinusoidal"), ("step", "Step"),
                ("triangular", "Triangular"), ("exponential", "Exponential"))
_FORCE_DEFAULTS = {"amp": 0.0, "freq": 1.0, "phase": 0.0, "offset": 0.0, "t0": 0.0, "tau": 1.0}
_SMOKE_STEPS = 100          # validation integration length (batch 1 -- milliseconds)


class _VarRow(QGroupBox):
    """One state variable's definition: drift | D | init | forcing kind + its parameters."""

    def __init__(self, var_name: str, parent=None):
        super().__init__(f"d{var_name}/dt", parent)
        self.var_name = var_name
        form = make_form(self)

        self.drift = QLineEdit()
        self.drift.setPlaceholderText(f"e.g. mu*{var_name} - {var_name}^3")
        self.noise = QLineEdit("0")
        self.init = FloatField(0.0)

        self.force_kind = QComboBox()
        for _, label in _FORCE_KINDS:
            self.force_kind.addItem(label)
        # AdaptiveStack: there is ONE of these per state variable, and the "None" page is empty while
        # the "exponential" page is three fields plus a radio row -- so every unforced variable used
        # to reserve the tallest page's height.
        self.force_stack = AdaptiveStack()
        self._force_fields = {}                      # kind -> {pname: FloatField}
        self._exp_grow = None
        for kind, _ in _FORCE_KINDS:
            page = QWidget()
            pf = make_form(page)
            pf.setContentsMargins(0, 0, 0, 0)
            if kind:
                fields = {}
                for pname in FORCING_PARAM_NAMES[kind]:
                    fields[pname] = FloatField(_FORCE_DEFAULTS[pname])
                    pf.addRow(pname, fields[pname])
                if kind == "exponential":
                    self._exp_grow = QRadioButton("Grow (+t/tau)")
                    self._exp_decay = QRadioButton("Decay (-t/tau)")
                    self._exp_decay.setChecked(True)
                    sign_row = QHBoxLayout()
                    sign_row.addWidget(self._exp_grow)
                    sign_row.addWidget(self._exp_decay)
                    pf.addRow(sign_row)
                self._force_fields[kind] = fields
            self.force_stack.addWidget(page)
        self.force_kind.currentIndexChanged.connect(self.force_stack.setCurrentIndex)

        add_help_row(form, "drift", self.drift, HELP["drift"])
        add_help_row(form, "D", self.noise, HELP["D"])
        add_help_row(form, "init", self.init, HELP["init"])
        add_help_row(form, "forcing", self.force_kind, HELP["forcing"])
        form.addRow(self.force_stack)

    def values(self) -> dict:
        kind = _FORCE_KINDS[self.force_kind.currentIndex()][0]
        forcing = None
        if kind:
            forcing = {"kind": kind,
                       "params": {p: f.value() for p, f in self._force_fields[kind].items()}}
            if kind == "exponential":
                forcing["sign"] = 1 if self._exp_grow.isChecked() else -1
        return {"name": self.var_name, "drift": self.drift.text(), "D": self.noise.text() or "0",
                "init": self.init.value(), "forcing": forcing}

    def populate(self, v: dict) -> None:
        self.drift.setText(str(v.get("drift", "")))
        self.noise.setText(str(v.get("D", "0")))
        self.init.setText(repr(float(v.get("init", 0.0))))
        forcing = v.get("forcing") or None
        kind = forcing["kind"] if forcing else ""
        idx = next((i for i, (k, _) in enumerate(_FORCE_KINDS) if k == kind), 0)
        self.force_kind.setCurrentIndex(idx)
        if forcing:
            for pname, fld in self._force_fields[kind].items():
                fld.setText(repr(float(forcing["params"].get(pname, _FORCE_DEFAULTS[pname]))))
            if kind == "exponential":
                # Radios are auto-exclusive: check the TARGET, never un-check the other.
                (self._exp_grow if forcing.get("sign", -1) == 1 else self._exp_decay).setChecked(True)


class _ParamRow(QWidget):
    """One parameter's ground-truth value, its SBI inference box, and that box's coordinate.

    TWO lines, deliberately: ``value | auto | min | max`` then ``box``. The first line already packs
    3 fields + 4 labels + a checkbox and is called out in PRISM_HANDOFF 10.2 (L4) as the worst
    offender in the app for splitting an already-narrow controls column N ways; a fifth control on it
    would leave each field a few characters wide again. Both lines are ``LabeledFieldRow``, so the
    minimum widths and growth policy stay in the one place that owns them.

    'auto' reproduces the placeholder box ``model_store.nd_bounds`` (v - pad, v + pad); unchecking it
    exposes an editable (min, max) for a tighter, physical prior. ``spec()`` returns
    (value, lo, hi, box) and is the single reuse of the placeholder rule when auto is on.
    """

    def __init__(self, value=1.0, lo=None, hi=None, box=None, parent=None):
        super().__init__(parent)
        self.value = FloatField(value)
        self.auto = QCheckBox("auto")
        self.lo = FloatField(0.0)
        self.hi = FloatField(0.0)
        self.box = QComboBox()
        self.box.addItems(list(model_store.BOX_KINDS))
        self.box.setToolTip(
            "The coordinate this parameter's box bijection works in.\n"
            "'log' linearizes a multiplicative parameter before the Fisher rotation and needs min > 0.\n"
            "It changes the coordinate the flow trains in — NOT where the prior puts its mass.")
        # A falsy caption omits the label, so the unlabelled "auto" checkbox sits in the same row
        # machinery as the labelled fields.
        top = LabeledFieldRow((("value", self.value), ("", self.auto),
                               ("min", self.lo), ("max", self.hi)))
        bottom = LabeledFieldRow((("box", self.box),))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE[0])
        layout.addWidget(top)
        layout.addWidget(bottom)
        self.auto.toggled.connect(self._on_auto)
        self.set_spec(value, lo, hi, box)

    def _on_auto(self, checked: bool) -> None:
        # Disable the fields under auto, and (re)seed them with the placeholder box so unchecking leaves
        # a sensible editable starting point rather than a blank/zero range.
        self.lo.setEnabled(not checked)
        self.hi.setEnabled(not checked)
        lo, hi = model_store.nd_bounds(self.value.value())
        self.lo.setText(repr(lo))
        self.hi.setText(repr(hi))

    def spec(self) -> tuple:
        """(value, lo, hi, box). lo/hi are None when the box is on but a field does not parse -- a
        blank "min" used to come back as a real bound of 0.0 (FloatField.value()'s documented hazard,
        which widgets/param_grid guards against and this row did not). _validate turns the None into
        a message; it must never reach model_store as a number the user did not type."""
        v = self.value.value()
        box = self.box.currentText()
        if self.auto.isChecked():
            lo, hi = model_store.nd_bounds(v)
            return v, lo, hi, box
        return v, self.lo.value_or_none(), self.hi.value_or_none(), box

    def set_spec(self, value, lo=None, hi=None, box=None) -> None:
        value = float(value)
        auto = lo is None or hi is None or (float(lo), float(hi)) == model_store.nd_bounds(value)
        self.value.setText(repr(value))
        self.auto.setChecked(auto)                 # fires _on_auto (seeds the placeholder box) if it flips
        if auto:
            alo, ahi = model_store.nd_bounds(value)
            self.lo.setText(repr(alo))
            self.hi.setText(repr(ahi))
        else:
            self.lo.setText(repr(float(lo)))
            self.hi.setText(repr(float(hi)))
        self.lo.setEnabled(not auto)
        self.hi.setEnabled(not auto)
        idx = self.box.findText(str(box or "linear"))
        self.box.setCurrentIndex(idx if idx >= 0 else 0)


class ModelBuilderScreen(QWidget):
    def __init__(self, on_saved=None, on_back=None, parent=None):
        """``on_saved(name)`` fires after a successful save (MainWindow refreshes the model combos);
        ``on_back()`` returns to the Settings screen."""
        super().__init__(parent)
        self._on_saved = on_saved
        self._on_back = on_back
        self._editing_name = None                 # set by load_existing(); None = creating new
        self._var_rows = []                       # [_VarRow] in declared order
        self._param_fields = {}                   # name -> _ParamRow (rebuilt by _detect_params)

        heading = QLabel("Model builder")
        heading.setProperty("type", "heading")

        definition = QGroupBox("Definition")
        dform = make_form(definition)
        self.name_edit = QLineEdit()
        self.vars_edit = QLineEdit()
        self.vars_edit.setPlaceholderText("x, y")
        btn_vars = QPushButton("Set variables")
        btn_vars.clicked.connect(self._set_variables)
        vrow = QHBoxLayout()
        vrow.addWidget(self.vars_edit, 1)
        vrow.addWidget(btn_vars)
        add_help_row(dform, "name", self.name_edit, HELP["name"])
        vrow_w = QWidget()
        vrow_w.setLayout(vrow)
        add_help_row(dform, "variables", vrow_w, HELP["variables"])

        self._rows_host = QVBoxLayout()           # the per-variable _VarRow stack

        params_box = QGroupBox("Parameter values")
        pv = QVBoxLayout(params_box)
        btn_detect = QPushButton("Detect parameters")
        btn_detect.clicked.connect(self._detect_params)
        self._params_form = make_form()
        pv.addWidget(btn_detect)
        pv.addLayout(self._params_form)

        scales_box = QGroupBox("Display scales")
        sform = make_form(scales_box)
        self.x_scale = FloatField(10.0)
        self.t_scale = FloatField(0.01)
        add_help_row(sform, "x_scale (nm)", self.x_scale, HELP["x_scale"])
        add_help_row(sform, "t_scale (s)", self.t_scale, HELP["t_scale"])

        self.status = QLabel("")
        self.status.setWordWrap(True)

        btn_validate = QPushButton("Validate")
        btn_validate.clicked.connect(self._validate_clicked)
        self.btn_save = QPushButton("Save model")
        self.btn_save.setProperty("accent", True)              # primary CTA (Fluent accent)
        self.btn_save.clicked.connect(self._save)
        btn_back = QPushButton("Back to settings")
        btn_back.clicked.connect(lambda: self._on_back and self._on_back())
        btns = QHBoxLayout()
        btns.addWidget(btn_validate)
        btns.addWidget(self.btn_save)
        btns.addStretch(1)
        btns.addWidget(btn_back)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(definition)
        iv.addLayout(self._rows_host)
        iv.addWidget(params_box)
        iv.addWidget(scales_box)
        iv.addWidget(self.status)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        # STICKY ACTION BAR, outside the scroll area. Validate/Save used to be the last thing INSIDE
        # the scrolled column, below one _VarRow per state variable and one _ParamRow per parameter --
        # both unbounded. With ~15 parameters they sat a full screen below the fold, which is why the
        # Save button was reported as missing entirely. Actions must not be something you have to
        # hunt for.
        action_bar = QWidget()
        action_bar.setObjectName("actionBar")
        ab = QHBoxLayout(action_bar)
        ab.setContentsMargins(8, 6, 8, 6)
        ab.addLayout(btns)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll, 1)
        layout.addWidget(action_bar)

    # ── dynamic rows ──────────────────────────────────────────────────────────
    def _set_variables(self):
        names = [n.strip() for n in self.vars_edit.text().split(",") if n.strip()]
        if not names:
            self._set_status("Type at least one variable name (e.g. `x` or `x, y`).", error=True)
            return
        old = {row.var_name: row.values() for row in self._var_rows}
        for row in self._var_rows:
            self._rows_host.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._var_rows = []
        for name in names:
            row = _VarRow(name)
            if name in old:
                row.populate(old[name])           # keep typed content across a variable-list edit
            self._rows_host.addWidget(row)
            self._var_rows.append(row)
        self._set_status(f"{len(names)} variable(s): fill in each equation, then Detect parameters.")

    def _detect_params(self):
        variables = [row.values() for row in self._var_rows]
        if not variables:
            self._set_status("Set the variables first.", error=True)
            return
        try:
            compiled = parse_user_model(variables)
        except ModelParseError as e:
            self._set_status(str(e), error=True)
            return
        old = {name: row.spec() for name, row in self._param_fields.items()}
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_fields = {}
        for name in compiled.param_names:
            # keep typed (value, lo, hi, box) -- spec() and __init__ are positionally matched, so a
            # field added to one has to be added to the other or a re-detect silently drops it.
            row = _ParamRow(*old[name]) if name in old else _ParamRow(1.0)
            self._param_fields[name] = row
            self._params_form.addRow(name, row)
        self._set_status(f"Found {len(compiled.param_names)} parameter(s): "
                         f"{', '.join(compiled.param_names) or '(none)'}. Set values, then Validate.")
        return compiled

    # ── validate / save ───────────────────────────────────────────────────────
    def _assemble_doc(self) -> dict:
        return {
            "schema_version": model_store.SCHEMA_VERSION,
            "name": self.name_edit.text().strip(),
            "variables": [row.values() for row in self._var_rows],
            "params": {name: dict(zip(("value", "lo", "hi", "box"), row.spec()))
                       for name, row in self._param_fields.items()},
            "rescale": {"x_scale": self.x_scale.value(), "t_scale": self.t_scale.value()},
        }

    def _validate(self):
        """Parse + compile + a short batch-1 smoke integration. Returns the doc, or None (status set)."""
        if BasePanel._running:
            # The smoke integration's tqdm writes to the process-wide redirected streams while a
            # worker runs -- it would interleave into that run's progress pane. Refuse instead.
            self._set_status("A task is running -- wait for it to finish before validating.", error=True)
            return None
        if not self._var_rows:
            self._set_status("Set the variables first.", error=True)
            return None
        try:
            name = model_store.validate_name(self.name_edit.text())
        except ValueError as e:
            self._set_status(str(e), error=True)
            return None
        doc = self._assemble_doc()
        doc["name"] = name
        try:
            compiled = parse_user_model(doc["variables"])
        except ModelParseError as e:
            self._set_status(str(e), error=True)
            return None
        missing = [p for p in compiled.param_names if p not in self._param_fields]
        if missing:
            self._set_status(f"New parameter(s) {missing} have no value yet -- "
                             "click 'Detect parameters' first.", error=True)
            return None
        doc["params"] = {p: doc["params"][p] for p in compiled.param_names}
        t_scale = doc["rescale"]["t_scale"]
        if doc["rescale"]["x_scale"] <= 0 or t_scale <= 0:
            self._set_status("x_scale and t_scale must be > 0.", error=True)
            return None
        for p, e in doc["params"].items():         # UX pre-check; model_store._check_schema re-enforces
            # None means the field did not parse -- a blank box, or "-" mid-typing. Caught here so it
            # is a message about THAT box rather than model_store rejecting a 0.0 the user never typed.
            if e["lo"] is None or e["hi"] is None:
                which = "min" if e["lo"] is None else "max"
                self._set_status(f"Parameter '{p}': {which} is blank or not a number.", error=True)
                return None
            if not e["lo"] < e["hi"]:
                self._set_status(f"Parameter '{p}': min must be < max.", error=True)
                return None
            if not e["lo"] <= e["value"] <= e["hi"]:
                self._set_status(f"Parameter '{p}': value {e['value']} is outside its bounds "
                                 f"[{e['lo']}, {e['hi']}].", error=True)
                return None
            # A log box needs a positive lower bound. reparam._log_mask would otherwise downgrade it
            # to linear with a warnings.warn that never reaches the GUI, so the model would train in a
            # coordinate the user did not pick, silently. model_store._check_schema re-enforces this.
            if e["box"] == "log" and e["lo"] <= 0:
                self._set_status(f"Parameter '{p}': a log box needs min > 0 (got {e['lo']}). "
                                 f"Raise min, or set its box to 'linear'.", error=True)
                return None

        # Smoke integration at the stream's fine ND step (dt_exp over the t_scale upper bound).
        try:
            n_vars = len(compiled.var_names)
            params = torch.tensor([[doc["params"][p]["value"] for p in compiled.param_names]])
            force = torch.zeros((1, n_vars, _SMOKE_STEPS + 1))
            model = UserModel(compiled, torch.unbind(params, dim=1), force, batch_size=1)
            inits = torch.tensor([[float(v["init"]) for v in doc["variables"]]])
            dt_nd = DT_EXP_S / (2.0 * t_scale)
            res = sdeint.Solver().euler(model, inits, (0.0, _SMOKE_STEPS * dt_nd), _SMOKE_STEPS + 1,
                                        state_dep_drift=compiled.state_dep_noise)
            if not bool(torch.isfinite(res).all()):
                self._set_status("Validation integration diverged (NaN/inf) -- check the drift/noise "
                                 "expressions or the initial conditions.", error=True)
                return None
        except Exception as e:                       # noqa: BLE001 -- any numeric failure is user-facing
            self._set_status(f"Validation integration failed: {e}", error=True)
            return None
        return doc

    def _validate_clicked(self):
        doc = self._validate()
        if doc is not None:
            self._set_status(f"'{doc['name']}' is valid: {len(doc['variables'])} variable(s), "
                             f"{len(doc['params'])} parameter(s). Ready to save.")

    def _save(self):
        if BasePanel._running:
            self._set_status("A task is running -- wait for it to finish before saving.", error=True)
            return
        doc = self._validate()
        if doc is None:
            return
        json_path = config.MODELS_PATH / f"{doc['name']}.json"
        if doc["name"] != self._editing_name and json_path.exists():
            self._set_status(f"A model named '{doc['name']}' already exists -- edit it from Settings "
                             "or pick another name.", error=True)
            return
        try:
            model_store.save_user_model(doc)
        except Exception as e:                       # noqa: BLE001 -- surface, keep the screen alive
            self._set_status(f"Save failed: {e}", error=True)
            return
        registry.load_user_models()                  # (re-)register; combos are refreshed by on_saved
        self._editing_name = doc["name"]
        self._set_status(f"Saved '{doc['name']}'. It is now available in the Simulate model list.")
        if self._on_saved:
            self._on_saved(doc["name"])

    # ── edit / reset ─────────────────────────────────────────────────────────
    def load_existing(self, name: str) -> None:
        """Populate the form from a saved model (the Settings 'Edit' flow). Renaming saves a copy."""
        doc = model_store.load_user_model(config.MODELS_PATH / f"{str(name).upper()}.json")
        self._editing_name = doc["name"]
        self.name_edit.setText(doc["name"])
        self.vars_edit.setText(", ".join(v["name"] for v in doc["variables"]))
        self._set_variables()
        for row, v in zip(self._var_rows, doc["variables"]):
            row.populate(v)
        self.x_scale.setText(repr(float(doc["rescale"]["x_scale"])))
        self.t_scale.setText(repr(float(doc["rescale"]["t_scale"])))
        self._detect_params()
        for pname, row in self._param_fields.items():
            if pname in doc["params"]:
                # load_user_model has already migrated v1/v2 -> {value, lo, hi, box}
                e = doc["params"][pname]
                row.set_spec(float(e["value"]), float(e["lo"]), float(e["hi"]), e["box"])
        self._set_status(f"Editing '{doc['name']}'. Renaming saves a copy under the new name.")

    def reset(self) -> None:
        """Blank the form (the Settings 'Open model builder' flow)."""
        self._editing_name = None
        self.name_edit.clear()
        self.vars_edit.clear()
        for row in self._var_rows:
            self._rows_host.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._var_rows = []
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_fields = {}
        self._set_status("")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(("⚠ " if error else "") + text)
