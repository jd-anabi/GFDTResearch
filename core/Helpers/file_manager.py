import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def _atomic_write(path, writer: Callable[..., None], *, retries: int = 3,
                  backoff_s: float = 0.1) -> Path:
    """Run ``writer(fh)`` against a sibling temp file, then ``os.replace`` it over ``path``.

    The one mechanism every ``atomic_*`` helper below shares. Writes a sibling ``<name>.tmp``, fsyncs
    it, then ``os.replace``s it over the destination -- which is atomic on POSIX and on Windows
    (MoveFileEx with REPLACE_EXISTING). A reader therefore sees either the whole old file or the whole
    new one, never a truncated mixture. The temp file is a SIBLING, not a file in the system temp dir,
    because os.replace is only atomic within one volume.

    fsync before the rename, not after, is the load-bearing order: the rename can be durable while the
    bytes it points at are still in the page cache, which is exactly how a power cut produces a file
    that exists, has the right size, and is full of zeros.

    :param writer: called with the open binary temp handle. Anything it raises propagates untouched,
                   after the partial temp file has been removed.
    :param retries: os.replace attempts on PermissionError. Windows-specific: a virus scanner or an
                    Explorer preview holding the destination open makes the rename fail transiently,
                    and a multi-day run must not die on that.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        # Clean up the partial temp file. It is harmless (nothing ever reads a .tmp) but a checkpoint
        # writes every N batches, so a recurring failure would otherwise leave one per attempt beside
        # the file it failed to replace -- noise exactly where someone is trying to diagnose a
        # failing run. The original exception is re-raised untouched.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (attempt + 1))
    return path                                          # unreachable; keeps the return type honest


def atomic_torch_save(obj, path, *, retries: int = 3, backoff_s: float = 0.1) -> Path:
    """``torch.save`` that a crash cannot leave half-written. Mechanism: :func:`_atomic_write`.

    Added for the training checkpoint (C-11), which rewrites its state file every N batches and so
    turns "non-atomic torch.save against a cancel" from a catalogued low-priority risk into a real
    one. It now also carries the END-OF-RUN artifacts -- ``save_mix_dist`` (the ND prior) and
    ``orchestrator.save_posterior_artifacts`` (the posterior and its ``.rot.pt`` sidecar). The window
    there is one write rather than one every 50 batches, but what it protects is the product of a
    multi-day run, and a torn ``.pt`` is not detectably torn: it is an unpickling error hours later,
    or a sidecar that loads with half its keys and decodes every latent sample through a default.
    """
    return _atomic_write(path, lambda fh: torch.save(obj, fh), retries=retries, backoff_s=backoff_s)


def atomic_savez(path, arrays: dict, *, retries: int = 3, backoff_s: float = 0.1) -> Path:
    """``np.savez`` that a crash cannot leave half-written. Mechanism: :func:`_atomic_write`.

    Arrays arrive as a DICT rather than ``**kwargs`` so that an array named ``path``/``retries`` can
    never collide with this function's own parameters -- np.savez's own ``**kwds`` signature has that
    hazard and there is no reason to inherit it.

    numpy appends ``.npz`` only when it is given a NAME; handed an open handle it writes exactly what
    it is given, so the temp file's ``.tmp`` suffix cannot end up baked into the destination.
    """
    return _atomic_write(path, lambda fh: np.savez(fh, **arrays), retries=retries, backoff_s=backoff_s)

# --- Regex Definitions ---
# Float Value (Scientific Notation)
FLOAT_REGEX = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Flexible Assignment: name [opt_units] = val
ASSIGNMENT_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s*$')
# Flexible Bounds: name [opt_units] = val [opt_in] (bounds)
BOUNDS_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s+(?:in\s+)?[\[\(](?P<tup>.*?)[\]\)]\s*$')
# Bounds-only (no value): name [opt_units] [opt_in] (bounds) -- for the decoupled BOUNDS file.
BOUNDS_ONLY_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*(?:in\s+)?[\[\(](?P<tup>.*?)[\]\)]\s*$')

def parse_model_file(file_name: str) -> tuple:
    """
    Parses a model configuration file to extract initialization variables, parameters, rescaling values,
    forcing parameters, and associated unit types. The function processes a file with sections defined
    by specific headers, and categorizes the data into corresponding dictionaries or structures.

    :param file_name: The path to the model file to be parsed.
    :return: A tuple containing extracted model data.
        - ``init_conditions``: An ordered dictionary of initial conditions mapping variable names to their values.
        - ``parameters``: An ordered dictionary where each key is a parameter name, and the value is a tuple of
          its initial value and bounds.
        - ``forcing_params``: An ordered dictionary of time-dependent forcing parameters.
        - ``collected_units``: A tuple of unit strings found during processing.
        If `nd` is True, the tuple includes:
        - ``init_conditions``: An ordered dictionary of initial conditions.
        - ``parameters``: Parameter data with values and bounds.
        - ``rescale_params``: Rescaling data for specific variables.
        - ``forcing_params``: Forcing parameter data.
        - ``collected_units``: Unit strings found during processing.
    :rtype: tuple
    """
    # --- Data Structures ---
    init_conditions = OrderedDict()
    parameters = OrderedDict()  # Format: {name: (val, (min, max))}
    rescale_params = OrderedDict()
    forcing_params = OrderedDict()
    collected_units = set()

    # --- State/Section Management ---
    current_section = None

    # split string content into lines (simulating file read)
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    def process_units(match_obj):
        if match_obj.group('units'):
            raw_units = match_obj.group('units').split()
            for u in raw_units:
                base_unit = u.split('^')[0]
                collected_units.add(base_unit)

    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # --- Section Detection ---
        if line.startswith("#"):
            if "Initial Conditions" in line:
                current_section = "INIT"
            elif "Parameters" in line and "Forcing" not in line:
                if line.startswith("# Dimensional"):
                    current_section = "RESCALE"
                else:
                    current_section = "PARAM"
            elif "Forcing Parameters" in line:
                current_section = "FORCING"
            continue

        # 1. Initial Conditions (Using ASSIGNMENT_PATTERN)
        if current_section == "INIT":
            match = ASSIGNMENT_PATTERN.search(line)
            if match:
                init_conditions[match.group('name')] = float(match.group('val'))
                process_units(match)

        # 2. Parameters, Forcing, Rescale (Using BOUNDS_PATTERN)
        elif current_section in ["PARAM", "FORCING", "RESCALE"]:
            match = BOUNDS_PATTERN.search(line)
            if match:
                name = match.group('name')
                val = float(match.group('val'))
                bounds = tuple(float(x) for x in re.findall(FLOAT_REGEX, match.group('tup')))
                if current_section == "PARAM":
                    target_dict = parameters
                elif current_section == "FORCING":
                    target_dict = forcing_params
                else:
                    target_dict = rescale_params
                target_dict[name] = (val, bounds)
                process_units(match)

    return init_conditions, parameters, rescale_params, forcing_params, tuple(collected_units)


# --- Decoupled bounds / units / values parsers (additive; parse_model_file is left untouched) ---
def _read_lines(file_name: str) -> list[str]:
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read().strip().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_name}")


def _section_of(header_line: str) -> str | None:
    """Map a '# ...' header line to a section key, or None. Mirrors parse_model_file's logic + Units."""
    if "Initial Conditions" in header_line:
        return "INIT"
    if "Units" in header_line:
        return "UNITS"
    if "Parameters" in header_line and "Forcing" not in header_line:
        return "RESCALE" if header_line.startswith("# Dimensional") else "PARAM"
    if "Forcing Parameters" in header_line:
        return "FORCING"
    return None


def parse_bounds_file(file_name: str) -> tuple:
    """
    Parse a BOUNDS-only file: parameter bounds for the ND / Dimensional / Forcing sections, no values,
    no initial conditions. Lines look like ``name (units) in (lo, hi)`` (units and ``in`` optional).

    :return: (parameters, rescale_params, forcing_params, collected_units) where each param dict maps
             {name: (None, (lo, hi))} -- the value slot is None until a cell file fills it.

    NOTE on ``collected_units``: inline ``(units)`` annotations are parsed but ALL THREE callers
    (``cli.parse_cell``, ``cli.make_sim_config``, ``inference_tabs``) discard the
    result, and no shipped bounds file carries them -- units are declared centrally (the per-model units
    file, or the Config tab's units control) precisely so one declaration governs every file. It is
    returned for backward compatibility; do not start relying on per-line units here, or a bounds file
    could silently disagree with the config's declared units.
    """
    parameters, rescale_params, forcing_params = OrderedDict(), OrderedDict(), OrderedDict()
    collected_units = set()
    targets = {"PARAM": parameters, "RESCALE": rescale_params, "FORCING": forcing_params}
    current_section = None
    for line in _read_lines(file_name):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = _section_of(line)
            continue
        if current_section in targets:
            match = BOUNDS_ONLY_PATTERN.search(line)
            if match:
                bounds = tuple(float(x) for x in re.findall(FLOAT_REGEX, match.group('tup')))
                targets[current_section][match.group('name')] = (None, bounds)
                if match.group('units'):
                    for u in match.group('units').split():
                        collected_units.add(u.split('^')[0])
    return parameters, rescale_params, forcing_params, tuple(collected_units)


def parse_units_file(file_name: str) -> tuple:
    """
    Parse a UNITS-only file: a ``# Units`` section of whitespace-separated unit tokens (one per physical
    dimension, e.g. ``nm ms pN Hz rad``). Returns the unit set as a tuple.
    """
    collected_units = set()
    current_section = None
    for line in _read_lines(file_name):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = "UNITS" if "Units" in line else None
            continue
        if current_section == "UNITS":
            for u in line.split():
                collected_units.add(u.split('^')[0])
    return tuple(collected_units)


def parse_values_file(file_name: str) -> tuple:
    """
    Parse a CELL / VALUES file: initial conditions + VALUES for the ND / Dimensional / Forcing params
    (the ground-truth point). A trailing ``in (lo, hi)`` on a line is ignored -- only the value is taken,
    so existing full cell files (value + bounds) work unchanged as a ground-truth source.

    :return: (init_conditions, parameters, rescale_params, forcing_params) where each param dict maps
             {name: val} (plain floats).
    """
    init_conditions, parameters, rescale_params, forcing_params = (
        OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict())
    targets = {"PARAM": parameters, "RESCALE": rescale_params, "FORCING": forcing_params}
    current_section = None
    for line in _read_lines(file_name):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = _section_of(line)
            continue
        if current_section == "INIT":
            match = ASSIGNMENT_PATTERN.search(line)
            if match:
                init_conditions[match.group('name')] = float(match.group('val'))
        elif current_section in targets:
            # accept both `name = val` and `name = val in (lo,hi)` -- take the value either way
            match = BOUNDS_PATTERN.search(line) or ASSIGNMENT_PATTERN.search(line)
            if match:
                targets[current_section][match.group('name')] = float(match.group('val'))
    return init_conditions, parameters, rescale_params, forcing_params


def list_dir(files_dir: str, return_list: bool = True,
             keep: Callable[[str], bool] | None = None) -> list[str] | list[None]:
    """
    Lists all files in the specified directory and its subdirectories, with an option to return a list of files.

    The function walks through the directory tree starting from the given directory. It prints the directory structure
    with files ordered and numbered. Optionally, it returns a list of all files found.

    :param files_dir: Path to the directory that needs to be traversed.
    :type files_dir: str
    :param return_list: A flag indicating whether to return the list of files. If True, the list of files is returned.
        Default is True.
    :type return_list: bool
    :param keep: Optional predicate applied to each file's path (relative to files_dir). Only files for which
        keep(rel) is True are numbered, printed, and returned -- so the printed ``(N)`` numbering stays in sync
        with the returned list (e.g. a posterior picker that hides ``.rot.pt`` sidecars / ``.loss.npz`` curves).
        Default None lists every file.
    :type keep: Callable[[str], bool] | None
    :return: A list of all files in the directory and its subdirectories if `return_list` is True; otherwise, None.
    :rtype: list[str] | None
    """
    # list files in directory
    model_files = [""]
    file_num = 1
    for root, dirs, files in os.walk(files_dir):
        level = root.replace(files_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}")
        subindent = " " * 2 * (level + 1)
        for file in files:
            # Return the path relative to files_dir so subfoldered layouts (e.g. Bounds/<model>/<cell>.txt)
            # resolve correctly and same-named files across subfolders don't collide. For a flat directory
            # this is just the basename, so callers doing `PATH / result[i]` are unaffected.
            rel = os.path.relpath(os.path.join(root, file), files_dir)
            if keep is not None and not keep(rel):
                continue          # skip before numbering so printed (N) matches the returned list
            model_files.append(rel)
            print(f"{subindent}({file_num}) {file}")
            file_num += 1
    model_files.pop(0)
    if return_list:
        return model_files
    return []

def load_experimental_data(file_path: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load a 1D experimental time series from CSV or NPY.

    Supported formats:
      - .npy: NumPy binary, expects a 1D array of values.
      - .csv: comma-separated. If single column, treated as values. If multiple
              columns, the LAST column is treated as the values (assumes time is
              in earlier columns and discarded since dt_exp is known).

    :param file_path: Path to the data file.
    :param dtype: Tensor data type. Defaults to torch.float32.
    :return: 1D torch.Tensor of values.
    :raises ValueError: If file extension is unsupported or data shape is invalid.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        arr = np.load(file_path)
        if arr.ndim != 1:
            arr = arr.squeeze()
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D array in {file_path}, got shape {arr.shape}")
    elif ext == ".csv":
        arr = np.loadtxt(file_path, delimiter=",", ndmin=2)
        # take last column (handles single-column or time+value layouts)
        arr = arr[:, -1]
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .npy or .csv.")

    return torch.tensor(arr, dtype=dtype)

def save_mix_dist(dist, filename: str, *, model: str = None, param_keys: list = None):
    """
    Serializes a (possibly transformed) ND prior.

    If dist is TransformedDistribution wrapping a MixtureSameFamily (post-reparam fix),
    saves the base GMM's means/covariances/weights plus the (lows, highs) that define
    the bijection. If dist is a bare MixtureSameFamily (legacy), saves GMM only.
    load_mix_dist discriminates on the presence of 'lows'/'highs' keys.

    ``model`` and ``param_keys`` make the file SELF-DESCRIBING. The GMM is fit in the box's own
    coordinate, so a prior is only meaningful against the exact (model, parameter set + ORDER, box)
    it was built for -- but nothing on the load path used to check any of that, and the file gives no
    hint either: means are latent, and lows/highs alone cannot say which parameter each column is.
    A prior built for one cell's box therefore loaded silently against another's, and the resulting
    posterior was describable only by comparing GMM component counts after the fact. Both are
    optional so legacy files still load; ``build_prior`` treats absence as "unverifiable, warn".
    """
    from torch.distributions.transforms import AffineTransform, ComposeTransform
    from core.SBI.reparam import UnitToBoxTransform
    if isinstance(dist, torch.distributions.TransformedDistribution):
        base = dist.base_dist

        # dist.transforms is a list; entries may be atomic Transforms or ComposeTransforms.
        # Walk one level deep to find the box bijection (UnitToBoxTransform; AffineTransform legacy).
        box = None
        for t in dist.transforms:
            for inner in (t.parts if isinstance(t, ComposeTransform) else [t]):
                if isinstance(inner, (UnitToBoxTransform, AffineTransform)):
                    box = inner
                    break
            if box is not None:
                break

        if box is None:
            raise ValueError("TransformedDistribution has no box transform; can't extract bounds.")

        if isinstance(box, UnitToBoxTransform):
            lows, highs, log_mask = box.lows, box.highs, box.log_mask
        else:  # legacy AffineTransform box (all-linear)
            lows, highs = box.loc, box.loc + box.scale
            log_mask = torch.zeros_like(box.loc, dtype=torch.bool)

        data_to_save = {
            'means':       base.component_distribution.loc,
            'covariances': base.component_distribution.covariance_matrix,
            'weights':     base.mixture_distribution.probs,
            'lows':        lows,
            'highs':       highs,
            'log_mask':    log_mask,   # per-param linear/log flags; absent in pre-log saved priors
        }
    else:
        # Legacy path: raw MixtureSameFamily
        data_to_save = {
            'means':       dist.component_distribution.loc,
            'covariances': dist.component_distribution.covariance_matrix,
            'weights':     dist.mixture_distribution.probs,
        }
    if model is not None:
        data_to_save['model'] = str(model)
    if param_keys is not None:
        data_to_save['param_keys'] = [str(k) for k in param_keys]
    # Atomic, because this is the file build_posterior's checkpoint identity fingerprints and that
    # SBC later draws theta* from: a prior half-replaced by a crashed re-save is not a broken run, it
    # is a run that resumes against a distribution nobody can name.
    atomic_torch_save(data_to_save, filename)

def read_prior_metadata(filename: str) -> dict:
    """The identity of a saved ND prior: ``model``, ``param_keys``, ``lows``, ``highs``, ``log_mask``.

    Separate from :func:`load_mix_dist` on purpose -- the caller that VALIDATES a prior against a
    config wants the metadata before paying to reconstruct the distribution, and load_mix_dist's
    return type (a Distribution) has several callers that must not change. Keys are absent for
    files written before they were recorded; the caller decides whether that is fatal.
    """
    data = torch.load(filename, map_location='cpu')
    return {k: data[k] for k in ('model', 'param_keys', 'lows', 'highs', 'log_mask') if k in data}


def load_mix_dist(filename: str, device: torch.device = torch.device('cpu')):
    """
    Loads a serialized ND prior. Returns a TransformedDistribution if bounds were saved
    (post-reparam), otherwise a raw MixtureSameFamily (legacy).
    """
    data = torch.load(filename, map_location=device)
    means   = data['means']
    covs    = data['covariances']
    weights = data['weights']
    comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
    mix_dist  = torch.distributions.Categorical(probs=weights)
    latent_prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)

    if 'lows' in data and 'highs' in data:
        from core.SBI.reparam import build_box_bijection
        log_mask = data.get('log_mask', None)          # absent in pre-log saved priors => linear box
        if log_mask is not None:
            log_mask = log_mask.to(device)
        T_nd = build_box_bijection(data['lows'].to(device), data['highs'].to(device), log_mask)
        return torch.distributions.TransformedDistribution(latent_prior, T_nd)
    else:
        return latent_prior  # legacy
