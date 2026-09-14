"""What a benchmark series *is*: its identity, its direction, its unit.

Every layer -- viewer, analysis, MCP, reports, the pytest suite -- has to
agree on these or they quietly disagree about the same number. They have not
agreed: before this module the "lower is better" rule was written out in six
places, the seconds-to-milliseconds rule in two (with different conditions),
and the series key tuple in twenty-five. Adding the ``*-infer`` family in
a7852ea then broke four call sites at once, because each of them had spelled
out "count the rows in perf" for itself.

So the rule here is not "these helpers exist" but "nothing outside this
module gets to spell these out again".
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# exec_mode families
# ---------------------------------------------------------------------------

# Two families are ingested per prompt, from the same llm_bench row.
#
# ``1st`` / ``2nd``
#     End-to-end token latency (first_latency / second_avg_latency). This is
#     *the* daily metric -- what a pipeline user actually waits for -- and the
#     only family the regression verdict is computed from.
#
# ``1st-infer`` / ``2nd-infer``
#     The infer-only slice of the same token. It excludes tokenization,
#     detokenization and multimodal embedding prep, so on VLMs it can
#     understate TTFT by more than an order of magnitude. Stored so a GPU
#     kernel regression can be told apart from pipeline-side overhead;
#     deliberately kept out of every verdict and every count.
TOKEN_EXEC_MODES = ("1st", "2nd")
INFER_EXEC_MODES = ("1st-infer", "2nd-infer")

INFER_SUFFIX = "-infer"


def is_infer_exec_mode(exec_mode: str | None) -> bool:
    """True for the diagnostic infer-only series."""
    return bool(exec_mode) and exec_mode.endswith(INFER_SUFFIX)


def infer_twin(exec_mode: str) -> str | None:
    """The infer counterpart of a token series, or None if there isn't one."""
    return f"{exec_mode}{INFER_SUFFIX}" if exec_mode in TOKEN_EXEC_MODES else None


def exclude_infer_sql(column: str = "exec_mode") -> str:
    """SQL predicate keeping only the families that carry a verdict.

    A suffix match rather than an ``IN`` list, so a future infer series (say
    ``pipeline-infer``) is excluded without a second edit here.
    """
    return f"{column} NOT LIKE '%{INFER_SUFFIX}'"


# ---------------------------------------------------------------------------
# Series identity
# ---------------------------------------------------------------------------

#: Columns that identify one benchmark series, in the order every query uses.
SERIES_KEY_COLUMNS = ("model", "precision", "in_token", "out_token", "exec_mode")


def series_key_sql(alias: str = "") -> str:
    """``model, precision, in_token, out_token, exec_mode``, optionally aliased.

    Use in SELECT / GROUP BY / USING so the column order cannot drift between
    a query and the code unpacking its rows.
    """
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{column}" for column in SERIES_KEY_COLUMNS)


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """One benchmark point, independent of which run produced it."""

    model: str
    precision: str
    in_token: int
    out_token: int
    exec_mode: str

    @classmethod
    def from_row(cls, row) -> "SeriesKey":
        """Build from anything indexable by the key column names or by order."""
        try:
            values = [row[column] for column in SERIES_KEY_COLUMNS]
        except (TypeError, KeyError, IndexError):
            values = list(row)[:len(SERIES_KEY_COLUMNS)]
        model, precision, in_token, out_token, exec_mode = values
        return cls(str(model), str(precision), int(in_token), int(out_token),
                   str(exec_mode))

    def as_tuple(self) -> tuple:
        return (self.model, self.precision, self.in_token, self.out_token,
                self.exec_mode)

    @property
    def is_infer(self) -> bool:
        return is_infer_exec_mode(self.exec_mode)

    def with_exec_mode(self, exec_mode: str) -> "SeriesKey":
        """Same point, different family -- used to pair a token series with
        its infer twin, which requires the token shape to match exactly."""
        return SeriesKey(self.model, self.precision, self.in_token,
                         self.out_token, exec_mode)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

#: Units where a larger number is worse. Everything else ('FPS', 'tps', ...)
#: is higher-is-better.
LOWER_IS_BETTER_UNITS = frozenset({"ms", "s", "%"})


def lower_is_better(unit: str | None) -> bool:
    return unit in LOWER_IS_BETTER_UNITS


def direction_label(unit: str | None) -> str:
    return "lower_is_better" if lower_is_better(unit) else "higher_is_better"


def direction_sign(unit: str | None) -> int:
    """+1 when a rise is a regression, -1 when a fall is.

    Multiply a raw delta by this to get a number where positive always means
    "worse", which is what makes sort-by-worst trivial.
    """
    return 1 if lower_is_better(unit) else -1


def direction_label_sql(column: str = "unit") -> str:
    units = ", ".join(f"'{unit}'" for unit in sorted(LOWER_IS_BETTER_UNITS))
    return (f"CASE WHEN {column} IN ({units}) THEN 'lower_is_better'"
            f" ELSE 'higher_is_better' END")


def worsening_pct(current: float, baseline: float, unit: str | None) -> float | None:
    """Signed so that positive means worse, whichever way the unit points."""
    if baseline in (None, 0) or current is None:
        return None
    return direction_sign(unit) * (current - baseline) / baseline


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

# The image-generation models report seconds; the viewer, the Excel paste and
# the report convention are all milliseconds.
#
# schema.sql gated this on a model list while queries.legacy_geomean_summary
# applied it to any 's' row. Checked against the whole DB: every 's' row
# belongs to a model on that list, so the two rules agree on all existing data
# and the list is redundant. The unconditional rule is kept because it also
# covers the next image-generation model without an edit here -- the list
# would have silently missed it.
SECONDS_UNIT = "s"
MILLISECONDS_UNIT = "ms"


def normalize_unit(unit: str | None) -> str | None:
    return MILLISECONDS_UNIT if unit == SECONDS_UNIT else unit


def normalize_value(value: float | None, unit: str | None) -> float | None:
    return None if value is None else (
        value * 1000.0 if unit == SECONDS_UNIT else value
    )


def normalize_value_sql(value_column: str = "value",
                        unit_column: str = "unit") -> str:
    return (f"CASE WHEN {unit_column} = '{SECONDS_UNIT}'"
            f" THEN {value_column} * 1000 ELSE {value_column} END")


def normalize_unit_sql(unit_column: str = "unit") -> str:
    return (f"CASE WHEN {unit_column} = '{SECONDS_UNIT}'"
            f" THEN '{MILLISECONDS_UNIT}' ELSE {unit_column} END")


# ---------------------------------------------------------------------------
# Token buckets
# ---------------------------------------------------------------------------

# The live short/long split, matching schema.sql's perf_with_buckets.
#
# Not to be confused with the <400 / 401-1200 banding inside
# queries.legacy_geomean_summary: that one reproduces a legacy artifact and
# has a blind spot -- it drops 10.4% of token rows (every in_token above 1200,
# and exactly 400) rather than bucketing them. It is scoped to that one
# function and goes away with the legacy viewer; do not generalise it.
SHORT_TOKEN_MAX = 100


def token_bucket(in_token: int | None) -> str:
    """``'0'`` / ``'short'`` / ``'long'`` for an input token count."""
    if not in_token:
        return "0"
    return "short" if in_token < SHORT_TOKEN_MAX else "long"
