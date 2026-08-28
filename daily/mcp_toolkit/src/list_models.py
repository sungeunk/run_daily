"""List model names available for filtering, optionally scoped to a machine."""
from _common import DB_PATH, emit, input_str
from viewer import queries

machine = input_str("MACHINE")
models = queries.list_models(DB_PATH, machine=machine)
emit(models)
