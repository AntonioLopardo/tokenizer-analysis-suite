# morphscore integration notes

## Duck-typing fix needed

`morphscore.py::encode_text` checks `isinstance(tokens_raw, dict)` before accessing `"input_ids"`.
This breaks for HuggingFace `BatchEncoding` objects, which are dict-like but not `dict` instances.

Fix: change to `hasattr(tokens_raw, '__getitem__')`.

Can't push upstream — `cimeister/morphscore` remote is not writable. Either:
- Get collaborator access / open a PR to `cimeister/morphscore`
- Fork to `AntonioLopardo/morphscore` and update the submodule URL
