# Transformers Compatibility

The previous workspace included a vendored fork at `workspace/transformers-fix-lerobot_openpi/`.

The refactored layout removes that embedded copy and expects compatibility to be handled through pinned package versions in the active environment. If specific OpenPI or SmolVLA workflows require a patched `transformers` build, document and pin that environment separately instead of vendoring the whole repository.
