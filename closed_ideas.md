# Closed setup approaches

- Do not initialize an unrelated Git history over the local research directory:
  GitHub has existing code and history. Use the branch based on its main commit.
- Do not replace root `requirements.txt` with CPU dependencies: the upstream
  installer still uses it. Separate hashed development locks preserve both flows.
- Do not commit caches, datasets, checkpoints or generated experiments to make a
  clone appear self-contained. Preserve their checksums and provenance through
  explicit artifact handoff, and use the small synthetic test fixtures in Git.

Earlier scoped negative research results remain in
[the archived notes](docs/archive/local-handoff-20261112/closed_ideas.md).
