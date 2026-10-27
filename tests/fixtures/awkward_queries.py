"""Known-answer queries over the awkward vault, from the vault-hardening
usability test: each query and the note that must rank first. The test's
queries for the Latin-1 note (skipped as unreadable) and the symlinked folder
(not followed, out of scope) are left out; the web clip has its own test."""

KNOWN_ANSWERS = [
    ("who invented the moka pot and when", "Cucina/Caffè è buono.md"),
    ("perché i pomodori hanno il marciume apicale", "Projects/Garden/Tomato watering.md"),
    ("ZX-4471", "Tech/Home network/Router troubleshooting.md"),
    ("lighthouse keeper who logged ships in the strait", "Reference/Maritime history (long).md"),
    ("climbing season for Mount Fuji", "日本語/日本語メモ.md"),
    ("quanto deve essere la pressione della caldaia", "Areas/Home/Boiler.md"),
    ("should cream go in carbonara", "Cucina/Ricette/Primi/Pasta/Carbonara autentica.md"),
    (
        "late delivery penalty in the Rossi contract",
        "Archive/2019/Q3/Clients/Rossi SRL/Contratto/Clausole.md",
    ),
    ("how many snapshots does the nightly backup keep", "Tech/Scripts/Backup script.md"),
    ("sourdough starter feeding ratio", "Legacy/With BOM.md"),
    ("printer jams with recycled paper", "Legacy/Windows CRLF.md"),
    ("heat pump service interval", "Inbox/Malformed yaml.md"),
    ("deployment uses blue-green switching", "Tech/Scripts/Code first.md"),
]

# Files the vault must leave out, each with a word only that file contains.
EXCLUDED = {
    ".trash/Old secret.md": "TRASHMARKER_77",
    ".git/notes.md": "GITMARKER_55",
    ".obsidian/plugins/dataview/README.md": "QUOKKA_PLUGIN_README",
    "Boards/Sketch.excalidraw.md": "EXCALIDRAWMARKER_63",
}

# Words that sit only in frontmatter, which is never indexed as text.
FRONTMATTER_ONLY = ["CRLFFRONTMATTER_88", "BOMFRONTMATTER_99"]
