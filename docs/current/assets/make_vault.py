"""Build a synthetic but realistic Obsidian vault for usability testing."""
import os
import random
import shutil
import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "vault"
EXT = HERE / "external_notes"
for d in (V, EXT):
    if d.exists():
        shutil.rmtree(d)
V.mkdir()
EXT.mkdir()


def w(rel, text, enc="utf-8", newline="\n", bom=False, base=V):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\n", newline).encode(enc)
    if bom:
        data = b"\xef\xbb\xbf" + data
    p.write_bytes(data)


def png(rel):
    raw = b"\x00\xff\x00\x00"
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    p = V / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# --- .obsidian config, trash, git ---
w(".obsidian/app.json", '{"attachmentFolderPath": "Attachments"}')
w(".obsidian/workspace.json", '{"main": {}}')
w(".obsidian/plugins/dataview/manifest.json", '{"id": "dataview"}')
w(".obsidian/plugins/dataview/README.md", "# Dataview plugin README\nThis plugin readme mentions QUOKKA_PLUGIN_README marker.\n")
w(".trash/Old secret.md", "# Old secret\nThe deleted note mentions TRASHMARKER_77 which should not be searchable.\n")
w(".git/HEAD", "ref: refs/heads/main\n")
w(".git/notes.md", "# Git internal\nGITMARKER_55 hidden git markdown.\n")

# --- attachments ---
png("Attachments/diagram.png")
png("Attachments/Screenshot 2024-05-01 at 10.00.00.png")
(V / "Attachments/manual.pdf").write_bytes(b"%PDF-1.4\n%fake\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
w("Boards/Project map.canvas", '{"nodes":[{"id":"a","type":"text","text":"CANVASMARKER_31 roadmap","x":0,"y":0,"width":200,"height":100}],"edges":[]}')
w("Boards/Data.base", "filters:\n  and:\n    - file.hasTag(\"book\")\nviews:\n  - type: table\n    name: Books\n")
excal_json = '{"type":"excalidraw","version":2,"elements":[' + ",".join(
    f'{{"id":"el{i}","type":"rectangle","x":{i*13},"y":{i*7},"width":120,"height":60,"seed":{random.Random(i).randint(1,10**9)}}}' for i in range(150)
) + "]}"
w("Boards/Sketch.excalidraw.md", f"""---
excalidraw-plugin: parsed
tags: [excalidraw]
---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠==

# Text Elements
Warehouse layout draft ^t1

%%
# Drawing
```json
{excal_json}
```
%%
""")

# --- Daily notes ---
for i, (d, txt) in enumerate([
    ("2024-03-01", "Called the plumber about the leaking boiler. He'll come Tuesday."),
    ("2024-03-02", "Ran 8 km along the river. Knee felt fine."),
    ("2024-03-03", "Read two chapters of *The Name of the Rose*."),
    ("2024-03-04", "Oggi ho rinnovato il passaporto in questura, ci vogliono 3 settimane."),
    ("2024-03-05", "Team retro: we agreed to move standup to 9:30."),
    ("2024-03-06", "Tried the new ramen place on Via Roma, broth too salty."),
]):
    w(f"Daily/2024/03/{d}.md", f"""---
date: {d}
tags: [daily]
---
# {d}

- [x] Morning pages
- [ ] Inbox zero
- [ ] Call mum

## Log
{txt}
""")

# --- Projects with known facts ---
w("Projects/Garden/Tomato watering.md", """---
tags: [garden, vegetables]
aliases: [Tomatoes, Pomodori]
created: 2024-04-12
---
# Tomato watering

Water tomatoes deeply twice a week rather than a little every day; shallow watering makes roots stay near the surface.

> [!warning] Blossom end rot
> Blossom end rot (the black leathery patch at the bottom of the fruit) is caused by calcium deficiency, usually because irregular watering stops the plant from moving calcium into the fruit. Adding eggshells does not fix it quickly.

## Mulch
Use straw mulch 5 cm thick to keep soil moisture even. See [[Compost|my compost notes]].

| Week | Litres per plant |
|------|------------------|
| 1-4  | 2 |
| 5-10 | 4 |
""")

w("Projects/Garden/Compost.md", """# Compost
Keep a ratio of roughly 30:1 carbon to nitrogen. Brown material: dry leaves, cardboard. Green material: kitchen scraps, grass.
Turn the pile every two weeks. ^turn-rule
""")

w("Cucina/Caffè è buono.md", """---
tags: [cucina, caffè]
---
# Caffè è buono

La moka fu inventata da Alfonso Bialetti nel 1933 a Omegna, sul lago d'Orta. Il caffè sale per la pressione del vapore che si forma nella caldaia inferiore.

## Consigli
Non pressare mai la polvere nel filtro della moka: va livellata, non compressa, altrimenti la valvola di sicurezza può aprirsi.

![[diagram.png]]
""")

w("Cucina/Ricette/Primi/Pasta/Carbonara autentica.md", """# Carbonara autentica
Ingredienti per 4 persone: 400 g di spaghetti, 150 g di guanciale, 4 tuorli e 1 uovo intero, 80 g di pecorino romano, pepe nero.

La panna non va mai messa nella carbonara. Il guanciale si rosola senza olio finché il grasso diventa trasparente.

> [!tip] Segreto
> Mescolare uova e pecorino fuori dal fuoco per evitare l'effetto frittata.
""")

w("Tech/Home network/Router troubleshooting.md", """---
tags: [tech, network]
---
# Router troubleshooting

When the Fritzbox shows error code ZX-4471 the DSL sync failed because the line profile changed upstream. Fix: reboot the modem, then force profile 17a under Internet > DSL Information.

## Wi-Fi channels
2.4 GHz: use channel 1, 6 or 11 only.
""")

w("Tech/Scripts/Backup script.md", """# Backup script
This note documents the nightly backup.

```bash
# Configuration section
SRC=/home/me
# Run rsync with delete
rsync -a --delete "$SRC" /mnt/nas/backup
```

```python
# Dataclass helpers
def checksum(path):
    # Compute sha256 of a file
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
```

## Schedule
Runs via cron at 03:15 every night and keeps 14 snapshots of retention.
""")

w("Tech/Scripts/Code first.md", """```python
# This comment is not a title
print("hello")
```

Body text after the code: the deployment uses blue-green switching.
""")

w("Tech/Math/Bayes.md", """# Bayes theorem

$$
P(A\\mid B) = \\frac{P(B\\mid A)\\,P(A)}{P(B)}
$$

Inline: the posterior $P(H|E)$ grows when the likelihood ratio exceeds one. A false positive rate of 5% with a prevalence of 1% gives a positive predictive value of about 16%.
""")

w("Reading/Books.md", """---
tags: [reading]
---
# Books

```dataview
TABLE author, rating FROM #book SORT rating DESC
```

```dataviewjs
dv.list(dv.pages("#book").file.name)
```

Favourite so far: *Il nome della rosa* by Umberto Eco, set in a Benedictine abbey in 1327.
""")

w("Reading/Book - Dune.md", """---
title: Dune
author: Frank Herbert
rating: 5
tags: [book, scifi]
---
# Dune
The spice melange extends life and enables space navigation by the Guild navigators. Arrakis is the only source. #book
""")

# Duplicate filenames
w("Work/Acme/Notes.md", "# Notes\nAcme kickoff: the budget for the warehouse robots is 240k EUR.\n")
w("Personal/Notes.md", "# Notes\nPersonal reminder: renew the gym membership in September.\n")
w("Work/Acme/Meeting 2024-05-10.md", """# Meeting 2024-05-10
Attendees: [[Notes|Acme notes]], Laura, Mark.
Decision: we picked the Kiva-style robots. Embed of the plan below:

![[Tomato watering#Mulch]]

Reference to a block: [[Compost#^turn-rule]]
""")

# Deep nesting
w("Archive/2019/Q3/Clients/Rossi SRL/Contratto/Clausole.md", """# Clausole del contratto Rossi
La penale per ritardata consegna è pari allo 0,5% del valore dell'ordine per ogni settimana di ritardo, fino a un massimo del 10%.
""")

# Non-ASCII filenames
w("🚀 Launch plan.md", """# 🚀 Launch plan
Launch date is set for 14 November. Marketing budget: 12k. Beta testers: 40 people from the waitlist.
""")
w("日本語/日本語メモ.md", """# 富士山の登山
富士山の登山シーズンは7月上旬から9月上旬までです。吉田ルートが最も人気があります。山頂の気温は夏でも5度前後です。
""")
w("Viaggi/Città d'arte/Firenze — note.md", """# Firenze
La cupola del Brunelleschi si visita prenotando online; sono 463 gradini fino in cima.
""")

# Frontmatter variants
w("Inbox/Frontmatter only.md", """---
title: Placeholder
tags: [todo]
status: draft
---
""")
w("Inbox/Frontmatter only no newline.md", "---\ntitle: Placeholder2\n---")
w("Inbox/Malformed yaml.md", """---
tags: [unclosed, list
aliases: "quote never closed
: weird
---
# Malformed yaml
The heat pump service interval is every 18 months; the filter is behind the front grille.
""")
w("Inbox/Empty.md", "")
w("Inbox/Only headings.md", "# Title\n## Section A\n## Section B\n### Sub\n")
w("Inbox/Whitespace only.md", "\n\n   \n")

# Encodings / line endings
w("Legacy/Vecchia nota latin1.md", "# Vecchia nota\nLa città di Ferrara è famosa per le mura e per la salama da sugo; perché non andarci in bicicletta?\n", enc="latin-1")
w("Legacy/Windows CRLF.md", """---
tags: [windows]
secretfm: CRLFFRONTMATTER_88
---
# Windows note
The office printer HP-M404 jams when using recycled paper heavier than 90 gsm.
""", newline="\r\n")
w("Legacy/With BOM.md", """---
tags: [bom]
secretfm: BOMFRONTMATTER_99
---
# BOM note
The sourdough starter needs feeding at a 1:5:5 ratio every 12 hours at room temperature.
""", bom=True)

# Callouts, tasks, block refs, highlights
w("Areas/Health/Running plan.md", """---
tags: [health]
---
# Running plan

> [!note] Goal
> Run a half marathon under 1:50 in the Ferrara marathon in October.

- [ ] Tuesday intervals 6x800m at 4:30/km ^intervals
- [x] Long run Sunday 18 km
- [ ] Buy new shoes (Pegasus 41)

==Important==: never increase weekly mileage by more than 10%.
""")

# English notes for Italian queries / generic filler
w("Areas/Finance/Taxes.md", """# Taxes
The flat-rate regime (regime forfettario) applies a 15% substitute tax, reduced to 5% for the first five years of a new business. Revenue cap: 85,000 EUR.
""")
w("Areas/Home/Boiler.md", """# Boiler
The boiler pressure should sit between 1 and 1.5 bar when cold. If it drops below 0.8 bar, open the filling loop under the boiler until the gauge reads 1.2.
""")

topics = [
    ("Photography", "Shoot at f/8 for landscapes; use a polarizer to cut reflections on water."),
    ("Bread", "A hydration of 75% gives an open crumb; bake at 250C with steam for 20 minutes."),
    ("Chess openings", "The Sicilian Defence starts 1.e4 c5 and leads to sharp asymmetric positions."),
    ("Bike maintenance", "Lube the chain every 300 km; a stretched chain wears the cassette."),
    ("Birdwatching", "The hoopoe has a crest of orange feathers and a distinctive oop-oop-oop call."),
    ("Aquarium", "Cycle a new tank for four weeks before adding fish so ammonia drops to zero."),
    ("Woodworking", "Cut with the grain; sand to 180 grit before applying Danish oil."),
    ("Guitar", "Drop D tuning lowers the low E string a whole step to D."),
    ("Languages", "Spaced repetition with Anki: review new cards daily, cap at 20 new per day."),
    ("Wine", "Barolo is made from Nebbiolo grapes and must age at least 38 months."),
    ("Camping", "A sleeping pad R-value of 4 or more is needed for winter camping."),
    ("Kubernetes", "A liveness probe failing restarts the container; a readiness probe only removes it from the Service."),
    ("Git tips", "git reflog lets you recover a branch you deleted by mistake."),
    ("Sleep", "Keep the bedroom at 18C and avoid screens an hour before bed."),
]
for i, (t, fact) in enumerate(topics):
    folder = ["Hobbies", "Hobbies/Outdoor", "Tech/Notes", "Areas/Misc"][i % 4]
    w(f"{folder}/{t}.md", f"---\ntags: [{t.lower().replace(' ', '-')}]\n---\n# {t}\n\n{fact}\n\nRelated: [[Books]], [[Running plan|plan]].\n")

# Very long note (~20k words) with a buried heading
rng = random.Random(42)
words = ("harbour tide rope keel mast anchor storm compass lantern gull reef shore current fog sail beacon chart "
         "sailor deck rudder ballast hull pier wave dawn dusk island strait channel buoy cargo merchant").split()
sections = []
for s in range(60):
    title = f"Chapter {s+1}: Coastal survey {s+1}"
    if s == 41:
        title = "Chapter 42: The Lighthouse of Vardø"
        body = ("The lighthouse of Vardø was relit in 1896 with a paraffin lamp visible for 19 nautical miles; "
                "the keeper, Ingrid Halvorsen, logged every ship passing the Bussesundet strait. ") + " ".join(rng.choice(words) for _ in range(300))
    else:
        body = "\n\n".join(" ".join(rng.choice(words) for _ in range(110)) + "." for _ in range(3))
    sections.append(f"## {title}\n\n{body}\n")
w("Reference/Maritime history (long).md", "# Maritime history\n\n" + "\n".join(sections))

# Symlinked folder (outside the vault)
w("Shared recipe.md", "# Shared recipe\nSymlinked note: the pesto needs Genovese basil, Ligurian olive oil and SYMLINKMARKER_42 pine nuts.\n", base=EXT)
os.symlink(EXT, V / "Linked")

print("notes:", sum(1 for _ in V.rglob("*.md")))
