# Gold spot-review sample

Tick a case only if **all three** hold: the query is answerable, the span answers it, and no *other* passage in the corpus answers it just as well.

The third is not a stylistic preference. `answer_spans` is a fixed list, and a retrieved chunk scores as a hit only if it sits in the labelled note and contains the labelled span. So if some other passage also answers the query, a retriever that ranks it first is *correct* and scored as a **miss** -- the case would punish the behaviour we want. Rejecting those keeps the metric honest.

## buried  (40 cases, showing 5: 5 en)

- [x] **What is the temperature range for ripening blue cheese?**  `en`
  - note: `cheese/blue-cheese.md`
  - span: 8-10 degrees Celsius

- [x] **What material can be used for the chessboard in FIDE tournaments other than championships?**  `en`
  - note: `board-games/chessboard.md`
  - span: wood, plastic, or cardboard boards

- [x] **What plant species is also known as Paneer Booti, Ashwagandha and the Indian Cheesemaker?**  `en`
  - note: `cheese/rennet.md`
  - span: *Withania coagulans* (also known as Paneer Booti, Ashwagandh and the Indian Cheesemaker)

- [x] **What keys do beginning violin students often start with?**  `en`
  - note: `string-instruments/violin.md`
  - span: A Major and G major

- [x] **What is the name of the coffee made from beans excreted by the Asian palm civet?**  `en`
  - note: `coffee/coffee-bean.md`
  - span: These beans are called *kopi luwak*, and are sold as a rare coffee at a high price

## cross_lingual  (46 cases, showing 4: 2 es, 2 it)

- [x] **¿Cuándo se cree que los mineros del período Hallstatt consumieron queso azul y cerveza?**  `es`
  - note: `cheese/blue-cheese.md`
  - span: miners of the Hallstatt Period (800 to 400 BC) already consumed blue cheese and beer

- [x] **¿Cómo se operaba el freno en las bicicletas de la época del boneshaker?**  `es`
  - note: `cycling/bicycle-brake.md`
  - span: by a lever or by a cord connecting to the handlebars

- [x] **Quali sono i due principali tipi di freni per biciclette?**  `it`
  - note: `cycling/bicycle-brake.md`
  - span: rim brakes and disc brakes

- [x] **Qual è il nome dato al brie che viene lasciato maturare per mesi o anni?**  `it`
  - note: `cheese/brie.md`
  - span: Brie noir ('black brie')

## exact_term  (42 cases, showing 5: 5 en)

- [x] **Who introduced the front brake on penny-farthings?**  `en`
  - note: `cycling/bicycle-brake.md`
  - span: John Kean in 1873

- [x] **What happens to a piece after it is jumped in international draughts?**  `en`
  - note: `board-games/checkers.md`
  - span: jumped pieces remain on the board until the turn is completed

- [ ] **What material are modern violin strings trying to combine the sound quality of?**  `en`  <- REJECTED: spot-review: English query over a Spanish span shares no lexical term with it, so it cannot test lexical matching; the English violin article is also a near-competitor
  - note: `string-instruments/violin-es.md`
  - span: tripa y la resistencia de los metales

- [x] **What is marked on the board to indicate where cannons start?**  `en`
  - note: `board-games/xiangqi.md`
  - span: starting points of the soldiers and cannons are usually, but not always, marked with small crosses

- [x] **What is Coffea arabica classified as in southeast Queensland due to its invasiveness?**  `en`
  - note: `coffee/coffea-arabica.md`
  - span: an environmental weed

## paraphrase  (41 cases, showing 5: 5 en)

- [x] **Are racing bikes permitted on public roads?**  `en`
  - note: `cycling/racing-bicycle.md`
  - span: Racing bicycles are generally legal for use on public roads

- [x] **What limitation did traditional main sails face before Herreshoff's innovation?**  `en`
  - note: `sailing/mainsail.md`
  - span: mainsails were limited in height

- [x] **What maneuver did Peter Claydon pioneer in narrow waterways that later proved beneficial in open seas?**  `en`
  - note: `sailing/tacking-sailing.md`
  - span: roll tacking technique he developed on the narrow river gave a distinct advantage in open water too

- [ ] **how do chess pieces take enemy units?**  `en`  <- REJECTED: spot-review: chess.md states twice that a piece captures by moving onto the enemy's square, answering the query as well as the labelled span
  - note: `board-games/chess-piece.md`
  - span: Pieces other than pawns capture in the same way that they move.

- [x] **When did steam-powered milk frothing in coffee drinks become feasible?**  `en`
  - note: `coffee/latte.md`
  - span: in 1903, which made it possible to add heat and texture to milk added to coffee
