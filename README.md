# MK Characters (private)

A private plugin for the Ballest plugin manager: Mario, Yoshi, Peach and Bowser balls in their Mario Kart Wii
colours, added to the **Customize** page through Cosmetic Kit. **Never publish this or add it to the registry.**

Everything is made from scratch by `tools/make_models.py` (shapes built in NumPy, written as glTF). No game models,
textures or downloaded models are used.

| Ball | Look | Triangles |
|---|---|---|
| Mario | the ball is his head: red cap with the white circle and red M, brim, blue eyes, brows, big nose, bushy moustache, ears, sideburns, brown hair under the cap, red collar, blue overalls with yellow buttons | 7,862 |
| Yoshi | the ball is his body: white belly, red saddle with a white rim, orange spikes down his back, a tail, orange shoes; his head on top with the big snout, nostrils, white chin, eyes on top and head spikes | 6,526 |
| Peach | the ball is her head: gold crown with points, a ruby and four sapphires, blonde hair with side-swept bangs, locks down the back and flicked ends, big blue eyes with lashes, brows, rosy cheeks, pink lips, sapphire earrings, pink dress collar with a blue brooch | 9,536 |
| Bowser | the ball is his body: green shell with a cream rim and white spikes on cream rings, cream belly with grooves, arms with black spiked cuffs and claws, feet with claws; his head on top with horns, red mane and angry brows, red eyes, snout, fangs | 10,172 |

Each one hides the game's own clear ball (`ball hidden`) and is the whole ball, in a `travel` group: it stays upright
and faces where the ball goes (the camera sees the character from behind, like in Mario Kart; on the Customize page it
faces the camera). The head or body sphere is the ball's size (radius 50), so it matches the ball the game rolls.

These are **local** cosmetics, like all custom ones: worn on the player's own ball only, never saved to the profile,
never sent to hiscores or leaderboards.

## Use

Customize > **local** > balls > the **custom** section at the bottom > Mario, Yoshi, Peach or Bowser.
Needs the plugin manager host 0.23.2 or newer and Cosmetic Kit.

## Files

- `main.as`: adds the four balls (one a frame).
- `models/<name>.glb` and `models/<name>.txt`: each character and its model file.
- `previews/<name>.png`, `icon.png`: tile pictures.
- `tools/meshkit.py`: the shapes, the glTF writer and the tile renderer. `tools/make_models.py`: the characters.
  `python tools/make_models.py [mario|yoshi|peach|bowser]` rebuilds them.
