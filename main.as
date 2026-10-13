// MK Characters: Mario, Yoshi, Peach and Bowser as balls on the Customize page, through Cosmetic Kit (a dependency:
// see info.toml). Each is a model made from scratch by tools/make_models.py: the game's clear ball is hidden and the
// character is the whole ball, kept upright and facing where the ball goes.
//
// Custom cosmetics are local: worn on the player's own ball only, never saved to the profile, never in hiscores or
// leaderboard uploads (the game's public pick is what others see).

import bool AddBall(const string &in, const string &in, const string &in, const string &in, const string &in) from "cosmetic-kit";

const array<string> kKeys = {"mario", "yoshi", "peach", "bowser"};
const array<string> kNames = {"Mario", "Yoshi", "Peach", "Bowser"};
int next = 0;

// One a frame: each model is read and built when it's added, and all four at once could go over the plugin's frame
// budget (Glass Balls does the same).
void Update(float dt)
{
    if (next >= int(kKeys.length())) return;
    string f = Plugins::Folder();
    string k = kKeys[next];
    AddBall("mk-cosmetics." + k, kNames[next], "", f + "previews/" + k + ".png", f + "models/" + k + ".txt");
    next++;
}
