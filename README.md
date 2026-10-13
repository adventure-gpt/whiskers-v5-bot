# Whiskers V5

A Rocket League bot for [RLBot v5](https://rlbot.org): a neural network trained from scratch by self-play
reinforcement learning (PPO). One policy plays 1v1, 2v2 and 3v3; it runs as a hivemind, so one process drives all of
its team's cars. It decides every 8 game ticks from the raw game state and runs on the CPU.

This repository holds the version after **33.3 billion training steps**, with everything needed to run it:
the bot, its observation builder, the policy network and its weights. It contains no training code.

Against the original Nexto in 1v1 this version scored 74 goals to 81 over 40 two-minute games, so it plays at about
Nexto's level in that mode. It has not been measured against Nexto in 2v2 or 3v3.

## Run it

1. Install Python 3.12 and Rocket League with RLBot v5.
2. `python -m venv .venv`
3. Windows: `.venv\Scripts\pip install -r requirements.txt` (Linux: `.venv/bin/pip install -r requirements.txt`)
4. Check the install without the game: `.venv\Scripts\python src\bot.py --selfcheck`
5. In the RLBot launcher, add `src/bot.toml` (or this folder) and put the bot in a match.

Options for `src/bot.py`: `--sample` picks actions at random from the policy's distribution as in training (the
default is its most likely action); `--temperature=T` samples from a sharpened distribution; `--kickoff-script`
uses the scripted speed-flip kickoff instead of the bot's own.

## Bot pack build

The repository is set up for the RLBot v5 bot pack's builder, [bob](https://github.com/swz-git/bob):
`bob.toml` points it at `bot.spec`, a PyInstaller spec that bundles `src/` with the weights and leaves PyTorch out,
because the pack ships one shared copy that `src/bot.py` falls back to. With bob and Docker installed:
`bob build bob.toml`.

The PyInstaller spec has been built and run on its own; the full bob build has not been run yet, so treat it as
untested and open an issue if it fails.

## Layout

| | |
|---|---|
| `src/bot.py` | the RLBot bot: reads packets, decides every 8 ticks, holds the controls in between |
| `src/whiskers/obs.py` | game state -> observation (own car, ball, boost pads, the other cars as a set) |
| `src/whiskers/model.py` | the policy network |
| `src/whiskers/actions.py` | the table of 90 discrete actions |
| `src/whiskers/live.py` | RLBot packet -> the bot's canonical game state |
| `src/whiskers/kickoff.py` | optional scripted kickoff |
| `src/model.pt` | the policy's weights |
| `bob.toml`, `bot.spec` | bot pack build |

## License

MIT, see [LICENSE](LICENSE).
