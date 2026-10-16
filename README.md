Proof of Concept for Model A: Poker Baseline Policy

Python: 3.12
Game: heads-up (2 player) no-limit Texas Hold’em.
    Will change to 2-9 player once proof of concept is complete.
Starting stack: 200 chips per player (100 BBs)
Blinds: 1/2 chips
One complete hand per episode
Reward: net chip profit divided by 2
Algorithm: MaskablePPO