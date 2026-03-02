# Chess-Playing Transformer Agent

A GPT-2-medium model fine-tuned and trained on Lichess games, submitted as part of the INFOMTALC 2025/26 midterm assignment at Utrecht University.

## Overview

`TransformerPlayer` is a chess agent that combines a fine-tuned language model with chess heuristics to select moves from any given board position in FEN format. The model scores all legal moves by log-probability and combines this with a heuristic bonus to select the best move.

## Model

- **Base model:** `openai-community/gpt2-medium` (345M parameters)
- **Fine-tuned model:** [`chiaraDG/gpt2-medium-chess-lora`](https://huggingface.co/chiaraDG/gpt2-medium-chess-lora)
- **Fine-tuning method:** LoRA (r=32, alpha=64), targeting the `c_attn` attention module
- **Training data:** [`Lichess/standard-chess-games`](https://huggingface.co/datasets/Lichess/standard-chess-games), filtered to ELO > 2000, 20,000 games
- **Training steps:** 10,000 — final loss: ~1.33
- **Custom tokens:** All 64 board squares (a1–h8) added as atomic tokens

## Approach

### Move Scoring
For each legal move, the model computes a log-probability score by concatenating the board prompt with the candidate move and extracting the summed log-probabilities of the move tokens:

```
prompt = "Board FEN: <fen>\nBest move:"
score = sum of log P(move tokens | prompt)
```

This is done in a single batched forward pass, for efficiency.

### Heuristics
A heuristic bonus is added to each move score to guide the player toward good chess principles:

- **Capture bonus:** rewards taking opponent pieces weighted by piece value (x2).
- **Center control:** bonuses for moves to e4/d4/e5/d5 and extended center squares.
- **Piece development:** rewards moving knights and bishops off the back rank in the opening.
- **Opening penalties:** penalizes bringing the queen out before move 5.
- **Pawn advancement:** bonus for advancing pawns, scaled by rank, extra +1.0 bonus in first 5 moves to encourage early pawn play.
- **Blunder prevention:** penalizes moving a valuable piece to an attacked square for a cheap or no capture.
- **Hanging piece detection:** afer each candidate move, scans remaining pieces to find any that are attacked by opponent but not defended. Catches situation where move indirectly creates vulnerability.
- **Repetition penalty:** strong penalty (-50) for moves that repeat a position: the model was struggling with shuffling pieces back and forth, which is why this repetition penalty was upped from -20 to -50, to break that loop.
- **Stalemate avoidance:** massive penalty (-1000) for stalemating when ahead on material.
- **Endgame king activity:** in endgame (less than 10 pieces on board), king becomes an active fighting pieces (instead of one needing protection). Centralized king control more squares and can support pawn advancement.
- **Material evaluation:** overall material balance weighted at 0.6.

### Checkmate Detection
Before scoring, `get_move()` scans all legal moves for an immediate checkmate. If found, it is returned instantly without consulting the model.

## Design Decisions
#### **Why GPT-2 medium?**

I first used GPT-2, which produced nearly identical log-probability scores across all legal moves, which  means the model contributed no useful signal. Upgrading to GPT-2-medium reduced the final training loss from ~1.85 to ~1.33 and produced more distinct move scores, which gives the heuristics better signal to work with.

#### **Why LoRA instead of full fine-tuning?**

LoRA freezes the original model weights and only trains small adapter matrices, which is more memory and time efficient. With rank `r=32` and `alpha=64`, the model learns chess-specific patterns without forgetting its language understanding. The training took about 35 minutes on T4 GPU.

#### **Why add chess square tokens**

By default GPT-2's tokenizer splits squares like `e4` into `e` and `4` as separate tokens. Adding all 64 squares as atomic tokens ensures that the model always sees a square as a single unit, which makes move representations more consistent and easier to learn from.

#### **Why combine model scores with heuristics?**

Even after fine-tuning, the 345M parameter model had limited chess understanding, especially comparing to dedicated chess engines. The heuristics provide a reliable signal for common chess principles like captures, development and avoiding blunders, while the model contributes learned patterns from high ELO games. In practice, the heuristics dominate move selection with the model score scaled to 0.1 to prevent the model noise from overriding important heuristic signals.

#### **Filtering on ELO > 2000**

Training on games from stronger players means the model learns from higher quality chess games. Games from weaker players contain more random or suboptimal moves, which would add noise.

#### **Why forced decoding?**

All legal moves are scored by log-probability rather than asking the model to freely generate a move, which might produce illegal moves. This guarantees that the output is always a legal move, without requiring any post-processing or retries.

#### **Why scale down model score?**
The raw model log-probability scores were in the range of -20 to -22, 
while heuristic bonuses are in the range of -10 to +10. Without scaling, the 
model contribution completely drowns out the heuristics with the consequence that a move the model slightly prefers will always win over a move the heuristics strongly recommend, even if the heuristic difference reflects something important like avoiding a big blunder. By scaling the model score down by a factor of 0.1, the model contributes roughly 2 units of difference between moves while the heuristics contribute 5-10, allowing the chess principles to dominate in the move selection while the model still provides a meaningful tiebreaker based on patterns learned from the high-ELO games from the Lichess dataset.

## Repository Structure

```
player.py           # TransformerPlayer class + other player classes
requirements.txt    # Dependencies
LICENSE             # MIT license
README.md           # this README file
finetune_train_GPT2_Medium.ipynb    # Notebook used to finetune GPT2-medium model
```

## Requirements (`requirements.txt`)

```
python-chess
requests
torch
transformers
bitsandbytes
huggingface-hub
accelerate
peft
```

## Usage

```python
from player import TransformerPlayer

player = TransformerPlayer("MyPlayer")
move = player.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(move)  # e.g. "e2e4"
```

The player loads the fine-tuned model from HuggingFace automatically. A GPU is used if available, otherwise CPU.
