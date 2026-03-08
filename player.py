from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import chess
import random
import requests
import torch
import re
import time
import os

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass

class TransformerPlayer(Player): # new player
    """
    Inherents from Player class.
    Only takes name (required by assignment).
    Fine-tuned/Trained gpt2-medium model on chess data (20,000 games),
    using LoRA (uses the link of my finetuned model).
    Uses forced decoding and mainly relies on the heuristics defined in heuristic_bonus function.
    """
    def __init__(self, name:str):
        """
        Initializes player by loding tokenizer and model from HuggingFace.
        Only takes name as argument, as required by the assignment.
        """
        super().__init__(name)

        # Choose GPU if available, otherwise fall back to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # My trained/fine-tuned transformer model from HF
        self.model_name = "chiaraDG/gpt2-medium-chess-lora"

        # Load GPT-2 tokenizer (converts text to token IDs)
        # was extended with all 64 board squares (atomic tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Reuse end-of-sequence token as padding for batched tokenization
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load GPT-2 model for causal language modeling, move to correct device
        self.model = AutoModelForCausalLM.from_pretrained(
          self.model_name).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Warmup call: forces model weights to be fully loaded into memory
        # during instantiation rather than during first get_move() call.
        # Prevents HF download eating into any possible per-move time budget 
        # during championship.
        dummy_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.get_move(dummy_fen)

    def heuristic_bonus(self, board, move):
        """
        Chess heuristics to improve quality.
        Move is temporarily pushed to evaluate resulting position, then popped
        to restore original state.
        
        board: chess.Board object representing the current position
        move: chess.Move object representing the candidate move

        Returns the computed heuristic bonus.
        """
        bonus = 0 # initilize bonus score

        piece_values = { # basic piece value system
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0 # not counted in material
        }

        # Store turn before making the move
        original_turn = board.turn

        # Center squares
        center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
        # The 12 squares surrounding the center
        extended_center = {chess.C3, chess.D3, chess.E3, chess.F3,
                       chess.C4, chess.F4, chess.C5, chess.F5,
                       chess.C6, chess.D6, chess.E6, chess.F6}

        # Capture bonus
        if board.is_capture(move): # check if move captures a piece
          captured_piece = board.piece_at(move.to_square) # get captured piece
          if captured_piece is not None: # safety check
            bonus += piece_values[captured_piece.piece_type] * 2.0
            # makes captures preferred over passive moves
        
        # Moving piece
        moving_piece = board.piece_at(move.from_square)
        
        # Center control bonus
        # reward moves that place piece on or near center of the board
        if move.to_square in center_squares:
            bonus += 0.8
        elif move.to_square in extended_center:
            bonus += 0.3

        # Piece development bonus
        # in opening, reward moving knights and bishops off back
        if board.fullmove_number <= 15:
            if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                back_rank = chess.BB_RANK_1 if original_turn == chess.WHITE else chess.BB_RANK_8
                if chess.BB_SQUARES[move.from_square] & back_rank:
                    bonus += 0.6

        # Penalty for moving queen out too early (first 5 moves)
        if board.fullmove_number <= 5:
            if moving_piece and moving_piece.piece_type == chess.QUEEN:
                bonus -= 0.5
        
        # Bonus for advancing passed pawns in winning positions
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            if original_turn == chess.WHITE:
                rank = chess.square_rank(move.to_square)
            else:
                rank = 7 - chess.square_rank(move.to_square) # flip for black
            bonus += rank * 0.1 # more bonus if further advanced

            # Extra bonus for pawn moves in opening
            if board.fullmove_number <= 5:
                bonus += 1.0

        moving_val = piece_values.get(moving_piece.piece_type, 0) if moving_piece else 0

        # Check if square moving to is attacked by opponent
        if board.is_attacked_by(not original_turn, move.to_square):
            captured_piece = board.piece_at(move.to_square)
            captured_val = piece_values.get(captured_piece.piece_type, 0) if (board.is_capture(move) and captured_piece) else 0
            
            # If we are moving a valuable piece to a defended square for a cheap capture 
            if moving_val > captured_val + 1:
                # Heavy penalty for hanging a piece!
                bonus -= (moving_val - captured_val) * 2.0


        board.push(move) # push move to evalute board after this move


        # Penalty for repeating a position
        if board.is_repetition(2): # repeated at least 2 times
            bonus -= 50.0

        # Check bonus
        if board.is_check(): # check if move gives check
          bonus += 0.5
          

        # # Checkmate bonus
        # if board.is_checkmate(): # check if move results in checkmate (ends game)
        #   bonus += 100 # forces selection
        
        # Stalemate penalty
        if board.is_stalemate():
            material = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == original_turn:
                        material += value
                    else:
                        material -= value
            if material > 0:
                bonus -= 1000 # never stalemate if ahead on material

        # Fleeing / Hanging piece detection
        # check if any pieces are left undefended and attacked by opponent
        # apply penalty proportional to its value
        for square, piece in board.piece_map().items():
            if piece.color == original_turn:
                if board.is_attacked_by(not original_turn, square):
                    val = piece_values.get(piece.piece_type, 0)
                    if not board.is_attacked_by(original_turn, square):
                        bonus -= val * 2.0 # undefended, heavily penalize

        # Bonus for king activity in endgame: should move towards center
        total_pieces = len(board.piece_map())
        if total_pieces < 10: # endgame
            king_square = board.king(original_turn)
            if king_square:
                king_rank = chess.square_rank(king_square)
                king_file = chess.square_file(king_square)
                center_dist = abs(king_rank - 3.5) + abs(king_file - 3.5)
                bonus -= center_dist * 0.1


        # Material evaluation
        material = 0 # initialize material difference

        for square in chess.SQUARES: # loop over all board squares
          piece = board.piece_at(square) # get piece on square
          if piece: # square not empty
            value = piece_values[piece.piece_type] # get piece value
            # evaluating from perspective of player who made the move:
            if piece.color == original_turn: 
              material += value
            else:
              material -= value
        bonus += material * 0.6 # weight material contribution

        board.pop() # undo temporary move to restore original board position

        return bonus # computed bonus score
      
    def get_move(self, fen: str) -> Optional[str]:
        """
        Selects best move for current board position given as a FEN string.
        - Checkmate scan: loop through legal moves, return immediately if any move
        results in checkmate (no scoring needed).
        - Scoring: for all remaining moves, compute combined scrore from
        (a) model's log-probability for move given board prompt
        (b) heuristic bonus encoding chess principles

        Returns a UCI move string or None (if no legal move exists).
        """

        # Create board from FEN string
        board = chess.Board(fen)
        # Get all legal moves in current position
        legal_moves = list(board.legal_moves)

        if not legal_moves: # checkmate or stalemate
          return None
        
        # Safe moves filter
        for legal_move in legal_moves:
            board.push(legal_move)
            
            # Do not miss an instant win
            if board.is_checkmate():
                board.pop()
                return legal_move.uci() # immediate win
            board.pop()

        ## Prompt ##
        # Prompt for GPT-2  
        prompt = f"Board FEN: {fen}\nBest move:" # matches prompt during training

        inputs = self.tokenizer( # tokenize prompt
          prompt,
          return_tensors="pt"
        ).to(self.device) # move tensors to device (gpu/cpu)

        ## Prepare batched move inputs ##
        # Create list of move strings
        move_texts = [" " + move.uci() for move in legal_moves]

        # Tokenize candidate moves (single batch)
        move_inputs = self.tokenizer(
          move_texts,
          return_tensors ="pt",
          padding = True # pad shorter moves, to match longest in batch
        ).to(self.device) # move to device

        ## Repeat prompt for each move ## (single forward pass)
        num_moves = len(legal_moves)
        prompt_len = inputs["input_ids"].shape[1]

        # Duplicate prompt tokens so each move has same context
        prompt_ids = inputs["input_ids"].repeat(num_moves, 1)
        # Duplicate attention mask
        attention_mask = inputs["attention_mask"].repeat(num_moves, 1)

        # Concatenate prompt and moves along dimension 1 (sequence length)
        full_input_ids = torch.cat([prompt_ids, move_inputs["input_ids"]], dim=1)
        full_attention_mask = torch.cat([attention_mask, move_inputs["attention_mask"]], dim=1)

        ## Forward pass (batched) ##
        with torch.no_grad(): # disable gradients for inference
          outputs = self.model(
            input_ids = full_input_ids,# prompt token as context
            attention_mask = full_attention_mask
          )
          logits = outputs.logits
        
        # Score each move
        scores = [] # store final scores

        for i, move in enumerate(legal_moves): # loop over moves
          move_ids = move_inputs["input_ids"][i] # get token ids for this move
          move_mask = move_inputs["attention_mask"][i]

          # Find actual length of move (ignoring the padding tokens)
          actual_move_len = move_mask.sum().item()

          # Slice logits that predict move tokens
          move_logits = logits[i, prompt_len - 1 : prompt_len - 1 + actual_move_len, :]
          # Convert logits to log-probabilities
          log_probs = torch.log_softmax(move_logits, dim=-1)

          # Get log-probability of actual move tokens
          token_log_probs = log_probs.gather( # select log prob of each token in move
            1,
            move_ids[:actual_move_len].unsqueeze(1) # make 1 column for gather
          ).squeeze(-1) # flatten back to 1D to sum log probs

          # Mean log-probs prevents penalizing longer token squences
          model_score = token_log_probs.mean().item()

          # Heuristic bonus
          heuristic_score = self.heuristic_bonus(board, move)

          # Combine both scores
          total_score = (model_score * 0.1) + heuristic_score

          scores.append(total_score)

        ## Select best move ##
        # Find index of highest scoring move
        best_index = int(torch.tensor(scores).argmax())
        # Convert to UCI string
        best_move = legal_moves[best_index].uci()

        # Return best legal move
        return best_move
        




        
            
        





class RandomPlayer(Player):
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None


class EnginePlayer(Player):
    """
    EnginePlayer now behaves like ANY Player:
    Input: FEN
    Output: move string (UCI) | "__NO_MOVES__" | None

    Internal failures are NOT visible to Game.
    """

    def __init__(
        self,
        name: str,
        blunder_rate: float = 0.0,
        ponder_rate: float = 0.0,
        base_delay: float = 0.9,
        enable_cache: bool = True,
    ):
        super().__init__(name)

        assert 0.0 <= blunder_rate <= 1.0
        assert 0.0 <= ponder_rate <= 1.0
        assert blunder_rate + ponder_rate <= 1.0

        self.blunder_rate = blunder_rate
        self.ponder_rate = ponder_rate
        self.base_delay = base_delay
        self.enable_cache = enable_cache

        self.api_key = os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY must be set")

        self.url = "https://chess-stockfish-16-api.p.rapidapi.com/chess/api"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "chess-stockfish-16-api.p.rapidapi.com",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        self.cache: Dict[str, Tuple[str, Optional[str]]] = {}

    def _sleep(self):
        time.sleep(self.base_delay)

    def _random_legal_from_fen(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
        except Exception:
            return None
        legal = list(board.legal_moves)
        if not legal:
            return None
        return random.choice(legal).uci()

    def _choose_move(self, best: str, ponder: Optional[str], fen: str) -> str:
        r = random.random()

        if r < self.blunder_rate:
            rm = self._random_legal_from_fen(fen)
            return rm if rm else best

        if r < self.blunder_rate + self.ponder_rate:
            return ponder if ponder else best

        return best

    def get_move(self, fen: str) -> Optional[str]:

        # CACHE
        if self.enable_cache and fen in self.cache:
            best, ponder = self.cache[fen]
            return self._choose_move(best, ponder, fen)

        self._sleep()

        try:
            r = requests.post(self.url, data={"fen": fen}, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None

            j = r.json()

        except Exception:
            return None

        # Engine says no moves
        result_field = j.get("result")
        if isinstance(result_field, str) and "bestmove (none)" in result_field.lower():

            rm = self._random_legal_from_fen(fen)
            if rm is None:
                return "__NO_MOVES__"

            return rm  # Game will treat as normal move

        best = j.get("bestmove")
        ponder = j.get("ponder")

        if not best:
            return None

        if self.enable_cache:
            self.cache[fen] = (best, ponder if ponder else None)

        return self._choose_move(best, ponder if ponder else None, fen)

class LMPlayer(Player):
    def __init__(
        self,
        name: str,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        quantization: Optional[str] = "4bit",
        temperature: float = 0.1,
        max_new_tokens: int = 6,
        retries: int = 5
    ):
        super().__init__(name)

        self.model_id = model_id
        self.quantization = quantization
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading {self.model_id} on {self.device}")
        print(f"[{self.name}] Quantization mode: {self.quantization}")

        # -------------------------
        # Quantization config
        # -------------------------
        quant_config = None

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        elif quantization is None:
            quant_config = None

        else:
            raise ValueError("quantization must be one of: None, '8bit', '4bit'")

        # -------------------------
        # Tokenizer
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -------------------------
        # Config
        # -------------------------
        config = AutoConfig.from_pretrained(model_id)
        config.pad_token_id = self.tokenizer.pad_token_id

        # -------------------------
        # Model loading
        # -------------------------
        if quant_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=dtype,
                device_map="auto"
            )

        # -------------------------
        # UCI regex
        # -------------------------
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.uci_re.search(text)
        return match.group(0) if match else None

    def get_move(self, fen: str) -> Optional[str]:
        prompt = self._build_prompt(fen)

        for attempt in range(1, self.retries + 1):

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
                return move

        return None

class SmolPlayer(Player):
    """
    LLMAPIPlayer using InferenceClient.chat_completion()
    Compatible with chat/instruct models.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str,
        model_id: str = 'moonshotai/Kimi-K2-Instruct',
        temperature: float = 0.2,
        max_tokens: int = 32,
    ):
        super().__init__(name)

        self.client = InferenceClient(
            model=model_id,
            token=os.environ.get("HF_TOKEN")
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_uci(self, text: str):
        if not text:
            return None

        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def get_move(self, fen: str):

        prompt = self._build_prompt(fen)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response.choices[0].message.content

            return self._extract_uci(text)

        except Exception as e:
            # Optional debug:
            print(f"[{self.name}] API error:", e)
            return None
          
