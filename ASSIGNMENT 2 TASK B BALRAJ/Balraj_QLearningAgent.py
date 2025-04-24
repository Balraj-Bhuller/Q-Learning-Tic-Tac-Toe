# 06/04/2025: Creating a TIC TAC TOE interface so 2 people can play against each other
# 12/04/2025: Creating a Random Number Generator AI and ROck Paper Scissors mini Gmae to decide first player
# 13/04/2025: Creating a QLearningAgent Class to play against the Player
# 13/04/2025: Restarting over with the knowledge I've ganed so far and make program more readable and work
# 15/04/2025: Creating GUI for TIC TAC TOE and using knowledge I gained from previous program + combing it with other programs to create wokring QlearningAgent
# 21/04/2025: Creating Value Agent to be able to compare it to QLearning Agent Later on
# 24/04/2025: Tried to create a Symmetric QLearning Agent but it didn't work

import tkinter as tk
import numpy as np
import copy
import pickle
import os

# --- Player Base Classes ---
class Player:
    def __init__(self, turn):
        self.turn = turn

    @property
    def opponentTurn(self):
        # Simple helper to get the other player's symbol
        return 'O' if self.turn == 'X' else 'X'


class User(Player):
    # Human player (will use GUI)
    pass


class AI(Player):
    # Base class for all AI players
    pass


class RandomAI(AI):
    # Picks a random legal move
    @staticmethod
    def getMove(board):
        moves = board.availableMoves()
        return moves[np.random.choice(len(moves))] if moves else None


class RuleBasedAI(AI):
    # Tries to win or block, otherwise picks randomly
    def getMove(self, board):
        for move in board.availableMoves():
            if self.winningMove(board, move, self.turn) or self.winningMove(board, move, self.opponentTurn):
                return move
            
        return RandomAI.getMove(board)

    @staticmethod
    def winningMove(board, move, turn):
        # Returns True if this move immediately wins for 'turn'
        return board.getNextState(move, turn).winner() == turn


class VAgent(AI):
    # Value-based learning agent
    def __init__(self, turn, Value = None, epsilon = 0.2):
        super().__init__(turn)
        self.Value = Value if Value is not None else {}
        self.epsilon = epsilon

    def getMove(self, board):
        # Epsilon-greedy move selection
        if np.random.uniform() < self.epsilon:
            return RandomAI.getMove(board)
        
        key = self.addKey(board, self.turn, self.Value)
        
        return self.selectMove(board, self.turn, self.Value)

    @staticmethod
    def addKey(board, turn, Value):
        # Ensure the current board state is in the Value table
        key = board.makeKey(turn)
        if key not in Value:
            Value[key] = 0.5  # neutral value for unexplored states
        return key

    @staticmethod
    def selectMove(board, turn, Value):
        # Choose the move with the best value (greedy), breaking ties randomly
        bestValue = -float('inf') if turn == 'X' else float('inf')
        bestMoves = []
        
        for move in board.availableMoves():
            nextBoard = board.getNextState(move, turn)
            
            if nextBoard.gameFinished():
                val = nextBoard.getReward()
                
            else:
                # Look up value from opponent's perspective
                key = nextBoard.makeKey('O' if turn == 'X' else 'X')
                val = Value.get(key, 0.5)
                
            if (turn == 'X' and val > bestValue) or (turn == 'O' and val < bestValue):
                bestValue = val
                bestMoves = [move]
                
            elif val == bestValue:
                bestMoves.append(move)
                
        return bestMoves[np.random.choice(len(bestMoves))] if bestMoves else None


class QAgent(AI):
    # Q-Learning agent  
    def __init__(self, turn, qTable = None, epsilon = 0.2):
        super().__init__(turn)
        self.qTable = qTable if qTable is not None else {}
        self.epsilon = epsilon

    def getMove(self, board):
        # Epsilon-greedy move selection
        if np.random.uniform() < self.epsilon:
            return RandomAI.getMove(board)
        key = self.addKey(board, self.turn, self.qTable)
        Q = self.qTable[key]
        # X tries to maximize, O tries to minimize (greedy move)
        return self.selectMove(Q, max if self.turn == 'X' else min)

    @staticmethod
    def addKey(board, turn, Q):
        # Ensure every state is in the Q-table
        key = board.makeKey(turn)
        if key not in Q:
            Q[key] = {move: 1.0 for move in board.availableMoves()}
            
        return key

    @staticmethod
    def selectMove(Q, func):
        # Choose move with best (or worst) Q-value, breaking ties randomly
        target_value = func(Q.values())
        best_moves = [move for move, val in Q.items() if val == target_value]
        return best_moves[np.random.choice(len(best_moves))]


# --- Board Class ---
class Board:
    def __init__(self, grid = None):
        # grid is a 3x3 numpy array of floats or None for empty
        self.grid = np.full((3, 3), np.nan) if grid is None else grid

    def winner(self):
        # Checks all lines for a win
        lines = [self.grid[r, :] for r in range(3)] + \
                [self.grid[:, c] for c in range(3)] + \
                [np.diag(self.grid), np.diag(np.fliplr(self.grid))]
        for line in lines:
            if np.all(line == 1): return 'X'
            if np.all(line == 0): return 'O'
        return None

    def gameFinished(self):
        # Game is finished if there's a winner or no empty cells
        return self.winner() is not None or not np.any(np.isnan(self.grid))

    def placeTurn(self, move, turn):
        # Places 'X' (1) or 'O' (0) at the specified cell
        self.grid[move] = 1 if turn == 'X' else 0

    def availableMoves(self):
        # Returns all empty spots as (row, col) tuples
        return [(r, c) for r in range(3) for c in range(3) if np.isnan(self.grid[r][c])]

    def getNextState(self, move, turn):
        # Returns a deep copy of the board after a move
        new_board = copy.deepcopy(self)
        new_board.placeTurn(move, turn)
        return new_board

    def makeKey(self, turn):
        # Key for the current state (used in Q/V tables, no symmetry)
        filled = np.nan_to_num(self.grid, nan = 9).astype(int).flatten()
        return ''.join(map(str, filled)) + turn

    def getReward(self):
        # Returns +1 for X win, -1 for O win, 0 for draw/unfinished
        winner = self.winner()
        if winner == 'X': return 1.0
        if winner == 'O': return -1.0
        if not np.any(np.isnan(self.grid)): return 0.0
        return 0.0


# --- Game GUI and Q-Learning Trainer ---
class LearningAgentAndGame:
    def __init__(self, root = None, P1 = None, P2 = None, QLearn = None, qTable = None, VLearn = False, Value = None, alpha = 0.3, gamma = 0.9):
        self.root = root
        self.P1 = P1
        self.P2 = P2
        self.currentPlayer = P1
        self.opponent = P2
        self.board = Board()
        self.qTable = qTable if qTable is not None else {}
        self._VLearn = VLearn
        self.Value = Value if Value is not None else {}
        self.alpha = alpha
        self.gamma = gamma
        self._QLearn = QLearn if QLearn is not None else isinstance(P1, QAgent) or isinstance(P2, QAgent)
        self.syncQTable()

        if self.root:
            # Set up GUI buttons for the board
            self.buttons = [[None for _ in range(3)] for _ in range(3)]
            frame = tk.Frame(root)
            frame.pack()
            for r in range(3):
                for c in range(3):
                    self.buttons[r][c] = tk.Button(frame, width = 10, height = 4, command = lambda r = r, c = c: self.onClick(r, c))
                    self.buttons[r][c].grid(row = r, column = c)
                    
            tk.Button(root, text = "Reset", command = self.reset).pack()


    def syncQTable(self):
        # Ensure both agents share the same Q-table (for learning)
        if isinstance(self.P1, QAgent): self.P1.qTable = self.qTable
        
        if isinstance(self.P2, QAgent): self.P2.qTable = self.qTable

    def onClick(self, r, c):
        # Handles user clicks in the GUI
        if self.board.gameFinished(): return
        
        if isinstance(self.currentPlayer, User) and np.isnan(self.board.grid[r][c]):
            self.makeMove((r, c))
            self.root.after(200, self.aiPlay)

    def aiPlay(self):
        # If it's the AI's turn, keep playing until the game is over or human's turn
        while not self.board.gameFinished() and isinstance(self.currentPlayer, AI):
            move = self.currentPlayer.getMove(self.board)
            self.makeMove(move)

    def makeMove(self, move):
        # Main move handler: updates board and switches player
        if self._QLearn:
            self.updateQ(move)
            
        elif self._VLearn:
            self.updateV(move)
            
        self.board.placeTurn(move, self.currentPlayer.turn)
        
        if self.root:
            r, c = move
            self.buttons[r][c]['text'] = self.currentPlayer.turn
            
        if self.board.gameFinished():
            self.announceWinner()
            
        else:
            self.switchPlayer()

    def switchPlayer(self):
        # Alternate between P1 and P2
        self.currentPlayer, self.opponent = self.opponent, self.currentPlayer

    def announceWinner(self):
        # Prints the winner to the console (could also use a GUI popup)
        if self.root:
            winner = self.board.winner()
            result = f"Winner: {winner}" if winner else "Draw"
            print(result)

    def reset(self):
        # Resets the board for a new game
        self.board = Board()
        self.currentPlayer = self.P1
        self.opponent = self.P2
        if self.root:
            for r in range(3):
                for c in range(3):
                    self.buttons[r][c]['text'] = ''

    def play(self):
        # For automated (self-play) training: let the AI play until the game ends
        if isinstance(self.P1, AI):
            move = self.P1.getMove(self.board)
            self.makeMove(move)
            
        while not self.board.gameFinished() and isinstance(self.currentPlayer, AI):
            move = self.currentPlayer.getMove(self.board)
            self.makeMove(move)

    def updateQ(self, move):
        # Standard Q-Learning update for two-player games
        key = QAgent.addKey(self.board, self.currentPlayer.turn, self.qTable)
        nextBoard = self.board.getNextState(move, self.currentPlayer.turn)
        reward = nextBoard.getReward()
        nextKey = QAgent.addKey(nextBoard, self.opponent.turn, self.qTable)

        # Here's the important part: use the opponent's turn for min/max!
        future = 0.0
        if nextBoard.availableMoves():
            nextQ = self.qTable[nextKey]
            # O will try to minimize X's value, X will try to maximize (from their next turn)
            if self.opponent.turn == 'X':
                future = max(nextQ.values())
                
            else:
                future = min(nextQ.values())

        # Q-learning update rule
        self.qTable[key][move] += self.alpha * ((reward + self.gamma * future) - self.qTable[key][move])

    def updateV(self, move):
        # Value iteration update for two-player games
        key = VAgent.addKey(self.board, self.currentPlayer.turn, self.Value)
        nextBoard = self.board.getNextState(move, self.currentPlayer.turn)
        reward = nextBoard.getReward()

        if nextBoard.gameFinished():
            future = 0.0
            
        else:
            # As in Q-learning, the opponent will try to minimize/maximize our value
            futureValues = []
            for opponentMove in nextBoard.availableMoves():
                afterBoard = nextBoard.getNextState(opponentMove, self.opponent.turn)
                afterKey = VAgent.addKey(afterBoard, self.currentPlayer.turn, self.Value)
                futureValues.append(self.Value[afterKey])
                
            if self.currentPlayer.turn == 'X':
                future = min(futureValues) if futureValues else 0.0
                
            else:
                future = max(futureValues) if futureValues else 0.0

        currentValue = self.Value[key]
        targetValue = reward + self.gamma * future
        self.Value[key] += self.alpha * (targetValue - currentValue)

# --- Mode Select: Train or Play ---
mode = input("Choose mode: \n(1) Train Q-Learning Agent \n(2) Play Q-Learning Agent \n(3) Train Value Agent \n(4) Play Value Agent \nChoice: ").strip()

if mode == "1":
    # Self-play training for Q-learning agent
    print("Training Q-Agent...")
    epsilon = 0.9
    episodes = 100000
    P1 = QAgent(turn = "X", epsilon = epsilon)
    P2 = QAgent(turn = "O", epsilon = epsilon)
    game = LearningAgentAndGame(P1 = P1, P2 = P2)
    for i in range(episodes):
        if i % 10000 == 0:
            print(f"Episode: {i}")
        game.play()
        game.reset()
        
    filename = "trained_qtable.p"
    pickle.dump(game.qTable, open(filename, "wb"))
    print(f"Training complete. Q-table saved to {filename}.")

elif mode == "2":
    # Load trained Q-table and play against the Q-agent
    if not os.path.exists("trained_qtable.p"):
        print("No trained Q-table found. Train the agent first.")
        
    else:
        qTable = pickle.load(open("trained_qtable.p", "rb"))
        root = tk.Tk()
        P1 = User(turn = "X")
        P2 = QAgent(turn = "O", epsilon = 0)
        game = LearningAgentAndGame(root, P1, P2, qTable=qTable)
        game.play()
        root.mainloop()

elif mode == "3":
    # Self-play training for Value agent
    print("Training V-Agent...")
    epsilon = 0.9
    episodes = 100000
    Value = {}
    P1 = VAgent(turn = "X", epsilon = epsilon, Value = Value)
    P2 = VAgent(turn = "O", epsilon = epsilon, Value = Value)
    game = LearningAgentAndGame(P1 = P1, P2 = P2, VLearn = True, Value = Value)
    for i in range(episodes):
        if i % 10000 == 0:
            print(f"Episode {i}")
        game.play()
        game.reset()
        
    filename = "trained_vtable.p"
    pickle.dump(game.Value, open(filename, "wb"))
    print(f"Training complete. V-table saved to {filename}")

elif mode == "4":
    # Load trained Value-table and play against the Value agent
    if not os.path.exists("trained_vtable.p"):
        print("No trained V-table found. Train the agent first.")
        
    else:
        Value = pickle.load(open("trained_vtable.p", "rb"))
        root = tk.Tk()
        P1 = User(turn = "X")
        P2 = VAgent(turn = "O", epsilon = 0, Value = Value)
        game = LearningAgentAndGame(root, P1, P2, VLearn = True, Value = Value)
        game.play()
        root.mainloop()

else:
    print("Invalid input. Choose one from the options given.")