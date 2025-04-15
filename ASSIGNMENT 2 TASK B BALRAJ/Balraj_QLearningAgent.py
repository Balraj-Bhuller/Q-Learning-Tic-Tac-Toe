# 06/04/2025: Creating a TIC TAC TOE interface so 2 people can play against each other
# 12/04/2025: Creating a Random Number Generator AI and ROck Paper Scissors mini Gmae to decide first player
# 13/04/2025: Creating a QLearningAgent Class to play against the Player
# 13/04/2025: Restarting over with the knowledge I've ganed so far and make program more readable and work
# 15/04/2025: Creating GUI for TIC TAC TOE and using knowledge I gained from previous program + combing it with other programs to create wokring QlearningAgent

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
        return 'O' if self.turn == 'X' else 'X'


class User(Player):
    pass


class AI(Player):
    pass


class RandomAI(AI):
    
    @staticmethod
    def getMove(board):
        moves = board.availableMoves()
        return moves[np.random.choice(len(moves))] if moves else None


class RuleBasedAI(AI):
    
    def getMove(self, board):
        for move in board.availableMoves():
            if self.winningMove(board, move, self.turn) or self.winningMove(board, move, self.opponentTurn):
                return move
            
        return RandomAI.getMove(board)

    @staticmethod
    def winningMove(board, move, turn):
        return board.getNextState(move, turn).winner() == turn


class QAgent(AI):
    def __init__(self, turn, qTable=None, epsilon=0.2):
        super().__init__(turn)
        self.qTable = qTable if qTable is not None else {}
        self.epsilon = epsilon

    def getMove(self, board):
        if np.random.uniform() < self.epsilon:
            return RandomAI.getMove(board)
        key = self.addKey(board, self.turn, self.qTable)
        Q = self.qTable[key]
        return self.selectMove(Q, max if self.turn == 'X' else min)

    @staticmethod
    def addKey(board, turn, Q):
        key = board.makeKey(turn)
        if key not in Q:
            Q[key] = {move: 1.0 for move in board.availableMoves()}
            
        return key

    @staticmethod
    def selectMove(Q, func):
        target_value = func(Q.values())
        best_moves = [move for move, val in Q.items() if val == target_value]
        return best_moves[np.random.choice(len(best_moves))]


# --- Board Class ---
class Board:
    def __init__(self, grid=None):
        self.grid = np.full((3, 3), np.nan) if grid is None else grid

    def winner(self):
        lines = [self.grid[i, :] for i in range(3)] + \
                [self.grid[:, j] for j in range(3)] + \
                [np.diag(self.grid), np.diag(np.fliplr(self.grid))]
        for line in lines:
            if np.all(line == 1): return 'X'
            if np.all(line == 0): return 'O'
        return None

    def gameFinished(self):
        return self.winner() is not None or not np.any(np.isnan(self.grid))

    def placeTurn(self, move, turn):
        self.grid[move] = 1 if turn == 'X' else 0

    def availableMoves(self):
        return [(i, j) for i in range(3) for j in range(3) if np.isnan(self.grid[i][j])]

    def getNextState(self, move, turn):
        new_board = copy.deepcopy(self)
        new_board.placeTurn(move, turn)
        return new_board

    def makeKey(self, turn):
        filled = np.nan_to_num(self.grid, nan=9).astype(int).flatten()
        return ''.join(map(str, filled)) + turn

    def getReward(self):
        winner = self.winner()
        if winner == 'X': return 1.0
        if winner == 'O': return -1.0
        if not np.any(np.isnan(self.grid)): return 0.5
        return 0.0


# --- Game GUI and Q-Learning Trainer ---
class QLearningAgentAndGame:
    def __init__(self, root=None, P1=None, P2=None, QLearn=None, qTable=None, alpha=0.3, gamma=0.9):
        self.root = root
        self.P1 = P1
        self.P2 = P2
        self.currentPlayer = P1
        self.opponent = P2
        self.board = Board()
        self.qTable = qTable if qTable is not None else {}
        self.alpha = alpha
        self.gamma = gamma
        self._QLearn = QLearn if QLearn is not None else isinstance(P1, QAgent) or isinstance(P2, QAgent)
        self.syncQTable()

        if self.root:
            self.buttons = [[None for _ in range(3)] for _ in range(3)]
            frame = tk.Frame(root)
            frame.pack()
            for r in range(3):
                for c in range(3):
                    self.buttons[r][c] = tk.Button(frame, width=10, height=4, command=lambda r=r, c=c: self.onClick(r, c))
                    self.buttons[r][c].grid(row=r, column=c)
                    
            tk.Button(root, text="Reset", command=self.reset).pack()

    def syncQTable(self):
        if isinstance(self.P1, QAgent): self.P1.qTable = self.qTable
        
        if isinstance(self.P2, QAgent): self.P2.qTable = self.qTable

    def onClick(self, r, c):
        if self.board.gameFinished(): return
        
        if isinstance(self.currentPlayer, User) and np.isnan(self.board.grid[r][c]):
            self.makeMove((r, c))
            self.root.after(200, self.aiPlay)
                
    def aiPlay(self):
        while not self.board.gameFinished() and isinstance(self.currentPlayer, AI):
            move = self.currentPlayer.getMove(self.board)
            self.makeMove(move)

    def makeMove(self, move):
        if self._QLearn:
            self.updateQ(move)
          
        self.board.placeTurn(move, self.currentPlayer.turn)
        
        if self.root:
            r, c = move
            self.buttons[r][c]['text'] = self.currentPlayer.turn
            
        if self.board.gameFinished():
            self.announceWinner()
            
        else:
            self.switchPlayer()

    def switchPlayer(self):
        self.currentPlayer, self.opponent = self.opponent, self.currentPlayer

    def announceWinner(self):
        winner = self.board.winner()
        result = f"Winner: {winner}" if winner else "Draw"
        print(result)

    def reset(self):
        self.board = Board()
        self.currentPlayer = self.P1
        self.opponent = self.P2
        if self.root:
            for r in range(3):
                for c in range(3):
                    self.buttons[r][c]['text'] = ''

    def play(self):
        if isinstance(self.P1, AI):
            move = self.P1.getMove(self.board)
            self.makeMove(move)
            
        while not self.board.gameFinished() and isinstance(self.currentPlayer, AI):
            move = self.currentPlayer.getMove(self.board)
            self.makeMove(move)

    def updateQ(self, move):
        key = QAgent.addKey(self.board, self.currentPlayer.turn, self.qTable)
        next_board = self.board.getNextState(move, self.currentPlayer.turn)
        reward = next_board.getReward()
        next_key = QAgent.addKey(next_board, self.opponent.turn, self.qTable)

        future = 0.0
        if next_board.availableMoves():
            next_Q = self.qTable[next_key]
            future = min(next_Q.values()) if self.currentPlayer.turn == 'X' else max(next_Q.values())

        self.qTable[key][move] += self.alpha * ((reward + self.gamma * future) - self.qTable[key][move])


# --- Mode Select: Train or Play ---
mode = input("Choose mode: (1) Train AI, (2) Play vs AI: ").strip()

if mode == "1":
    print("Training Q-Agent...")
    epsilon = 0.9
    episodes = 200000
    P1 = QAgent(turn="X", epsilon=epsilon)
    P2 = QAgent(turn="O", epsilon=epsilon)
    game = QLearningAgentAndGame(P1=P1, P2=P2)
    for _ in range(episodes):
        game.play()
        game.reset()

    filename = "trained_qtable.p"
    pickle.dump(game.qTable, open(filename, "wb"))
    print(f"Training complete. Q-table saved to {filename}.")
    
elif mode == "2":
    if not os.path.exists("trained_qtable.p"):
        print("No trained Q-table found. Train the agent first.")
        
    else:
        qTable = pickle.load(open("trained_qtable.p", "rb"))
        root = tk.Tk()
        P1 = User(turn="X")
        P2 = QAgent(turn="O", epsilon=0)
        game = QLearningAgentAndGame(root, P1, P2, qTable=qTable)
        game.play()
        root.mainloop()
        
else:
    print("Invalid input. Choose 1 or 2.")