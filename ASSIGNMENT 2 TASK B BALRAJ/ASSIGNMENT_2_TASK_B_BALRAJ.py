#06/04/2022 Creating a TIC TAC TOE interface so 2 people can play against each other

def printBoard(board):
    print()
    for i in range(0, 9, 3):
        print(f"{board[i]} | {board[i+1]} | {board[i+2]}")
        if i < 6:
            print("==+===+==")
    print()
    
def checkWinner(p):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7 ,8),
            (0, 3, 6), (1, 4 ,7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
            ]
    return any(board[a] == board[b] == board[c] == p for a, b ,c in wins)

board = [" " for _ in range(9)]
turn = "X"

while True:
    printBoard(board)
    move = input(f"Player {turn}, enter position (0 - 8): ")
    if not move.isdigit() or int(move) not in range(9) or board[int(move)] != " ":
        print("Invalid Move. Try Again!")
        continue
    
    move = int(move)
    board[move] = turn
    
    if checkWinner(turn):
        printBoard(board)
        print(f"Player {turn} wins!")
        break
    
    if " " not in board:
        printBoard(board)
        print("It's a draw!")
        break
    
    turn = "O" if turn == "X" else "X"

