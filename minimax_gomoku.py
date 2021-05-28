import numpy as np
import random
import datetime
import pandas as pd
import copy
import math

board_idx = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
             [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
             [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], 
             [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], 
             [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75], 
             [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90], 
             [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105], 
             [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], 
             [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135], 
             [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150], 
             [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165], 
             [166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180], 
             [181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195], 
             [196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210], 
             [211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]]

board_value = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
               [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]

board_idx = np.array(board_idx)
board_value = np.array(board_value)


# Print the board values.
def print_board():
    df = pd.DataFrame(np.concatenate((board_value,board_idx),axis=1))
    print(df.to_string(index=False, header=False))


# Reset the board values.
def reset_board():
    for i in range(15):
        for j in range(15):
            board_value[i,j] = ' '


# Check if the game is in a terminal state.
def terminal(board_value):
    for row in board_value:
        row_str = ''.join(row)
        if 'xxxxx' in row_str.lower():
            return 'X'
        elif 'ooooo' in row_str.lower():
            return 'O'
    for i in range(0,board_value.shape[1]):
        col = board_value[:,i]
        col_str = ''.join(col)
        if 'xxxxx' in col_str.lower():
            return 'X'
        elif 'ooooo' in col_str.lower():
            return 'O'

    for i in range(0,board_value.shape[1]):
        diagonal = board_value.diagonal(offset=i)
        diagonal_str = ''.join(diagonal)
        if 'xxxxx' in diagonal_str.lower():
            return 'X'
        elif 'ooooo' in diagonal_str.lower():
            return 'O'
    
    for i in range(0,board_value.shape[0]):
        diagonal = board_value.diagonal(offset=i,axis1=1,axis2=0)
        diagonal_str = ''.join(diagonal)
        
        if 'xxxxx' in diagonal_str.lower():
            return 'X'
        elif 'ooooo' in diagonal_str.lower():
            return 'O'
    
    flipped_board_value = np.flip(board_value,axis=1)
    for i in (range(flipped_board_value.shape[0])):
        diagonal = flipped_board_value.diagonal(offset=i,axis1=0,axis2=1)
        diagonal_str = ''.join(diagonal)
        
        if 'xxxxx' in diagonal_str.lower():
            return 'X'
        elif 'ooooo' in diagonal_str.lower():
            return 'O'
        
    for i in (range(flipped_board_value.shape[1])):
        diagonal = flipped_board_value.diagonal(offset=i,axis1=1,axis2=0)
        diagonal_str = ''.join(diagonal)
        
        if 'xxxxx' in diagonal_str.lower():
            return 'X'
        elif 'ooooo' in diagonal_str.lower():
            return 'O'
    
    if np.where(board_value==' '):
        return "ongoing"
    else:
        return 'tie'


# Identify strings of turns and reward them appropriately. In order to 
# incentivise achieving longer strings, the rewards increase quadratically.
def pattern_finder(reward_str, turn, c):
    # Offer twice as much reward for each consecutive piece.
    reward_str = reward_str.lower()
    rewards = [0]

    # Winning state.
    if turn * 5 in reward_str:
        rewards.append(c * 2 ** 7)

    # 4 pieces with a space either side.
    elif ' ' + turn * 4 + ' ' in reward_str:
        rewards.append(c * 2 ** 6)

    # 4 pieces
    elif turn * 4 + ' ' in reward_str or\
    ' ' + turn * 4 in reward_str or\
    turn * 2 + ' ' + turn * 2 in reward_str:
        rewards.append(c * 2 ** 5)

    # 3 pieces with a space either side.
    elif ' ' + turn * 3 + ' ' in reward_str:
        rewards.append(c * 2 ** 4)

    # 3 pieces.
    elif turn * 3 + ' ' * 2 in reward_str or\
    ' ' * 2 + turn * 3 in reward_str or\
    turn * 2 + ' ' + turn + ' ' in reward_str or\
    ' ' + turn + ' ' + turn * 2 in reward_str or\
    turn + ' ' + turn * 2 + ' ' in reward_str or\
    ' ' + turn * 2 + ' ' + turn in reward_str or\
    turn + ' ' + turn + ' ' + turn in reward_str or\
    turn * 2 + ' ' * 2 + turn in reward_str or\
    turn + ' ' * 2 + turn * 2 in reward_str:
        rewards.append(c * 2 ** 3)

    # 2 consecutive pieces.
    elif turn * 2 + ' ' * 3 in reward_str or\
    ' ' + turn * 2 + ' ' * 2 in reward_str or\
    ' ' * 2 + turn * 2 + ' ' in reward_str or\
    ' ' * 3 + turn * 2 in reward_str:
        rewards.append(c * 2 ** 2)

    return rewards


# Send formatted strings to the pattern finder.
def matts_reward(board, my_turn):
    # I have just played so check for o's. I am maximising so my coefficient is 1.
    if my_turn:
        turn = 'o'
        c = 1
    else:
        turn = 'x'
        c = -1

    rewards = [0]

    # Check rows.
    for row in board:
        row_str = ''.join(row)
        rewards += pattern_finder(row_str, turn, c)

    # Check columns.
    for i in range(0, board.shape[1]):
        col = board[:, i]
        col_str = ''.join(col)
        rewards += pattern_finder(col_str, turn, c)

    # Check diagonals.
    for i in range(0, board.shape[1]):
        diagonal = board.diagonal(offset=i)
        diagonal_str = ''.join(diagonal)
        rewards += pattern_finder(diagonal_str, turn, c)

    for i in range(0, board.shape[0]):
        diagonal = board.diagonal(offset=i, axis1=1, axis2=0)
        diagonal_str = ''.join(diagonal)
        rewards += pattern_finder(diagonal_str, turn, c)

    flipped_board_value = np.flip(board, axis=1)

    for i in (range(flipped_board_value.shape[0])):
        diagonal = flipped_board_value.diagonal(offset=i, axis1=0, axis2=1)
        diagonal_str = ''.join(diagonal)
        rewards += pattern_finder(diagonal_str, turn, c)

    for i in (range(flipped_board_value.shape[1])):
        diagonal = flipped_board_value.diagonal(offset=i, axis1=1, axis2=0)
        diagonal_str = ''.join(diagonal)
        rewards += pattern_finder(diagonal_str, turn, c)

    if my_turn:
        return max(rewards)
    else:
        return min(rewards)
    

# Calculate the magnitude of the distance to the last move
# In practice playing around the opponents move was too defensive 
# and resulted in longer games. **NOT USED IN FINAL IMPLEMENTATION**
def distance_to_last_move(move, last_move):
    if move is None:
        # Return largest possible distance
        return 20
    else:
        row_idx = np.where(board_idx == move)[0][0]
        col_idx = np.where(board_idx == move)[1][0]
        last_row_idx = np.where(board_idx == last_move)[0][0]
        last_col_idx = np.where(board_idx == last_move)[1][0]
        vertical_diff = abs(row_idx - last_row_idx)
        horizontal_diff = abs(col_idx - last_col_idx)
        diff = math.sqrt(vertical_diff ** 2 + horizontal_diff ** 2)
        return diff


# Calculate the magnitude of the distance to the centre of the board, this keeps play
# tight, reducing search space and focuses the play aggressively around the centre
def distance_to_centre(move):
    if move is None:
        # Return largest possible distance
        return 10
    else:
        row_idx = np.where(board_idx == move)[0][0]
        col_idx = np.where(board_idx == move)[1][0]
        central_row_idx = 7
        central_col_idx = 7
        vertical_diff = abs(row_idx - central_row_idx)
        horizontal_diff = abs(col_idx - central_col_idx)
        diff = math.sqrt(vertical_diff ** 2 + horizontal_diff ** 2)
        return diff


# Generate a default sub-board around a move, this function calculates the
# sub-board boundaries using the maximum grid size and provides a "worst case" 
# search space. These boundaries are refined in the dynamic_sub_board function.
def get_default_sub_board(last_move, max_grid_size):
    # Get the row and column of the last move
    row = np.where(board_idx == last_move)[0][0]
    col = np.where(board_idx == last_move)[1][0]

    # Calculate sub-board boundaries using max grid size.
    top_limit = row - max_grid_size + 1
    bottom_limit = row + max_grid_size
    left_limit = col - max_grid_size + 1
    right_limit = col + max_grid_size

    # Set any boundaries that are invalid indices to the edge of the board.
    if top_limit < 0:
        top_limit = 0
    if bottom_limit > 15:
        bottom_limit = 15
    if left_limit < 0:
        left_limit = 0
    if right_limit > 15:
        right_limit = 15

    # Return default boundaries.
    return top_limit, bottom_limit, left_limit, right_limit


# Refine the sub-board by pruning any unpopulated space. This function
# removes any empty rows/columns and leaves a row + column either side 
# to ensure that these moves can be explored.
def get_dynamic_sub_board(board, last_move, max_grid_size):
    # First generate a default sub board using the maximum grid size
    top_limit, bottom_limit, left_limit, right_limit = get_default_sub_board(last_move, max_grid_size)

    # Pruning variables.
    top_prune = 0
    bottom_prune = 15
    left_prune = 0
    right_prune = 15

    # Prune empty rows from the top.
    for row in board:
        row_str = ''.join(row)
        if row_str == ' ' * 15:
            top_prune += 1
        else:
            break

    # Prune empty rows from the bottom.
    for row in board[::-1]:
        row_str = ''.join(row)
        if row_str == ' ' * 15:
            bottom_prune -= 1
        else:
            break

    # Prune empty columns from the left.
    for i in range(0, board.shape[1]):
        col = board[:, i]
        col_str = ''.join(col)
        if col_str == ' ' * 15:
            left_prune += 1
        else:
            break

    # Prune empty columns from the right.
    for i in range(0, board.shape[1])[::-1]:
        col = board[:, i]
        col_str = ''.join(col)
        if col_str == ' ' * 15:
            right_prune -= 1
        else:
            break

    # Adjust so that moves are simulated around moves at the edge of the sub-board.
    top_prune -= 1
    bottom_prune += 1
    left_prune -= 1
    right_prune += 1

    # Use the calculated prune variables to remove any empty rows/cols from the default search space.
    if top_prune > top_limit:
        top_limit = top_prune
    if bottom_prune < bottom_limit:
        bottom_limit = bottom_prune
    if left_prune > left_limit:
        left_limit = left_prune
    if right_prune < right_limit:
        right_limit = right_prune

    # Return a list of sub board id's used to simulate moves
    sub_board_ids = board_idx[top_limit:bottom_limit, left_limit:right_limit]
    return sub_board_ids


# Get a list of available moves using the board and cross-referencing the
# sub-board id list to make sure we only simulate moves within the boundaries.
def get_available_moves(board, sub_board_ids):
    available_moves = []
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row, col] == ' ' and board_idx[row, col] in sub_board_ids:
                available_moves.append(board_idx[row, col])
    return available_moves


# Minimax function. This function assumes the opponent will play optimally and chooses
# its path by alternating between turns to a specified depth. Minimax uses a DFS where
# node costs and rewards are summed as the recursion unwinds. Alpha-beta pruning reduces
# computation by no longer searching branches if a better option for the opponent is already
# found (assumes they pick it because they are playing optimally).
def matts_minimax(board, sub_board_ids, last_move, my_turn=True, depth=0, current_best=-1000):
    # Initialise to worst case values
    best_move = None
    if my_turn:
        max_min_eval = -1000
    else:
        max_min_eval = 1000

    # For each recursion make a copy of the board.
    new_board = copy.copy(board)

    # Use the sub-board id's to identify available moves.
    available_moves = get_available_moves(new_board, sub_board_ids)

    # Calculate depth limit, only increase depth when it is feasible to do so
    board_population = len(available_moves) / (len(sub_board_ids[0]) * len(sub_board_ids[1]))
    if board_population >= 0.4:
        depth_limit = 2
    elif 0.2 <= board_population < 0.4:
        depth_limit = 3
    else:
        depth_limit = 5
    
    depth = depth + 1

    for move in available_moves:
        row = np.where(board_idx == move)[0][0]
        col = np.where(board_idx == move)[1][0]
        if my_turn:
            new_board[row, col] = 'o'
            if depth < depth_limit:
                # Keep calling minimax swapping for each player until and end node is reached.
                _, next_node_cost = matts_minimax(new_board, sub_board_ids, last_move, my_turn=False, depth=depth, current_best=max_min_eval)

            else:
                # Next node cost is 0 because it is an end node.
                next_node_cost = 0

            # Calculate the net cost of a node by summing its reward and next node costs.
            node_cost = next_node_cost + matts_reward(new_board, my_turn)
            
            # Alpha-beta pruning. I assume the other player is playing optimally, so if there is a previously
            # seen better option for the minimiser, I  assume they will take it. This means I don't need to 
            # search the rest of the tree.
            if current_best!=-1000 and node_cost > current_best:
                best_move = move
                max_min_eval = node_cost
                new_board[row ,col] = ' '
                break

            # As we are the maximiser, take the path which evaluates to the highest value.
            if node_cost > max_min_eval:
                best_move = move
                max_min_eval = node_cost
            
            # If the best node costs are equal, playing closer to the centre allows for more efficient
            # search times and focused move generation.
            elif node_cost == max_min_eval:
                closer_to_centre = distance_to_centre(best_move) - distance_to_centre(move)
                if closer_to_centre > 0:
                    best_move = move
                    max_min_eval = node_cost

            # Reset the coordinate to empty once the move has been simulated.
            new_board[row, col] = ' '

        else:
            # ---REPEAT THIS PROCESS FOR THE MINIMISER---
            new_board[row, col] = 'x'
            if depth < depth_limit:
                # Keep calling minimax swapping for each player until and end node is reached.
                _, next_node_cost = matts_minimax(new_board, sub_board_ids, last_move, my_turn=True, depth=depth, current_best=max_min_eval)

            else:
                next_node_cost = 0

            node_cost = next_node_cost + matts_reward(new_board, my_turn)
            
            # Alpha-beta pruning. Minimax plays optimally so doesn't need to search when there are 
            # previously seen better options.
            if current_best!=-1000 and node_cost < current_best:
                best_move = move
                max_min_eval = node_cost
                new_board[row ,col] = ' '
                break

            # The minimiser chooses that path which evaluates to the lowest value.
            if node_cost < max_min_eval:
                best_move = move
                max_min_eval = node_cost

            new_board[row, col] = ' '
    
    # Return the best move id and its associated reward.
    return best_move, max_min_eval


def matts_agent(board, last_move):
    # If we are the first turn, play a random move.
    if last_move is None:
        return random.randint(1, 225)

    # Generate a grid relative to the previous move.
    max_grid_size = 9
    sub_board_ids = get_dynamic_sub_board(board, last_move, max_grid_size)

    move, reward = Matts_minimax(board, sub_board_ids, last_move)
    return move, reward


# Main game loop.
def game_autoplay(board):
    global depth_limit,grid_size
    reset_board()
    print("Game starts")
    turn = 'X'
    count = 0
    max = True
    min = False
    x_last_move = None
    o_last_move = None
    
    for i in range(225):            
        if(i%2==0):
            print("Round:",i/2)
        if max:
            before = datetime.datetime.now()
            move, reward = Lis_Agent_Policy(board_value,o_last_move)
            after = datetime.datetime.now()
            delta = after - before
            print("It's X's move:,", move,". Took ",delta.total_seconds()," seconds to react\n")
            row_idx = np.where(board_idx==move)[0][0]
            col_idx = np.where(board_idx==move)[1][0]
            if board_value[row_idx,col_idx] == ' ':
                board_value[row_idx,col_idx] = turn
                x_last_move = move
                print_board()
                count += 1
                max = False
                min = True
            else:
                print("That place is already filled.\nMove to which place?")
                continue
  
        else:
            before = datetime.datetime.now()
            move, reward = Matts_agent(board, x_last_move)
            after = datetime.datetime.now()
            row_idx = np.where(board_idx==move)[0][0]
            col_idx = np.where(board_idx==move)[1][0]
            delta = after - before
            print("It's O's move:,", move,". Took ",delta.total_seconds()," seconds to react\n")
            board_value[row_idx,col_idx] = turn
            print_board()
            o_last_move = move
            max = True
            min = False
            count += 1
            
        # Check if player X or O has won for every move after 5 moves. 
        if count >= 9:
            terminal_status = terminal(board_value)
            if terminal_status == 'X':
                print_board()
                print("\nGame Over.\n")                
                print(" **** " +turn + " won. ****")                
                break
            elif terminal_status == 'O':
                print_board()
                print("\nGame Over.\n")                
                print(" **** " +turn + " won. ****")                
                break
            elif terminal_status == 'tie':
                print("\nGame Over.\n")                
                print("It's a Tie!!")
                break;

        if turn =='X':
            turn = 'O'
        else:
            turn = 'X'        
    
    restart = input("Do want to play Again?(y/n)")
    if restart == "y" or restart == "Y":  
        game_autoplay(board_value)
if __name__ == "__main__":
    game_autoplay(board_value)