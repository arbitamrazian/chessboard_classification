from cairosvg import svg2png
from PIL import Image
import random
import numpy as np
import chess
import chess.svg
import io
import os
from random import randint

DIR = 'training_data/standard/test/'
if not os.path.exists(DIR):
    os.makedirs(DIR)
    os.makedirs(DIR+'images/')

NUM_SAMPLES = 2000
NUM_PCS = None
BOARD_SIZE = 360
FINAL_BOARD_SIZE = 180
NUM_ROTATIONS = 10

piece_type_map = {
        0: 'x',
        1: 'p',
        2: 'n',
        3: 'b',
        4: 'r',
        5: 'q',
        6: 'k',
        7: 'P',
        8: 'N',
        9: 'B',
        10: 'R',
        11: 'Q',
        12: 'K',
}

def get_filename_from_board(board):
    filename = ['x']*64
    for i in list(range(64)):
        piece = board.piece_at(i)
        if piece is not None:
            piece_type = piece.piece_type
            piece_color = piece.color
            filename[i] = piece_type_map[piece_type+ 6*piece_color]
    return ''.join(filename)

def get_positions_from_board(board):
    output = [0]*64
    for i in list(range(64)):
        piece = board.piece_at(i)
        if(piece is not None):
            output[i] = 1
    return output

def get_random_board(num_pcs = None):
    board = chess.Board(None)
    lst = list(range(64))
    random.shuffle(lst)

    kingw = lst.pop()
    kingb = lst.pop()

    piecew = chess.Piece(chess.KING,True)
    board_map = {kingw:piecew}
    pieceb = chess.Piece(chess.KING,False)
    board_map[kingb]=pieceb

    if num_pcs is None:
        num_pcs = random.randint(0,len(lst))
    for pos in lst[0:num_pcs]:
        min_pcs = 1
        if chess.square_rank(pos) == 0 or chess.square_rank(pos) == 7:
            min_pcs = 2
        p = random.randint(min_pcs,5)
        col = random.randint(0,1)
        piece = chess.Piece(p,col)
        board_map[pos] = piece

    board.set_piece_map(board_map)
    return board


count = 0
while(count < NUM_SAMPLES):
    image = io.BytesIO()
    board = get_random_board(num_pcs = NUM_PCS)
    filename = get_filename_from_board(board)
    board_arr = get_positions_from_board(board)
    svg = chess.svg.board(board=board,coordinates=False,size=BOARD_SIZE)
    #dark=d18b47
    #light=ffce9e

    svg2png(svg, write_to=image)
    img = Image.open(image)
    for k in range(NUM_ROTATIONS):
        print(count)
        count = count + 1
        rand_rotate = randint(-30, 30)
        img_rot = img.rotate(rand_rotate,expand=True,resample=Image.BICUBIC).resize((FINAL_BOARD_SIZE,FINAL_BOARD_SIZE)).convert('L')
        img_rot.save(DIR+'images/%s.png' % (filename+'_'+str(rand_rotate)))

