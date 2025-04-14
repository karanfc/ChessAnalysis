import pandas as pd
import chess
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import plotly.graph_objects as go

# === CONFIG ===
DATA_PATH = "/home/karan/Desktop/Projects/Chess/games_metadata_profile.csv"  # <<-- Change this to your CSV file path
MAX_GAMES = None

capture_map = defaultdict(lambda: np.zeros((8, 8)))
label_to_info = {}
files = "abcdefgh"
ranks = "87654321"  # Static board display

piece_names = {
    chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
    chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
}

# Standard chess layout (White's POV)
static_layout = [
    'Black_Rook_a8', 'Black_Knight_b8', 'Black_Bishop_c8', 'Black_Queen_d8', 'Black_King_e8', 'Black_Bishop_f8', 'Black_Knight_g8', 'Black_Rook_h8',
    'Black_Pawn_a7', 'Black_Pawn_b7', 'Black_Pawn_c7', 'Black_Pawn_d7', 'Black_Pawn_e7', 'Black_Pawn_f7', 'Black_Pawn_g7', 'Black_Pawn_h7',
    'White_Pawn_a2', 'White_Pawn_b2', 'White_Pawn_c2', 'White_Pawn_d2', 'White_Pawn_e2', 'White_Pawn_f2', 'White_Pawn_g2', 'White_Pawn_h2',
    'White_Rook_a1', 'White_Knight_b1', 'White_Bishop_c1', 'White_Queen_d1', 'White_King_e1', 'White_Bishop_f1', 'White_Knight_g1', 'White_Rook_h1'
]

def clean_moves(move_str):
    move_str = re.sub(r'\{[^}]*\}', '', move_str)
    move_str = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*', '', move_str)
    move_str = re.sub(r'\d+\.(\.\.)?', '', move_str)
    move_str = re.sub(r'[\?!#+=]+', '', move_str)
    return move_str.strip().split()

def process_game(move_str):
    board = chess.Board()
    piece_ids = {}
    king_labels = {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            name = piece_names[piece.piece_type]
            prefix = "White" if piece.color == chess.WHITE else "Black"
            square_name = chess.square_name(square)
            label = f"{prefix}_{name}_{square_name}"
            title = f"Final Position Heatmap: {prefix} King ({square_name})" if piece.piece_type == chess.KING else f"Capture Heatmap: {prefix} {name} ({square_name})"
            label_to_info[label] = (title, square_name)
            piece_ids[square] = label
            if piece.piece_type == chess.KING:
                king_labels[piece.color] = label

    try:
        move_list = clean_moves(move_str)
        for san in move_list:
            move = board.parse_san(san)
            from_sq = move.from_square
            to_sq = move.to_square

            if board.is_capture(move):
                captured_sq = to_sq
                if board.is_en_passant(move):
                    captured_sq = to_sq + (-8 if board.turn else 8)

                captured_piece = board.piece_at(captured_sq)
                if captured_piece and captured_piece.piece_type != chess.KING:
                    label = piece_ids.get(captured_sq)
                    if label:
                        file_idx = chess.square_file(captured_sq)
                        rank_idx = chess.square_rank(captured_sq)
                        capture_map[label][rank_idx][file_idx] += 1

            if from_sq in piece_ids:
                piece_ids[to_sq] = piece_ids.pop(from_sq)

            board.push(move)

        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is not None:
                label = king_labels[color]
                file_idx = chess.square_file(king_sq)
                rank_idx = chess.square_rank(king_sq)
                if label:
                    capture_map[label][rank_idx][file_idx] += 1
    except Exception:
        pass

def draw_chessboard_background(ax):
    light_color = "#f0d9b5"
    dark_color = "#b58863"
    for rank in range(8):
        for file in range(8):
            color = light_color if (rank + file) % 2 == 0 else dark_color
            rect = plt.Rectangle((file, rank), 1, 1, facecolor=color, zorder=0)
            ax.add_patch(rect)
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(files)
    ax.set_yticklabels(ranks)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.invert_yaxis()
    ax.grid(False)

def plot_capture_heatmap(data, label):
    title, _ = label_to_info.get(label, ("", ""))
    if np.sum(data) == 0:
        print(f"⚠️ Skipping: {title}")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_chessboard_background(ax)
    im = ax.imshow(data, cmap='Reds', interpolation='nearest',
                   alpha=0.7, extent=(0, 8, 0, 8), zorder=5)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Captures", rotation=270, labelpad=15)
    ax.set_title(title)
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"heatmap_{safe_title}.png")
    plt.close()

def generate_hover_text(data):
    return [
        [f"{files[col]}{row + 1}<br>Captures: {int(data[row][col])}" for col in range(8)]
        for row in range(8)
    ]

# === LOAD AND PROCESS ===
df = pd.read_csv(DATA_PATH)

for idx, row in df.iterrows():
    process_game(row["Moves"])
    if MAX_GAMES and idx + 1 >= MAX_GAMES:
        break
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} games...")

# === STATIC HEATMAPS ===
for label, data in sorted(capture_map.items()):
    plot_capture_heatmap(data, label)

# === STATIC GRID OF 32 ===
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))

for i, label in enumerate(static_layout):
    ax = axes[i // 8][i % 8]
    data = capture_map.get(label, np.zeros((8, 8)))
    title, _ = label_to_info.get(label, ("", ""))
    draw_chessboard_background(ax)
    if np.sum(data) > 0:
        ax.imshow(data, cmap='Reds', interpolation='nearest', alpha=0.7,
                  extent=(0, 8, 0, 8), zorder=5)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("All_Capture_Heatmaps_Grid.png", dpi=300)
plt.close()

# === INTERACTIVE DROPDOWN (FULLY FIXED) ===
dropdown_fig = go.Figure()
buttons = []

for i, label in enumerate(static_layout):
    data = capture_map.get(label, np.zeros((8, 8)))
    title, _ = label_to_info.get(label, ("", ""))
    hover = generate_hover_text(data)
    visible = [False] * len(static_layout)
    visible[i] = True

    dropdown_fig.add_trace(go.Heatmap(
        z=data,  # 
        colorscale="Reds",
        zmin=0,
        zmax=max(1, data.max()),
        text=hover,
        hoverinfo="text",
        visible=(i == 0),
        showscale=True
    ))

    buttons.append(dict(
        label=title,
        method="update",
        args=[{"visible": visible}, {"title": title}]
    ))

dropdown_fig.update_layout(
    title=label_to_info[static_layout[0]][0],
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.0,
        xanchor="left",
        y=1.1,
        yanchor="top"
    )],
    height=600,
    width=600,
    xaxis=dict(
        tickvals=list(range(8)),
        ticktext=list("abcdefgh")
    ),
    yaxis=dict(
        tickvals=list(range(8)),
        ticktext=list("12345678"),  # rank 1 at bottom
        autorange=True              
    )
)

dropdown_fig.write_html("Capture_Heatmap_Dropdown.html")
