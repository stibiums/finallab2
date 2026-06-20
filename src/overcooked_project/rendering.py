from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


TERRAIN_COLORS = {
    "X": (101, 88, 74),
    " ": (238, 232, 218),
    "O": (228, 185, 86),
    "P": (196, 92, 85),
    "D": (105, 150, 197),
    "S": (99, 173, 119),
}

TERRAIN_LABELS = {
    "X": "",
    " ": "",
    "O": "O",
    "P": "POT",
    "D": "D",
    "S": "S",
}

PLAYER_COLORS = [(55, 99, 184), (214, 92, 60)]


def _font(size: int):
    try:
        return ImageFont.truetype("Arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_state(mdp, state, cell_size: int = 72) -> Image.Image:
    rows = len(mdp.terrain_mtx)
    cols = len(mdp.terrain_mtx[0])
    width = cols * cell_size
    height = rows * cell_size
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    label_font = _font(max(12, cell_size // 5))
    player_font = _font(max(14, cell_size // 4))

    for y, row in enumerate(mdp.terrain_mtx):
        for x, terrain in enumerate(row):
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=TERRAIN_COLORS.get(terrain, (230, 230, 230)))
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
            label = TERRAIN_LABELS.get(terrain, terrain)
            if label:
                draw.text((x0 + 6, y0 + 6), label, fill=(45, 45, 45), font=label_font)

    objects = state.objects.values() if isinstance(state.objects, dict) else state.objects
    for obj in objects:
        x, y = obj.position
        cx = x * cell_size + cell_size // 2
        cy = y * cell_size + cell_size // 2
        radius = cell_size // 5
        color = (239, 209, 95) if obj.name == "onion" else (215, 235, 245)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=(45, 45, 45), width=2)
        draw.text((cx - radius, cy - radius - 14), obj.name[:1].upper(), fill=(20, 20, 20), font=label_font)

    for idx, player in enumerate(state.players):
        x, y = player.position
        cx = x * cell_size + cell_size // 2
        cy = y * cell_size + cell_size // 2
        radius = cell_size // 3
        color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=(20, 20, 20), width=3)
        draw.text((cx - 8, cy - 10), str(idx), fill=(255, 255, 255), font=player_font)
        dx, dy = player.orientation
        draw.line([cx, cy, cx + dx * radius, cy + dy * radius], fill=(255, 255, 255), width=4)
        if player.held_object is not None:
            draw.text((cx - radius, cy + radius - 6), player.held_object.name[:1].upper(), fill=(255, 255, 255), font=label_font)

    return image
