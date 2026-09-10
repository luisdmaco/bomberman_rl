import numpy as np 
from collections import deque

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
# Helper functions needed to extract the features from the game state
#---------------------------------------------------------------------

# avoids that it drops bombs in coin-heaven scenario 
def get_action_mask(game_state):
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    _, _, can_bomb, (x, y) = game_state['self']

    dmap = danger_map(field, bombs, explosion_map)
    directions = {
        'UP':     (x, y-1),
        'DOWN':   (x, y+1),
        'LEFT':   (x-1, y),
        'RIGHT':  (x+1, y),
    }

    mask = {a : True for a in ACTIONS}

    # Don't walk into a tile that is going to explode
    for action, pos in directions.items():
        if not get_walkable(field, pos[0], pos[1], bombs, others):
            mask[action] = False
        elif danger_at(pos, dmap) == 0:
            mask[action] = False

    # Never wait inside a blast zone or when is playing solo
    if danger_at((x,y), dmap) is not None or len(others) == 0:
        mask['WAIT'] = False

    # Bomb
    if not can_bomb:
        mask['BOMB'] = False
    elif np.sum(field == 1) == 0 and len(others) == 0:
        mask['BOMB'] = False    # nothing to gain, only risk

    elif len(others) == 0 and not any(field[bx, by] == 1
                                      for (bx, by) in get_blast_coords(x, y, field)):
        mask['BOMB'] = False   # never spend a bomb on zero crates 
    
    elif not escape_exists((x, y), field, bombs, others):
        mask['BOMB'] = False

    # Never hand back zero legal actions
    if not any(mask.values()):
        mask['WAIT'] = True

    return mask 

# TODO: only accounts for one bomb at a time
# change later for multi-bomb scenario
def escape_exists(pos, field, bombs, others, bomb_power=3, bomb_timer=4):

    blast = get_blast_coords(pos[0], pos[1], field, bomb_power)

    frontier = deque([(pos, 0)])
    visited = {pos}
    while frontier:
        (cx, cy), dist = frontier.popleft()
        if (cx, cy) not in blast:
            return True
        if dist == bomb_timer:
            continue
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            npos = (cx + dx, cy + dy)
            if npos not in visited and get_walkable(field, npos[0], npos[1], bombs, others):
                visited.add(npos)
                frontier.append((npos, dist + 1))
    return False


# 1. FEATURE
# Check whether tile (x,y) is currently available to enter by an agent
def get_walkable(field, x, y, bombs, others):
    """
    field: for every tile -1 = wall; 1 = crate; 0 = free
    x, y:  coordinates to check
    bombs: game_state['bombs']
    others: game_state['others'] -> other agents
    """
    width, height = field.shape
    # out of bounds
    if not (0 <= x < width and 0 <= y < height):
        return False

    # tile is not free
    if field[x, y] != 0:
        return False

    # there is a bomb in that tile
    bomb_positions = {pos for pos, _ in bombs}
    if (x, y) in bomb_positions:
        return False

    # there is another agent in that position
    other_positions = {o[3] for o in others}
    if (x, y) in other_positions:
        return False

    return True

# 2. FEATURE
# Given a bomb at (bomb_x, bomb_y), return the set of tiles
# that would be hit by its explosion, respecting walls.
def get_blast_coords(bomb_x, bomb_y, field, bomb_power=3):
    blast = {(bomb_x, bomb_y)}

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, bomb_power + 1):
            nx, ny = bomb_x + dx * i, bomb_y + dy * i
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:  # wall stops the blast
                break
            blast.add((nx, ny))
            # note: crates absorb the blast visually/logically but the
            # tile itself still becomes dangerous; blast stops AFTER a crate
            if field[nx, ny] == 1:
                break
    return blast

# creates a danger map from the active explosions and ticking bombs
# tile = 0 --> dangerous now
def danger_map(field, bombs, explosion_map, bomb_power=3):
    danger = {}

    # 1. Active explosions --> dangerous right now
    xs, ys = np.nonzero(explosion_map)
    for x, y in zip(xs, ys):
        danger[(x, y)] = 0

    # 2. Ticking bombs --> will be dangerouss in 't' steps
    for (bx, by), t in bombs:
        for (x, y) in get_blast_coords(bx, by, field, bomb_power):
            if (x, y) not in danger or danger[(x, y)] > t:
                danger[(x, y)] = t

    return danger

# give the danger countdown for a given position
def danger_at(pos, danger_map_dict):

    return danger_map_dict.get(pos, None)

def danger_feature(pos, dmap):
    d = danger_at(pos, dmap)
    if d is None:
        return 0.0          # safe
    return 1.0 / (d + 1)    # closer danger -> higher value; tune as needed

# 4. Feature
# BFS from start over walkable tiles to nearest targets
# TODO: for next task: precompute statics wall_adjacency one per round 
# and only re-check dynamic obstacles --> less time need for each step 
def bfs_direction_and_distance(start, targets, field, bombs, others):

    if not targets:
        return 'NONE', None

    targets = set(targets)
    if start in targets:
        return 'NONE', 0

    # each query entry = (position, first_step_direction)
    queue = deque()
    visited = {start}
    neighbors = {
        'UP':     (0,-1),
        'DOWN':   (0, 1),
        'LEFT':   (-1,0),
        'RIGHT':  (1, 0),
    }

    # BFS 
    for direction, (dx, dy) in neighbors.items():
        npos = (start[0] + dx, start[1] + dy)
        if get_walkable(field, npos[0], npos[1], bombs, others):
            queue.append((npos, direction, 1))
            visited.add(npos)

    while queue:
        pos, first_dir, dist = queue.popleft()

        if pos in targets:
            return first_dir, dist

        for direction, (dx, dy) in neighbors.items():
            npos = (pos[0] + dx, pos[1] + dy)
            if npos not in visited and get_walkable(field, npos[0], npos[1], bombs, others):
                visited.add(npos)
                queue.append((npos, first_dir, dist + 1))

    return 'NONE', None # unreachable

# 5. Feature: 
# finds free tiles touching at least one crate
def crate_adjacent_free_tiles(field):
    w, h = field.shape
    tiers = {3: [], 2: [], 1: []}

    for xx in range(1, w-1):
        for yy in range(1, h-1):
            if field[xx, yy] != 0:
                continue
            n = sum(1 for (bx, by) in get_blast_coords(xx, yy, field) 
                    if field[bx, by] == 1)
            if n >= 3:
                tiers[3].append((xx, yy))
            elif n == 2:
                tiers[2].append((xx, yy))
            elif n == 1:
                tiers[1].append((xx, yy))

    for  t in (3, 2, 1):
        if tiers[t]:
            return tiers[t]

    return []

def bfs_all_distances(start, field, bombs, others):
    dist = {start: (0, 'NONE')}
    queue = deque([start])
    neighbors = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    while queue:
        pos = queue.popleft()
        d, first = dist[pos]
        for direction, (dx, dy) in neighbors.items():
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in dist:
                continue
            if not get_walkable(field, npos[0], npos[1], bombs, others):
                continue
            dist[npos] = (d + 1, direction if pos == start else first)
            queue.append(npos)
    return dist

# direction toward the spot maximising crates / (distance + cooldown) 
def best_bomb_spot(start, field, bombs, others, cooldown=6):
    best_score, best_dir = 0.0, 'NONE'
    for pos, (d, first) in bfs_all_distances(start, field, bombs, others). items():
        n = sum(1 for (bx, by) in get_blast_coords(pos[0], pos[1], field)
                if field[bx, by] == 1)
        if n == 0:
            continue
        score = n / (d + cooldown)
        if score > best_score:
            best_score, best_dir = score, first

    return best_dir, best_score

# 6. Feature: finds free safe tiles, when threatened (to escape bomb)
def safe_free_tiles(field, bombs, explosion_map):

    dmap = danger_map(field, bombs, explosion_map)
    w, h = field.shape

    return [(xx, yy) for xx in range(1, w-1) for yy in range(1, h-1)
            if field[xx, yy] == 0 and (xx, yy) not in dmap]


def state_to_features(game_state):

    if game_state is None:
        return None

    # get state information
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    _, _, can_bomb, (x, y) = game_state['self']

    # neighbor coordinates — confirm UP/DOWN direction convention against the GUI
    neighbors = {
        'UP':    (x, y - 1),
        'DOWN':  (x, y + 1),
        'LEFT':  (x - 1, y),
        'RIGHT': (x + 1, y),
    }

    # 1. FEATURE: one value per direction 
    walkable_features = []
    for direction, (nx, ny) in neighbors.items():
        walkable = get_walkable(field, nx, ny, bombs, others)
        walkable_features.append(1.0 if walkable else 0.0)

    # 2. FEATURE
    dmap = danger_map(field, bombs, explosion_map)

    danger_features = [danger_feature((x, y), dmap)]  # danger at own tile
    for direction, pos in neighbors.items():
        danger_features.append(danger_feature(pos, dmap))

    # 3. FEATURE: can place bomb?
    bomb_feature = [1.0 if can_bomb else 0.0]

    # 4. FEATURE
    coin_dir, coin_dist = bfs_direction_and_distance(
        (x,y), game_state['coins'], field, bombs, others
    )
    coin_dir_features = [
        1.0 if coin_dir == d else 0.0
        for d in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
    ]
    # distance as normalized scalar
    if coin_dist is None:
        coin_dist_feature = [1.0] # far
    else:
        coin_dist_feature = [1.0 / (coin_dist + 1)] # closer --> higher value

    # 5. FEATURE: Direction to nearest crate-bombing spot
    crate_dir, _ = best_bomb_spot((x, y), field, bombs, others)

    crate_dir_features = [1.0 if crate_dir == d else 0.0
                          for d in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']]

    # 6. FEATURE: escape danger
    if danger_at((x,y), dmap) is not None:
        safe_dir, _ = bfs_direction_and_distance(
            (x, y), safe_free_tiles(field, bombs, explosion_map), field, bombs, others)
    else:
        safe_dir = 'NONE'

    safe_dir_features = [1.0 if safe_dir == d else 0.0 
                         for d in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']]

    # 7. FEATURE: How many crates destroy a bomb dropped at (x,y)
    crates_in_blast = sum(1 for (bx, by) in get_blast_coords(x, y, field) if field[bx, by] == 1)
    crate_blast_feature = [crates_in_blast / 4.0] 

    # --- combine into one fixed-length vector ---
    features = np.array(
        walkable_features +   # 4
        danger_features +     # 5
        bomb_feature +        # 1
        coin_dir_features +   # 5
        coin_dist_feature +   # 1
        crate_dir_features +  # 5
        safe_dir_features +   # 5
        crate_blast_feature,  # 1
        dtype=np.float32
    )

    return features
