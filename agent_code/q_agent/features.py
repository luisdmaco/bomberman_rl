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

    # P2: voluntary WAIT is banned outright. It was the #1 suicide driver
    # (standing still next to your own ticking bomb) and a linear model never
    # uses it strategically. It is only re-enabled by the empty-mask fallback
    # at the bottom, when literally no move is possible.
    mask['WAIT'] = False

    # Don't walk into a tile that is going to explode
    for action, pos in directions.items():
        if not get_walkable(field, pos[0], pos[1], bombs, others):
            mask[action] = False
        elif danger_at(pos, dmap) == 0:
            mask[action] = False

    # Prune moves that walk into a spot with no escape route, but only
    # while at least one other move still survives -- otherwise pruning
    # the last one would fall through to the WAIT fallback below, which
    # is worse than trying even a doomed move. If every option is doomed,
    # don't prune here; the block below keeps only the move whose blast
    # arrives latest instead of leaving the choice to chance.
    if bombs:
        movable = [a for a in directions if mask[a]]
        if movable:
            doomed = [a for a in movable
                      if not escape_exists(directions[a], field, bombs, others,
                                           explosion_map, assume_own_bomb=False)]
            survivors = [a for a in movable if a not in doomed]
            if survivors: # only prunes if there is somewhere to move
                for action in doomed:
                    mask[action] = False
            # if survivors is empty, all doomed are legal, bc at least
            # the agent can try and run 
            else:
                # every move is doomed: don't let argmax pick one at random
                # (possibly straight into the blast) -- keep only the move
                # toward the tile that explodes LAST, so the agent runs as
                # far from the incoming blast as physics allow
                def _deadline(a):
                    d = danger_at(directions[a], dmap)
                    return 99 if d is None else d
                best_t = max(_deadline(a) for a in movable)
                for action in movable:
                    if _deadline(action) < best_t:
                        mask[action] = False

    # WAIT is already banned at the top, only at the bottom can it be re-enabled
    # Bomb: ask what the blast catches
    blast = get_blast_coords(x, y, field)
    hits_crate = any(field[bx, by] == 1 for (bx, by) in blast)
    hits_opponent = any(o[3] in blast for o in others)

    if not can_bomb:
        mask['BOMB'] = False
    elif not hits_crate and not hits_opponent:
        mask['BOMB'] = False    # nothing to gain, only risk
    elif not escape_exists((x, y), field, bombs, others, explosion_map):
        mask['BOMB'] = False

    # Never hand back zero legal actions
    # CHANGE: 
    if not any(mask.values()):
        safe_moves = [a for a, pos in directions.items()
            if get_walkable(field, pos[0], pos[1], bombs, others)
            and danger_at(pos, dmap) != 0]
        if safe_moves: # keep the one that is less near to explode
            best = max(safe_moves,
                       key=lambda a: 99 if danger_at(directions[a], dmap) is None
                            else danger_at(directions[a], dmap))
            mask = {a: False for a in ACTIONS}
            mask[best] = True
        else:
            mask['WAIT'] = True

    return mask 

# Would dropping a bomb here still leave us a way out?
# Timing from environment.do_step: bomb with timer t is lethal at step t AND
# t+1 (the explosion goes one more step), our own bomb at steps 4 and 5,
# explosion_map entry at step 0 only
# assume_own_bomb=False asks the same about tile we want to step onto
def escape_exists(pos, field, bombs, others, explosion_map=None,
                  bomb_power=3, bomb_timer=4, assume_own_bomb=True):

    lethal = {}          # step -> tiles lethal at the end of that step

    def mark(step, tiles):
        lethal.setdefault(step, set()).update(tiles)

    horizon = 1

    # the bomb we are thinking about dropping
    if assume_own_bomb:
        own_blast = get_blast_coords(pos[0], pos[1], field, bomb_power)
        mark(bomb_timer, own_blast)
        mark(bomb_timer + 1, own_blast)
        horizon = bomb_timer + 1

    # every bomb already ticking, ours and theirs
    for (bx, by), t in bombs:
        blast = get_blast_coords(bx, by, field, bomb_power)
        mark(t, blast)
        mark(t + 1, blast)
        horizon = max(horizon, t + 1)

    # explosions burning right now
    if explosion_map is not None:
        xs, ys = np.nonzero(explosion_map)
        mark(0, {(int(bx), int(by)) for bx, by in zip(xs, ys)})

    if pos in lethal.get(0, ()):
        return False     # we are already dead this step, wherever we go

    # where can we stand alive after each step; waiting counts as a move,
    # walking back onto our own bomb does not
    frontier = {pos}
    for step in range(1, horizon + 1):
        deadly = lethal.get(step, ())
        reachable = set()
        for (cx, cy) in frontier:
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                npos = (cx + dx, cy + dy)
                if (dx, dy) != (0, 0):
                    if assume_own_bomb and npos == pos:
                        continue    # our own bomb occupies that tile now
                    if not get_walkable(field, npos[0], npos[1], bombs, others):
                        continue
                if npos in deadly:
                    continue
                reachable.add(npos)
        if not reachable:
            return False
        frontier = reachable

    return True


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
# only walls stop the blast, crates do not: they are destroyed and the
# tiles behind them still explode
def get_blast_coords(bomb_x, bomb_y, field, bomb_power=3):
    blast = {(bomb_x, bomb_y)}

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, bomb_power + 1):
            nx, ny = bomb_x + dx * i, bomb_y + dy * i
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            blast.add((nx, ny))
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

# 8. Feature: direction + distance to the nearest opponent
# get_walkable reports an occupied tile as blocked
def bfs_to_opponents(start, field, bombs, others):
    opponents = {o[3] for o in others}
    if not opponents:
        return 'NONE', None

    neighbors = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    queue = deque()
    visited = {start}

    for direction, (dx, dy) in neighbors.items():
        npos = (start[0] + dx, start[1] + dy)
        if npos in opponents:
            return direction, 1
        if get_walkable(field, npos[0], npos[1], bombs, others):
            visited.add(npos)
            queue.append((npos, direction, 1))

    while queue:
        pos, first_dir, dist = queue.popleft()
        for direction, (dx, dy) in neighbors.items():
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in visited:
                continue
            if npos in opponents:
                return first_dir, dist + 1
            if get_walkable(field, npos[0], npos[1], bombs, others):
                visited.add(npos)
                queue.append((npos, first_dir, dist + 1))

    return 'NONE', None   # no opponent reachable

# 9. Feature: does a bomb dropped at (x, y) catch anybody?
def opponent_in_blast(x, y, field, others, bomb_power=3):
    if not others:
        return False
    blast = get_blast_coords(x, y, field, bomb_power)
    return any(o[3] in blast for o in others)


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
    # /12 = BOMB_POWER tiles in each of the 4 directions, the true maximum
    crates_in_blast = sum(1 for (bx, by) in get_blast_coords(x, y, field) if field[bx, by] == 1)
    crate_blast_feature = [crates_in_blast / 12.0] 

    # 8. FEATURE: direction and distance to the nearest opponent
    opp_dir, opp_dist = bfs_to_opponents((x, y), field, bombs, others)

    opp_dir_features = [1.0 if opp_dir == d else 0.0
                        for d in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']]

    # 0.0 = none reachable, 1.0 = adjacent. coin_dist maps none to 1.0 instead
    if opp_dist is None:
        opp_dist_feature = [0.0]
    else:
        opp_dist_feature = [1.0 / opp_dist]

    # 9. FEATURE: would a bomb dropped here catch an opponent?
    opp_blast_feature = [1.0 if opponent_in_blast(x, y, field, others) else 0.0]

    # --- combine into one fixed-length vector ---
    features = np.array(
        walkable_features +   # 4
        danger_features +     # 5
        bomb_feature +        # 1
        coin_dir_features +   # 5
        coin_dist_feature +   # 1
        crate_dir_features +  # 5
        safe_dir_features +   # 5
        crate_blast_feature + # 1
        opp_dir_features +    # 5
        opp_dist_feature +    # 1
        opp_blast_feature,    # 1
        dtype=np.float32
    )

    return features
