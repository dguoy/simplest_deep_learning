import numpy as np
import math


class Node(object):

    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count  

class ReplayBuffer(object):

    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history) - 1)) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

def play_game(config, game, network):
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game

def run_mcts(config, game, network):
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)
    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        if scratch_game.terminal():
            value = scratch_game.terminal_value()
        else:
            value = evaluate(node, scratch_game, network)

        backpropagate(search_path, value)

    return select_action(config, game, root), root

def select_action(config, game, root):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)

    return action

def select_child(config, node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child

def ucb_score(config, parent, child):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score

def evaluate(node, game, network):
    board, player = game.make_image(-1)
    value, policy_logits = network.inference([board], [player])
    value, policy_logits = value[0].numpy(), policy_logits[0].numpy()

    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
        node.children[action].to_play = game.to_play()

    return value

def backpropagate(search_path, value):
    for node in search_path:
        node.value_sum += value * node.to_play
        node.visit_count += 1

def add_exploration_noise(config, node):
    actions = node.children.keys()
    noise = np.random.dirichlet(config.root_dirichlet_alpha * np.ones(len(actions)))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

def softmax_sample(d):
    sum_visits = sum(visit for (visit, action) in d)
    visit_count_distribution = [visit / sum_visits for (visit, action) in d]
    idx = np.random.choice(range(len(d)), p=visit_count_distribution)
    return d[idx]
