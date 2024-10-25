import numpy as np
from collections import defaultdict

class GridWorld:
    def __init__(self, grid_size=(4, 4), walls=[], key_location=None, door_location=None,
                 terminal_state=(3, 3), transition_probs=None, default_prob=0.25, discount_factor=1.0):
        self.grid_size = grid_size
        self.walls = walls
        self.key_location = key_location
        self.door_location = door_location
        self.terminal_state = terminal_state
        self.discount_factor = discount_factor
        self.actions = ['up', 'down', 'left', 'right']
        self.states = []
        self.rewards = defaultdict(float)
        self.default_reward = -1.0  # Default reward for each step
        self.key_collected = False
        self.transition_probs = transition_probs or {}
        self.default_prob = default_prob  # Default probability for unspecified transitions

        # Initialize states
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if (i, j) not in walls:
                    self.states.append((i, j))

    def is_terminal(self, state):
        return state == self.terminal_state

    def get_possible_actions(self, state):
        if self.is_terminal(state):
            return [None]
        else:
            return self.actions

    def get_next_states_and_probs(self, state, action):
        """
        Returns a list of (next_state, probability) tuples based on the action taken at the given state.
        """
        if action is None:
            return [(state, 1.0)]

        # Get the transition probabilities for this state-action pair
        probs = self.transition_probs.get((state, action), {})
        if not probs:
            # Use default probabilities (uniform over possible moves)
            probs = {a: self.default_prob for a in self.actions}

        # Normalize probabilities
        total_prob = sum(probs.values())
        probs = {a: p / total_prob for a, p in probs.items()}

        next_states = []
        for a, prob in probs.items():
            next_state = self.attempt_move(state, a)
            next_states.append((next_state, prob))
        return next_states

    def attempt_move(self, state, action):
        """
        Returns the next state after attempting to move in the given action from the given state.
        If the move leads to a wall or outside the grid, the state doesn't change.
        """
        i, j = state
        if action == 'up':
            i_new = max(i - 1, 0)
            next_state = (i_new, j)
        elif action == 'down':
            i_new = min(i + 1, self.grid_size[0] - 1)
            next_state = (i_new, j)
        elif action == 'left':
            j_new = max(j - 1, 0)
            next_state = (i, j_new)
        elif action == 'right':
            j_new = min(j + 1, self.grid_size[1] - 1)
            next_state = (i, j_new)
        else:
            next_state = state

        # Check for walls
        if next_state in self.walls:
            next_state = state

        # Check for door
        if next_state == self.door_location and not self.key_collected:
            next_state = state

        return next_state

    def get_reward(self, state, action, next_state):
        if next_state == self.terminal_state:
            return 0.0  # No reward for entering terminal state
        else:
            return self.rewards.get((state, action, next_state), self.default_reward)

    def collect_key(self, state):
        if state == self.key_location:
            self.key_collected = True

def policy_evaluation(env, policy, V_init=None, theta=0.0001, num_eval_iterations=None):
    V = V_init or {state: 0.0 for state in env.states}
    iterations = 0
    while True:
        delta = 0
        for state in env.states:
            if env.is_terminal(state):
                V[state] = 0.0
                continue
            v = V[state]
            action = policy[state]
            env.collect_key(state)
            next_states_probs = env.get_next_states_and_probs(state, action)
            V[state] = sum(
                prob * (env.get_reward(state, action, next_state) + env.discount_factor * V[next_state])
                for next_state, prob in next_states_probs
            )
            delta = max(delta, abs(v - V[state]))
        iterations += 1
        if delta < theta or (num_eval_iterations is not None and iterations >= num_eval_iterations):
            break
    return V

def policy_iteration(env, eval_iterations=1):
    policy = {}
    for state in env.states:
        if env.is_terminal(state):
            policy[state] = None
        else:
            policy[state] = np.random.choice(env.get_possible_actions(state))
    V = {state: 0.0 for state in env.states}

    while True:
        # Policy Evaluation
        V = policy_evaluation(env, policy, V_init=V, num_eval_iterations=eval_iterations)
        policy_stable = True
        # Policy Improvement
        for state in env.states:
            if env.is_terminal(state):
                continue
            old_action = policy[state]
            action_values = {}
            for action in env.get_possible_actions(state):
                env.collect_key(state)
                next_states_probs = env.get_next_states_and_probs(state, action)
                action_value = sum(
                    prob * (env.get_reward(state, action, next_state) + env.discount_factor * V[next_state])
                    for next_state, prob in next_states_probs
                )
                action_values[action] = action_value
            best_action = max(action_values, key=action_values.get)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return policy, V

# Example usage
if __name__ == "__main__":
    # Define walls, key location, door location, and terminal state
    walls = [(1, 1), (2, 2)]
    key_location = (0, 2)
    door_location = (1, 2)
    terminal_state = (3, 3)
    discount_factor = 0.9

    # Define transition probabilities
    # For example, at state (2,2), action 'up' has specific probabilities
    transition_probs = {
        ((2, 2), 'up'): {'up': 0.7, 'left': 0.1, 'right': 0.1, 'down': 0.1},
        # You can add more state-action specific probabilities here
    }

    # Create the grid world environment
    env = GridWorld(
        grid_size=(4, 4),
        walls=walls,
        key_location=key_location,
        door_location=door_location,
        terminal_state=terminal_state,
        transition_probs=transition_probs,
        discount_factor=discount_factor
    )

    # Perform policy iteration with 1 evaluation iteration (approximating value iteration)
    policy_1, V_1 = policy_iteration(env, eval_iterations=1)
    print("Policy after policy iteration with 1 evaluation iteration:")
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            state = (i, j)
            action = policy_1.get(state, 'N/A')
            if state in env.walls:
                action = 'Wall'
            elif state == env.key_location:
                action = f"{action}/Key"
            elif state == env.door_location:
                action = f"{action}/Door"
            elif state == env.terminal_state:
                action = 'Terminal'
            print(f"State ({i},{j}): {action}", end=' | ')
        print()

    # Perform policy iteration with full policy evaluation (standard policy iteration)
    policy_inf, V_inf = policy_iteration(env, eval_iterations=None)
    print("\nPolicy after policy iteration with full evaluation iterations:")
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            state = (i, j)
            action = policy_inf.get(state, 'N/A')
            if state in env.walls:
                action = 'Wall'
            elif state == env.key_location:
                action = f"{action}/Key"
            elif state == env.door_location:
                action = f"{action}/Door"
            elif state == env.terminal_state:
                action = 'Terminal'
            print(f"State ({i},{j}): {action}", end=' | ')
        print()
