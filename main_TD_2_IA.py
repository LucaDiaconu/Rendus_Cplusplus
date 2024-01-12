import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Met à jour la table Q selon l'algorithme Q-learning.
    """
    current_q_value = Q[s][a]
    max_future_q_value = max(Q[sprime])
    new_q_value = (1 - alpha) * current_q_value + alpha * (r + gamma * max_future_q_value)
    Q[s][a] = new_q_value
    return Q


def epsilon_greedy(Q, s, epsilon):
    """
    Implémente la politique epsilon-greedy pour la sélection d'actions.
    """
    if random.uniform(0, 1) < epsilon:
        # Exploration : choisir une action au hasard
        action = random.choice(range(len(Q[s])))
    else:
        # Exploitation : choisir l'action avec la plus grande valeur Q
        action = np.argmax(Q[s])
    return action

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    state = env.s
    env.render()

    # Définir l'ensemble des États
    states = [(taxi_x, taxi_y, passenger_loc, destination_loc)
              for taxi_x in range(5)
              for taxi_y in range(5)
              for passenger_loc in range(5)  # 0-4 représentent les 4 emplacements du passager
              for destination_loc in range(5)  # 0-4 représentent les 4 destinations possibles
              ]

    Q = np.zeros((len(states), env.action_space.n))

    alpha = 0.01  # choisissez la valeur que vous voulez
    gamma = 0.8  # choisissez la valeur que vous voulez
    epsilon = 0.2  # choisissez la valeur que vous voulez

    n_epochs = 20  # choisissez la valeur que vous voulez
    max_itr_per_epoch = 100  # choisissez la valeur que vous voulez
    rewards = []

    for e in range(n_epochs):
        r = 0

        env.reset()
        state = env.s

        for _ in range(max_itr_per_epoch):
            action = epsilon_greedy(Q=Q, s=state, epsilon=epsilon)

            next_state, reward, done, _, info = env.step(action)

            r += reward

            next_s = next_state  # Pas besoin de chercher l'index, car next_state est déjà l'indice unique

            Q = update_q_table(
                Q=Q, s=state, a=action, r=reward, sprime=next_s, alpha=alpha, gamma=gamma
            )

            state = next_s

            if done:
                break

        print("Épisode #", e, " : r = ", r)

        rewards.append(r)

    print("Récompense moyenne = ", np.mean(rewards))

    # tracer les récompenses en fonction des époques

    print("Entraînement terminé.\n")

    """

    Évaluer l'algorithme Q-learning

    """

    env.close()
