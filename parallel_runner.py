import numpy as np
import matplotlib.pyplot as plt
import datetime, time, json, sys, multiprocessing

from env import StochasticFrozenLake
from agents import UCBVI
from sleepingagents import SleepingUCBVI


def sliding_window_mean(arr, window_size):
    if window_size > len(arr) or window_size <= 0:
        raise ValueError("Array too short")

    result = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    padding = np.zeros(window_size - 1)
    return np.concatenate((padding, result))


def trial(grid, H, K, no_hole_prob, run_lst, use_reward=False, trial_id=1):

    np.random.seed(trial_id)

    env = StochasticFrozenLake(grid, no_hole_prob, H)
    if not use_reward:
        opt = env.computeOptimalValueFunction()
        opt_rewards = opt * np.ones(K)

    results_all = {}

    for alg in run_lst:
        start_time = time.time()
        time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")
        print(time_print + " Starting trial " + str(trial_id+1) + " for " + alg + ".")
        if alg == "UCBVI":
            agent = UCBVI(grid*grid, 5, H, K, env.getRewards())
        elif alg == "SleUCBVI":
            agent = SleepingUCBVI(grid*grid, 5, H, K, env.getRewards())
        else:
            raise ValueError("Error in input")
        rewards = np.zeros((K, H))
        for k in range(K):
            env.reset()
            agent.newEpisode()
            state = env.getCurrentState()
            for h in range(H):
                if alg == "SleUCBVI":
                    action = agent.choose(state, h, env.getAllowedActions())
                else:
                    action = agent.choose(state, h)
                state, reward = env.step(action)
                agent.update(state, reward)
                rewards[k, h] = reward
        ep_rewards = rewards.sum(axis=1)
        if not use_reward:
            metric = opt_rewards - ep_rewards
            metric = metric.cumsum()
        else:
            metric = ep_rewards
        time_print_end = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")
        print(time_print_end + " Terminating trial " + str(trial_id+1) + " for " + alg + ". Elapsed time: " + str(int(time.time() - start_time)) + " sec.")
        results_all[alg] = metric

    return results_all


if __name__ == '__main__':

    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)

    time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")

    grid = config["grid"]
    H = config["H"]
    K = config["K"]
    no_hole_prob = config["no_hole_prob"]
    alg_lst = config["alg_lst"]
    n_trials = config["n_trials"]
    n_cores = config["n_cores"]
    multiproc = n_cores != 1
    use_reward = True
    win_size = 1000
    config["use_reward"] = int(use_reward)

    print(f"{time_print} Starting run with config file : {str(config)}")

    if multiproc:
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(trial, [(grid, H, K, no_hole_prob, alg_lst, use_reward, i)
                                           for i in range(n_trials)])
    else:
        results = [trial(grid, H, K, no_hole_prob, alg_lst, use_reward, i) for i in range(n_trials)]

    results_dict = {alg : [] for alg in alg_lst}
    results_dict["config"] = config

    for i in range(len(results)):
        for alg in alg_lst:
            results_dict[alg].append(list(results[i][alg]))

    with open(f"results/rawdata_{str(config)}_time={time_print}.json", "w") as fp:
        json.dump(results_dict, fp, indent=4)

    plt.figure()

    env = StochasticFrozenLake(grid, no_hole_prob, H)
    opt = env.computeOptimalValueFunction()

    for alg in alg_lst:

        x_plt = np.linspace(0, K-1, K, dtype=int) # take care of this for number of samples to visual
        results_alg = np.array(results_dict[alg]).T
        for i in range(n_trials):
            results_alg[:, i] = sliding_window_mean(results_alg[:, i], win_size)
        results_mean = results_alg.mean(axis=1)
        results_std = 1.96 * results_alg.std(axis=1) / np.sqrt(n_trials)
        plt.plot(x_plt, results_mean[x_plt], label=alg)
        plt.fill_between(x_plt, results_mean[x_plt] - results_std[x_plt], results_mean[x_plt] + results_std[x_plt], alpha=0.3)
        if use_reward:
            plt.ylabel("Instantanous Reward")
            plt.axhline(opt, color="green")
        else:
            plt.ylabel("Cumulative Regret")
        plt.xlabel("Episodes")

    plt.legend()
    plt.savefig(f"results/figplot_{str(config)}_time={time_print}.jpg")
    # tkz.save(f"results/figplot_{str(config)}_time={time_print}.tex")
