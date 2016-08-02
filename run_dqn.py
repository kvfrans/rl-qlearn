import gym
import os
import argparse
import q_conv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Breakout-v0", help="OpenAI gym env to run on")
    parser.add_argument("--submit", type=bool, default=False, help="record the trial for submission?")
    parser.add_argument("--render", type=bool, default=False, help="render the environment while training?")
    parser.add_argument("--episodes", type=int, default=2000, help="how many episodes to train for")
    parser.add_argument("--maxframes", type=int, default=200, help="max frames to run an episode for")
    parser.add_argument("--discount", type=float, default=0.95, help="discount factor for later timesteps")
    parser.add_argument("--batchsize", type=int, default=64, help="batches for experience replay")
    parser.add_argument("--training_iterations", type=int, default=50, help="how many times to train per episode")
    parser.add_argument("--updaterate", type=int, default=16, help="when to swap frozen and updated models")
    parser.add_argument("--epsilon_decay", type=float, default=0.99, help="scale down factor for epsilon-greedy")
    parser.add_argument("--memory_size", type=int, default=300, help="memory size for experience replay")
    parser.add_argument("--learningrate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--learningrate_decay", type=int, default=0.9995, help="decay learningrate by multiply")
    parser.add_argument("--history", type=int, default=4, help="how many pastframes to use")

    args = parser.parse_args()

    env = gym.make(args.env)
    if args.submit:
        env.monitor.start('monitor/', force=True)

    # q_basic.learn(env,args)
    q_conv.learn(env,args)

    if args.submit:
        env.monitor.close()
