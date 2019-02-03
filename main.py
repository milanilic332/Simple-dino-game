import argparse

import ai_runner_easy
import ai_runner_hard

parser = argparse.ArgumentParser(description='Choose difficulty, if initial data is provided and if model has been made')

parser.add_argument('--difficulty', help='easy or hard')
parser.add_argument('--initial-data', help='if data has been provided')
parser.add_argument('--model', help='if model has been provided')

args = parser.parse_args()

if args.difficulty == 'easy':
    ai_runner_easy.run(args.initial_data == 'True', args.model == 'True')
elif args.difficulty == 'hard':
    ai_runner_hard.run(args.initial_data == 'True', args.model == 'True')
else:
    print('Difficulty should be either easy or hard')
    exit(1)
