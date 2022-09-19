# Generate pseudo-randomn numbers for Qualtrics

import numpy as np
import csv
import scipy.stats

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from enum import Enum

class DistType(Enum):
    UNIFORM_INT = 0 # low, high (both inclusive)
    NORMAL_TRUNCATE = 1 # low, high, mean, std

def generate_percentiles(pool, seed=0):
	np.random.seed(seed)
	return np.random.uniform(size=pool)

# NO_REPEATED_NUMBER_SAME_PAGE: no repeated number shown to workers within the same page
def write_csv_numbers(numbers, percentiles, num_per_worker, mode, num_worker_print=-1, num_question_print=-1, seed=0):
	L = len(numbers)
	assert(L % num_per_worker == 0)
	num_group = L // num_per_worker
	if num_worker_print >= 0:
		num_group = np.min((num_group, num_worker_print))

	if num_question_print == -1:
		num_question_print = num_per_worker

	str_mode = [mode[0].name,] + [str(x) for x in mode[1:]]
	# assume GT dir exist
	# allow repeat
	fname = 'gt/%s_size%d_seed%d.csv' % ('_'.join(str_mode), num_per_worker, seed)
	fname_frac = 'gt/%s_size%d_seed%d_percent.csv' % ('_'.join(str_mode), num_per_worker, seed)

	# write num + percent
	with open(fname, 'w') as csv_num, open(fname_frac, 'w') as csv_percent:
		writer_num = csv.writer(csv_num)
		writer_percent = csv.writer(csv_percent)


		writer_num.writerow(['',] + ['Q_%d' % (i+1) for i in range(num_per_worker)])
		writer_percent.writerow(['',] + ['Q_%d' % (i+1) for i in range(num_per_worker)])
		for g in range(num_group):
			writer_num.writerow(['worker_%d' % (g+1),] + list(numbers[ g * num_per_worker : g * num_per_worker + num_question_print]) )
			writer_percent.writerow(['worker_%d' % (g+1),] + list(percentiles[ g * num_per_worker : g * num_per_worker + num_question_print]) )

# MODE: {uniform, normal}
# (uniform, )
def convert_percentiles_to_numbers(percentiles, mode):
	print('Mode: %s' % mode[0].name)

	if mode[0] is DistType.UNIFORM_INT:
		(low, high) = mode[1:]

		size = high - low + 1
		numbers = percentiles * size
		numbers = np.floor(numbers) 
		numbers = numbers + low
		numbers = numbers.astype(int)

	elif mode[0] is DistType.NORMAL_TRUNCATE:
		(low, high, mean, std) = mode[1:]

		low = low - 0.499
		high = high + 0.499
		unit_low = (low - mean) / std
		unit_high = (high - mean) / std

		# unit (low, high), then scaled back as numbers * std + mean
		numbers = scipy.stats.truncnorm.ppf(percentiles, unit_low, unit_high, loc=mean, scale=std)
		numbers = np.around(numbers)
		numbers = numbers.astype(int)
	else:
		raise Exception('Unknown mode:' + str(mode))

	if True:
		print('Min: %d' % np.min(numbers))
		print('Max: %d' % np.max(numbers))
		plt.figure(1)
		plt.hist(numbers)
		plt.title(str(mode))
		plt.show()

	assert(np.all(np.logical_and(numbers <= high, numbers >= low)))
	return numbers

# Read numbers from CSV 
# Format and print for Qualtrics
def print_for_qualtrics(csv_file):

	NUM_QUESTION = 20
	NUM_SETTING = 100
	# load gt
	gt = np.zeros((NUM_SETTING, NUM_QUESTION), dtype=int)

	with open(csv_file, 'r') as f:
		file = csv.DictReader(f)
		workers = []
		for row in file:
			workers.append(dict(row))

	for i in range(NUM_SETTING):
		for q in range(NUM_QUESTION):
			gt[i, q] = int(workers[i]['Q_%d' % (q+1)]) # 1-indexed questions

	for q in range(NUM_QUESTION):
		print('Q%d' % (q+1))
		print(np.array2string(gt[:, q], separator=', ') + ';')
		print('================================================')
	return gt


if __name__ == '__main__':

	POOL = 20000 # just some large number
	NUM_PER_WORKER = 100 # number of questions per worker (another relatively large number)

	seed = 0
	## Distributions
	MODE = (DistType.NORMAL_TRUNCATE, 200, 300, 230, 25)

	num_worker_print=100
	num_question_print=20


	percentiles = generate_percentiles(pool=POOL, seed=seed)
	numbers = convert_percentiles_to_numbers(percentiles, MODE)

	if False:
		write_csv_numbers(numbers, percentiles, num_per_worker=NUM_PER_WORKER, mode=MODE, 
					  num_worker_print=num_worker_print, num_question_print=num_question_print, seed=seed)
	if True:
		csv_file = 'gt/NORMAL_TRUNCATE_200_300_230_25_size100_seed0.csv'
		print_for_qualtrics(csv_file)

	print('Done.')
