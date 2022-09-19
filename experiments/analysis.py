import numpy as np

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import os
import csv

np.set_printoptions(precision=3, suppress=True)

fontsize=25
legendsize=20
ticksize=17.5
linewidth=2.5
markersize=10
markeredgewidth=4
axissize=17.5
handlelength=3

MARKERS = ['x','o', 's', '<', 'd']
LINESTYLES = ['solid', 'dotted','dashed','dashdot', (0, (1, 5))]
COLORS = ['C3', 'C9', 'C0']
PLOT_DIR = 'plots'


DATA_DIR = 'data'
CSV_FILE = 'expt_old.csv'
CSV_FILE = 'expt.csv'

CSV_FILE = DATA_DIR + '/' + CSV_FILE

NUM_QUESTION = 20
NUM_PAGE = 4
NUM_QUESTION_PER_PAGE = int(NUM_QUESTION / NUM_PAGE)
NUM_SETTING = 100 # 100 workers per group (2 groups in total)

GT_DIR = 'gt'
GT_PERCENT_FILE = '%s/NORMAL_TRUNCATE_200_300_230_25_size100_seed0_percent.csv' % GT_DIR

# DATA: dict
def read_csv(csv_file):
	with open(csv_file, 'r') as f:
		file = csv.DictReader(f)
		workers = []
		for row in file:
			workers.append(dict(row))

	workers = clean_format(workers)
	return workers

# Convert strings to numbers
def clean_format(workers):
	for w in workers:
		# int/float
		for k, v in w.items():
			if v is None:
				continue # in GT_PERCENT_FILE
			try:
				w[k] = int(v)
				continue
			except ValueError: pass

			try:
				w[k] = float(v)
			except ValueError: pass

	return workers

# Return:
# 	CORRECT_ATTENTION: people answering att'n check correctly the first round
# 	CORRECT_ATTENTION2: people answering att'n check wrong the first round, and correct the second round
def parse_data(workers):
	W = len(workers)

	(history_answers, answers, history_times) = parse_responses(workers)

	# load buttons
	buttons = np.zeros((W, NUM_PAGE-1))
	history_buttons_back = np.full((W, NUM_PAGE-1), None)
	history_buttons_forward = np.full((W, NUM_PAGE-1), None)

	has_button_history = ('time_button_2_to_1' in workers[0])
	for iw in range(W):
		worker = workers[iw]
		for p in np.arange(NUM_PAGE-1):
			txt = 'button_%d_to_%d' % () # TODO -- make sure that first time turning forward is the same as first time turning backward

			# back button -- possible that it doesn't exist if a worker has never turned back
			txt = 'button_%d_to_%d' % (p+2, p+1)
			if worker[txt] == '':
				continue

			if has_button_history:
				history_buttons[iw, p] = parse_trace(worker['time_button_%d_to_%d' % (p+2, p+1)])

			assert(len(hisotry_buttons[iw, p] == worker[txt])) # number of clicks equals length of history


	num_answered = np.sum(np.logical_not(np.isnan(answers)), axis=1)

	# load setting
	settings = np.full(W, -1)
	for iw in range(W):
		if workers[iw]['setting'] == '':
			continue
		settings[iw] = workers[iw]['setting']

	# correct_attention 
	correct_attention = np.full(W, False)
	correct_attention2 = np.full(W, False) # answer correctly in the second chance

	gt = [2, 1, 2] # correct answers for attention check
	G = len(gt)
	answers_attn = np.zeros((W, G))

	# attention (round 1)
	for iw in range(W):
		w = workers[iw]

		for q in range(G):
			trace = parse_trace(w['history_a%d'% (q+1)]) # attention questions 1-idxed
			if trace is not None:
				answers_attn[iw, q] = trace[-1]

	correct_attention = np.all(np.equal(answers_attn, gt), axis=1)

	# attention (round 2)
	answers_attn2 = np.zeros((W, G))
	for iw in range(W):
		w = workers[iw]
		for q in range(G):
			trace = parse_trace(w['history_a%d_fail' % (q+1)]) # attention questions 1-idxed
			if trace is not None:
				answers_attn2[iw, q] = trace[-1]

	correct_attention2 = np.all(np.equal(answers_attn2, gt), axis=1)

	# complete
	select = np.logical_or(num_answered==5, num_answered==20) # 5 or 20
	select = np.logical_and(select, settings != -1)

	num_answered = num_answered[select]
	correct_attention = correct_attention[select]
	correct_attention2 = correct_attention2[select]
	answers = answers[select, :]
	settings = settings[select]
	history_answers = history_answers[select, :]
	history_times = history_times[select, :]
	history_buttons = history_buttons[select, :]

	buttons = buttons[select, :]


	return num_answered, correct_attention, correct_attention2,\
		answers, settings, history_answers, history_times, buttons, history_buttons

def parse_responses(workers):
	W = len(workers)

	# parse history
	history_answers = np.full((W, NUM_QUESTION), None)
	for iw in range(W):
		for q in range(NUM_QUESTION):
			history_answers[iw, q] = parse_trace(workers[iw]['history_q%d'%(q+1)])

	# convert history to answer -- answer is the last response in the history
	answers = np.full((W, NUM_QUESTION), np.nan)
	for iw in range(W):
		for q in range(NUM_QUESTION):
			if history_answers[iw, q] is not None:
				answers[iw, q] = history_answers[iw, q][-1]

	# parse time
	history_times = np.full((W, NUM_QUESTION), None)
	for iw in range(W):
		for q in range(NUM_QUESTION):
			history_times[iw, q] = parse_trace(workers[iw]['time_q%d'% (q + 1)])

	return (history_answers, answers, history_times)

# TRACE: comma-separated string to int np.array
def parse_trace(trace):
	if trace == '':
		return None

	try: # one number
		trace = int(trace)
		trace = np.array([trace], dtype=int)
	except ValueError: # multipe numbers
		trace = trace.split(',')
		trace = [t.strip() for t in trace]
		trace = tuple(map(int, trace))
		trace = np.array(trace, dtype=int)

	return trace

# Read from CSV for percentiles
# Return: GT percentiles
def read_gt(csv_file):

	# load gt
	gt = np.zeros((NUM_SETTING, NUM_QUESTION))

	with open(csv_file, 'r') as f:
		file = csv.DictReader(f)
		workers = []
		for row in file:
			workers.append(dict(row))

	workers = clean_format(workers)

	for i in range(NUM_SETTING):
		for q in range(NUM_QUESTION):
			gt[i, q] = workers[i]['Q_%d' % (q+1)] # 1-indexed questions

	return gt

# QS1, QS2: 0-indexed
# Return: True/False
def questions_on_different_pages(qs1, qs2):
	if np.isscalar(qs1):
		qs1 = [qs1]
	if np.isscalar(qs2):
		qs2 = [qs2]

	for i1 in range(len(qs1)):
		for i2 in range(len(qs2)):
			q1 = qs1[i1]
			q2 = qs2[i2]
			if questions_to_pages(q1) != questions_to_pages(q2):
				return True

	return False

# Q: 0-idxed question-idx scalar (or an array of question indices)
# Returns:
# 	P: 0-idxed {0, 1, 2, 3}
def questions_to_pages(q):
	return q // NUM_QUESTION_PER_PAGE

# P: {0, 1, 2, 3}
# Return: 0-indexed questions
def page_to_questions(p):
	qs = np.arange( p*NUM_QUESTION_PER_PAGE, (p+1) * NUM_QUESTION_PER_PAGE )
	return qs.astype(int)

# Compute accuracy as if the first response is actual answer
def analyze_edit_history(num_answered, settings, gt, history_answers, history_times, answers, buttons, history_buttons):
	W = len(settings)
	answers_final = np.full((W, NUM_QUESTION), np.nan)

	for iw in range(W):
		for q in range(NUM_QUESTION):
			if history_answers[iw, q] is not None:
				answers_final[iw, q] = history_answers[iw, q][-1] # final answer


	# Construct same page
	(questions_flatten, answers_flatten, times_flatten, _) = \
		flatten_worker_history(history_answers, history_times) # sorted w.r.t. time

	answers_same_page_proxy = np.full((W, NUM_QUESTION), np.nan) # include same page
	answers_same_page_corrected = np.full((W, NUM_QUESTION), np.nan) # 8/25/2022 corrected for page turning proxy

	final_clicks = np.zeros((W, NUM_QUESTION), dtype=int)

	for iw in range(W):
		assert(np.all(np.diff(times_flatten[iw]) >= 0))
		questions_worker = questions_flatten[iw]
		answers_worker = answers_flatten[iw]
		L = len(questions_worker)

		questions_unique = np.unique(questions_worker) # 0-idxed Q
		for q in questions_unique:

			idxs = np.where(questions_worker == q)[0]
			# final click
			final_clicks[iw, q]= idxs[-1]
			assert(answers_worker[final_clicks[iw, q]] == answers_final[iw, q])

 			# same page - proxy (last idx before ever seeing next page)
			p = questions_to_pages(q) # 0-idxed Q, 0-idxed page
			q_last = page_to_questions(p)[-1] # 0-idxed Q + next page hasn't been responded at all
			locations = np.where(questions_flatten[iw] > q_last)[0] # further pages
			if len(locations) == 0:
				idx = idxs[-1]
			else: # cut
				location = locations[0] # first time cliking on next page
				idxs_cut = idxs[idxs < location]
				idx = idxs_cut[-1]

			answers_same_page_proxy[iw, q] = answers_flatten[iw][idx]

			# same page (only working for the expt.csv and not expt_old.csv)
			p = questions_to_pages(q) # 0-idxed page
			back_first_time = np.inf
			if p < NUM_PAGE - 1:
				if history_buttons[iw, p] is not None:
					back_first_time = history_buttons[iw, p][0]

			select_before = (history_times[iw, q] < back_first_time)
			answers_same_page_corrected[iw, q] = history_answers[iw, q][select_before][-1]

	print('\n====== Accuracy (same page -- proxy) ======')
	errs_bin_same_proxy = compute_acc_quintiles(answers_same_page_proxy, num_answered, settings, gt)
	compute_accs(errs_bin_same_proxy, num_answered)

	print('\n====== Accuracy (same page) ======')
	errs_bin_same_correct = compute_acc_quintiles(answers_same_page_corrected, num_answered, settings, gt)
	compute_accs(errs_bin_same_correct, num_answered)

	print('\n====== Accuracy (final) ======')
	errs_bin_final = compute_acc_quintiles(answers_final, num_answered, settings, gt)
	compute_accs(errs_bin_final, num_answered)

	if np.all(history_buttons == None): # expt_old.csv
		return errs_bin_same_proxy, errs_bin_final
	else: # expt.csv
		return errs_bin_same_correct, errs_bin_final

# Concatenate all answers from all workers in a 1d array
# Return:
# 	QUESTIONS_FLATTEN: 0-idx
# 					   15-19 for 5Q | 0-19 for 20Q
def flatten_worker_answers(num_answered, answers, settings, gt):
	W = len(num_answered)

	num_answered_flatten = np.zeros(0)
	answers_flatten = np.zeros(0)
	gt_flatten = np.zeros(0)
	idxs_worker_flatten = np.zeros(0)
	questions_flatten = np.zeros(0)
	settings_flatten = np.zeros(0)

	for iw in range(W):
		answers_worker = answers[iw, :]
		setting = settings[iw]
		gt_worker = gt[setting]
		
		questions_worker = np.arange(NUM_QUESTION)

		if num_answered[iw] == 5:
			gt_worker = gt_worker[-5:]
			answers_worker = answers_worker[-5:]
			questions_worker = questions_worker[-5:]

		L = len(gt_worker) # 5 or 20

		num_answered_flatten = np.concatenate((num_answered_flatten, np.ones(L) * num_answered[iw]))
		answers_flatten = np.concatenate((answers_flatten, answers_worker))
		gt_flatten = np.concatenate((gt_flatten, gt_worker))
		idxs_worker_flatten = np.concatenate((idxs_worker_flatten, np.ones(L) * iw))
		settings_flatten = np.concatenate((settings_flatten, np.ones(L) * setting))

		questions_flatten = np.concatenate((questions_flatten, questions_worker))
	return num_answered_flatten, answers_flatten, questions_flatten, gt_flatten, idxs_worker_flatten, settings_flatten

# Each worker's response is a 1d-array (flatten by the separate questions), sorted in the chronological order of clicks
# Return:
# 	QUESTIONS_FLATTEN: 0-idxed question idxs
# 	PAGES_FLATTEN: 0-idxed pages (0-3 for 20Q and 3 for 5Q)
def flatten_worker_history(history_answers, history_times):
	W = history_answers.shape[0]

	times_flatten = np.full(W, None)
	questions_flatten = np.full(W, None)
	answers_flatten = np.full(W, None)
	pages_flatten = np.full(W, None)

	for iw in range(W):
		times_worker = np.array([])
		questions_worker = np.array([])
		answers_worker = np.array([])
		pages_worker = np.array([])

		for iq in range(NUM_QUESTION):
			assert (history_answers[iw, iq] is not None) == (history_times[iw, iq] is not None)
			if history_answers[iw, iq] is not None:
				L = len(history_answers[iw, iq])
				questions_worker = np.concatenate((questions_worker, np.ones(L) * iq )) # 0-idxed QUESTIONS_FLATTEN
				pages_worker = np.concatenate((pages_worker, np.ones(L) * questions_to_pages(iq))) # (0-19 or 15-19) -> 0-idxed page
				times_worker = np.concatenate((times_worker, history_times[iw, iq]))
				answers_worker = np.concatenate((answers_worker, history_answers[iw, iq]))

		# sort chronologically
		idxs = np.argsort(times_worker)

		times_worker = times_worker[idxs]
		questions_worker = questions_worker[idxs].astype(int)
		pages_worker = pages_worker[idxs].astype(int)
		answers_worker = answers_worker[idxs].astype(int)

		times_flatten[iw] = times_worker
		questions_flatten[iw] = questions_worker
		pages_flatten[iw] = pages_worker
		answers_flatten[iw] = answers_worker

	return questions_flatten, answers_flatten, times_flatten, pages_flatten

# PERCENTS: (any shape) percent in [0, 1]
def percentile_to_bins(percents):
	return np.floor(percents * 100 / 20).astype(int) + 1

# 5 settings (for GT)
def compute_acc_quintiles(answers, num_answered, settings, gt_percentile, verbose=False):
	W = len(settings)

	gt_bins = percentile_to_bins(gt_percentile)

	# compute the error
	errs_bin = np.zeros((W, NUM_QUESTION))
	for iw in range(W):
		setting = settings[iw]

		errs_bin[iw, :] = np.abs(answers[iw, :] - gt_bins[setting, :])

	if verbose:
		print("~~~Spearman's footrule~~~ (Size %d)" % len(num_answered))
		compute_accs(errs_bin, num_answered)

	return errs_bin

# ERRS: (W x Q)
def plot_errs(errs, errs_bin_final, num_answered):
	(fig, ax) = plt.subplots()

	(err_5q, sem_5q), (err_20q, sem_20q), (errs_page, sems_page), d = compute_accs(errs, num_answered, verbose=False)
	(err_5q, sem_5q), (err_20q, sem_20q), (errs_final_page, sems_final_page), (d_5q_vs_20q, d_page_first_vs_last) = compute_accs(errs_bin_final, num_answered, verbose=False)
	# plot 5Q
	ax.errorbar([0-0.04], [err_5q], yerr=sem_5q,
				label='5Q', color=COLORS[0],
				marker=MARKERS[0], linestyle='None', markersize=markersize, 
				markeredgewidth=markeredgewidth, linewidth=linewidth)

	# Plot 20Q (init)
	xvals = np.arange(NUM_PAGE, dtype=float)
	xvals[:-1] += 0.04
	plt.errorbar(xvals, errs_page, yerr=sems_page,
				label='20Q-initial', color=COLORS[1],
				marker=MARKERS[1], linestyle=LINESTYLES[1], markersize=markersize, 
				markeredgewidth=markeredgewidth, linewidth=linewidth)
	# 20Q (final)
	xvals = np.arange(NUM_PAGE, dtype=float)
	# xvals -= 0.02
	plt.errorbar(xvals, errs_final_page, yerr=sems_final_page,
				label='20Q', color=COLORS[2],
				marker=MARKERS[2], linestyle=LINESTYLES[0], markersize=markersize, 
				markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.set_ylim([0, None])
	plt.xticks(np.arange(4), ['Q1-5', 'Q6-10', 'Q11-15', 'Q16-20'], fontsize=axissize)
	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)

	plt.xlabel("Page", fontsize=fontsize)

	plt.ylabel("Mean error", fontsize=fontsize)

	plt.legend(fontsize=legendsize, handlelength=handlelength)
	# plt.savefig('%s/experiment.pdf' % PLOT_DIR, bbox_inches='tight')
	plt.savefig('%s/experiment_old.pdf' % PLOT_DIR, bbox_inches='tight')

	plt.show()

	# perm test
	errs_5q = errs[num_answered==5, :]
	errs_20q = errs[num_answered==20, :]

	errs_20q_early = errs[num_answered==20, :5]
	errs_20q_late = errs[num_answered==20, -5:]
	
	print('\n====== Test (5q vs 20q) ======')
	print('p-value: %.3f' % perm_test(errs_5q.flatten(), errs_20q.flatten()))
	print('effect size: %.3f' % d_5q_vs_20q)

	print('\n====== Test (first vs last page) ======')
	print('p-value: %.3f' % perm_test(errs_20q_early.flatten(), errs_20q_late.flatten()))
	print('effect size: %.3f' % d_page_first_vs_last)

# Compute the statistics for the *mean* accuracy of workers
# Input:
# 	ERRS: W x 20
# Output:
# 	ERR_5/20Q, SEM_5/20Q: scalar
# 	ERRS_PAGE: 4-dim vector for 20Q workers
def compute_accs(errs, num_answered, verbose=True):
	# filter
	errs_5q = errs[num_answered==5, -5:]
	errs_20q = errs[num_answered==20, :]

	errs_5q_per_worker = np.mean(errs_5q, axis=1)
	err_5q = np.mean(errs_5q_per_worker)
	sem_5q = np.std(errs_5q_per_worker) / np.sqrt(len(errs_5q_per_worker))

	errs_20q = errs[num_answered==20, :]
	errs_20q_per_worker = np.mean(errs_20q, axis=1)
	err_20q = np.mean(errs_20q_per_worker)
	sem_20q = np.std(errs_20q_per_worker) / np.sqrt(len(errs_20q_per_worker))

	d_5q_vs_20q = compute_effect_size(errs_5q_per_worker, errs_20q_per_worker)

	if verbose:
		print('err (5q): %.2f (+/- %.2f)' % (err_5q, sem_5q))
		print('err (20q): %.2f (+/- %.2f)' % (err_20q, sem_20q))

	errs_page = np.zeros(NUM_PAGE)
	sems_page = np.zeros(NUM_PAGE)

	for p in range(NUM_PAGE):
		vals_page = errs_20q[:, 5*p : 5 * (p+1)]
		vals_page_per_worker = np.mean(vals_page, axis=1)
		(errs_page[p], sems_page[p]) = np.mean(vals_page_per_worker), np.std(vals_page_per_worker) / np.sqrt(len(vals_page_per_worker))

		if verbose:
			print('  err (20q - page %d): %.2f (+/- %.2f)' % (p+1, errs_page[p], sems_page[p]))

	vals_first = errs_20q[:, 0:5]
	vals_last = errs_20q[:, -5:]
	vals_per_worker_first = np.mean(vals_first, axis=1)
	vals_per_worker_last = np.mean(vals_last, axis=1)
	d_page_first_vs_last = compute_effect_size(vals_per_worker_first, vals_per_worker_last)

	return (err_5q, sem_5q), (err_20q, sem_20q), (errs_page, sems_page), (d_5q_vs_20q, d_page_first_vs_last)

# https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
def compute_effect_size(arr1, arr2):
	n1 = len(arr1)
	n2 = len(arr2)
	mean1 = np.mean(arr1)
	mean2 = np.mean(arr2)

	s1_squared = np.sum(np.square(arr1-mean1)) / (n1-1)
	s2_squared = np.sum(np.square(arr2-mean2)) / (n2-1)

	s = np.sqrt( ((n1-1) * s1_squared + (n2-1) * s2_squared) / (n1+n2-2))
	d = (mean1 - mean2) / s
	return d

# filter all data according to CORRECT_ATTENTION{1,2} (answer correctly in either 1st or 2nd round)
def filter_attention(num_answered, correct_attention1, correct_attention2, answers, settings, history_answers, history_times, buttons, history_buttons):
	correct_attention = np.logical_or(correct_attention1, correct_attention2)

	num_answered = num_answered[correct_attention]
	answers = answers[correct_attention, :]
	settings = settings[correct_attention]
	history_answers = history_answers[correct_attention, :]
	history_times = history_times[correct_attention, :]
	buttons = buttons[correct_attention, :]
	history_buttons = history_buttons[correct_attention, :]

	return num_answered, answers, settings, history_answers, history_times, buttons, history_buttons

# Perform permutation test (unpaired, one-sided)
# 	H0: ys1 == ys2
# 	H1: ys1 > ys2
# Return: p-value
def perm_test(ys1, ys2, repeat=100000):
	n1 = len(ys1)
	n2 = len(ys2)
	diff_test = np.mean(ys1) - np.mean(ys2)

	ys = np.concatenate((ys1, ys2))
	count = 0
	for r in range(repeat):
		perm = np.random.permutation(n1 + n2)
		ys_permuted = ys[perm]

		diff = np.mean(ys_permuted[:n1]) - np.mean(ys_permuted[n1:])
		if diff > diff_test:
			count += 1

	return count / repeat

###
if __name__ == '__main__':

	if not os.path.isdir(PLOT_DIR):
		print('mkdir %s...' % PLOT_DIR)
		os.mkdir(PLOT_DIR)

	###### read data
	workers = read_csv(CSV_FILE)
	print('Number of workers (unfiltered): %d' % len(workers))

	(num_answered, correct_attention, correct_attention2, \
		answers, settings, history_answers, history_times, buttons, history_buttons) = parse_data(workers)

	gt = read_gt(GT_PERCENT_FILE)

	###### filter by attention check
	## Filter by correct attention (1st + 2nd round)
	(num_answered, answers, settings, history_answers, history_times, buttons, history_buttons) = \
		filter_attention(num_answered, correct_attention, correct_attention2, answers, settings, history_answers, history_times, buttons, history_buttons)

	print('Number of workers (filtered by attention check): %d (5Q %d | 20Q %d)' % (len(settings), np.sum(num_answered==5), np.sum(num_answered==20)))
	assert(len(np.unique(settings[num_answered==5]) == NUM_SETTING) and np.sum(num_answered == 5) == NUM_SETTING) # each setting appearing exactly once
	assert(len(np.unique(settings[num_answered==20]) == NUM_SETTING) and np.sum(num_answered == 5) == NUM_SETTING)

	###### analysis
	## Accuracy stats
	errs_bin_same, errs_bin_final = analyze_edit_history(num_answered, settings, gt, history_answers, history_times, answers, buttons, history_buttons)

	## Plot
	plot_errs(errs_bin_same, errs_bin_final, num_answered)
	print('\nDone.')
