# Simulation for bias (Fig. 5)

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import time
import os
import pickle

from distribution import *

fontsize=25
legendsize=20
ticksize=17.5
linewidth=2.5
markersize=10
markeredgewidth=4
axissize=17.5
colorbarsize=30

MARKERS = ['x','o', 's', '<', 'd']
LINESTYLES = ['solid', 'dotted','dashed','dashdot', (0, (1, 5))]
COLORS = ['C0', 'C1', 'C2','C3', 'C4']
PLOT_DIR = 'plots'
PICKLE_FILE = 'data_fairness.p'

TEXT_SEG_ACC = 'Accuracy\n(Segmented)'
TEXT_HOL_ACC = 'Accuracy\n(Holistic)'
TEXT_DIFF_ACC = 'Accuracy\n(Seg. - Hol.)'

TEXT_AXIS_DELTA = r'$\delta$' + '\nPower law distribution'

t = 1 # alpha value
DECIMALS = 10 # avoid numerical issues

# Run all simulations
def run_fairness():
	print('DEFAULT CONFIG:', default_config())
	deltas = np.around(np.linspace(0.05, 1, 20), decimals=DECIMALS)
	D = len(deltas)

	# set up colormap
	cmap = matplotlib.cm.get_cmap('seismic')

	tic = time.time()

	if os.path.isfile(PICKLE_FILE):
		(dct_hol, dct_seg) = pickle.load(open(PICKLE_FILE, 'rb'))
	else:
		dct_hol = {}
		dct_seg = {}

	## 1 -- (correlation, delta)
	print('1 -- (correlation, delta)')
	(n, d, corr, beta, alpha, lda, repeat) = default_config()

	corrs = np.around(np.linspace(0, 1, 21), decimals=DECIMALS)
	C = len(corrs)

	corrects_seg = np.zeros((C, D, repeat))
	corrects_hol = np.zeros((C, D, repeat))

	for idd in range(D):
		delta = deltas[idd]
		print('delta %d/%d (%.1f sec)' % (idd + 1, D, time.time() - tic))

		for ic in range(C):
			print('  corr %d/%d (%.1f sec)' % (ic+1, C, time.time() - tic))
			corr = corrs[ic]

			repeat_existing = 0
			config =  (n, d, delta, corr, beta, alpha, lda)
			if config in dct_hol: # hol and seg the same
				repeat_existing = len(dct_hol[config])
				repeat_copy = np.min((repeat, repeat_existing))
				print('    Loading %d/%d from pickle...' % (repeat_copy, repeat))

				corrects_hol[ic, idd, :repeat_copy] = dct_hol[config][:repeat_copy]
				corrects_seg[ic, idd, :repeat_copy] = dct_seg[config][:repeat_copy]

			if repeat_existing < repeat:
				corrects_hol[ic, idd, repeat_existing:repeat], corrects_seg[ic, idd, repeat_existing:repeat] = compute_fairness_accuracy(n, d, delta, corr, alpha, lda, beta, repeat=repeat - repeat_existing)

				if os.path.isfile(PICKLE_FILE):
					(dct_hol, dct_seg) = pickle.load(open(PICKLE_FILE, 'rb'))

				dct_hol[config] = corrects_hol[ic, idd, :]
				dct_seg[config] = corrects_seg[ic, idd, :]					
				pickle.dump((dct_hol, dct_seg), open(PICKLE_FILE, 'wb'))
				print('    Saved pickle.')

	accs_hol, accs_seg = np.mean(corrects_hol, axis=2), np.mean(corrects_seg, axis=2)
	sems_hol, sems_seg = np.array([np.std(corrects_hol, axis=2), np.std(corrects_seg, axis=2)]) / np.sqrt(repeat)

	better_seg = 1 * np.greater(accs_seg, accs_hol) - 1 * np.less(accs_seg, accs_hol - sems_hol)
	
	# make 2D plot
	(fig, axes) = plt.subplots(1, 3)

	# acc
	vmax = np.max(np.maximum(accs_hol, accs_seg))
	vmin = np.min(np.minimum(accs_hol, accs_seg))

	# plot seg
	ax = axes[0]
	im_seg = ax.imshow(accs_seg, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(C)[0::4])
	ax.set_yticklabels(['%.1f' % corr for corr in corrs[0::4]])
	ax.set_ylabel(r'$\sigma$'+'\nCorrelation')

	# plot hol
	ax = axes[1]
	im = ax.imshow(accs_hol, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(C)[0::4])
	ax.set_yticklabels([])

	dval = np.max(np.abs(accs_seg - accs_hol))

	# plot diff
	ax = axes[2]
	im_diff = ax.imshow(accs_seg - accs_hol, origin='lower', cmap=cmap, vmin=-dval, vmax=dval)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(C)[0::4])
	ax.set_yticklabels([])

	plt.savefig('%s/fairness_2d_sigma_repeat%d.pdf' % (PLOT_DIR, repeat), bbox_inches='tight')

	# plot colorbars
	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_seg, orientation='horizontal', shrink=2, aspect=20*2) # aspect (default): 20
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_sigma_cb.pdf' % (PLOT_DIR), bbox_inches='tight')

	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_diff, orientation='horizontal')
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_sigma_cb_diff.pdf' % (PLOT_DIR), bbox_inches='tight')

	## Plot titles
	# title seg
	fig, ax = plt.subplots()
	ax.set_title(TEXT_SEG_ACC)
	ax.set_axis_off()
	plt.savefig('%s/fairness_2d_title_seg.pdf' % (PLOT_DIR), bbox_inches='tight')

	# title hol
	fig, ax = plt.subplots()
	ax.set_title(TEXT_HOL_ACC)
	ax.set_axis_off()
	plt.savefig('%s/fairness_2d_title_hol.pdf' % (PLOT_DIR), bbox_inches='tight')

	# title diff
	fig, ax = plt.subplots()
	ax.set_title(TEXT_DIFF_ACC)
	ax.set_axis_off()
	plt.savefig('%s/fairness_2d_title_diff.pdf' % (PLOT_DIR), bbox_inches='tight')

	## 2 -- (beta, delta)
	print('2 -- (beta, delta)')
	(n, d, corr, beta, alpha, lda, repeat) = default_config()

	betas = np.around(np.linspace(0, 0.95, 20), decimals=DECIMALS)
	B = len(betas)

	corrects_seg = np.zeros((B, D, repeat))
	corrects_hol = np.zeros((B, D, repeat))

	for idd in range(D):
		delta = deltas[idd]
		print('delta %d/%d (%.1f sec)' % (idd+1, D, time.time() -tic))

		for ib in range(B):
			beta = betas[ib]
			print('  beta %d/%d (%.1f sec)' % (ib + 1, B, time.time() - tic))

			repeat_existing = 0
			config =  (n, d, delta, corr, beta, alpha, lda)
			if config in dct_hol: # hol and seg the same
				repeat_existing = len(dct_hol[config])
				repeat_copy = np.min((repeat, repeat_existing))
				print('    Loading %d/%d from pickle...' % (repeat_copy, repeat))

				corrects_hol[ib, idd, :repeat_copy] = dct_hol[config][:repeat_copy]
				corrects_seg[ib, idd, :repeat_copy] = dct_seg[config][:repeat_copy]

			if repeat_existing < repeat:
				corrects_hol[ib, idd, repeat_existing:repeat], corrects_seg[ib, idd, repeat_existing:repeat] = compute_fairness_accuracy(n, d, delta, corr, alpha, lda, beta, repeat=repeat - repeat_existing)
				
				if os.path.isfile(PICKLE_FILE):
					(dct_hol, dct_seg) = pickle.load(open(PICKLE_FILE, 'rb'))

				dct_hol[config] = corrects_hol[ib, idd, :]
				dct_seg[config] = corrects_seg[ib, idd, :]					
				pickle.dump((dct_hol, dct_seg), open(PICKLE_FILE, 'wb'))
				print('    Saved pickle.')

	accs_hol, accs_seg = np.mean(corrects_hol, axis=2), np.mean(corrects_seg, axis=2)
	sems_hol, sems_seg = np.array([np.std(corrects_hol, axis=2), np.std(corrects_seg, axis=2)]) / np.sqrt(repeat)
				
	better_seg = 1 * np.greater(accs_seg, accs_hol) - 1 * np.less(accs_seg, accs_hol)
	# make 2D plot
	(fig, axes) = plt.subplots(1, 3)

	# acc
	vmax = np.max(np.maximum(accs_hol, accs_seg))
	vmin = np.min(np.minimum(accs_hol, accs_seg))

	# plot seg
	ax = axes[0]
	im_seg = ax.imshow(accs_seg, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(B)[::4])
	ax.set_yticklabels(['%.1f' % beta for beta in betas[::4]])
	ax.set_ylabel(r'$\beta$' + '\nDiscount')

	# plot hol
	ax = axes[1]
	im = ax.imshow(accs_hol, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(B)[::4])
	ax.set_yticklabels([])

	dval = np.max(np.abs(accs_seg - accs_hol))
	# plot diff
	ax = axes[2]
	im_diff = ax.imshow(accs_seg - accs_hol, origin='lower', cmap=cmap, vmin=-dval, vmax=dval)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(B)[::4])
	ax.set_yticklabels([])

	plt.savefig('%s/fairness_2d_beta_repeat%d.pdf' % (PLOT_DIR, repeat), bbox_inches='tight')

	# plot colorbars
	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_seg, orientation='horizontal', shrink=2, aspect=20*2)
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_beta_cb.pdf' % (PLOT_DIR), bbox_inches='tight')

	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_diff, orientation='horizontal')
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_beta_cb_diff.pdf' % (PLOT_DIR), bbox_inches='tight')

	## 3 -- (alpha, delta)
	print('3 -- (alpha, delta)')
	(n, d, corr, beta, alpha, lda, repeat) = default_config()

	alphas = np.around(np.linspace(0.05, 1, 20), decimals=DECIMALS)
	A = len(alphas)

	corrects_seg = np.zeros((A, D, repeat))
	corrects_hol = np.zeros((A, D, repeat))

	for idd in range(D):
		delta = deltas[idd]
		print('delta %d/%d (%.1f sec)' % (idd+1, D, time.time() - tic))

		for a in range(A):
			alpha = alphas[a]
			print('  alpha %d/%d (%.1f sec)' % (a + 1, A, time.time() - tic))

			repeat_existing = 0
			config =  (n, d, delta, corr, beta, alpha, lda)
			if config in dct_hol: # hol and seg the same
				repeat_existing = len(dct_hol[config])
				repeat_copy = np.min((repeat, repeat_existing))
				print('    Loading %d/%d from pickle...' % (repeat_copy, repeat))

				corrects_hol[a, idd, :repeat_copy] = dct_hol[config][:repeat_copy]
				corrects_seg[a, idd, :repeat_copy] = dct_seg[config][:repeat_copy]

			if repeat_existing < repeat:
				corrects_hol[a, idd, repeat_existing:repeat], corrects_seg[a, idd, repeat_existing:repeat] = compute_fairness_accuracy(n, d, delta, corr, alpha, lda, beta, repeat=repeat - repeat_existing)
				
				if os.path.isfile(PICKLE_FILE):
					(dct_hol, dct_seg) = pickle.load(open(PICKLE_FILE, 'rb'))

				dct_hol[config] = corrects_hol[a, idd, :]
				dct_seg[config] = corrects_seg[a, idd, :]					
				pickle.dump((dct_hol, dct_seg), open(PICKLE_FILE, 'wb'))
				print('    Saved pickle.')

	accs_hol, accs_seg = np.mean(corrects_hol, axis=2), np.mean(corrects_seg, axis=2)
	sems_hol, sems_seg = np.array([np.std(corrects_hol, axis=2), np.std(corrects_seg, axis=2)]) / np.sqrt(repeat)

	better_seg = 1 * np.greater(accs_seg, accs_hol) - 1 * np.less(accs_seg, accs_hol)
	# make 2D plot
	(fig, axes) = plt.subplots(1, 3)

	# acc
	vmax = np.max(np.maximum(accs_hol, accs_seg))
	vmin = np.min(np.minimum(accs_hol, accs_seg))

	# plot seg
	ax = axes[0]
	im_seg = ax.imshow(accs_seg, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(A)[3::4])
	ax.set_yticklabels(['%.1f' % alpha for alpha in alphas[3::4]])
	ax.set_ylabel(r'$\alpha$' +' (%)\nDisadv. applicants')

	# plot hol
	ax = axes[1]
	im = ax.imshow(accs_hol, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(A)[3::4])
	ax.set_yticklabels([])

	dval = np.max(np.abs(accs_seg - accs_hol))
	# plot diff
	ax = axes[2]
	im_diff = ax.imshow(accs_seg - accs_hol, origin='lower', cmap=cmap, vmin=-dval, vmax=dval)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_yticks(np.arange(A)[3::4])
	ax.set_yticklabels([])

	plt.savefig('%s/fairness_2d_alpha_repeat%d.pdf' % (PLOT_DIR, repeat), bbox_inches='tight')

	# plot colorbars
	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_seg, orientation='horizontal', shrink=2, aspect=20*2) # aspect (default): 20
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_alpha_cb.pdf' % (PLOT_DIR), bbox_inches='tight')

	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_diff, orientation='horizontal')
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_alpha_cb_diff.pdf' % (PLOT_DIR), bbox_inches='tight')

	## 4 -- (lda, delta)
	print('4 -- (lambda, delta)')
	(n, d, corr, beta, alpha, lda, repeat) = default_config()

	ldas = np.around(np.linspace(0.05, 1, 20), decimals=DECIMALS)
	L = len(ldas)

	corrects_seg = np.zeros((L, D, repeat))
	corrects_hol = np.zeros((L, D, repeat))

	for idd in range(D):
		delta = deltas[idd]
		print('delta %d/%d (%.1f sec)' % (idd + 1, D, time.time() - tic))

		for il in range(L):
			print('  lambda %d/%d (%.1f sec)' % (il+1, L, time.time() - tic))
			lda = ldas[il]

			repeat_existing = 0
			config =  (n, d, delta, corr, beta, alpha, lda)
			if config in dct_hol: # hol and seg the same
				repeat_existing = len(dct_hol[config])
				repeat_copy = np.min((repeat, repeat_existing))
				print('    Loading %d/%d from pickle...' % (repeat_copy, repeat))

				corrects_hol[il, idd, :repeat_copy] = dct_hol[config][:repeat_copy]
				corrects_seg[il, idd, :repeat_copy] = dct_seg[config][:repeat_copy]

			if repeat_existing < repeat:
				corrects_hol[il, idd, repeat_existing:repeat], corrects_seg[il, idd, repeat_existing:repeat] = compute_fairness_accuracy(n, d, delta, corr, alpha, lda, beta, repeat=repeat - repeat_existing)
				
				if os.path.isfile(PICKLE_FILE):
					(dct_hol, dct_seg) = pickle.load(open(PICKLE_FILE, 'rb'))

				dct_hol[config] = corrects_hol[il, idd, :]
				dct_seg[config] = corrects_seg[il, idd, :]					
				pickle.dump((dct_hol, dct_seg), open(PICKLE_FILE, 'wb'))
				print('    Saved pickle.')

	accs_hol, accs_seg = np.mean(corrects_hol, axis=2), np.mean(corrects_seg, axis=2)
	sems_hol, sems_seg = np.array([np.std(corrects_hol, axis=2), np.std(corrects_seg, axis=2)]) / np.sqrt(repeat)

	better_seg = 1 * np.greater(accs_seg, accs_hol) - 1 * np.less(accs_seg, accs_hol)
	# make 2D plot
	(fig, axes) = plt.subplots(1, 3)

	# acc
	vmax = np.max(np.maximum(accs_hol, accs_seg))
	vmin = np.min(np.minimum(accs_hol, accs_seg))

	# plot seg
	ax = axes[0]
	im_seg = ax.imshow(accs_seg, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_xlabel(TEXT_AXIS_DELTA)
	ax.set_yticks(np.arange(L)[3::4])
	ax.set_yticklabels(['%.1f' % lda for lda in ldas[3::4]])
	ax.set_ylabel(r'$\lambda$' + ' (%)\nProtected attributes')

	# plot hol
	ax = axes[1]
	im = ax.imshow(accs_hol, origin='lower', vmin=vmin, vmax=vmax)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_xlabel(TEXT_AXIS_DELTA)
	ax.set_yticks(np.arange(L)[3::4])
	ax.set_yticklabels([])

	dval = np.max(np.abs(accs_seg - accs_hol))
	# plot diff
	ax = axes[2]
	im_diff = ax.imshow(accs_seg - accs_hol, origin='lower', cmap=cmap, vmin=-dval, vmax=dval)
	ax.set_xticks(np.arange(D)[3::4])
	ax.set_xticklabels(['%.1f' % delta for delta in deltas[3::4]])
	ax.set_xlabel(TEXT_AXIS_DELTA)
	ax.set_yticks(np.arange(L)[3::4])
	ax.set_yticklabels([])

	plt.savefig('%s/fairness_2d_lda_repeat%d.pdf' % (PLOT_DIR, repeat), bbox_inches='tight')

	# plot colorbars
	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_seg, orientation='horizontal', shrink=2, aspect=20*2)
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_lda_cb.pdf' % (PLOT_DIR), bbox_inches='tight')

	(fig_cb, ax_cb) = plt.subplots()
	cb = fig_cb.colorbar(im_diff, orientation='horizontal')
	cb.ax.tick_params(labelsize=colorbarsize)
	ax_cb.remove()
	plt.savefig('%s/fairness_2d_lda_cb_diff.pdf' % (PLOT_DIR), bbox_inches='tight')

	plt.show()

# Compute accuracy in a specific given configuration
# 2 reviewers (1 biased + 1 unbiased)
# Configuration: (n, d, delta, corr, alpha, lda, beta)
# 	DELTA: power law parameter
# 	CORR: correlation between two attributes
# 	ALPHA: fraction of disadvantaged applicants (rounded to nearest integer)
# 	LAMBDA: fraction of protected attributes (rounded to nearest integer)
# 	BETA: discount factor
def compute_fairness_accuracy(n, d, delta, corr, alpha, lda, beta, repeat):
	n_reviewer = 2
	frac_biased_reviewer = 0.5
	n_biased_reviewer = int(np.round(n_reviewer * frac_biased_reviewer)) # 1 biased reviewer | 1 unbiased
	assert(n % n_reviewer == 0)
	assert(d % n_reviewer == 0) # either horizontal or vertical chunks

	dist_powerlaw = DistPowerLaw(delta, corr=corr)

	corrects_hol = np.zeros(repeat)
	corrects_seg = np.zeros(repeat)
	for r in range(repeat):
		mtx = dist_powerlaw.sample(size=(n, d))

		argmax_true = np.argmax(np.mean(mtx, axis=1))

		# disadvantaged applicant
		n_disadvantage = int(np.round(n * alpha))
		is_disadvantage = np.full(n, False)
		is_disadvantage[np.random.choice(n, size=n_disadvantage, replace=False)] = True

		d_protected = int(np.round(d * lda))
		is_protected = np.full(d, False)
		is_protected[np.random.choice(d, size=d_protected, replace=False)] = True

		# mask of (disadvantaged x protected)
		mask_discount = np.tile(is_disadvantage[:, np.newaxis], (1, d))
		mask_discount[:, np.logical_not(is_protected)] = False

		# mask of biased reviewer
		# seg
		mask_seg = np.full((n, d), False)
		mask_seg[:, :int(d / n_reviewer * n_biased_reviewer)] = True

		# hol
		mask_hol = np.full((n, d), False)
		mask_hol[:int(n / n_reviewer * n_biased_reviewer), :] = True

		# construct discount factor (1 for undiscounted, and beta for discounted)
		factors_seg = np.ones((n, d)) - (1-beta) * np.logical_and(mask_discount, mask_seg)
		factors_hol = np.ones((n, d)) - (1-beta) * np.logical_and(mask_discount, mask_hol)

		means_seg = np.mean(mtx * factors_seg, axis=1)
		means_hol = np.mean(mtx * factors_hol, axis=1)

		corrects_hol[r] = (np.argmax(means_hol) == argmax_true)
		corrects_seg[r] = (np.argmax(means_seg) == argmax_true)

	return corrects_hol, corrects_seg

# Return the default configuration
def default_config():
	n = 20
	d = 20

	corr = 0.5
	beta = 0. # discount
	alpha = 0.5
	lda = 1

	repeat = 50000

	return n, d, corr, beta, alpha, lda, repeat

if __name__ == '__main__':
	np.random.seed(0)

	if not os.path.isdir(PLOT_DIR):
		print('mkdir %s...' % PLOT_DIR)
		os.mkdir(PLOT_DIR)

	run_fairness()
