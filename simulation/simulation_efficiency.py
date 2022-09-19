# Smulation for efficiency (Fig. 4)

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import os

from distribution import *

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
COLORS = ['C0', 'C1', 'C2','C3', 'C4']
PLOT_DIR = 'plots'

def run_efficiency():
	n = 200
	d = 2
	delta = 1

	repeat = 1000
	# x-axis: correlation
	corrs = [0, 0.25, 0.5, 0.75, 1]
	cutoffs = np.array([1, 10, 100])

	# x-axis: tau
	cutoffs = np.array([1, 20, 40, 80, 120, 160, 200])
	F = len(cutoffs)
	C = len(corrs)

	errs = np.zeros((C, F, repeat))
	workloads = np.zeros((C, F, repeat)) # fractions
	for ic in range(C):
		corr = corrs[ic]
		dist = DistPowerLaw(delta=delta, corr=corr)

		for r in range(repeat):
			mtx = dist.sample((n, d))
			scores_gt = np.mean(mtx, axis=1)

			for iff in range(F):
				cutoff = cutoffs[iff]
				# early pruning
				thresh = np.sort(mtx[:, 0])[::-1][cutoff-1]
				idxs_prune = mtx[:, 0] < thresh
				mtx_prune = np.copy(mtx)
				mtx_prune[idxs_prune, :] = -np.inf
				scores_prune = np.mean(mtx_prune, axis=1)

				argmax_true = np.argmax(scores_gt)
				errs[ic, iff, r] = (np.argmax(scores_prune) != argmax_true)
				workloads[ic, iff, r] = (n * d - np.sum(idxs_prune)) / (n*d)

	EPS = 0.05
	(fig, ax) = plt.subplots()
	for c in range(C):
		idx = C - 1 - c
		workloads = cutoffs / n
		ax.errorbar(workloads, 1 - np.mean(errs[idx, :, :], axis=1), np.std(errs[idx, :, :], axis=1) / np.sqrt(repeat),
					label=r'$\sigma=$' + ' ' + '%s' % str(corrs[idx]),
					color=COLORS[c], marker=MARKERS[c],
					linestyle=LINESTYLES[c], linewidth=linewidth, 
					markersize=markersize, markeredgewidth=markeredgewidth)

	ax.set_ylim([0-EPS, 1+EPS])
	ax.set_xlabel('% Applicants evaluated for the second attr. ' + r'($\tau$)', fontsize=axissize)
	ax.set_ylabel('Top-1 accuracy', fontsize=axissize)
	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)

	ax.legend(fontsize=legendsize, handlelength=handlelength)
	plt.savefig('%s/efficiency.pdf' % (PLOT_DIR), bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	# np.random.seed(0)

	if not os.path.isdir(PLOT_DIR):
		print('mkdir %s...' % PLOT_DIR)
		os.mkdir(PLOT_DIR)

	run_efficiency()
