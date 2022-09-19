# Smulation for calibration (Fig. 3)

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import os
import scipy
import scipy.stats

ticksize=17.5
linewidth=2.5
markersize=10
markeredgewidth=4
axissize=17.5

PLOT_DIR = 'plots'

def run_calibration():
		ns = [5, 10, 100, 1000, 10000]
		N = len(ns)
		repeat = 1000

		errs = np.zeros((N, repeat))

		for i in range(N):
			n = ns[i]
			assert(n % 5 == 0)

			for r in range(repeat):
				xs = np.random.uniform(size=n)
				est = scipy.stats.rankdata(xs)

				# applicants in each quintile
				for k in range(1, 6):
					size = n // 5
					est[np.logical_and(est <= k * size, est > (k-1) * size)] = k

				errs[i, r] = np.mean(np.abs(np.ceil(xs * 5) - est))


		(fig, ax) = plt.subplots()
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(ns), np.log(np.mean(errs, axis=1)))

		ax.errorbar(ns, np.mean(errs, axis=1), yerr=np.std(errs, axis=1) / np.sqrt(repeat),
			linewidth=linewidth, marker='x', markersize=markersize, markeredgewidth=markeredgewidth)
		ax.set_xlabel('Number of applicants ' + r'($n$)', fontsize=axissize)
		ax.set_ylabel('Mean error', fontsize=axissize)
		ax.set_xticks(ns)
		ax.set_xticklabels([r'$%d$' % n for n in ns])
		ax.tick_params(axis='x', labelsize=ticksize)
		ax.tick_params(axis='y', labelsize=ticksize)
		ax.set_xscale('log')
		ax.set_yscale('log')
		print('Regression: y = %.3f * x + %.3f' % (slope, intercept))
		plt.savefig('%s/calibration.pdf' % PLOT_DIR, bbox_inches='tight')
		plt.show()

if __name__ == '__main__':
	# np.random.seed(0)
	if not os.path.isdir(PLOT_DIR):
		print('mkdir %s...' % PLOT_DIR)
		os.mkdir(PLOT_DIR)

	run_calibration()
