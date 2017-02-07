#include "criterion.h"
#include <algorithm>

namespace Yuki {

	Criterion::Criterion(std::vector<Tuple *>& data, const DTParam &param)
		: tuples(data), param(param) {
		reset();
	}

	void Criterion::reset() {
		// reset the position
		pos = param.min_leaf_samples();
		// reset the members
		impurity_cache = NAN;
		ch_impurity_cache.first = NAN;
		ch_impurity_cache.second = NAN;
		best_improvment = -INFINITY;
		
		// zeros sum
		sum_total.resize(param.label_size());
		memset(sum_total.data(), 0, sizeof(double) * sum_total.size());
		sq_sum_total = 0.f;
		weighted_n_samples = 0.f;

		// accumulate
		for (int i = 0; i < tuples.size(); ++i) {
			double w = tuples[i]->weight;

			for (int k = 0; k < param.label_size(); ++k) {
				double y_ik = tuples[i]->X[k];
				double w_y_ik = w * y_ik;
				sum_total[k] += w_y_ik;
				sq_sum_total += w_y_ik * y_ik;
			}

			weighted_n_samples += w;
		}

		// calculate the impurity of the node
		impurity();
		// calculate the initial split
		children_impurity();
		best_improvment = impurity_cache - ch_impurity_cache.first - ch_impurity_cache.second;
		best_pos = pos;
	}

	void Criterion::sort(int dim) {
		std::sort(tuples.begin(), tuples.end(), TupleSorter(dim));
	}

	double Criterion::impurity() {

		double ret = sq_sum_total / weighted_n_samples;

	}

	
}