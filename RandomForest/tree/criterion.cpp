#include "criterion.h"
#include <algorithm>

namespace Yuki {

	Criterion::Criterion(DataSet& data, const Param &param, double all_samples_weight)
		: tuples(data), param(param), weighted_all_samples(all_samples_weight),
		  sum_total(param.label_size()), sum_left(param.label_size()), sum_right(param.label_size()) {
		// default weight -1, means no weight.
		if (weighted_all_samples < 0) weighted_all_samples = tuples.size();

		init();
	}

	void Criterion::init() {
		reset();
		
		// zeros total sum
		memset(sum_total.data(), 0, sizeof(double) * sum_total.size());
		sq_sum_total = 0.f;
		weighted_n_total = 0.f;

		// accumulate for all
		Range(i, tuples.size()) {
			double w = tuples[i]->weight;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				double w_y_ik = w * y_ik;
				// total
				sum_total[k] += w_y_ik;
				sq_sum_total += w_y_ik * y_ik;
			}
			weighted_n_total += w;
		}
	}

	void Criterion::reset() {
		// reset the position
		pos = 0;

		// reset the sum
		memset(sum_left.data(), 0, sizeof(double) * sum_left.size());
		memset(sum_right.data(), 0, sizeof(double) * sum_right.size());
		weighted_n_left = weighted_n_right = 0;
	}

	void Criterion::sort(int dim) {
		std::sort(tuples.begin(), tuples.end(), TupleSorter(dim, param.mask()));
		reset();
	}

	double Criterion::impurity() {
		double ret = sq_sum_total / weighted_n_total;

		Range(k, param.label_size()) {
			ret -= sqr(sum_total[k] / weighted_n_total);
		}

		ret /= param.label_size();

		return ret;
	}

	std::pair<double, double> Criterion::children_impurity() {
		std::pair<double, double> ret;

		double sq_sum_left = 0;
		double sq_sum_right = 0;

		Range(i, pos) {
			double w = tuples[i]->weight;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				sq_sum_left += w * sqr(y_ik);
			}
		}
		sq_sum_right = sq_sum_total - sq_sum_left;
		
		ret.first  = sq_sum_left  / weighted_n_left;
		ret.second = sq_sum_right / weighted_n_right;

		Range(k, param.label_size()) {
			ret.first  -= sqr(sum_left[k]  / weighted_n_left);
			ret.second -= sqr(sum_right[k] / weighted_n_right);
		}

		ret.first /= param.label_size();
		ret.second /= param.label_size();

		return ret;
	}

	void Criterion::update(int new_pos) {
		if (new_pos == pos) return;

		int f, l, r;
		if (pos < new_pos) { l = pos; r = new_pos; f = 1; }
		else			   { l = new_pos; r = pos; f = -1; }

		// update left part
		for (int i = l; i < r; ++i) {
			double w = tuples[i]->weight;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				double w_y_ik = w * y_ik;
				sum_left[k] += w_y_ik * f;
			}
			weighted_n_left += w * f;
		}

		// calc right part
		weighted_n_right = weighted_n_total - weighted_n_left;
		Range(k, param.label_size()) {
			sum_right[k] = sum_total[k] - sum_left[k];
		}

		// update success
		pos = new_pos;
	}

	double Criterion::impurity_improvement() {
		double imp = impurity();
		std::pair<double, double> ch_imp = children_impurity();
		return ((weighted_n_total / weighted_all_samples) *
				(imp - (weighted_n_left  / weighted_n_total * ch_imp.first)
					 - (weighted_n_right / weighted_n_total * ch_imp.second)));
	}

	double Criterion::proxy_impurity_improvement() const {
		double proxy_left = 0;
		double proxy_right = 0;

		Range(k, param.label_size()) {
			proxy_left += sqr(sum_left[k]);
			proxy_right += sqr(sum_right[k]);
		}

		return (proxy_left / weighted_n_left + 
				proxy_right / weighted_n_right);
	}
}