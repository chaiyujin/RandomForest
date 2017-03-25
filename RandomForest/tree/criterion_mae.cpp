#include "criterion.h"
#include <algorithm>

namespace Yuki {

	CriterionMAE::CriterionMAE(DataSet& data, const Param &param, double all_samples_weight)
		: tuples(data), param(param), weighted_all_samples(all_samples_weight) {
		// default weight -1, means no weight.
		if (weighted_all_samples < 0) weighted_all_samples = (double)tuples.size();

		init();
	}

	void CriterionMAE::init() {
		reset();

		Range(i, tuples.size()) {
			double w = tuples[i]->weight;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				sum_right[k] += y_ik * w;
			}
			weighted_n_right += w;
		}

		pos = 0;
		weighted_n_left = 0;
		weighted_n_total = weighted_n_right;

		Range(k, param.label_size()) {
			node_average[k] = sum_right[k] / weighted_n_right;
		}

		update(1);
	}

	void CriterionMAE::reset() {
		pos = 0;
		weighted_n_left = weighted_n_right = 0;
		weighted_n_total = 0;

		node_average.clear();
		sum_left.clear();
		sum_right.clear();
		node_average.resize(param.label_size());
		sum_left.resize(param.label_size());
		sum_right.resize(param.label_size());
		memset(sum_left.data(), 0, sizeof(double) * sum_left.size());
		memset(sum_right.data(), 0, sizeof(double) * sum_right.size());
	}

	void CriterionMAE::sort(int dim) {
		std::sort(tuples.begin(), tuples.end(), TupleSorter(dim, param.mask()));
		init();
	}

	void CriterionMAE::sort(int dim, const SetMask &set_mask) {
		std::sort(tuples.begin(), tuples.end(), SetSorter(dim, set_mask, param.mask()));
		init();
	}

	double CriterionMAE::impurity() {
		double ret = 0;
		Range(i, tuples.size()) {
			double w = tuples[i]->weight;
			double err = 0;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				err += abs(y_ik - node_average[k]);
			}
			err /= (double)param.label_size();
			ret += w * err;
		}
		ret /= weighted_n_total;
		return ret;
	}

	std::pair<double, double> CriterionMAE::children_impurity() {
		std::pair<double, double> ret;

		double imp_left = 0;
		double imp_right = 0;

		std::vector<double> left_average(param.label_size());
		std::vector<double> right_average(param.label_size());
		Range(i, param.label_size()) {
			left_average[i] = sum_left[i] / weighted_n_left;
			right_average[i] = sum_right[i] / weighted_n_right;
		}

		Range(i, pos) {
			double w = tuples[i]->weight;
			double err = 0;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				err += abs(y_ik - left_average[k]);
			}
			err /= (double)param.label_size();
			imp_left += w * err;
		}
		imp_left /= weighted_n_left;

		for (int i = pos; i < tuples.size(); ++i) {
			double w = tuples[i]->weight;
			double err = 0;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				err += abs(y_ik - right_average[k]);
			}
			err /= (double)param.label_size();
			imp_right += w * err;
		}
		imp_right /= weighted_n_right;
		
		ret.first = imp_left;
		ret.second = imp_right;

		return ret;
	}

	void CriterionMAE::update(int new_pos) {
		if (new_pos == pos) return;

		int l = pos, r = new_pos;
		bool remove_left = false;
		if (new_pos < pos) {
			l = new_pos; r = pos;
			remove_left = true;
		}
		// update left part
		for (int i = l; i < r; ++i) {
			double w = tuples[i]->weight;
			Range(k, param.label_size()) {
				double y_ik = tuples[i]->Y[k];
				if (remove_left) {
					sum_left[k] -= w * y_ik;
					sum_right[k] += w * y_ik;
					weighted_n_left -= w;
				}
				else {
					sum_left[k] += w * y_ik;
					sum_right[k] -= w * y_ik;
					weighted_n_left += w;
				}
			}
		}

		// calc right part
		weighted_n_right = weighted_n_total - weighted_n_left;

		// update success
		pos = new_pos;
	}

	double CriterionMAE::impurity_improvement() {
		double imp = impurity();
		std::pair<double, double> ch_imp = children_impurity();
		return ((weighted_n_total / weighted_all_samples) *
				(imp - (weighted_n_left  / weighted_n_total * ch_imp.first)
					 - (weighted_n_right / weighted_n_total * ch_imp.second)));
	}

	double CriterionMAE::proxy_impurity_improvement() {
		auto ch_imp = children_impurity();

		return (- weighted_n_right * ch_imp.second
			    - weighted_n_left * ch_imp.first);
	}
}