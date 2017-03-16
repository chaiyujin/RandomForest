#include "splitter.h"

namespace Yuki {
	Splitter::Splitter(DataSet &data, const Param &param, double all_samples_weight)
		: tuples(data), param(param), best_pos_(-1), best_dim_(-1), 
		  criterion(data, param, all_samples_weight) {
	}

	bool Splitter::split_best(DataSet &set_a, DataSet &set_b) {
		bool found = false;
		double best_proxy = -DBL_MAX;

		// find best test of several iterations
		Range(itr, param.iterations()) {
			// random the dim
			int dim = random.random<int>(param.feature_size());
			SetMask set_mask(param.feature_types());
			if (param.max_category_sets() >= param.min_category_sets() &&
				param.max_category_sets() < param.feature_size()) {
				int sets = random.random<int>(param.max_category_sets() - param.min_category_sets() + 1) + param.min_category_sets();
				if (sets < 2) sets = 2;
				for (size_t i = 0; i < set_mask.size(); ++i) {
					set_mask[i] = (unsigned char)random.random<int>(sets);
				}
			}
			else if (param.max_category_sets() == param.min_category_sets() &&
					 param.max_category_sets() == param.feature_size()) {
				std::vector<int> flag(param.feature_types());
				for (size_t i = 0; i < flag.size(); ++i) {
					flag.push_back(0);
				}
				for (size_t i = 0; i < set_mask.size(); ++i) {
					int x = random.random<int>(param.feature_types());
					while (flag[x] > 0) {
						x = random.random<int>(param.feature_types());
					}
					flag[x] = 1;
					set_mask[i] = x;
				}
			}
			else {
				for (size_t i = 0; i < set_mask.size(); ++i) {
					set_mask[i] = i;
				}
			}
			// sort by this dim
			criterion.sort(dim, set_mask);
		
			// find the best split
			for (int pos = param.min_leaf_samples();
				 pos + param.min_leaf_samples() <= tuples.size(); ++pos) {

				criterion.update(pos);
				double proxy = criterion.proxy_impurity_improvement();

				if (proxy > best_proxy) {
					best_proxy = proxy;
					best_pos_ = pos;
					best_dim_ = dim;
					best_set_mask_ = set_mask;
					found = true;
				}
			}
		}

		// if sort is stable
		if (found) {
			criterion.sort(best_dim_);
			criterion.update(best_pos_);

			// update split feature
			new (&best_split_feature_) DFeature(tuples[best_pos_]->X);

			new (&set_a) DataSet(tuples.begin(), tuples.begin() + best_pos_);
			new (&set_b) DataSet(tuples.begin() + best_pos_, tuples.end());

			// update the impurity improvement
			best_improvement_ = criterion.impurity_improvement();
		}

		return found;
	}

}