#include "splitter.h"

namespace Yuki {
	Splitter::Splitter(DataSet &data, const DTParam &param, double all_samples_weight)
		: tuples(data), param(param), best_pos_(-1), best_dim_(-1), 
		  criterion(data, param, all_samples_weight) {
	}

	bool Splitter::split_best(DataSet &set_a, DataSet &set_b, std::pair<double, double> &child_impurity) {
		bool found = false;
		double best_proxy = -DBL_MAX;

		// find best test of several iterations
		Range(itr, param.iterations()) {
			// random the dim
			int dim = random.random<int>(param.feature_size());
			// sort by this dim
			criterion.sort(dim);
		
			// find the best split
			for (int pos = 1;
				 pos + 1 <= tuples.size(); ++pos) {

				criterion.update(pos);
				double proxy = criterion.proxy_impurity_improvement();
				if (proxy > best_proxy) {
					best_proxy = proxy;
					best_pos_ = pos;
					best_dim_ = dim;
					found = true;

					// update split feature
					new (&best_split_feature_) DFeature(tuples[pos]->X);

					// if sort is not stable
					new (&set_a) DataSet(tuples.begin(), tuples.begin() + pos);
					new (&set_b) DataSet(tuples.begin() + pos, tuples.end());

					// update the child impurity
					child_impurity = criterion.children_impurity_cache();
				}

			}
		}

		// if sort is stable
		/*if (found) {
		}*/

		return found;
	}

}