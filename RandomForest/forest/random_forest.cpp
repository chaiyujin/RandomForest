#include "random_forest.h"

namespace Yuki {

	RandomForest::RandomForest(const char *config)
		: param(config) {
		
		// construct several trees
#pragma omp parallel for
		Range(i, param.trees()) {
			// random pick several features for tree
			Mask mask;
			mask.resize(param.feature_size());
			Range(k, mask.size()) mask[k] = false;

			Random random;
			Range(k, param.tree_feature_size()) {
				do {
					int dim = random.random<int>(param.feature_size());
					if (mask[dim] == false) {
						mask[dim] = true;
						break;
					}
				} while (true);
			}

			// copy parameters and set mask
			Param tree_param(param);
			tree_param.set_mask(mask);

			// construct a new tree
			trees.emplace_back(tree_param);
		}

	}

	bool RandomForest::fit(DataSet &data_set) {
		bool succ = true;

#pragma omp parallel for schedule(dynamic)
		Range(i, param.trees()) {
			// to-do bootstrap
			// directly use all tuples
			bool tree_succ = trees[i].fit(data_set);
#pragma omp critical
			succ &= tree_succ;
		}
		
		return true;
	}

	DLabel RandomForest::predict(const DFeature &query) {
		DLabel res;
		res.zeros(param.label_size());

#pragma omp parallel for schedule(dynamic)
		Range(i, param.trees()) {
			DLabel tree_res = trees[i].predict(query);
#pragma omp critical
			res += tree_res;
		}

		// average
		res *= 1.f / (float)trees.size();

		return res;
	}

}