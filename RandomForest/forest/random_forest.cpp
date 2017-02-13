#include "random_forest.h"

namespace Yuki {

	RandomForest::RandomForest(const char *config)
		: param(config) {
		init();
	}

	RandomForest::RandomForest(const Param &param)
		: param(param) {
		init();
	}

	void RandomForest::init() {
		// construct several trees

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

		// openmp speed up
#pragma omp parallel for schedule(dynamic)
		Range(i, param.trees()) {

			DataSet sub_set;
			if (param.use_bootstrap()) {
				// bootstrap
				std::vector<bool> visit(data_set.size());
				Range(k, visit.size()) visit[k] = false;
				// randomly choose N tuples with putting back
				Random rand;
				Range(j, data_set.size()) {
					int k = rand.random<int>((int)data_set.size());
					if (!visit[k]) {
						visit[k] = true;
						sub_set.emplace_back(data_set[k]);
					}
				}
			}
			else {
				new (&sub_set) DataSet(data_set);
			}

			bool tree_succ = trees[i].fit(sub_set);

#pragma omp critical
			succ &= tree_succ;
		}

		return succ;
	}

	DLabel RandomForest::predict(const DFeature &query) {
		DLabel res;
		res.zeros(param.label_size());

		// openmp speed up
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

	void RandomForest::save(const char *file_name) {
		FILE *fp;
		fopen_s(&fp, file_name, "wb");
		// save param
		param.save(fp);
		// save trees
		Range(i, trees.size()) {
			trees[i].save(fp);
		}
		fclose(fp);
	}
	
	void RandomForest::load(RandomForest *forest, const char *file_name) {
		// open
		FILE *fp;
		fopen_s(&fp, file_name, "rb");
		// load param
		Param::load(&forest->param, fp);

		// load trees
		forest->trees.clear();
		Range(i, forest->param.trees()) {
			forest->trees.push_back(DecisionTree());
			DecisionTree::load(&forest->trees[i], fp);
		}
		fclose(fp);
		return;
	}
}