#include "tree.h"

namespace Yuki {

	DecisionTree::DecisionTree(const char *config_file)
		: param(config_file), is_trained(false) {}

	bool DecisionTree::fit(const DataSet &data_set) {
		// copy the DataSet, it's fast due the pointer representation
		new (&tuples) DataSet(data_set);

		if (param.max_leaves >= 0) {
			// with the max_leaves limitation, use best first method
			bfs_grow();
		}
		else {
			dfs_grow();
		}
	}

}