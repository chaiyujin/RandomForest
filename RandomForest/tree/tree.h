#pragma once
#ifndef __TREE_TREE_H__
#define __TREE_TREE_H__

#include "feature_label.h"
#include "criterion.h"
#include "splitter.h"

namespace Yuki {

	class DecisionTree {
	public:
		DecisionTree(const char *config_file);
		virtual ~DecisionTree() {}

		// simple fit with data
		bool fit(const DataSet &data_set);
		// predict with feature
		DLabel predict(const DFeature & feature);
		
	protected:
		// grow the decision tree with mode
		bool dfs_grow();
		bool bfs_grow();

		// the parameter for the decision tree
		DTParam param;
		// store the read data
		DataSet tuples;

		// fit with data?
		bool is_trained;
	};

	class DecisionTreeRegression : public DecisionTree {

	};
}

#endif  // !__TREE_TREE_H__