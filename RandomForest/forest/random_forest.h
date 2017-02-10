#pragma once
#ifndef __FOREST_RANDOM_FOREST_H__
#define __FOREST_RANDOM_FOREST_H__

#include "../tree/tree.h"
#include <vector>

namespace Yuki {

	// to-do classification

	// only regression version
	class RandomForest {
	public:
		RandomForest(const char *config);
		RandomForest(const Param &param);

		bool fit(DataSet &data_set);
		DLabel predict(const DFeature &query);

	private:
		void init();

		Param param;
		std::vector<DecisionTree> trees;

	};

}


#endif  // !__FOREST_RANDOM_FOREST_H__