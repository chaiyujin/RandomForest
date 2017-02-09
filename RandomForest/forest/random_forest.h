#pragma once
#ifndef __FOREST_RANDOM_FOREST_H__
#define __FOREST_RANDOM_FOREST_H__

#include "../tree/tree.h"
#include <vector>

namespace Yuki {

	class RandomForest {
	public:
		RandomForest(const char *config);

	private:
		
		Param param;
		std::vector<DecisionTree> trees;

	};

}


#endif  // !__FOREST_RANDOM_FOREST_H__