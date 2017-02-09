#include "random_forest.h"

namespace Yuki {

	RandomForest::RandomForest(const char *config)
		: param(config) {
		
		// construct several trees
#pragma omp parallel for
		Range(i, param.trees()) {
			// random pick several features for tree

		}

	}

}