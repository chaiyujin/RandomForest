#pragma once
#ifndef __TREE_SPLITTER_H__
#define __TREE_SPLITTER_H__

#include "../utils/random.h"
#include "criterion.h"

namespace Yuki {

	class Splitter {
	public:
		Splitter(DataSet &data, const DTParam &param, double all_samples_weight = -1);
		virtual ~Splitter() {}

		// to-do better for un-stable sorting
		// return true if successfully split, the result will be written into set_a and set_b
		bool split_best(DataSet &set_a, DataSet &set_b, std::pair<double, double> &child_impurity);

		int best_dim() { return best_dim_; }
		int best_pos() { return best_pos_; }

		DFeature best_split_feature() { return best_split_feature_; }

	private:
		DataSet &tuples;
		const DTParam &param;
		int best_pos_;
		int best_dim_;
		DFeature best_split_feature_;

		Criterion criterion;
		Random random;

	};
	
}

#endif  // !__TREE_SPLITTER_H__