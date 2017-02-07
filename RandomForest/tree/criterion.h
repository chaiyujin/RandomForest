#pragma once
#ifndef __TREE_CRITERION_H__
#define __TREE_CRITERION_H__

#include "feature_label.h"
#include <vector>

namespace Yuki {

	
	// the class to calculate the impurity
	class Criterion {
	public:

		Criterion(std::vector<Tuple *>& data, const DTParam &param);

		virtual ~Criterion() {}

		double impurity();
		std::pair<double, double> children_impurity();

		// sort features by dim
		void sort(int dim);
		// update the split position
		void update(int new_pos);

	protected:
		// tuples for current node
		std::vector<Tuple *> &tuples;
		// global param, passed.
		const DTParam &param;
		// split position, [0, pos - 1] in left and [pos, size - 1] in right
		int pos;
		// best improvment
		double best_improvment;
		int best_pos;

		// for impurity
		double sq_sum_total;
		std::vector<double> sum_total;
		double weighted_n_samples;


		// cache
		double impurity_cache;
		std::pair<double, double> ch_impurity_cache;

		// reset
		void reset();
	};
}

#endif  // !__TREE_CRITERION_H__