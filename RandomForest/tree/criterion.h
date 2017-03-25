#pragma once
#ifndef __TREE_CRITERION_H__
#define __TREE_CRITERION_H__

#include "../utils/yuki.h"
#include "feature_label.h"
#include <vector>

namespace Yuki {

	class Criterion {
	public:
		Criterion() {}
		virtual ~Criterion() {}

		virtual void sort(int dim) = 0;
		virtual void sort(int dim, const SetMask &set_mask) = 0;
		virtual void update(int new_pos) = 0;
		virtual double proxy_impurity_improvement() = 0;
		virtual double impurity_improvement() = 0;
	};
	
	// the class to calculate the impurity
	class CriterionMSE : public Criterion {
	public:

		CriterionMSE(DataSet &data, const Param &param, double all_samples_weight = -1);

		virtual ~CriterionMSE() {}

		// sort features by dim (for numeric)
		void sort(int dim);
		// sort features by dim and set mask (for category)
		void sort(int dim, const SetMask &set_mask);
		// update the split position, it's better to update in one direction
		void update(int new_pos);

		double proxy_impurity_improvement();
		double impurity_improvement();

		// call once for the real improvement calculation
		double impurity();
		std::pair<double, double> children_impurity();

	protected:
		// tuples for current node
		DataSet &tuples;
		// global param, passed.
		const Param &param;
		// total weight of all samples
		double weighted_all_samples;
		// split position, [0, pos - 1] in left and [pos, size - 1] in right
		int pos;

		// for impurity
		double sq_sum_total;
		std::vector<double> sum_total;
		std::vector<double> sum_left, sum_right;
		double weighted_n_total;
		double weighted_n_left, weighted_n_right;

		// init and reset
		void init();
		void reset();

	};


	class CriterionMAE : public Criterion {
	public:

		CriterionMAE(DataSet &data, const Param &param, double all_samples_weight = -1);

		virtual ~CriterionMAE() {}

		// sort features by dim (for numeric)
		void sort(int dim);
		// sort features by dim and set mask (for category)
		void sort(int dim, const SetMask &set_mask);
		// update the split position, it's better to update in one direction
		void update(int new_pos);

		double proxy_impurity_improvement();
		double impurity_improvement();

		// call once for the real improvement calculation
		double impurity();
		std::pair<double, double> children_impurity();

	protected:
		// tuples for current node
		DataSet &tuples;
		// global param, passed.
		const Param &param;
		// total weight of all samples
		double weighted_all_samples;
		// split position, [0, pos - 1] in left and [pos, size - 1] in right
		int pos;

		// for impurity
		std::vector<double> node_average;
		std::vector<double> sum_left;
		std::vector<double> sum_right;
		double weighted_n_total;
		double weighted_n_left, weighted_n_right;

		// init and reset
		void init();
		void reset();

	};
}

#endif  // !__TREE_CRITERION_H__