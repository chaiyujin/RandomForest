#pragma once
#ifndef __TREE_TREE_H__
#define __TREE_TREE_H__

#include "feature_label.h"
#include "criterion.h"
#include "splitter.h"

namespace Yuki {
#define LEFT_CHILD  0
#define RIGHT_CHILD 1

	class TreeNode {
	public:
		TreeNode(bool leaf)
			: is_leaf_(leaf),
			  left_child_(nullptr), right_child_(nullptr) {}

		TreeNode *which_child(const DFeature &query_feature) {
			if (less_than(query_feature, split_feature_, split_dim_))
				return left_child_;
			else 
				return right_child_;
		}

		/* set method */

		void set_depth(int d) { depth_ = d; }
		void set_left_child(TreeNode *p) { left_child_ = p; }
		void set_right_child(TreeNode *p) { right_child_ = p; }
		void set_split_feature(const DFeature &f) { new (&split_feature_) DFeature(f); }
		void set_split_dim(int dim) { split_dim_ = dim; }
		void set_label(const DLabel &l) { new (&label_) DLabel(l); }

		/* get method */

		int depth() const { return depth_; }
		bool is_leaf() const { return is_leaf_; }
		TreeNode *left_child() { return left_child_; }
		TreeNode *right_child() { return right_child_; }
		//const DFeature &split_feature() { return split_feature_; }
		//int split_dim() { return split_dim_; }
		const DLabel &label() const { return label_; }

	private:
		int depth_;
		bool is_leaf_;
		// non-leaf has left and right children
		TreeNode *left_child_, *right_child_;
		// for non-leaf to determine which child
		DFeature split_feature_;
		int      split_dim_;
		// for leaf, the representation label
		DLabel label_;
	};

	class GrowJob {
	public:
		GrowJob(const DataSet &data_set,
				const DTParam &param,
				TreeNode *father, int child_idx,
				int depth, double priority = DBL_MAX)
			: tuples(data_set), param(param),
			  father(father), child_idx(child_idx),
			  depth(depth), priority(priority) {}

		// return the node made, and the jobs for children if any exists.
		TreeNode *work(std::vector<GrowJob *> &children_jobs);

		double priority; // small first

	private:
		TreeNode *make_leaf();

		int depth;
		DataSet tuples;
		const DTParam &param;

		// to complete the father pointer
		TreeNode *father;
		int child_idx;
	};

	struct GrowJobCMP {
	public:
		bool operator()(GrowJob * const &j0, GrowJob * const &j1) {
			return j1->priority < j0->priority;
		}
	};


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

		TreeNode *root;
	};

	class DecisionTreeRegression : public DecisionTree {

	};
}

#endif  // !__TREE_TREE_H__