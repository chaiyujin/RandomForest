#pragma once
#ifndef __TREE_TREE_H__
#define __TREE_TREE_H__

#include "feature_label.h"
#include "criterion.h"
#include "splitter.h"

namespace Yuki {
#define LEFT_CHILD  0
#define RIGHT_CHILD 1
#define CHILDREN_NUM 2

	class TreeNode {
	public:
		TreeNode(bool leaf, const Mask &m)
			: is_leaf_(leaf),
			  children(CHILDREN_NUM),
			  mask_(m) {
			Range(i, children.size()) children[i] = nullptr;
		}

		TreeNode *which_child(const DFeature &query_feature) {
			if (less_than(query_feature, split_feature_, split_dim_, mask_))
				return children[LEFT_CHILD];
			else 
				return children[RIGHT_CHILD];
		}

		/* set method */

		void set_depth(int d) { depth_ = d; }
		void set_child(int i, TreeNode *p) { children[i] = p; }
		void set_split_feature(const DFeature &f) { new (&split_feature_) DFeature(f); }
		void set_split_dim(int dim) { split_dim_ = dim; }
		void set_label(const DLabel &l) { new (&label_) DLabel(l); }

		/* get method */

		int depth() const { return depth_; }
		bool is_leaf() const { return is_leaf_; }
		TreeNode *child(int i) { return children[i]; }
		const DLabel &label() const { return label_; }

	private:
		int depth_;
		bool is_leaf_;
		// non-leaf has left and right children
		std::vector<TreeNode *> children;
		// for non-leaf to determine which child
		DFeature split_feature_;
		int      split_dim_;
		// for leaf, the representation label
		DLabel label_;

		// mask
		const Mask &mask_;
	};

	class GrowJob {
	public:
		GrowJob(const DataSet &data_set,
				const Param &param,
				TreeNode *father, int child_idx,
				int depth, double priority = DBL_MAX)
			: priority(priority),
			  tuples(data_set), param(param),
			  father(father), child_idx(child_idx),
			  depth(depth), 
			  pre_calced(false), pre_calc_succ(false) {}

		// return the node made, and the jobs for children if any exists.
		TreeNode *work(std::vector<GrowJob *> &children_jobs);
		// directly make leaf
		TreeNode *abandon();

		double priority; // small first

	private:
		TreeNode *make_leaf();

		DataSet tuples;
		const Param &param;
		// to complete the father pointer
		TreeNode *father;
		int child_idx;
		int depth;

		/* for pre calculate */
		bool pre_calced;
		DataSet set_left, set_right;
		bool pre_calc_succ;
		DFeature pre_calc_best_feature;
		int pre_calc_best_dim;
		double pre_best_improvement;

		void pre_calc();
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
		DecisionTree(const Param &);
		virtual ~DecisionTree() {}

		// simple fit with data
		bool fit(const DataSet &data_set);
		// predict with feature
		DLabel predict(const DFeature & feature);
		
	protected:
		// grow the decision tree with mode
		bool dfs_grow();
		bool bfs_grow();
		void dfs(GrowJob *job);

		// the parameter for the decision tree
		Param param;
		// store the read data
		DataSet tuples;

		// fit with data?
		bool is_trained;

		TreeNode *root;

		/* for debug */
		int debug_count_leaves(TreeNode *root);
	};

}

#endif  // !__TREE_TREE_H__