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
			: priority(priority),
			  tuples(data_set), param(param),
			  father(father), child_idx(child_idx),
			  depth(depth) {}

		// return the node made, and the jobs for children if any exists.
		TreeNode *work(std::vector<GrowJob *> &children_jobs);

		double priority; // small first

	private:
		TreeNode *make_leaf();

		DataSet tuples;
		const DTParam &param;
		// to complete the father pointer
		TreeNode *father;
		int child_idx;
		int depth;
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

		// template for grow, both bfs and dfs
		template <class T>
		bool grow(T job_queue, int max_leaves) {
			bool succ = true;
			int leaves_count = 0;

			// first make the root
			root = nullptr;
			{
				GrowJob *job = new GrowJob(tuples, param, nullptr, 0, 0);
				job_queue.push(job);
			}
			// work on the queue until max_leaves or empty queue
			while (!job_queue.empty() &&
				   (!max_leaves || leaves_count < max_leaves)) {
				// get and pop the top job
				GrowJob *job_ptr = job_queue.top();
				job_queue.pop();

				std::vector<GrowJob *> children_jobs;
				TreeNode *ret = job_ptr->work(children_jobs);
				if (!ret) {
					error_exit("Error occurs in grow job!");
				}
				if (!root) root = ret; // update root
				Range(i, children_jobs.size()) {
					job_queue.push(children_jobs[i]);
				}
				if (children_jobs.size() == 0) {
					++leaves_count;
				}

				// delete the job
				delete job_ptr;
			}

			return succ;
		}
	};

	class DecisionTreeRegression : public DecisionTree {

	};
}

#endif  // !__TREE_TREE_H__