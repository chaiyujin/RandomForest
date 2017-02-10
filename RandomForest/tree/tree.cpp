#include "tree.h"
#include <queue>
#include <stack>
using std::priority_queue;
using std::stack;

namespace Yuki {

	DecisionTree::DecisionTree(const char *config_file)
		: param(config_file), is_trained(false),
		  root(nullptr) {}

	DecisionTree::DecisionTree(const Param &param)
		: param(param), is_trained(false),
		  root(nullptr) {}

	bool DecisionTree::fit(const DataSet &data_set) {
		// copy the DataSet, it's fast due the pointer representation
		new (&tuples) DataSet(data_set);

		bool ret;
		if (param.max_leaves()) {
			// with the max_leaves limitation, use best first method
			ret = bfs_grow();
		}
		else {
			ret = dfs_grow();
		}

		is_trained = ret;
		return ret;
	}

	// use the priority queue
	bool DecisionTree::bfs_grow() {
		priority_queue<GrowJob *, std::vector<GrowJob *>, GrowJobCMP> job_queue;
		bool succ = true;
		int split_count = 0;
		int max_split = param.max_leaves() - 1;
		
		// first make the root
		root = nullptr;
		{
			GrowJob *job = new GrowJob(tuples, param, nullptr, 0, 0);
			job_queue.push(job);
		}
		// work on the queue until max_leaves or empty queue
		while (!job_queue.empty() && split_count < max_split) {
			// get and pop the top job
			GrowJob *job_ptr = job_queue.top();
			job_queue.pop();
			++split_count;

			std::vector<GrowJob *> children_jobs;
			TreeNode *ret = job_ptr->work(children_jobs);
			if (!ret) {
				error_exit("Error occurs in grow job!");
			}

			if (!root) root = ret; // update root
			Range(i, children_jobs.size()) {
				job_queue.push(children_jobs[i]);
			}

			// delete the job
			delete job_ptr;
		}

		// convert the non-leaf into leaf
		while (!job_queue.empty()) {
			GrowJob *job_ptr = job_queue.top();
			job_queue.pop();

			TreeNode *ret = job_ptr->abandon();
			if (!ret) {
				error_exit("Error occurs in grow job!");
			}
			if (!root) root = ret; // update root

			delete job_ptr;
		}

		//LOG::log("Tree with %d leaves.\n", debug_count_leaves(root));

		return succ;
	}

	bool DecisionTree::dfs_grow() {
		bool succ = true;

		GrowJob *job = new GrowJob(tuples, param, nullptr, 0, 0);
		dfs(job);

		return succ;
	}

	void DecisionTree::dfs(GrowJob *job_ptr) {
		std::vector<GrowJob *> children_jobs;
		TreeNode *ret = job_ptr->work(children_jobs);
		if (!ret) {
			error_exit("Error occurs in grow job!");
		}

		if (!root) root = ret; // update root
		// delete the job
		delete job_ptr;

		Range(i, (int)children_jobs.size()) {
			dfs(children_jobs[i]);
		}
		return;
	}

	DLabel DecisionTree::predict(const DFeature & feature) {
		DLabel ret;

		if (!root) {
			LOG::error("Not trained yet!\n");
			return ret;
		}

		TreeNode *cur = root;
		while (cur != nullptr) {
			if (cur->is_leaf()) {
				new (&ret) DLabel(cur->label());
				cur = cur->child(LEFT_CHILD);
			}
			else {
				cur = cur->which_child(feature);
			}
		}
		
		return ret;
	}

	int DecisionTree::debug_count_leaves(TreeNode *root) {
		int sum = 0;
		if (!root) return sum;
		if (root->is_leaf()) {
			sum = 1;
		}
		else {
			Range(i, CHILDREN_NUM) {
				sum += debug_count_leaves(root->child(i));
			}
		}
		return sum;
	}

	/* Grow Job */

	TreeNode *GrowJob::work(std::vector<GrowJob *> &children_jobs) {
		TreeNode *node = nullptr;

		if ((param.max_depth() > 0 && depth >= param.max_depth()) || // reach the max depth
			tuples.size() < param.split_limit()/* no enough to split */) {
			node = make_leaf();
		}
		else {
			// make a non-leaf
			node = new TreeNode(false, param.mask());
			if (!pre_calced) {
				pre_calc();
			}

			if (pre_calc_succ) {
				// split the node, add new jobs
				// small impurity first (smaller priority)
				GrowJob *left_job = new GrowJob(set_left, param, node, LEFT_CHILD, depth + 1);
				GrowJob *right_job = new GrowJob(set_right, param, node, RIGHT_CHILD, depth + 1);
				left_job->pre_calc();
				right_job->pre_calc();
				children_jobs.emplace_back(left_job);
				children_jobs.emplace_back(right_job);
				
				// split
				node->set_split_feature(pre_calc_best_feature);
				node->set_split_dim(pre_calc_best_dim);
			}
			else {
				node = make_leaf();
			}
		}

		// update the node
		if (node) {
			node->set_depth(depth);
			// update father
			if (father) {
				father->set_child(child_idx, node);
			}
		}

		return node;
	}

	void GrowJob::pre_calc() {
		// new a split
		Splitter splitter(tuples, param);
		pre_calc_succ = splitter.split_best(set_left, set_right);
		if (pre_calc_succ) {
			new (&pre_calc_best_feature) DFeature(splitter.best_split_feature());
			pre_calc_best_dim = splitter.best_dim();
			pre_best_improvement = splitter.best_improvement();
			// bigger improvement better
			priority = -pre_best_improvement;
		}
		pre_calced = true;
	}

	TreeNode *GrowJob::abandon() {
		TreeNode *node = make_leaf();

		if (node) {
			node->set_depth(depth);
			// update father
			if (father) {
				father->set_child(child_idx, node);
			}
		}

		return node;
	}

	TreeNode *GrowJob::make_leaf() {
		// make a leaf
		TreeNode *node = new TreeNode(true, param.mask());

		// average the label
		DLabel label(tuples[0]->Y);
		for (size_t i = 1; i < tuples.size(); ++i) {
			label += tuples[i]->Y;
		}
		float inv_size = 1.f / (float)tuples.size();
		label *= inv_size;
		// set the leaf label
		node->set_label(label);

		return node;
	}

	// save the param, and the tree
	void DecisionTree::save(FILE *fp) {
		/* save param */
		param.save(fp);
		/* save tree */
		TreeNode *node = root;
		
		// save node's father id; if father == self, it's root.
		
		// save node info
		// 1 byte: is_leaf

		// 1 int: depth
		
		// if non_leaf
		//    save feature and dim

		//    else save label


	}

	// load a decision tree
	DecisionTree DecisionTree::load(FILE *fp) {
		DecisionTree tree;
		return tree;
	}
}

