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
		int leaves_count = 0;
		int max_leaves = param.max_leaves();

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

		LOG::log("Tree with %d leaves.\n", leaves_count);

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
			// new a split
			Splitter splitter(tuples, param);

			DataSet set_left, set_right;
			std::pair<double, double> ch_impurity;
			if (splitter.split_best(set_left, set_right, ch_impurity)) {
				// split the node, add new jobs
				// small impurity first (smaller priority)
				GrowJob *left_job = new GrowJob(set_left, param, node, LEFT_CHILD, depth + 1, ch_impurity.first);
				GrowJob *right_job = new GrowJob(set_right, param, node, RIGHT_CHILD, depth + 1, ch_impurity.second);
				children_jobs.emplace_back(left_job);
				children_jobs.emplace_back(right_job);
				
				// split
				node->set_split_feature(splitter.best_split_feature());
				node->set_split_dim(splitter.best_dim());
			}
			else {
				node = make_leaf();
			}
		}

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
}

