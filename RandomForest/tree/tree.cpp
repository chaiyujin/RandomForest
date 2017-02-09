#include "tree.h"
#include <queue>
using std::priority_queue;

namespace Yuki {

	DecisionTree::DecisionTree(const char *config_file)
		: param(config_file), is_trained(false),
		  root(nullptr) {}

	bool DecisionTree::fit(const DataSet &data_set) {
		// copy the DataSet, it's fast due the pointer representation
		new (&tuples) DataSet(data_set);

		bool ret;
		if (param.max_leaves() >= 0) {
			// with the max_leaves limitation, use best first method
			ret = bfs_grow();
		}
		else {
			//ret = dfs_grow();
		}

		is_trained = ret;
		return ret;
	}

	// use the priority queue
	bool DecisionTree::bfs_grow() {
		bool succ = true;
		int leaves_count = 0;

		priority_queue<GrowJob *, std::vector<GrowJob *>, GrowJobCMP> job_queue;

		// first make the root
		{
			GrowJob job(tuples, param, nullptr, 0, 0);
			std::vector<GrowJob *> children_jobs;
			// make root
			root = job.work(children_jobs);
			// insert the children job into queue
			if (children_jobs.size()) {
				Range(i, children_jobs.size()) {
					job_queue.push(children_jobs[i]);
				}
			}
			else {
				++leaves_count;
				LOG::warning("Too few data tuples.\n");
			}
		}
		// work on the queue until max_leaves or empty queue
		while (!job_queue.empty() && leaves_count < param.max_leaves()) {
			// get and pop the top job
			GrowJob *job_ptr = job_queue.top();
			job_queue.pop();

			// debug
			LOG::log("%priority: %f\n", job_ptr->priority);

			std::vector<GrowJob *> children_jobs;
			TreeNode *ret = job_ptr->work(children_jobs);
			if (!ret) {
				error_exit("Error occurs in grow job!");
			}
			Range(i, children_jobs.size()) {
				job_queue.push(children_jobs[i]);
			}

			// delete the job
			delete job_ptr;
		}

		return succ;
	}

	// to-do implement dfs grow method
	bool DecisionTree::dfs_grow() {
		bool succ = true;

		return succ;
	}


	/* Grow Job */

	TreeNode *GrowJob::work(std::vector<GrowJob *> &children_jobs) {
		TreeNode *node = nullptr;

		if (depth >= param.max_depth() || // reach the max depth
			tuples.size() < param.split_limit()/* no enough to split */) { 
			node = make_leaf();
		}
		else {
			// make a non-leaf
			node = new TreeNode(false);
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

		// update father
		if (father) {
			if (child_idx == 0)
				father->set_left_child(node);
			else if (child_idx == 1)
				father->set_right_child(node);
		}

		return node;
	}

	TreeNode *GrowJob::make_leaf() {
		// make a leaf
		TreeNode *node = new TreeNode(true);

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