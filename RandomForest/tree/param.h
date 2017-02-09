#pragma once
#ifndef __TREE_PARAM_H__
#define __TREE_PARAM_H__

#include "../utils/config.h"
namespace Yuki {

#define CLASSIFICATION 0
#define REGRESSION 1

	class DTParam {
	public:
		
		DTParam(const char *cfg_file)
			: type_(REGRESSION), // default as regression
			  max_depth_(0), max_leaves_(0), // 0 as not define
			  min_leaf_samples_(1), // at least 1 sample on a leaf
			  split_limit_(0), // no split limit
			  feature_size_(-1), label_size_(-1), // must define
			  iterations_(1) // default 1 iteration 
		{
			Config config(cfg_file);
			config.get("TYPE", type_);
			config.get("MAX_DEPTH", max_depth_);
			config.get("MAX_LEAVES", max_leaves_);
			config.get("MIN_LEAF_SAMPLES", min_leaf_samples_);
			config.get("FEATURE_SIZE", feature_size_);
			config.get("LABEL_SIZE", label_size_);
			config.get("ITERATIONS", iterations_);
			config.get("SPLIT_LIMIT", split_limit_);

			// must config
			CHECK(feature_size_ != -1);
			CHECK(label_size_ != -1);
		}

		int type()				const { return type_; }
		bool is_regression()	const { return type_ == REGRESSION; }
		int max_depth()			const { return max_depth_; }
		int max_leaves()		const { return max_leaves_; }
		int min_leaf_samples()	const { return min_leaf_samples_; }
		int feature_size()		const { return feature_size_; }
		int label_size()		const { return label_size_; }
		int iterations()		const { return iterations_; }
		int split_limit()		const { return split_limit_; }

	private:
		/* for tree */
		int type_;
		int max_depth_;
		int max_leaves_;
		int min_leaf_samples_;
		int split_limit_;
	
		/* for data */
		int feature_size_;
		int label_size_;

		/* for train */
		int iterations_;
	};

}

#endif  // !__TREE_PARAM_H__