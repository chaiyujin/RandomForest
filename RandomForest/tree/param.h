#pragma once
#ifndef __TREE_PARAM_H__
#define __TREE_PARAM_H__

#include "../utils/config.h"
#include "../utils/yuki.h"
namespace Yuki {

#define CLASSIFICATION 0
#define REGRESSION 1
	
	typedef std::vector<bool> Mask;
	typedef std::vector<unsigned char> SetMask;
	class Param {
	public:
		
		Param(const char *cfg_file = nullptr)
			: trees_(1), tree_feature_size_(0),
			  bootstrap_(0),
			  type_(REGRESSION), // default as regression
			  max_depth_(0), max_leaves_(0), // 0 as not define
			  min_leaf_samples_(1), // at least 1 sample on a leaf
			  split_limit_(0), // no split limit
			  feature_size_(-1), label_size_(-1), // must define
			  min_category_sets_(2), max_category_sets_(-1),
			  iterations_(1) // default 1 iteration 
		{
			if (cfg_file) {
				Config config(cfg_file);
				config.get("TYPE", type_);
				config.get("MAX_DEPTH", max_depth_);
				config.get("MAX_LEAVES", max_leaves_);
				config.get("MIN_LEAF_SAMPLES", min_leaf_samples_);
				config.get("FEATURE_SIZE", feature_size_);
				config.get("LABEL_SIZE", label_size_);
				config.get("ITERATIONS", iterations_);
				config.get("SPLIT_LIMIT", split_limit_);
				config.get("TREES", trees_);
				config.get("TREE_FEATURE_SIZE", tree_feature_size_);
				config.get("BOOTSTRAP", bootstrap_);
				config.get("FEATURE_TYPES", feature_types_);
				config.get("MAX_CATEGORY_SETS", max_category_sets_);
				config.get("MIN_CATEGORY_SETS", min_category_sets_);
				// must config
				CHECK(feature_size_ != -1);
				CHECK(label_size_ != -1);

				// update tree feature size for forest
				if (tree_feature_size_ == 0 ||
					tree_feature_size_ > feature_size_) {
					tree_feature_size_ = feature_size_;
				}
				split_limit_ = min_leaf_samples_ * 2;

				// default mask as true, all features work
				mask_.resize(feature_size_);
				for (int i = 0; i < mask_.size(); ++i) mask_[i] = true;
			}
		}

		int type()				const { return type_; }
		bool is_regression()	const { return type_ == REGRESSION; }

		int trees()				const { return trees_; }
		int tree_feature_size() const { return tree_feature_size_; }
		bool use_bootstrap()	const { return bootstrap_ > 0; }
		int max_depth()			const { return max_depth_; }
		int max_leaves()		const { return max_leaves_; }
		int min_leaf_samples()	const { return min_leaf_samples_; }
		int feature_size()		const { return feature_size_; }
		int label_size()		const { return label_size_; }
		int iterations()		const { return iterations_; }
		int split_limit()		const { return split_limit_; }
		int feature_types()		const { return feature_types_; }
		int max_category_sets()	const { return max_category_sets_; }
		int min_category_sets() const { return min_category_sets_; }
		const Mask &mask()		const { return mask_; }

		// set mask for tree
		void set_mask(const Mask &m) {
			new (&mask_) Mask(m);
		}

		void save(FILE *fp); 
		static void load(Param *param, FILE *fp);

	private:
		/* for forest */
		int trees_;
		int tree_feature_size_;
		int bootstrap_;

		/* for tree */
		int type_;
		int max_depth_;
		int max_leaves_;
		int min_leaf_samples_;
		int split_limit_;
		// mask
		Mask mask_;
	
		/* for data */
		int feature_size_;
		int label_size_;
		int feature_types_;
		int min_category_sets_;
		int max_category_sets_;

		/* for train */
		int iterations_;
	};

}

#endif  // !__TREE_PARAM_H__