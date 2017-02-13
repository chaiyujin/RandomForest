#include "param.h"

namespace Yuki {

	void Param::save(FILE *fp) {
#define FWriteInt(x) \
		fwrite(&x, sizeof(int), 1, fp)

		/* for data */
		FWriteInt(feature_size_);
		FWriteInt(label_size_);

		/* for forest */
		FWriteInt(trees_);
		FWriteInt(tree_feature_size_);
		FWriteInt(bootstrap_);

		/* for train */
		FWriteInt(iterations_);

		/* for tree */
		FWriteInt(type_);
		FWriteInt(max_depth_);
		FWriteInt(max_leaves_);
		FWriteInt(min_leaf_samples_);
		FWriteInt(split_limit_);

		// mask
		int mask_size = (int)mask_.size();
		FWriteInt(mask_size);
		Range(i, mask_.size()) {
			int x = mask_[i];
			FWriteInt(x);
		}

#undef FWriteInt
	}

	void Param::load(Param *param, FILE *fp) {
#define FReadInt(x) \
		fread(&param->x, sizeof(int), 1, fp);

		/* for data */
		FReadInt(feature_size_);
		FReadInt(label_size_);

		/* for forest */
		FReadInt(trees_);
		FReadInt(tree_feature_size_);
		FReadInt(bootstrap_);

		/* for train */
		FReadInt(iterations_);

		/* for tree */
		FReadInt(type_);
		FReadInt(max_depth_);
		FReadInt(max_leaves_);
		FReadInt(min_leaf_samples_);
		FReadInt(split_limit_);

		// mask
		int mask_size;
		fread(&mask_size, sizeof(int), 1, fp);
		param->mask_.resize(mask_size);

		Range(i, mask_size) {
			int x;
			fread(&x, sizeof(int), 1, fp);
			param->mask_[i] = (x > 0);
		}
		
#undef FReadInt
		return;
	}
}