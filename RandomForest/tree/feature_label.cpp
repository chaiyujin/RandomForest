#include "feature_label.h"
#include <fstream>

namespace Yuki {

	DataSet read_data(
		const char *feat_file_name,
		const char *label_file_name,
		const Param &param) {

		LOG::log("Reading data...\n\t"
				 "X from: %s\n\t"  "Y from: %s\n", 
				 feat_file_name, label_file_name);

		DataSet ret;

		if (!std::ifstream(feat_file_name) || 
			!std::ifstream(label_file_name)) {
			error_exit("Cannot find data file!");
		}

		FILE *x_fp, *y_fp;
		fopen_s(&x_fp, feat_file_name, "rb");
		fopen_s(&y_fp, label_file_name, "rb");

		// begin to read
		int tuple_cnt = 0;
		while (true) {
			Tuple *tuple = new Tuple();
			size_t x_size = tuple->X.read(x_fp, param.feature_size());
			size_t y_size = tuple->Y.read(y_fp, param.label_size());

			if (x_size != param.feature_size() ||
				y_size != param.label_size()) {
				// delete the un-used pointer
				delete tuple;
				if (x_size || y_size) {
					// bad situation
					error_exit("Data size is not correct!");
				}
				// eof
				break;
			}
			
			tuple->id = tuple_cnt++;
			// push back the useful data
			ret.emplace_back(tuple);
		}

		fclose(x_fp);
		fclose(y_fp);

		LOG::log("All %d tuples.\n", ret.size());

		return ret;
	}

	DataSet read_data(
		const char *feat_file_name,
		const Param &param) {

		LOG::log("Reading data...\n\t"
			"X from: %s\n",
			feat_file_name);

		DataSet ret;

		if (!std::ifstream(feat_file_name)) {
			error_exit("Cannot find data file!");
		}

		FILE *x_fp;
		fopen_s(&x_fp, feat_file_name, "rb");

		// begin to read
		int tuple_cnt = 0;
		while (true) {
			Tuple *tuple = new Tuple();
			size_t x_size = tuple->X.read(x_fp, param.feature_size());

			if (x_size != param.feature_size()) {
				// delete the un-used pointer
				delete tuple;
				if (x_size) {
					// bad situation
					error_exit("Data size is not correct!");
				}
				// eof
				break;
			}

			tuple->id = tuple_cnt++;
			// push back the useful data
			ret.emplace_back(tuple);
		}

		fclose(x_fp);

		LOG::log("All %d tuples.\n", ret.size());

		return ret;
	}

}