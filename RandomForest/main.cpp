#include "forest/random_forest.h"
#include "utils/timer.h"
#include <iostream>
using namespace std;
using namespace Yuki;

#include "Sample/check.h"

int main(int argc, char **argv) {
	
	check_load_forest();

	//// train
	//{
	//	Param param("config.cfg");
	//	auto tuples = read_data("X.bin", "EX.bin", param);
	//	RandomForest forest(param);
	//	{
	//		StopWatch watch("Fit");
	//		forest.fit(tuples);
	//	}

	//	forest.save("expression.rf");
	//}

	//// predict
	//{
	//	Param param("config.cfg");
	//	auto tuples = read_data("x_input", param);
	//	RandomForest forest;
	//	RandomForest::load(&forest, "expression.rf");

	//	std::vector<DLabel> results;
	//	Range(i, tuples.size()) {
	//		DFeature &query = tuples[i]->X;
	//		results.emplace_back(forest.predict(query));
	//	}

	//	FILE *fp;
	//	fopen_s(&fp, "output.txt", "w");
	//	fprintf(fp, "%d %d\n", results.size(), param.label_size());
	//	Range(i, results.size()) {
	//		Range(k, param.label_size()) {
	//			fprintf(fp, "%f ", results[i][k]);
	//		}
	//		fprintf(fp, "\n");
	//	}
	//	fclose(fp);
	//}

	return 0;
}