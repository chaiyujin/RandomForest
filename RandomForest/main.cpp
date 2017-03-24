#include "forest/random_forest.h"
#include "utils/timer.h"
#include <iostream>
#include <string>
using namespace std;
using namespace Yuki;

int main(int argc, char **argv) {

	// train
	{
		string X_path = "X.bin";
		string Y_path = "EX.bin";
		string save_path = "expression.rf";
		if (argc > 1) X_path = argv[1];
		if (argc > 2) Y_path = argv[2];
		if (argc > 3) save_path = argv[3];

		Param param("config.cfg");
		auto tuples = read_data(X_path.c_str(), Y_path.c_str(), param);
		RandomForest forest(param);
		{
			StopWatch watch("Fit");
			forest.fit(tuples);
		}

		forest.save(save_path.c_str());
	}

	// predict
	/*{
		Param param("config.cfg");
		auto tuples = read_data("x_input", param);
		RandomForest forest;
		RandomForest::load(&forest, "expression.rf");

		std::vector<DLabel> results;
		{
			StopWatch watch("Predict");
			Range(i, tuples.size()) {
				DFeature &query = tuples[i]->X;
				results.emplace_back(forest.predict(query));
			}
		}

		FILE *fp;
		fopen_s(&fp, "output.txt", "w");
		fprintf(fp, "%d %d\n", results.size(), param.label_size());
		Range(i, results.size()) {
			Range(k, param.label_size()) {
				fprintf(fp, "%f ", results[i][k]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}*/

	return 0;
}