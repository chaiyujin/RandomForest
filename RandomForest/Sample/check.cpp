#include "check.h"

using namespace std;
using namespace Yuki;

void check_criterion() {
	DTParam param("Sample/config.cfg");
	cout << param.max_depth() << " "
		<< param.max_leaves() << " "
		<< param.min_leaf_samples() << endl;
	cout << param.is_regression() << endl;

	auto tuples = read_data("Sample/X.bin", "Sample/Y.bin", param);
	DataSet tuples_sub;
	for (int i = 0; i < 2000; ++i) {
		tuples_sub.emplace_back(tuples[i]);
	}

	Criterion criterion(tuples_sub, param);
	{
		criterion.sort(0);
		Range(i, 20) {
			cout << tuples_sub[i]->X[0] << endl;
		}
		double p_improve_h = -INFINITY;
		double improve_h = -INFINITY;
		int best_i = -1;
		for (int i = param.min_leaf_samples(); i < tuples_sub.size() - param.min_leaf_samples(); ++i) {
			criterion.update(i);
			double p_improve = criterion.proxy_impurity_improvement();
			double improve = criterion.impurity_improvement();
			if (improve > improve_h) {
				if (!(p_improve > p_improve_h)) {
					cout << "error!\n";
				}
				improve_h = improve;
				p_improve_h = p_improve;
				best_i = i;
			}
		}
		cout << improve_h << endl;
		cout << best_i << endl;
	}

	{
		criterion.sort(9);
		double p_improve_h = -INFINITY;
		double improve_h = -INFINITY;
		int best_i = -1;
		for (int i = param.min_leaf_samples(); i < tuples_sub.size() - param.min_leaf_samples(); ++i) {
			criterion.update(i);
			double p_improve = criterion.proxy_impurity_improvement();
			double improve = criterion.impurity_improvement();
			if (improve > improve_h) {
				if (!(p_improve > p_improve_h)) {
					cout << "error!\n";
				}
				improve_h = improve;
				p_improve_h = p_improve;
				best_i = i;
			}
		}
		cout << best_i << endl;
	}

	{
		criterion.sort(0);
		double p_improve_h = -INFINITY;
		double improve_h = -INFINITY;
		int best_i = -1;
		for (int i = param.min_leaf_samples(); i < tuples_sub.size() - param.min_leaf_samples(); ++i) {
			criterion.update(i);
			double p_improve = criterion.proxy_impurity_improvement();
			double improve = criterion.impurity_improvement();
			if (improve > improve_h) {
				if (!(p_improve > p_improve_h)) {
					cout << "error!\n";
				}
				improve_h = improve;
				p_improve_h = p_improve;
				best_i = i;
			}
		}
		cout << best_i << endl;
	}

	{
		criterion.sort(9);
		double p_improve_h = -INFINITY;
		double improve_h = -INFINITY;
		int best_i = -1;
		for (int i = param.min_leaf_samples(); i < tuples_sub.size() - param.min_leaf_samples(); ++i) {
			criterion.update(i);
			double p_improve = criterion.proxy_impurity_improvement();
			double improve = criterion.impurity_improvement();
			if (improve > improve_h) {
				if (!(p_improve > p_improve_h)) {
					cout << "error!\n";
				}
				improve_h = improve;
				p_improve_h = p_improve;
				best_i = i;
			}
		}
		cout << best_i << endl;
	}


	system("pause");
}

void check_split() {
	DTParam param("Sample/config.cfg");
	cout << param.max_depth() << " "
		<< param.max_leaves() << " "
		<< param.min_leaf_samples() << endl;
	cout << param.is_regression() << endl;

	auto tuples = read_data("Sample/X.bin", "Sample/Y.bin", param);
	DataSet tuples_sub;
	DataSet set_a, set_b;

	for (int i = 0; i < 20; ++i) {
		tuples_sub.emplace_back(tuples[i]);
	}

	Splitter splitter(tuples_sub, param);
	cout << splitter.split_best(set_a, set_b) << endl;

	int dim = splitter.best_dim();
	int pos = splitter.best_pos();

	cout << dim << " " << pos << endl;

	Range(k, param.feature_size()) cout << k << "\t";
	cout << endl;
	cout << endl;

	Range(i, set_a.size()) {
		Range(k, param.feature_size())
			cout << set_a[i]->X[k] << "\t";
		cout << endl;
	}

	cout << endl;

	Range(i, set_b.size()) {
		Range(k, param.feature_size())
			cout << set_b[i]->X[k] << "\t";
		cout << endl;
	}

	cout << endl;

	DFeature s = splitter.best_split_feature();
	Range(k, param.feature_size())
		cout << s[k] << "\t";


	system("pause");
	return;
}

void check_random() {

	Yuki::Random random;

	int count[10];
	memset(count, 0, sizeof(int) * 10);
	Range(i, 1000000) {
		int x = random.random<int>(10);
		++count[x];
		if (x < 0 || x > 9) cout << x << "?\n";
	}
	int sum = 0;
	Range(i, 10) {
		Yuki::LOG::log("%d has shown %d\n", i, count[i]);
		sum += count[i];
	}
	Yuki::LOG::log("Total %d\n", sum);

	system("pause");
}