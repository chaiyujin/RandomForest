#include "tree/param.h"
#include "tree/feature_label.h"
#include <iostream>
using namespace std;
using namespace Yuki;

int main() {
	
	DTParam param("Sample/config.cfg");
	cout << param.max_depth() << " "
		 << param.max_leaves() << " " 
		 << param.min_leaf_samples() << endl;
	cout << param.is_regression() << endl;

	read_data("Sample/X.bin", "Sample/Y.bin", param);

	system("pause");
	return 0;
}