#include "config.h"
#include <fstream>

namespace Yuki {
	Config::Config(const char *file_path) {
		std::ifstream fin(file_path);
		if (!fin.is_open()) {
			error_exit("Cannot find config file!");
		}

		std::string s;
		float val;
		while (fin >> s) {
			if (s[0] == '#') {
				std::getline(fin, s);
				continue;
			}
			fin >> val;
			values[s] = val;
			std::getline(fin, s);
		}
	}
}