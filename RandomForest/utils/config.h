#pragma once
#ifndef __YUKI_CONFIG_H__
#define __YUKI_CONFIG_H__

#include "log.h"
#include <map>
#include <string>

namespace Yuki {
	class Config {
	public:
		// initialize from config file
		Config(const char *file_path);

		// get the value of "key", write into v
		template <class T>
		void get(std::string key, T &v) {
			std::map<std::string, float>::iterator it;
			it = values.find(key);
			if (it != values.end()) {
				v = (T)it->second;
			}
		}
	private:
		std::map<std::string, float> values;
	};
}

#endif  // !__YUKI_CONFIG_H__