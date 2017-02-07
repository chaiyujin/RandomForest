#pragma once
#ifndef __TREE_FEATURE_LABEL_H__
#define __TREE_FEATURE_LABEL_H__

#include "../utils/log.h"
#include "param.h"
#include <vector>

/* Feature and label should change according to data */
namespace Yuki {

	template <class T>
	class Label {
	public:
		Label() {}
		size_t read(FILE *fp, int size) {
			v.resize(size);
			return fread(v.data(), sizeof(T), size, fp);
		}
		T &operator[](int index) {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		const T &operator[](int index) const {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
	private:
		std::vector<T> v;
	};

	template <class T>
	class Feature {
	public:
		Feature() {}
		size_t read(FILE *fp, int size) {
			v.resize(size);
			return fread(v.data(), sizeof(T), size, fp);
		}
		T &operator[](int index) {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		const T &operator[](int index) const {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
	private:
		std::vector<T> v;
	};

	class Tuple {
	public:
		// basic data
		Feature<float> X;
		Label<int>     Y;
		
		// auxillary info
		size_t  id;
		float   weight; // default as 1.f

		Tuple() : weight(1.f) {}
		
		// no copy, which is a waste of time
		Tuple(const Tuple &) = delete;
		Tuple &operator=(const Tuple &) = delete;
	};

	class TupleSorter {
	public:
		TupleSorter(int i) : dim(i) {}

		bool operator()(const Tuple *& t0, const Tuple *& t1) {
			return t0->X[dim] < t1->X[dim];
		}

	private:
		int dim;
	};


	std::vector<Tuple *> read_data(
		const char *feat_file_name,
		const char *label_file_name,
		const DTParam &param);
	
}

#endif  // !__TREE_FEATURE_LABEL_H__