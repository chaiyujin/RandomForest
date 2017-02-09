#pragma once
#ifndef __TREE_FEATURE_LABEL_H__
#define __TREE_FEATURE_LABEL_H__

#include "../utils/log.h"
#include "param.h"
#include <vector>
#include <algorithm>

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
		Label &operator+=(const Label &l) {
			CHECK(v.size() == l.v.size());
			for (size_t i = 0; i < v.size(); ++i) {
				v[i] += l.v[i];
			}
			return *this;
		}
		Label &operator*=(float f) {
			for (size_t i = 0; i < v.size(); ++i) {
				v[i] *= f;
			}
			return *this;
		}
		bool operator==(const Label &l) {
			for (size_t i = 0; i < v.size(); ++i) {
				if (!equal(v[i], l.v[i])) return false;
			}
			return true;
		}
		T &operator[](int index) {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		const T &operator[](int index) const {
			CHECK(0 <= index && index < v.size());
			return v[index];
		}
		size_t size() const { return v.size(); }
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

		bool operator==(const Feature &f) {
			for (size_t i = 0; i < v.size(); ++i) {
				if (!equal(v[i], f.v[i])) return false;
			}
			return true;
		}

		size_t size() const { return v.size(); }
		template <class U>
		friend bool less_than(const Feature<U> &f0, const Feature<U> &f1, int dim);
	private:
		std::vector<T> v;
	};

	/* the sort is designed for the feature */
	template <class T>
	inline bool less_than(const Feature<T> &f0, const Feature<T> &f1, int dim) {
#define CMP_FEATURE(i) \
		{if (f0[i] < f1[i]) return true;\
		else if (f0[i] > f1[i]) return false;}

		CMP_FEATURE(dim);

		int delta = std::max(dim, (int)f0.size() - dim - 1);
		for (int i = 1; i <= delta; ++i) {
			int j = dim - i;
			if (j >= 0) CMP_FEATURE(j);
			j = dim + i;
			if (j < f0.size()) CMP_FEATURE(j);
		}
#undef CMP_FEATURE
		return false;
	}

	/* define the data feature and label */
	typedef Feature<int> DFeature;
	typedef Label<float> DLabel;

	class Tuple {
	public:
		// basic data
		DFeature X;
		DLabel   Y;
		
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
		bool operator()(Tuple * const &t0, Tuple * const &t1) {
			return less_than(t0->X, t1->X, dim);
		}
	private:
		int dim;
	};

	typedef std::vector<Tuple *> DataSet;

	DataSet read_data(
		const char *feat_file_name,
		const char *label_file_name,
		const DTParam &param);
	
}

#endif  // !__TREE_FEATURE_LABEL_H__